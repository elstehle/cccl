// WarpMergeSort optimization ablation, single warp (32 threads), keys-only float, sizes 32..384.
// The in-line clock64 breakdown (proto_wms_bench.cu) is perturbed by the stamps' __syncwarps, so
// per-change attribution is done here by ABLATION on the full primitive: a policy-templated mirror
// measured with the clean slope method (no stamps), one policy flipped at a time.
//
// Policies (independently toggled):
//   MP  : MergePath diagonal search. 0 = data-dependent while-loop (baseline). 1 = statically
//         unrolled to ceil(log2(range+1)) fixed iterations (StaticUpperBound-style, see
//         block_run_length_decode.cuh): removes loop overhead and the divergent trip count, so the
//         warp stays converged (no reconvergence stall after the search).
//   NET : thread-local sort. 0 = StableOddEvenSort (odd-even transposition, ~IPT^2/2 compares).
//         1 = Batcher odd-even mergesort network (branchless min/max, fewer compares for large IPT).
//   PAD : shared exchange layout. 0 = blocked keys_shared[IPT*lane+item]. 1 = the same logical
//         layout with one dead word inserted every 32 elements (idx + idx/32) to rotate banks and
//         break the strided STS/LDS conflicts of the blocked layout. All phases index through the
//         same sidx() map, so the search and merge see a consistent view.
//
// Modes: ./proto_wms_opt [correct|lat|all]

#include <cub/util_type.cuh> // cub::Log2

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

constexpr int kWarp   = 32;
constexpr int kRounds = 5;

__device__ int g_sink;

__device__ __forceinline__ float lcg_f(unsigned& s)
{
  s = 1664525u * s + 1013904223u;
  return (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
}

// ------------------------------------------------------------------ shared index map
template <int PAD>
__device__ __forceinline__ int sidx(int i)
{
  return PAD ? (i + (i >> 5)) : i;
}

// ------------------------------------------------------------------ thread-local networks
template <int IPT>
__device__ __forceinline__ void net_oddeven(float (&k)[IPT])
{
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
#pragma unroll
    for (int j = (1 & i); j < IPT - 1; j += 2)
    {
      if (k[j + 1] < k[j])
      {
        const float t = k[j];
        k[j]          = k[j + 1];
        k[j + 1]      = t;
      }
    }
  }
}

template <int NP, int I, int J>
__device__ __forceinline__ void cas(float (&a)[NP])
{
  const float lo = ::fminf(a[I], a[J]);
  const float hi = ::fmaxf(a[I], a[J]);
  a[I]           = lo;
  a[J]           = hi;
}

template <int NP, int LO, int HI, int R, int M>
struct OEMergeLoop
{
  __device__ __forceinline__ static void go(float (&a)[NP])
  {
    if constexpr (LO + R < HI)
    {
      cas<NP, LO, LO + R>(a);
      OEMergeLoop<NP, LO + M, HI, R, M>::go(a);
    }
  }
};

template <int NP, int LO, int N, int R>
struct OEMerge
{
  __device__ __forceinline__ static void go(float (&a)[NP])
  {
    constexpr int M = R * 2;
    if constexpr (M < N)
    {
      OEMerge<NP, LO, N, M>::go(a);
      OEMerge<NP, LO + R, N, M>::go(a);
      OEMergeLoop<NP, LO + R, LO + N - R, R, M>::go(a);
    }
    else
    {
      cas<NP, LO, LO + R>(a);
    }
  }
};

template <int NP, int LO, int N>
struct OESort
{
  __device__ __forceinline__ static void go(float (&a)[NP])
  {
    if constexpr (N > 1)
    {
      constexpr int M = N / 2;
      OESort<NP, LO, M>::go(a);
      OESort<NP, LO + M, N - M>::go(a);
      OEMerge<NP, LO, N, 1>::go(a);
    }
  }
};

__host__ __device__ constexpr int npow2(int n)
{
  int p = 1;
  while (p < n)
  {
    p <<= 1;
  }
  return p;
}

// Batcher odd-even mergesort is only correct for power-of-2 lengths; for other IPT we pad up to
// the next power of two with +INF sentinels (they sink to the top, leaving the first IPT slots
// sorted). A production impl would instead use a per-size optimal network to avoid the pad cost.
template <int IPT>
__device__ __forceinline__ void net_batcher(float (&k)[IPT])
{
  constexpr int NP = npow2(IPT);
  float a[npow2(IPT)];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    a[i] = k[i];
  }
#pragma unroll
  for (int i = IPT; i < NP; ++i)
  {
    a[i] = __int_as_float(0x7f800000); // +inf
  }
  OESort<NP, 0, NP>::go(a);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = a[i];
  }
}

// ------------------------------------------------------------------ MergePath (logical indices via sidx)
template <int PAD>
__device__ __forceinline__ int mp_while(const float* sh, int k1b, int k2b, int c1, int c2, int diag)
{
  int b = diag < c2 ? 0 : diag - c2;
  int e = ::min(diag, c1);
  while (b < e)
  {
    const int mid    = (b + e) >> 1;
    const float key1 = sh[sidx<PAD>(k1b + mid)];
    const float key2 = sh[sidx<PAD>(k2b + diag - 1 - mid)];
    if (key2 < key1)
    {
      e = mid;
    }
    else
    {
      b = mid + 1;
    }
  }
  return b;
}

template <int RANGE, int PAD>
__device__ __forceinline__ int mp_static(const float* sh, int k1b, int k2b, int c1, int c2, int diag)
{
  int b = diag < c2 ? 0 : diag - c2;
  int e = ::min(diag, c1);
#pragma unroll
  for (int i = 0; i <= cub::Log2<RANGE + 1>::VALUE; ++i)
  {
    const int mid    = (b + e) >> 1;
    const bool go    = b < e;
    const float key1 = sh[sidx<PAD>(k1b + mid)];
    const float key2 = sh[sidx<PAD>(k2b + diag - 1 - mid)];
    const bool up    = go && (key2 < key1);
    e                = up ? mid : e;
    b                = (go && !up) ? mid + 1 : b;
  }
  return b;
}

template <int IPT, int PAD>
__device__ __forceinline__ void
serial_merge(const float* sh, int k1b, int k2b, int k1c, int k2c, float (&out)[IPT], float oob)
{
  const int k1e = k1b + k1c;
  const int k2e = k2b + k2c;
  float key1    = k1c != 0 ? sh[sidx<PAD>(k1b)] : oob;
  float key2    = k2c != 0 ? sh[sidx<PAD>(k2b)] : oob;
#pragma unroll
  for (int item = 0; item < IPT; ++item)
  {
    const bool p = (k2b < k2e) && ((k1b >= k1e) || (key2 < key1));
    out[item]    = p ? key2 : key1;
    if (p)
    {
      key2 = sh[sidx<PAD>(++k2b)];
    }
    else
    {
      key1 = sh[sidx<PAD>(++k1b)];
    }
  }
}

// ------------------------------------------------------------------ the templated mirror
// Rounds are compile-time recursion so `size` (= IPT * 2^round) can parameterize the static search.
template <int IPT, int MP, int NET, int PAD, int ROUND>
__device__ __forceinline__ void do_round(float (&keys)[IPT], float* sh)
{
  if constexpr (ROUND < kRounds)
  {
    const int lane       = threadIdx.x & 31;
    constexpr int merged = (1 << ROUND);
    constexpr int size   = IPT * merged;
    const int mask       = (2 << ROUND) - 1;
    __syncwarp();
#pragma unroll
    for (int item = 0; item < IPT; ++item)
    {
      sh[sidx<PAD>(IPT * lane + item)] = keys[item];
    }
    __syncwarp();

    const int first = ~mask & lane;
    const int start = IPT * first;
    const int tig   = mask & lane;
    const int diag  = IPT * tig;
    const int k1b   = start;
    const int k2b   = start + size;

    int pdiag;
    if constexpr (MP)
    {
      pdiag = mp_static<size, PAD>(sh, k1b, k2b, size, size, diag);
    }
    else
    {
      pdiag = mp_while<PAD>(sh, k1b, k2b, size, size, diag);
    }
    const int k1b_loc = k1b + pdiag;
    const int k2b_loc = k2b + diag - pdiag;
    serial_merge<IPT, PAD>(sh, k1b_loc, k2b_loc, (k1b + size) - k1b_loc, (k2b + size) - k2b_loc, keys, keys[0]);

    do_round<IPT, MP, NET, PAD, ROUND + 1>(keys, sh);
  }
}

template <int IPT, int MP, int NET, int PAD>
__device__ __forceinline__ void mirror(float (&keys)[IPT], float* sh)
{
  if constexpr (NET)
  {
    net_batcher(keys);
  }
  else
  {
    net_oddeven(keys);
  }
  do_round<IPT, MP, NET, PAD, 0>(keys, sh);
}

// shared size: blocked TILE, padded TILE + TILE/32, +1 guard for the "read one past" in SerialMerge
template <int IPT, int PAD>
struct ShWords
{
  static constexpr int TILE = kWarp * IPT;
  static constexpr int N    = (PAD ? (TILE + (TILE >> 5)) : TILE) + 2;
};

// ------------------------------------------------------------------ kernels
template <int IPT, int MP, int NET, int PAD>
__global__ void __launch_bounds__(kWarp) correct_k(const float* in, float* out)
{
  __shared__ float sh[ShWords<IPT, PAD>::N];
  const int lane = threadIdx.x & 31;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in[lane * IPT + i];
  }
  mirror<IPT, MP, NET, PAD>(k, sh);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out[lane * IPT + i] = k[i];
  }
}

constexpr int kLatReps = 32;

template <int IPT, int MP, int NET, int PAD, int DO>
__global__ void __launch_bounds__(kWarp) lat_k(unsigned seed0, int chain, long long* out)
{
  __shared__ float sh[ShWords<IPT, PAD>::N];
  const int lane = threadIdx.x & 31;
  unsigned s     = seed0 + (lane + 7) * 2654435761u;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s);
  }
  if (DO)
  {
    mirror<IPT, MP, NET, PAD>(k, sh);
  }
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int rep = 0; rep < kLatReps; ++rep)
  {
    __syncwarp();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll 1
    for (int n = 0; n < chain; ++n)
    {
      unsigned ss = s ^ __float_as_uint(k[0]);
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        k[i] = lcg_f(ss);
      }
      if (DO)
      {
        mirror<IPT, MP, NET, PAD>(k, sh);
      }
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      asm volatile("" ::"f"(k[i]));
    }
    asm volatile("" ::: "memory");
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (lane == 0)
  {
    *out = best;
  }
}

// ------------------------------------------------------------------ host
#define CHECK(call)                                                          \
  do                                                                         \
  {                                                                          \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess)                                                    \
    {                                                                        \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static void fit(const double* x, const double* y, int m, double& a, double& b)
{
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = 0; i < m; ++i)
  {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  b = (m * sxy - sx * sy) / (m * sxx - sx * sx);
  a = (sy - b * sx) / m;
}

template <int IPT, int MP, int NET, int PAD>
bool check_one()
{
  const int N = kWarp * IPT;
  std::vector<float> in(N), out(N), ref;
  unsigned s = 777u + IPT;
  for (int i = 0; i < N; ++i)
  {
    s     = 1664525u * s + 1013904223u;
    in[i] = (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
  }
  ref = in;
  std::sort(ref.begin(), ref.end());
  float *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CHECK(cudaMemcpy(d_in, in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  correct_k<IPT, MP, NET, PAD><<<1, kWarp>>>(d_in, d_out);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < N; ++i)
  {
    if (out[i] != ref[i])
    {
      ok = false;
      break;
    }
  }
  cudaFree(d_in);
  cudaFree(d_out);
  return ok;
}

template <int IPT, int MP, int NET, int PAD>
double slope()
{
  long long* d_out;
  CHECK(cudaMalloc(&d_out, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], yf[5], yg[5];
  for (int i = 0; i < 5; ++i)
  {
    lat_k<IPT, MP, NET, PAD, 1><<<1, kWarp>>>(12345u, chains[i], d_out);
    CHECK(cudaDeviceSynchronize());
    long long c;
    CHECK(cudaMemcpy(&c, d_out, sizeof c, cudaMemcpyDeviceToHost));
    x[i]  = chains[i];
    yf[i] = (double) c;
    lat_k<IPT, MP, NET, PAD, 0><<<1, kWarp>>>(12345u, chains[i], d_out);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&c, d_out, sizeof c, cudaMemcpyDeviceToHost));
    yg[i] = (double) c;
  }
  double a, bf, bg;
  fit(x, yf, 5, a, bf);
  fit(x, yg, 5, a, bg);
  cudaFree(d_out);
  return bf - bg;
}

// baseline (0,0,0), +MP, +NET, +PAD (each alone), and the winning combo MP+NET (no pad)
template <int IPT>
void row()
{
  const double base  = slope<IPT, 0, 0, 0>();
  const double mp    = slope<IPT, 1, 0, 0>();
  const double net   = slope<IPT, 0, 1, 0>();
  const double pad   = slope<IPT, 0, 0, 1>();
  const double mpnet = slope<IPT, 1, 1, 0>();
  printf(
    "  %3d (IPT %2d):  base=%6.0f  +MP=%+5.0f  +NET=%+5.0f  +PAD=%+5.0f  |  MP+NET=%6.0f (%+5.0f, %.0f%%)\n",
    kWarp * IPT,
    IPT,
    base,
    mp - base,
    net - base,
    pad - base,
    mpnet,
    mpnet - base,
    100.0 * (mpnet - base) / base);
}

template <int IPT, int MP, int NET, int PAD>
void res_one(const char* tag)
{
  cudaFuncAttributes a;
  CHECK(cudaFuncGetAttributes(&a, (const void*) lat_k<IPT, MP, NET, PAD, 1>));
  printf("  size %3d (IPT %2d) %-8s regs=%3d  smem=%5zu B  spill=%zu B\n",
         kWarp * IPT,
         IPT,
         tag,
         a.numRegs,
         a.sharedSizeBytes,
         a.localSizeBytes);
}

template <int IPT>
void rrow()
{
  res_one<IPT, 0, 0, 0>("base");
  res_one<IPT, 1, 1, 0>("MP+NET");
}

template <int IPT>
void crow()
{
  const bool b   = check_one<IPT, 0, 0, 0>();
  const bool mp  = check_one<IPT, 1, 0, 0>();
  const bool net = check_one<IPT, 0, 1, 0>();
  const bool pad = check_one<IPT, 0, 0, 1>();
  const bool all = check_one<IPT, 1, 1, 1>();
  const bool ok  = b && mp && net && pad && all;
  printf("  size %3d (IPT %2d): %s   [base %s +MP %s +NET %s +PAD %s ALL %s]\n",
         kWarp * IPT,
         IPT,
         ok ? "PASS" : "FAIL",
         b ? "ok" : "X",
         mp ? "ok" : "X",
         net ? "ok" : "X",
         pad ? "ok" : "X",
         all ? "ok" : "X");
}

#define FOR_SIZES(X) X(1) X(2) X(3) X(4) X(5) X(6) X(8) X(10) X(12)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness (each policy combo vs std::sort) ===\n");
#define C(IPT) crow<IPT>();
    FOR_SIZES(C)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== ablation: mirror slope cyc/call (sort = full - gen); one policy at a time ===\n");
#define R(IPT) row<IPT>();
    FOR_SIZES(R)
  }
  if (mode == "res" || mode == "all")
  {
    printf("\n=== resources: lat kernel, base vs MP+NET ===\n");
#define RR(IPT) rrow<IPT>();
    FOR_SIZES(RR)
  }
  return 0;
}
