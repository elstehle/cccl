// WarpMergeSort — round 3: attacking the MergePath binary search itself. Single warp, keys-only
// float, sizes 32..384 (IPT 1..12). See WARP_MERGE_SORT_RESULTS.md §14 (written from this file).
//
// The search dominates because it is a dependent chain of ~log2(S) shared loads. Ideas seeded by
// the user, analyzed and distilled into variants:
//   * threads cooperating on split points -> hierarchical splitting SERIALIZES levels (sum of
//     logs > log2(S)); all 32 independent searches already run in parallel. What beats log2(S)
//     latency is MULTIPLE INDEPENDENT PROBES PER LANE PER STEP: k-ary search. 3 (or 7) probes at
//     interior points are independent loads -> one shared latency per step, log4/log8(S) steps.
//   * redux/intrinsics -> no fit for 32 distinct split values; usable only as a presorted fast
//     path (max(A) <= min(B) skips a round) — documented, not measured (random inputs).
//   * remembering split points / bounds -> neighbor bounds are circular under lockstep (§11);
//     cross-round bounds are loose. BUT the idea mutates into RANK-AND-SCATTER: each thread
//     already holds its IPT sorted elements in registers; compute each element's rank in the
//     OTHER run (IPT independent searches that PIPELINE -> ~latency of one search) and scatter
//     to own_index + rank. Eliminates SerialMerge entirely; MergePath disappears as a phase.
//
// Variants (V):
//   V0  baseline: while-loop MergePath + odd-even thread sort + serial merge (cub-equivalent)
//   V1  best-known: static BINARY MergePath + Batcher net + prefetch-shift merge (round 1+2)
//   V2  V1 with static 4-ARY MergePath
//   V3  V1 with static 8-ARY MergePath
//   V4  rank-and-scatter rounds (binary rank search, search-major loops) + Batcher net
//   V5  rank-and-scatter rounds (4-ary rank search, search-major) + Batcher net
//   V6  rank-and-scatter, step-major ("transposed") binary search + Batcher net
//   V7  rank-and-scatter, step-major 4-ary search + Batcher net
//
// OUTCOME (B200, CUDA 13.3, see WARP_MERGE_SORT_RESULTS.md §14): V2/V3 lose (probe cost
// exceeds step savings at S<=192); V4..V7 lose for IPT>=2 in every structural arrangement --
// ptxas serializes the "independent" search chains (register-minimizing allocation), full
// unrolling explodes code size, and rolling puts loop overhead on the dependent chain. The
// single win: V4 at IPT==1 (-12% vs V1). Static binary MergePath (V1) remains the recommendation.

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

// ------------------------------------------------------------------ thread-local sorting networks
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

__host__ __device__ constexpr int npow2(int n)
{
  int p = 1;
  while (p < n)
  {
    p <<= 1;
  }
  return p;
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
    a[i] = __int_as_float(0x7f800000);
  }
  OESort<NP, 0, NP>::go(a);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = a[i];
  }
}

// ------------------------------------------------------------------ searches
__device__ __forceinline__ int mp_while(const float* sh, int k1b, int k2b, int c1, int c2, int diag)
{
  int b = diag < c2 ? 0 : diag - c2;
  int e = ::min(diag, c1);
  while (b < e)
  {
    const int mid = (b + e) >> 1;
    if (sh[k2b + diag - 1 - mid] < sh[k1b + mid])
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

__host__ __device__ constexpr int kary_steps(int S, int K)
{
  int s       = 0;
  long long r = 1;
  while (r < (long long) S + 1)
  {
    r *= K;
    ++s;
  }
  return s + 1; // +1 safety for clamped/duplicated probes near convergence
}

// first x in [b,e] with monotone P(x)==true (P(e) implicitly true); K-1 independent probes/step
template <int S, int K, typename Pred>
__device__ __forceinline__ int kary_first(int b, int e, Pred P)
{
  constexpr int STEPS = kary_steps(S, K);
#pragma unroll
  for (int s = 0; s < STEPS; ++s)
  {
    const int w   = e - b;
    const bool go = w > 0;
    int m[K - 1];
    bool p[K - 1];
#pragma unroll
    for (int j = 0; j < K - 1; ++j)
    {
      int mj = b + (w * (j + 1)) / K;
      mj     = ::min(mj, e - 1);
      mj     = ::max(mj, b);
      m[j]   = mj;
      p[j]   = go && P(mj); // independent loads across j -> pipelined
    }
    int nb = m[K - 2] + 1;
    int ne = e;
#pragma unroll
    for (int j = K - 2; j >= 1; --j)
    {
      nb = p[j] ? m[j - 1] + 1 : nb;
      ne = p[j] ? m[j] : ne;
    }
    nb = p[0] ? b : nb;
    ne = p[0] ? m[0] : ne;
    b  = go ? nb : b;
    e  = go ? ne : e;
  }
  return b;
}

// ------------------------------------------------------------------ merges
template <int IPT>
__device__ __forceinline__ void
serial_merge(const float* sh, int k1b, int k2b, int k1c, int k2c, float (&out)[IPT], float oob)
{
  const int k1e = k1b + k1c;
  const int k2e = k2b + k2c;
  float key1    = k1c != 0 ? sh[k1b] : oob;
  float key2    = k2c != 0 ? sh[k2b] : oob;
#pragma unroll
  for (int item = 0; item < IPT; ++item)
  {
    const bool p = (k2b < k2e) && ((k1b >= k1e) || (key2 < key1));
    out[item]    = p ? key2 : key1;
    if (p)
    {
      key2 = sh[++k2b];
    }
    else
    {
      key1 = sh[++k1b];
    }
  }
}

template <int IPT>
__device__ __forceinline__ void
merge_pfshift(const float* sh, int k1b, int k2b, int k1c, int k2c, float (&out)[IPT], float oob)
{
  float A[IPT];
  float B[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    A[i] = (i < k1c) ? sh[k1b + i] : oob;
    B[i] = (i < k2c) ? sh[k2b + i] : oob;
  }
  int na = k1c < IPT ? k1c : IPT;
  int nb = k2c < IPT ? k2c : IPT;
#pragma unroll
  for (int item = 0; item < IPT; ++item)
  {
    const bool takeB = (nb > 0) && ((na == 0) || (B[0] < A[0]));
    out[item]        = takeB ? B[0] : A[0];
#pragma unroll
    for (int i = 0; i < IPT - 1; ++i)
    {
      A[i] = takeB ? A[i] : A[i + 1];
      B[i] = takeB ? B[i + 1] : B[i];
    }
    na -= takeB ? 0 : 1;
    nb -= takeB ? 1 : 0;
  }
}

// ------------------------------------------------------------------ rounds
// V0..V3: MergePath rounds. V4/V5: rank-and-scatter rounds.
template <int IPT, int V, int ROUND>
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
      sh[IPT * lane + item] = keys[item];
    }
    __syncwarp();

    const int first = ~mask & lane;
    const int start = IPT * first;
    const int tig   = mask & lane;

    if constexpr (V <= 3)
    {
      const int diag = IPT * tig;
      const int k1b  = start;
      const int k2b  = start + size;
      int pdiag;
      if constexpr (V == 0)
      {
        pdiag = mp_while(sh, k1b, k2b, size, size, diag);
      }
      else
      {
        constexpr int K = (V == 1) ? 2 : (V == 2) ? 4 : 8;
        const int b0    = diag < size ? 0 : diag - size;
        const int e0    = ::min(diag, size);
        pdiag           = kary_first<size, K>(b0, e0, [&](int x) {
          return sh[k2b + diag - 1 - x] < sh[k1b + x];
        });
      }
      const int k1b_loc = k1b + pdiag;
      const int k2b_loc = k2b + diag - pdiag;
      const int k1c     = (k1b + size) - k1b_loc;
      const int k2c     = (k2b + size) - k2b_loc;
      if constexpr (V == 0)
      {
        serial_merge<IPT>(sh, k1b_loc, k2b_loc, k1c, k2c, keys, keys[0]);
      }
      else
      {
        merge_pfshift<IPT>(sh, k1b_loc, k2b_loc, k1c, k2c, keys, keys[0]);
      }
    }
    else
    {
      // rank-and-scatter: my IPT (sorted, register-resident) elements each rank themselves in
      // the OTHER run; dest = own idx + rank. Stability: run-A elements use lower_bound (count of
      // B strictly less), run-B upper_bound.
      //   V4/V5: search-major loops (one full search per item). MEASURED PITFALL: ptxas minimizes
      //          registers by SERIALIZING the "independent" searches -> IPT x full search latency.
      //   V6/V7: step-major (transposed) loops: all IPT searches advance one step together, so
      //          each step issues all IPT probes as a batch -> pipelining is structural.
      constexpr int K      = (V == 4 || V == 6) ? 2 : 4;
      const bool in_a      = tig < merged;
      const int idx_in_run = (in_a ? tig : tig - merged) * IPT;
      const int other      = start + (in_a ? size : 0);
      int dest[IPT];
      if constexpr (V <= 5)
      {
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          const float a = keys[i];
          const int r   = kary_first<size, K>(0, size, [&](int x) {
            const float o = sh[other + x];
            return in_a ? !(o < a) : (a < o);
          });
          dest[i] = start + idx_in_run + i + r;
        }
      }
      else
      {
        constexpr int STEPS = kary_steps(size, K);
        int lo[IPT];
        int hi[IPT];
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          lo[i] = 0;
          hi[i] = size;
        }
        // ROLLED step loop: full unrolling of STEPS x IPT probe bodies explodes code size
        // (icache thrash measured as a 2-5x blowup); one compact body + loop keeps it resident.
#pragma unroll 1
        for (int s2 = 0; s2 < STEPS; ++s2)
        {
          int m[IPT][K - 1];
          float pr[IPT][K - 1];
          // phase 1: issue ALL probes of this step (independent loads -> one shared latency)
#pragma unroll
          for (int i = 0; i < IPT; ++i)
          {
            const int w = hi[i] - lo[i];
#pragma unroll
            for (int j = 0; j < K - 1; ++j)
            {
              int mj      = lo[i] + (w * (j + 1)) / K;
              mj          = ::min(mj, hi[i] - 1);
              mj          = ::max(mj, lo[i]);
              m[i][j]     = mj;
              pr[i][j]    = sh[other + mj];
            }
          }
          // phase 2: fold predicates, shrink ranges (pure ALU)
#pragma unroll
          for (int i = 0; i < IPT; ++i)
          {
            const bool go = hi[i] > lo[i];
            bool pd[K - 1];
#pragma unroll
            for (int j = 0; j < K - 1; ++j)
            {
              pd[j] = go && (in_a ? !(pr[i][j] < keys[i]) : (keys[i] < pr[i][j]));
            }
            int nb = m[i][K - 2] + 1;
            int ne = hi[i];
#pragma unroll
            for (int j = K - 2; j >= 1; --j)
            {
              nb = pd[j] ? m[i][j - 1] + 1 : nb;
              ne = pd[j] ? m[i][j] : ne;
            }
            nb    = pd[0] ? lo[i] : nb;
            ne    = pd[0] ? m[i][0] : ne;
            lo[i] = go ? nb : lo[i];
            hi[i] = go ? ne : hi[i];
          }
        }
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          dest[i] = start + idx_in_run + i + lo[i];
        }
      }
      __syncwarp(); // all reads done; in-place scatter is safe (values live in registers)
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        sh[dest[i]] = keys[i];
      }
      __syncwarp();
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        keys[i] = sh[IPT * lane + i];
      }
    }

    do_round<IPT, V, ROUND + 1>(keys, sh);
  }
}

template <int IPT, int V>
__device__ __forceinline__ void mirror(float (&keys)[IPT], float* sh)
{
  if constexpr (V == 0)
  {
    net_oddeven(keys);
  }
  else
  {
    net_batcher(keys);
  }
  do_round<IPT, V, 0>(keys, sh);
}

// ------------------------------------------------------------------ kernels
template <int IPT>
struct ShWords
{
  static constexpr int N = kWarp * IPT + 2;
};

template <int IPT, int V>
__global__ void __launch_bounds__(kWarp) correct_k(const float* in, float* out)
{
  __shared__ float sh[ShWords<IPT>::N];
  const int lane = threadIdx.x & 31;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in[lane * IPT + i];
  }
  mirror<IPT, V>(k, sh);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out[lane * IPT + i] = k[i];
  }
}

constexpr int kLatReps = 32;

template <int IPT, int V, int DO>
__global__ void __launch_bounds__(kWarp) lat_k(unsigned seed0, int chain, long long* out)
{
  __shared__ float sh[ShWords<IPT>::N];
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
    mirror<IPT, V>(k, sh);
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
        mirror<IPT, V>(k, sh);
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

template <int IPT, int V>
bool check_one()
{
  const int N = kWarp * IPT;
  std::vector<float> in(N), out(N), ref;
  unsigned s = 777u + IPT * 13 + V;
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
  correct_k<IPT, V><<<1, kWarp>>>(d_in, d_out);
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

template <int IPT, int V>
double slope()
{
  long long* d;
  CHECK(cudaMalloc(&d, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], yf[5], yg[5];
  for (int i = 0; i < 5; ++i)
  {
    lat_k<IPT, V, 1><<<1, kWarp>>>(12345u, chains[i], d);
    CHECK(cudaDeviceSynchronize());
    long long c;
    CHECK(cudaMemcpy(&c, d, sizeof c, cudaMemcpyDeviceToHost));
    x[i]  = chains[i];
    yf[i] = (double) c;
    lat_k<IPT, V, 0><<<1, kWarp>>>(12345u, chains[i], d);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&c, d, sizeof c, cudaMemcpyDeviceToHost));
    yg[i] = (double) c;
  }
  double a, bf, bg;
  fit(x, yf, 5, a, bf);
  fit(x, yg, 5, a, bg);
  cudaFree(d);
  return bf - bg;
}

template <int IPT>
void row()
{
  const double v0 = slope<IPT, 0>();
  const double v1 = slope<IPT, 1>();
  const double v6 = slope<IPT, 6>();
  const double v7 = slope<IPT, 7>();
  printf("  %3d (IPT %2d): base=%5.0f  bin=%5.0f  RS-T-bin=%5.0f  RS-T-4ary=%5.0f   best vs bin: %+5.0f (%.0f%%)\n",
         kWarp * IPT,
         IPT,
         v0,
         v1,
         v6,
         v7,
         std::min(v6, v7) - v1,
         100.0 * (std::min(v6, v7) - v1) / v1);
}

template <int IPT, int V>
void res_one(const char* tag)
{
  cudaFuncAttributes a;
  CHECK(cudaFuncGetAttributes(&a, (const void*) lat_k<IPT, V, 1>));
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
  res_one<IPT, 1>("bin");
  res_one<IPT, 6>("RS-T-bin");
  res_one<IPT, 7>("RS-T-4ary");
}

template <int IPT>
void crow()
{
  const bool a = check_one<IPT, 0>();
  const bool b = check_one<IPT, 1>();
  const bool c = check_one<IPT, 2>();
  const bool d = check_one<IPT, 3>();
  const bool e = check_one<IPT, 4>();
  const bool f = check_one<IPT, 5>();
  const bool g = check_one<IPT, 6>();
  const bool h = check_one<IPT, 7>();
  printf("  size %3d (IPT %2d): %s  [V0 %s V1 %s V2 %s V3 %s V4 %s V5 %s V6 %s V7 %s]\n",
         kWarp * IPT,
         IPT,
         (a && b && c && d && e && f && g && h) ? "PASS" : "FAIL",
         a ? "ok" : "X",
         b ? "ok" : "X",
         c ? "ok" : "X",
         d ? "ok" : "X",
         e ? "ok" : "X",
         f ? "ok" : "X",
         g ? "ok" : "X",
         h ? "ok" : "X");
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
    printf("\n=== correctness vs std::sort ===\n");
#define C(IPT) crow<IPT>();
    FOR_SIZES(C)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== latency: slope cyc/call (sort = full - gen) ===\n");
    printf("  (bin/4ary/8ary = static k-ary MergePath + net + prefetch-shift; RS = rank-and-scatter)\n");
#define R(IPT) row<IPT>();
    FOR_SIZES(R)
  }
  if (mode == "res" || mode == "all")
  {
    printf("\n=== resources ===\n");
#define RR(IPT) rrow<IPT>();
    FOR_SIZES(RR)
  }
  return 0;
}
