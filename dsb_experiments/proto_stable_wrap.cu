// Stable warp sort via the REAL WarpBitonicSort (exp/sub-warp-bitonic-sort) + wrapper helpers,
// vs the hand-rolled stable network from proto_hybrid_block_sort.
//
//   M0 mine        : hand-rolled stable network, separate rank channel (the hybrid prototype's
//                    bootstrap — the thing to beat)
//   M1 cub+packed  : WarpBitonicSort<u64> on a bit-twiddled (key, rank) pack — the "builtin
//                    types + builtin comparator" fast path: 2 shuffles, ONE integer compare
//   M2 cub+wrapper : WarpBitonicSort<{KeyT,rank} struct> with a generic two-call wrapper
//                    comparator — the arbitrary-comparator path
//   M3 cub unstable: WarpBitonicSort<KeyT> as-is — network-quality reference & stability cost
//
// Full warp (LW=32), IPT in {1,2,4,8} (sizes 32..256 = the hybrid bootstrap range), float keys,
// pairs variant carries int values through cub's ValueT channel. Input order = striped position
// (lane + 32*item) — the arrangement-correct rank convention. Stability verified vs
// std::stable_sort on (key, position); M3 checks keys only.
//
// Build: nvcc -std=c++17 -arch=sm_100 -O3 -I<branch>/cub -I<branch>/libcudacxx/include \
//        -I<branch>/thrust proto_stable_wrap.cu
// Modes: ./proto_stable_wrap [correct|lat|thr|all]

#include <cub/warp/warp_bitonic_sort.cuh>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

__device__ int g_dcesink[16];

__device__ __forceinline__ unsigned read_smid()
{
  unsigned x;
  asm("mov.u32 %0, %%smid;" : "=r"(x));
  return x;
}

template <int W>
__device__ __forceinline__ void sink_f(float (&v)[W])
{
  if (read_smid() == (unsigned) -1)
  {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < W; ++i)
    {
      sum += v[i];
    }
    *reinterpret_cast<float*>(g_dcesink) += sum;
  }
}

__device__ __forceinline__ float lcg_f(unsigned& s)
{
  s = 1664525u * s + 1013904223u;
  return (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
}

// order-preserving float <-> u32 twiddle (ascending unsigned order; NaN-free data)
__device__ __forceinline__ unsigned twiddle_in(float f)
{
  const unsigned u = __float_as_uint(f);
  return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
__device__ __forceinline__ float twiddle_out(unsigned u)
{
  u ^= ((u >> 31) ? 0x80000000u : 0xFFFFFFFFu);
  return __uint_as_float(u);
}

struct LessAny
{
  template <typename T>
  __device__ __forceinline__ bool operator()(const T& a, const T& b) const
  {
    return a < b;
  }
};

// generic stable wrapper: (key, rank) struct + two-call lexicographic comparator
struct KR
{
  float k;
  int r;
};
struct StableWrapLess
{
  __device__ __forceinline__ bool operator()(const KR& a, const KR& b) const
  {
    const bool ab = a.k < b.k; // compare_op(a, b)
    const bool ba = b.k < a.k; // compare_op(b, a)  (generic form: no operator== assumed)
    return ab || (!ba && a.r < b.r);
  }
};

// ------------------------------------------------------------------ M0: hand-rolled network
template <int IPT, bool PAIRS>
__device__ __forceinline__ void my_stable_sort(float (&sk)[IPT], int (&sr)[IPT], int (&sv)[IPT])
{
  const int l     = threadIdx.x & 31;
  constexpr int N = 32 * IPT;
#pragma unroll
  for (int stage = 2; stage <= N; stage <<= 1)
  {
#pragma unroll
    for (int j = stage >> 1; j >= 1; j >>= 1)
    {
      if (j >= 32)
      {
        const int jb = j / 32;
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          if ((i & jb) == 0 && (i | jb) < IPT)
          {
            const int i2   = i | jb;
            const int p    = l + 32 * i;
            const bool asc = (p & stage) == 0 || stage == N;
            const bool sw  = (sk[i2] < sk[i]) || (sk[i2] == sk[i] && sr[i2] < sr[i]);
            if (sw == asc)
            {
              float tk = sk[i]; sk[i] = sk[i2]; sk[i2] = tk;
              int tr = sr[i]; sr[i] = sr[i2]; sr[i2] = tr;
              if (PAIRS) { int tv = sv[i]; sv[i] = sv[i2]; sv[i2] = tv; }
            }
          }
        }
      }
      else
      {
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          const float pk   = __shfl_xor_sync(~0u, sk[i], j);
          const int pr     = __shfl_xor_sync(~0u, sr[i], j);
          const int pv     = PAIRS ? __shfl_xor_sync(~0u, sv[i], j) : 0;
          const int p      = l + 32 * i;
          const bool lower = (p & j) == 0;
          const bool asc   = (p & stage) == 0 || stage == N;
          const bool pless = (pk < sk[i]) || (pk == sk[i] && pr < sr[i]);
          const bool take  = (lower == asc) ? pless : !pless;
          if (take)
          {
            sk[i] = pk;
            sr[i] = pr;
            if (PAIRS)
            {
              sv[i] = pv;
            }
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------ dispatch (striped, in-place)
// METHOD: 0 = mine, 1 = cub packed u64, 2 = cub struct wrapper, 3 = cub unstable
template <int IPT, int METHOD, bool PAIRS>
__device__ __forceinline__ void do_sort(float (&k)[IPT], int (&v)[IPT])
{
  const int lane = threadIdx.x & 31;
  if constexpr (METHOD == 0)
  {
    int sr[IPT], sv[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      sr[i] = lane + 32 * i;
      sv[i] = v[i];
    }
    my_stable_sort<IPT, PAIRS>(k, sr, sv);
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      v[i] = sv[i];
    }
  }
  else if constexpr (METHOD == 1)
  {
    unsigned long long a[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      a[i] = ((unsigned long long) twiddle_in(k[i]) << 32) | (unsigned) (lane + 32 * i);
    }
    // NOTE: keys-only Sort must be called on a ValueT=NullType instantiation — on a ValueT!=NullType
    // class it dereferences its internal nullptr values pointer (API-hardening note for the branch)
    if constexpr (PAIRS)
    {
      using BS = cub::detail::WarpBitonicSort<unsigned long long, IPT, 32, int>;
      typename BS::TempStorage ts;
      BS(ts).Sort(a, v, LessAny{});
    }
    else
    {
      using BS = cub::detail::WarpBitonicSort<unsigned long long, IPT, 32>;
      typename BS::TempStorage ts;
      BS(ts).Sort(a, LessAny{});
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = twiddle_out((unsigned) (a[i] >> 32));
    }
  }
  else if constexpr (METHOD == 2)
  {
    KR a[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      a[i] = {k[i], lane + 32 * i};
    }
    if constexpr (PAIRS)
    {
      using BS = cub::detail::WarpBitonicSort<KR, IPT, 32, int>;
      typename BS::TempStorage ts;
      BS(ts).Sort(a, v, StableWrapLess{});
    }
    else
    {
      using BS = cub::detail::WarpBitonicSort<KR, IPT, 32>;
      typename BS::TempStorage ts;
      BS(ts).Sort(a, StableWrapLess{});
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = a[i].k;
    }
  }
  else
  {
    if constexpr (PAIRS)
    {
      using BS = cub::detail::WarpBitonicSort<float, IPT, 32, int>;
      typename BS::TempStorage ts;
      BS(ts).Sort(k, v, LessAny{});
    }
    else
    {
      using BS = cub::detail::WarpBitonicSort<float, IPT, 32>;
      typename BS::TempStorage ts;
      BS(ts).Sort(k, LessAny{});
    }
  }
}

// ------------------------------------------------------------------ kernels
template <int IPT, int METHOD, bool PAIRS>
__global__ void correct_k(const float* ink, const int* inv, float* outk, int* outv)
{
  const int lane = threadIdx.x & 31;
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = ink[lane + 32 * i]; // striped input order
    v[i] = inv[lane + 32 * i];
  }
  do_sort<IPT, METHOD, PAIRS>(k, v);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    outk[lane + 32 * i] = k[i];
    outv[lane + 32 * i] = v[i];
  }
}

constexpr int kLatReps = 32;

template <int IPT, int METHOD, bool PAIRS, int DO>
__global__ void __launch_bounds__(32) lat_k(unsigned seed0, int chain, long long* out)
{
  const int lane = threadIdx.x & 31;
  unsigned s     = seed0 + (lane + 7) * 2654435761u;
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s); // random init once; timed loop is an IN-PLACE chain (all methods oblivious)
    v[i] = i;
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
      k[0] = k[0] + (float) (n & 7); // per-iteration perturbation: non-idempotent RAW chain
      if (DO)
      {
        do_sort<IPT, METHOD, PAIRS>(k, v);
      }
    }
    // consume EVERY output through the unprovable-side-effect sink (partial consumption let the
    // compiler eliminate/hoist the register-only cub sorts: measured -34 cyc / 92 Telem/s)
    float cv[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      cv[i] = k[i] + (PAIRS ? (float) v[i] : 0.f);
    }
    sink_f(cv);
    asm volatile("" ::: "memory");
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (lane == 0)
  {
    *out = best;
  }
}

constexpr int kThrBlock = 128;
constexpr int kThrIters = 200;

template <int IPT, int METHOD, bool PAIRS>
__global__ void __launch_bounds__(kThrBlock) thr_k(int num_iter)
{
  unsigned s = 12345u + (threadIdx.x + 7) * 2654435761u;
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    v[i] = i;
  }
#pragma unroll 1
  for (int iter = 0; iter < num_iter; ++iter)
  {
    unsigned ss = s + iter * 2654435761u;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = lcg_f(ss);
    }
    do_sort<IPT, METHOD, PAIRS>(k, v);
    float cv[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      cv[i] = k[i] + (PAIRS ? (float) v[i] : 0.f); // consume every output
    }
    sink_f(cv);
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

template <int IPT, int METHOD, bool PAIRS>
static bool check_one(unsigned seed, int pattern)
{
  const int N = 32 * IPT;
  std::vector<float> ink(N);
  std::vector<int> inv(N);
  unsigned s = seed;
  for (int i = 0; i < N; ++i)
  {
    s      = 1664525u * s + 1013904223u;
    ink[i] = pattern == 0 ? (float) ((s >> 8) & 0xffffu) : pattern == 1 ? (float) ((s >> 8) % 5) : 1.f;
    inv[i] = i;
  }
  std::vector<std::pair<float, int>> ref(N);
  for (int i = 0; i < N; ++i)
  {
    ref[i] = {ink[i], i};
  }
  std::stable_sort(ref.begin(), ref.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  float *d_ik, *d_ok;
  int *d_iv, *d_ov;
  CHECK(cudaMalloc(&d_ik, N * 4));
  CHECK(cudaMalloc(&d_ok, N * 4));
  CHECK(cudaMalloc(&d_iv, N * 4));
  CHECK(cudaMalloc(&d_ov, N * 4));
  CHECK(cudaMemcpy(d_ik, ink.data(), N * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_iv, inv.data(), N * 4, cudaMemcpyHostToDevice));
  correct_k<IPT, METHOD, PAIRS><<<1, 32>>>(d_ik, d_iv, d_ok, d_ov);
  CHECK(cudaDeviceSynchronize());
  std::vector<float> ok(N);
  std::vector<int> ov(N);
  CHECK(cudaMemcpy(ok.data(), d_ok, N * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ov.data(), d_ov, N * 4, cudaMemcpyDeviceToHost));
  cudaFree(d_ik);
  cudaFree(d_ok);
  cudaFree(d_iv);
  cudaFree(d_ov);
  for (int i = 0; i < N; ++i)
  {
    const bool key_ok = ok[i] == ref[i].first;
    const bool val_ok = !PAIRS || (METHOD == 3) || (ov[i] == ref[i].second); // unstable: keys only
    if (!key_ok || !val_ok)
    {
      printf("      M%d IPT%d pat%d pos %d: got (%g,%d) want (%g,%d)\n",
             METHOD, IPT, pattern, i, ok[i], ov[i], ref[i].first, ref[i].second);
      return false;
    }
  }
  return true;
}

template <int IPT, int METHOD>
static bool check_method()
{
  bool ok = true;
  for (int pat = 0; pat <= 2; ++pat)
  {
    for (unsigned t = 0; t < (pat <= 1 ? 10u : 1u); ++t)
    {
      ok = ok && check_one<IPT, METHOD, true>(4242u + t * 131u + pat * 7919u, pat);
      ok = ok && check_one<IPT, METHOD, false>(4242u + t * 131u + pat * 7919u, pat); // keys-only path too
    }
  }
  return ok;
}

template <int IPT>
static void run_correct()
{
  const bool m0 = check_method<IPT, 0>();
  const bool m1 = check_method<IPT, 1>();
  const bool m2 = check_method<IPT, 2>();
  const bool m3 = check_method<IPT, 3>();
  printf("  size %3d (IPT %d):  mine %s   cub+packed %s   cub+wrapper %s   cub-unstable(keys) %s\n",
         32 * IPT, IPT, m0 ? "PASS" : "FAIL", m1 ? "PASS" : "FAIL", m2 ? "PASS" : "FAIL", m3 ? "PASS" : "FAIL");
}

template <int IPT, int METHOD, bool PAIRS, int DO>
static double slope_raw()
{
  long long* d;
  CHECK(cudaMalloc(&d, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], y[5];
  for (int i = 0; i < 5; ++i)
  {
    lat_k<IPT, METHOD, PAIRS, DO><<<1, 32>>>(12345u, chains[i], d);
    CHECK(cudaDeviceSynchronize());
    long long c;
    CHECK(cudaMemcpy(&c, d, sizeof c, cudaMemcpyDeviceToHost));
    x[i] = chains[i];
    y[i] = (double) c;
  }
  double a, b;
  fit(x, y, 5, a, b);
  cudaFree(d);
  return b;
}

template <int IPT, bool PAIRS>
static void run_lat()
{
  const double gen = slope_raw<IPT, 0, PAIRS, 0>();
  const double m0  = slope_raw<IPT, 0, PAIRS, 1>() - gen;
  const double m1  = slope_raw<IPT, 1, PAIRS, 1>() - gen;
  const double m2  = slope_raw<IPT, 2, PAIRS, 1>() - gen;
  const double m3  = slope_raw<IPT, 3, PAIRS, 1>() - gen;
  printf("  %s size %3d (IPT %d):  mine=%7.1f  cub+packed=%7.1f  cub+wrapper=%7.1f  cub-unstable=%7.1f\n",
         PAIRS ? "pairs" : "keys ", 32 * IPT, IPT, m0, m1, m2, m3);
}

template <int IPT, int METHOD, bool PAIRS>
static double thr_gelems(int num_SMs)
{
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_k<IPT, METHOD, PAIRS>, kThrBlock, 0));
  const int grid = maxblk * num_SMs;
  thr_k<IPT, METHOD, PAIRS><<<grid, kThrBlock>>>(kThrIters);
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best_ms = 1e30f;
  for (int rep = 0; rep < 5; ++rep)
  {
    cudaEventRecord(e0);
    thr_k<IPT, METHOD, PAIRS><<<grid, kThrBlock>>>(kThrIters);
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best_ms = std::min(best_ms, ms);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  const double elems = (double) grid * (kThrBlock / 32) * kThrIters * (double) (32 * IPT);
  return elems / (best_ms * 1e-3) / 1e9;
}

template <int IPT, bool PAIRS>
static void run_thr(int num_SMs)
{
  const double m0 = thr_gelems<IPT, 0, PAIRS>(num_SMs);
  const double m1 = thr_gelems<IPT, 1, PAIRS>(num_SMs);
  const double m2 = thr_gelems<IPT, 2, PAIRS>(num_SMs);
  const double m3 = thr_gelems<IPT, 3, PAIRS>(num_SMs);
  printf("  %s size %3d (IPT %d):  mine=%6.1f  cub+packed=%6.1f  cub+wrapper=%6.1f  cub-unstable=%6.1f  Gelem/s\n",
         PAIRS ? "pairs" : "keys ", 32 * IPT, IPT, m0, m1, m2, m3);
}

#define FOR_IPT(X) X(1) X(2) X(4) X(8)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness + stability vs std::stable_sort (random / ties / all-equal) ===\n");
#define C(IPT) run_correct<IPT>();
    FOR_IPT(C)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== LATENCY: single-warp slope cyc/sort (gen-subtracted) ===\n");
#define LK(IPT) run_lat<IPT, false>();
    FOR_IPT(LK)
#define LP(IPT) run_lat<IPT, true>();
    FOR_IPT(LP)
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== THROUGHPUT: one occupancy wave, Gelem/s ===\n");
#define TK(IPT) run_thr<IPT, false>(p.multiProcessorCount);
    FOR_IPT(TK)
#define TP(IPT) run_thr<IPT, true>(p.multiProcessorCount);
    FOR_IPT(TP)
  }
  return 0;
}
