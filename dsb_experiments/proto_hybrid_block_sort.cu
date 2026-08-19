// Hybrid BlockMergeSort prototype: stable warp-scope bitonic phase + cross-warp MergePath rounds.
//
// Hypothesis (from BALLOT_MERGE_RESULTS.md + WARP_MERGE_SORT_RESULTS.md + WMS_STATIC_SWITCH_RESULTS.md):
// BlockMergeSort spends its first log2(32)=5 merge rounds on groups that live entirely within
// single warps, paying block barriers + smem searches for work a (stable) register network does
// 2-3x faster at small IPT. Replace thread-sort + intra-warp rounds with ONE rank-augmented
// bitonic warp sort (stable by rank tiebreak), keep MergePath rounds only across warps.
//
// Variants (BLOCK=256 threads, TILE = 256*IPT):
//   V0: thread odd-even sort + 8 MergePath rounds, dynamic search   (stock mirror)
//   V1: V0 with statically-unrolled MergePath                        (the SEARCH_STATIC switch)
//   V2: stable warp bitonic phase + 3 cross-warp rounds, dynamic search
//   V3: V2 with statically-unrolled MergePath                        (the full stack)
// All variants: stable, blocked in/out, full tiles. KeyT float, ValueT int (PAIRS variant).
// Self-contained mirror (no cub includes); earlier mirrors tracked cub within ~1%.
//
// Methodology per DEVICE_SIDE_BENCHMARKING_ISSUE.md: latency = single block, back-to-back random
// input, chain serialized on previous output, generate-only control subtracted, chain-length
// slope, min of reps; throughput = one occupancy wave, element-normalized; correctness/stability
// vs std::stable_sort on (key, index) with heavy-tie patterns.
//
// Modes: ./proto_hybrid_block_sort [correct|lat|thr|res|all]

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

constexpr int kBlock = 256;

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

__host__ __device__ constexpr int log2_ceil(int n)
{
  int l = 0;
  while ((1 << l) < n)
  {
    ++l;
  }
  return l;
}

// ------------------------------------------------------------------ thread-local stable sort
template <int IPT, bool PAIRS>
__device__ __forceinline__ void thread_odd_even(float (&k)[IPT], int (&v)[IPT])
{
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
#pragma unroll
    for (int j = 1 & i; j < IPT - 1; j += 2)
    {
      if (k[j + 1] < k[j])
      {
        float tk = k[j]; k[j] = k[j + 1]; k[j + 1] = tk;
        if (PAIRS) { int tv = v[j]; v[j] = v[j + 1]; v[j + 1] = tv; }
      }
    }
  }
}

// ------------------------------------------------------------------ MergePath searches
__device__ __forceinline__ int mp_dyn(const float* sk, int k1b, int k2b, int c1, int c2, int diag)
{
  int lo = ::max(0, diag - c2);
  int hi = ::min(diag, c1);
  while (lo < hi)
  {
    const int mid = (lo + hi) >> 1;
    if (sk[k2b + diag - 1 - mid] < sk[k1b + mid])
    {
      hi = mid;
    }
    else
    {
      lo = mid + 1;
    }
  }
  return lo;
}

template <int RANGE>
__device__ __forceinline__ int mp_static(const float* sk, int k1b, int k2b, int c1, int c2, int diag)
{
  int lo = ::max(0, diag - c2);
  int hi = ::min(diag, c1);
#pragma unroll
  for (int it = 0; it <= log2_ceil(RANGE + 1); ++it)
  {
    const int mid = (lo + hi) >> 1;
    const bool go = lo < hi;
    const int i2  = ::max(0, diag - 1 - mid); // clamps only matter on converged steps (values unused)
    const bool up = go && (sk[k2b + i2] < sk[k1b + mid]);
    hi            = up ? mid : hi;
    lo            = (go && !up) ? mid + 1 : lo;
  }
  return lo;
}

// ------------------------------------------------------------------ serial merge (cub mirror)
template <int IPT>
__device__ __forceinline__ void
serial_merge(const float* sk, int b1, int b2, int c1, int c2, float (&out)[IPT], int (&idx)[IPT])
{
  const int e1 = b1 + c1;
  const int e2 = b2 + c2;
  float k1     = c1 != 0 ? sk[b1] : out[0];
  float k2     = c2 != 0 ? sk[b2] : out[0];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const bool p = (b2 < e2) && ((b1 >= e1) || (k2 < k1)); // ties -> run 1 (stable)
    out[i]       = p ? k2 : k1;
    idx[i]       = p ? b2++ : b1++;
    if (p)
    {
      k2 = sk[b2];
    }
    else
    {
      k1 = sk[b1];
    }
  }
}

// ------------------------------------------------------------------ one block merge round
// TARGET = threads per merging group; SEGL_STORE != 0: registers are striped over SEGL_STORE-lane
// segments (bootstrap-phase exit), 0: blocked
template <int IPT, bool PAIRS, bool STAT, int TARGET, int SEGL_STORE, int TILE>
__device__ __forceinline__ void block_round(float (&k)[IPT], int (&v)[IPT], float* tk, int* tv)
{
  const int tid = threadIdx.x;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const int l   = tid & 31;
    const int idx = SEGL_STORE
                    ? ((tid & ~31) * IPT + (l / SEGL_STORE) * (SEGL_STORE * IPT) + (l & (SEGL_STORE - 1))
                       + i * SEGL_STORE)
                    : (IPT * tid + i);
    tk[idx]       = k[i];
  }
  if (tid == 0)
  {
    tk[TILE] = k[0]; // defined pad for the one-past prefetch
  }
  __syncthreads();

  int idxs[IPT];
  constexpr int merged = TARGET / 2;
  constexpr int size   = IPT * merged;
  const int mask       = TARGET - 1;
  const int start      = IPT * (~mask & tid);
  const int diag       = IPT * (mask & tid);
  const int k1b        = start;
  const int k2b        = start + size;

  const int pd = STAT ? mp_static<size>(tk, k1b, k2b, size, size, diag) : mp_dyn(tk, k1b, k2b, size, size, diag);

  const int b1 = k1b + pd;
  const int b2 = k2b + diag - pd;
  serial_merge<IPT>(tk, b1, b2, k1b + size - b1, k2b + size - b2, k, idxs);

  if (PAIRS)
  {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      const int l   = tid & 31;
      const int idx = SEGL_STORE
                      ? ((tid & ~31) * IPT + (l / SEGL_STORE) * (SEGL_STORE * IPT) + (l & (SEGL_STORE - 1))
                         + i * SEGL_STORE)
                      : (IPT * tid + i);
      tv[idx]       = v[i];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      v[i] = tv[idxs[i]];
    }
  }
}

// compile-time round recursion: TARGET, TARGET*2, ..., kBlock
template <int IPT, bool PAIRS, bool STAT, int TARGET, int FIRST_SEGL, int TILE>
__device__ __forceinline__ void rounds_from(float (&k)[IPT], int (&v)[IPT], float* tk, int* tv)
{
  block_round<IPT, PAIRS, STAT, TARGET, FIRST_SEGL, TILE>(k, v, tk, tv);
  if constexpr (TARGET < kBlock)
  {
    rounds_from<IPT, PAIRS, STAT, TARGET * 2, 0, TILE>(k, v, tk, tv);
  }
}

// ------------------------------------------------------------------ stable segment bitonic sort
// striped arrangement over a SEGL-lane segment: (sub-lane s, item i) holds position s + SEGL*i of
// the segment's SEGL*IPT elements; stability via static input-rank tiebreak (ranks make the
// network's order strict). SEGL = 32 sorts the whole warp tile; SEGL < 32 sorts 32/SEGL
// independent segments concurrently (xor partners stay within a power-of-two segment).
template <int SEGL, int IPT, bool PAIRS>
__device__ __forceinline__ void seg_stable_bitonic(float (&sk)[IPT], int (&sr)[IPT], int (&sv)[IPT])
{
  const int l     = threadIdx.x & (SEGL - 1); // sub-lane within segment
  constexpr int N = SEGL * IPT;
#pragma unroll
  for (int stage = 2; stage <= N; stage <<= 1)
  {
#pragma unroll
    for (int j = stage >> 1; j >= 1; j >>= 1)
    {
      if (j >= SEGL)
      {
        const int jb = j / SEGL; // register-local exchange between items i and i|jb
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          if ((i & jb) == 0 && (i | jb) < IPT)
          {
            const int i2   = i | jb;
            const int p    = l + SEGL * i;
            const bool asc = (p & stage) == 0 || stage == N;
            const bool sw  = (sk[i2] < sk[i]) || (sk[i2] == sk[i] && sr[i2] < sr[i]); // partner-less
            if (sw == asc) // put min at the lower position iff ascending
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
          const float pk = __shfl_xor_sync(~0u, sk[i], j);
          const int pr   = __shfl_xor_sync(~0u, sr[i], j);
          const int pv   = PAIRS ? __shfl_xor_sync(~0u, sv[i], j) : 0;
          const int p    = l + SEGL * i;
          const bool lower = (p & j) == 0;
          const bool asc   = (p & stage) == 0 || stage == N;
          const bool pless = (pk < sk[i]) || (pk == sk[i] && pr < sr[i]);
          const bool take  = (lower == asc) ? pless : !pless; // ranks make the order strict
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

// ------------------------------------------------------------------ full tile sort, all variants
// VARIANT: 0 = merge+dyn, 1 = merge+static, 2 = hybrid+dyn (bootstrap = whole warp tile),
//          3 = hybrid+static, 4 = capped hybrid (bootstrap = 64-element chunks) + dyn
template <int IPT, int VARIANT, bool PAIRS>
__device__ __forceinline__ void sort_tile(float (&k)[IPT], int (&v)[IPT], float* tk, int* tv)
{
  constexpr int TILE  = kBlock * IPT;
  constexpr bool STAT = (VARIANT & 1) != 0 && VARIANT < 4;
  if constexpr (VARIANT < 2)
  {
    thread_odd_even<IPT, PAIRS>(k, v);
    rounds_from<IPT, PAIRS, STAT, 2, 0, TILE>(k, v, tk, tv);
  }
  else
  {
    // bootstrap phase: sorted runs of CAP elements via a stable segment network
    constexpr int CAP  = (VARIANT == 4) ? (IPT == 1 ? 32 : 64) : 32 * IPT; // bootstrap run length
    constexpr int SEGL = CAP / IPT;                                       // lanes per segment
    const int tid  = threadIdx.x;
    const int l    = tid & 31;
    const int subl = l & (SEGL - 1);
    const int base = (tid & ~31) * IPT + (l / SEGL) * CAP; // segment's tile offset
    __syncthreads(); // tile buffer handoff between chained calls
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      tk[IPT * tid + i] = k[i]; // blocked -> striped-in-segment via the warp's own smem section
    }
    __syncwarp();
    float sk[IPT];
    int sr[IPT], sv[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      sk[i] = tk[base + subl + SEGL * i];
      sr[i] = subl + SEGL * i; // static input rank within the segment -> stability
    }
    if (PAIRS)
    {
      __syncwarp();
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        tv[IPT * tid + i] = v[i]; // union with tk: keys already reloaded to registers
      }
      __syncwarp();
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        sv[i] = tv[base + subl + SEGL * i];
      }
    }
    seg_stable_bitonic<SEGL, IPT, PAIRS>(sk, sr, sv);
    // merge rounds from run length CAP; first store converts segment-striped -> blocked for free
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = sk[i];
      if (PAIRS)
      {
        v[i] = sv[i];
      }
    }
    rounds_from<IPT, PAIRS, STAT, 2 * CAP / IPT, SEGL, TILE>(k, v, tk, tv);
  }
}

// ------------------------------------------------------------------ kernels
template <int IPT, int VARIANT>
__global__ void __launch_bounds__(kBlock) correct_k(const float* ink, const int* inv, float* outk, int* outv)
{
  constexpr int TILE = kBlock * IPT;
  __shared__ union
  {
    float k[TILE + 1];
    int v[TILE + 1];
  } sm;
  const int tid = threadIdx.x;
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = ink[IPT * tid + i];
    v[i] = inv[IPT * tid + i];
  }
  sort_tile<IPT, VARIANT, true>(k, v, sm.k, sm.v);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    outk[IPT * tid + i] = k[i];
    outv[IPT * tid + i] = v[i];
  }
}

constexpr int kLatReps = 24;

template <int IPT, int VARIANT, bool PAIRS, int DO_SORT>
__global__ void __launch_bounds__(kBlock) lat_k(unsigned seed0, int chain, long long* out)
{
  constexpr int TILE = kBlock * IPT;
  __shared__ union
  {
    float k[TILE + 1];
    int v[TILE + 1];
  } sm;
  unsigned s = seed0 + (threadIdx.x + 7) * 2654435761u;
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = 0.f;
    v[i] = i;
  }
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int rep = 0; rep < kLatReps; ++rep)
  {
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll 1
    for (int n = 0; n < chain; ++n)
    {
      unsigned ss = s ^ __float_as_uint(k[0]) ^ (unsigned) (n * 2654435761u);
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        k[i] = lcg_f(ss);
      }
      if (DO_SORT)
      {
        sort_tile<IPT, VARIANT, PAIRS>(k, v, sm.k, sm.v);
      }
    }
    float cv[2] = {k[0] + (float) v[0], k[IPT - 1] + (float) v[IPT - 1]};
    sink_f(cv);
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    *out = best;
  }
}

constexpr int kThrIters = 100;

template <int IPT, int VARIANT, bool PAIRS>
__global__ void __launch_bounds__(kBlock) thr_k(int num_iter)
{
  constexpr int TILE = kBlock * IPT;
  __shared__ union
  {
    float k[TILE + 1];
    int v[TILE + 1];
  } sm;
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
    sort_tile<IPT, VARIANT, PAIRS>(k, v, sm.k, sm.v);
    float cv[2] = {k[0] + (float) v[0], k[IPT - 1] + (float) v[IPT - 1]};
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

template <int IPT, int VARIANT>
static bool check_one(unsigned seed, int pattern)
{
  const int N = kBlock * IPT;
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
  correct_k<IPT, VARIANT><<<1, kBlock>>>(d_ik, d_iv, d_ok, d_ov);
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
    if (ok[i] != ref[i].first || ov[i] != ref[i].second)
    {
      printf("      V%d IPT%d pat%d pos %d: got (%g,%d) want (%g,%d)\n",
             VARIANT, IPT, pattern, i, ok[i], ov[i], ref[i].first, ref[i].second);
      return false;
    }
  }
  return true;
}

template <int IPT, int VARIANT>
static bool check_variant()
{
  bool ok = true;
  for (int pat = 0; pat <= 2; ++pat)
  {
    for (unsigned t = 0; t < (pat <= 1 ? 8u : 1u); ++t)
    {
      ok = ok && check_one<IPT, VARIANT>(4242u + t * 131u + pat * 7919u, pat);
    }
  }
  return ok;
}

template <int IPT>
static void run_correct()
{
  const bool v0 = check_variant<IPT, 0>();
  const bool v1 = check_variant<IPT, 1>();
  const bool v2 = check_variant<IPT, 2>();
  const bool v3 = check_variant<IPT, 3>();
  const bool v4 = check_variant<IPT, 4>();
  printf("  tile %4d (IPT %d):  V0 %s  V1 %s  V2 %s  V3 %s  V4 %s   (incl. stability, heavy ties)\n",
         kBlock * IPT, IPT, v0 ? "PASS" : "FAIL", v1 ? "PASS" : "FAIL", v2 ? "PASS" : "FAIL",
         v3 ? "PASS" : "FAIL", v4 ? "PASS" : "FAIL");
}

template <int IPT, int VARIANT, bool PAIRS, int DO>
static double slope_raw()
{
  long long* d;
  CHECK(cudaMalloc(&d, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], y[5];
  for (int i = 0; i < 5; ++i)
  {
    lat_k<IPT, VARIANT, PAIRS, DO><<<1, kBlock>>>(12345u, chains[i], d);
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
  const double v0  = slope_raw<IPT, 0, PAIRS, 1>() - gen;
  const double v1  = slope_raw<IPT, 1, PAIRS, 1>() - gen;
  const double v2  = slope_raw<IPT, 2, PAIRS, 1>() - gen;
  const double v3  = slope_raw<IPT, 3, PAIRS, 1>() - gen;
  const double v4  = slope_raw<IPT, 4, PAIRS, 1>() - gen;
  printf("  %s tile %4d (IPT %d):  V0=%8.1f  V1(stat)=%8.1f  V2(hyb)=%8.1f  V3(h+s)=%8.1f  V4(cap64)=%8.1f   best vs V0: %+.1f%%\n",
         PAIRS ? "pairs" : "keys ", kBlock * IPT, IPT, v0, v1, v2, v3, v4,
         100.0 * (std::min(std::min(std::min(v1, v2), v3), v4) - v0) / v0);
}

template <int IPT, int VARIANT, bool PAIRS>
static double thr_gelems(int num_SMs)
{
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_k<IPT, VARIANT, PAIRS>, kBlock, 0));
  const int grid = maxblk * num_SMs;
  thr_k<IPT, VARIANT, PAIRS><<<grid, kBlock>>>(kThrIters);
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best_ms = 1e30f;
  for (int rep = 0; rep < 5; ++rep)
  {
    cudaEventRecord(e0);
    thr_k<IPT, VARIANT, PAIRS><<<grid, kBlock>>>(kThrIters);
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best_ms = std::min(best_ms, ms);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  const double elems = (double) grid * kThrIters * (double) (kBlock * IPT);
  return elems / (best_ms * 1e-3) / 1e9;
}

template <int IPT, bool PAIRS>
static void run_thr(int num_SMs)
{
  const double v0 = thr_gelems<IPT, 0, PAIRS>(num_SMs);
  const double v1 = thr_gelems<IPT, 1, PAIRS>(num_SMs);
  const double v2 = thr_gelems<IPT, 2, PAIRS>(num_SMs);
  const double v3 = thr_gelems<IPT, 3, PAIRS>(num_SMs);
  const double v4 = thr_gelems<IPT, 4, PAIRS>(num_SMs);
  printf("  %s tile %4d (IPT %d):  V0=%6.1f  V1=%6.1f  V2=%6.1f  V3=%6.1f  V4=%6.1f  Gelem/s\n",
         PAIRS ? "pairs" : "keys ", kBlock * IPT, IPT, v0, v1, v2, v3, v4);
}

template <int IPT, int VARIANT, bool PAIRS>
static void res_one()
{
  cudaFuncAttributes at{};
  CHECK(cudaFuncGetAttributes(&at, thr_k<IPT, VARIANT, PAIRS>));
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_k<IPT, VARIANT, PAIRS>, kBlock, 0));
  printf("    V%d %s: regs=%3d spills=%4zuB smem=%6zuB occ=%d blk/SM\n",
         VARIANT, PAIRS ? "pairs" : "keys ", at.numRegs, (size_t) at.localSizeBytes, (size_t) at.sharedSizeBytes, maxblk);
}

template <int IPT>
static void run_res()
{
  printf("  tile %4d (IPT %d):\n", kBlock * IPT, IPT);
  res_one<IPT, 0, false>();
  res_one<IPT, 1, false>();
  res_one<IPT, 2, false>();
  res_one<IPT, 3, false>();
  res_one<IPT, 4, false>();
  res_one<IPT, 0, true>();
  res_one<IPT, 1, true>();
  res_one<IPT, 2, true>();
  res_one<IPT, 3, true>();
  res_one<IPT, 4, true>();
}

#define FOR_IPT(X) X(1) X(2) X(4) X(8)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d, %d SMs), block=%d\n", p.name, p.major, p.minor, p.multiProcessorCount, kBlock);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness + stability vs std::stable_sort (random / 5-distinct ties / all-equal) ===\n");
#define C(IPT) run_correct<IPT>();
    FOR_IPT(C)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== LATENCY: single-block slope cyc/sort (gen-subtracted) ===\n");
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
  if (mode == "res" || mode == "all")
  {
    printf("\n=== RESOURCES (thr kernels) ===\n");
#define R(IPT) run_res<IPT>();
    FOR_IPT(R)
  }
  return 0;
}
