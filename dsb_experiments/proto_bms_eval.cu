// BlockMergeSort partial-tile performance evaluation for PR #10733 (fix/block-merge-sort-partial-tile).
//
// Three implementations, one harness, two builds:
//   BASE build (-I<main>):                    Sort(..., valid_items, oob_default)  [old clamped impl]
//   FIX build  (-DFIXED=1 -I<shadow> -I<main>):
//       overload A: Sort(..., valid_items, oob_default)  [pad-then-sort-full: no boundary logic in rounds]
//       overload B: Sort(..., valid_items)                [clamped rounds + identity-round early exit]
//
// Configs: BlockMergeSort<float, 256, IPT[, int]>, IPT in {1,2,4,8} (tiles 256..2048), keys-only
// and pairs, valid_items in {N, N-3, N/2+3, N/8}:
//   N     = full tile           (pure overhead comparison of the three paths)
//   N-3   = nearly full         (boundary machinery, no early-exit help)
//   N/2+3 = mid partial
//   N/8   = small partial       (early-exit showcase for B; small-input DeviceMergeSort shape)
// oob_default = +inf (satisfies the documented precondition on both builds).
//
// Latency: single block, back-to-back random input (data-dependent sort), chain serialized on
// previous output, generate-only control subtracted, chain-length slope, min of reps.
// Throughput: 8 occupancy waves (tail-amortized), fixed iters, best of 5 events, Gelem/s
// normalized by VALID elements (useful work). res: regs/spills/occupancy.
// Correctness (slim; full suite = proto_merge_fix.cu): sorted-valid-prefix gate per overload and
// valid_items, plus informational oob-suffix==inf flag for overload A.
//
// Modes: ./bms_[base|fix] [correct|lat|thr|res|all]

#include <cub/block/block_merge_sort.cuh>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef FIXED
#  define FIXED 0
#endif

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

struct LessK
{
  template <typename T>
  __device__ __forceinline__ bool operator()(const T& a, const T& b) const
  {
    return a < b;
  }
};

// ------------------------------------------------------------------ sort dispatch
// OVERLOAD: 0 = Sort(..., valid, oob_default)  [both builds]
//           1 = Sort(..., valid)               [FIX build only]
template <int IPT, int OVERLOAD, bool PAIRS>
__device__ __forceinline__ void do_sort(float (&k)[IPT], int (&v)[IPT], int valid_items, void* smem)
{
  if constexpr (PAIRS)
  {
    using BMS = cub::BlockMergeSort<float, kBlock, IPT, int>;
    auto& ts  = *reinterpret_cast<typename BMS::TempStorage*>(smem);
    if constexpr (OVERLOAD == 0)
    {
      BMS(ts).Sort(k, v, LessK{}, valid_items, INFINITY);
    }
#if FIXED
    else
    {
      BMS(ts).Sort(k, v, LessK{}, valid_items);
    }
#endif
  }
  else
  {
    using BMS = cub::BlockMergeSort<float, kBlock, IPT>;
    auto& ts  = *reinterpret_cast<typename BMS::TempStorage*>(smem);
    if constexpr (OVERLOAD == 0)
    {
      BMS(ts).Sort(k, LessK{}, valid_items, INFINITY);
    }
#if FIXED
    else
    {
      BMS(ts).Sort(k, LessK{}, valid_items);
    }
#endif
  }
}

// max TempStorage across the two class flavors (union of KeyT/ValueT tiles + padding)
template <int IPT>
struct SmemBox
{
  using A = typename cub::BlockMergeSort<float, kBlock, IPT, int>::TempStorage;
  using B = typename cub::BlockMergeSort<float, kBlock, IPT>::TempStorage;
  union U
  {
    A a;
    B b;
  };
};

// ------------------------------------------------------------------ correctness
template <int IPT, int OVERLOAD, bool PAIRS>
__global__ void correct_k(const float* ink, const int* inv, float* outk, int* outv, int valid_items)
{
  __shared__ typename SmemBox<IPT>::U sm;
  const int tid = threadIdx.x;
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = ink[IPT * tid + i];
    v[i] = inv[IPT * tid + i];
  }
  do_sort<IPT, OVERLOAD, PAIRS>(k, v, valid_items, &sm);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    outk[IPT * tid + i] = k[i];
    outv[IPT * tid + i] = v[i];
  }
}

// ------------------------------------------------------------------ latency
constexpr int kLatReps = 24;

template <int IPT, int OVERLOAD, bool PAIRS, int DO_SORT>
__global__ void __launch_bounds__(kBlock) lat_k(unsigned seed0, int chain, int valid_items, long long* out)
{
  __shared__ typename SmemBox<IPT>::U sm;
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
      // serialize on previous output; n-mix keeps inputs varying even where outputs are constant
      unsigned ss = s ^ __float_as_uint(k[0]) ^ (unsigned) (n * 2654435761u);
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        k[i] = lcg_f(ss);
      }
      if (DO_SORT)
      {
        do_sort<IPT, OVERLOAD, PAIRS>(k, v, valid_items, &sm);
      }
    }
    float cv[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      cv[i] = k[i] + (PAIRS ? (float) v[i] : 0.f); // consume every output
    }
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

// ------------------------------------------------------------------ throughput
constexpr int kThrIters = 48;
constexpr int kWaves    = 8;

template <int IPT, int OVERLOAD, bool PAIRS>
__global__ void __launch_bounds__(kBlock) thr_k(int num_iter, int valid_items)
{
  __shared__ typename SmemBox<IPT>::U sm;
  unsigned s = 12345u + (threadIdx.x + blockIdx.x * kBlock + 7) * 2654435761u;
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
    do_sort<IPT, OVERLOAD, PAIRS>(k, v, valid_items, &sm);
    float cv[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      cv[i] = k[i] + (PAIRS ? (float) v[i] : 0.f);
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

static void valids_for(int N, int (&v)[4])
{
  v[0] = N;
  v[1] = N - 3;
  v[2] = N / 2 + 3;
  v[3] = N / 8 < 32 ? 32 : N / 8;
}

template <int IPT, int OVERLOAD>
static void check_one(int valid_items, int& fails, char* buf)
{
  const int N = kBlock * IPT;
  std::vector<float> ink(N);
  std::vector<int> inv(N);
  unsigned s = 4242u + IPT * 13 + OVERLOAD + valid_items;
  for (int i = 0; i < N; ++i)
  {
    s      = 1664525u * s + 1013904223u;
    ink[i] = (float) ((s >> 8) & 0xffffu);
    inv[i] = i;
  }
  std::vector<float> ref(ink.begin(), ink.begin() + valid_items);
  std::sort(ref.begin(), ref.end());
  float *d_ik, *d_ok;
  int *d_iv, *d_ov;
  CHECK(cudaMalloc(&d_ik, N * 4));
  CHECK(cudaMalloc(&d_ok, N * 4));
  CHECK(cudaMalloc(&d_iv, N * 4));
  CHECK(cudaMalloc(&d_ov, N * 4));
  CHECK(cudaMemcpy(d_ik, ink.data(), N * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_iv, inv.data(), N * 4, cudaMemcpyHostToDevice));
  correct_k<IPT, OVERLOAD, true><<<1, kBlock>>>(d_ik, d_iv, d_ok, d_ov, valid_items);
  CHECK(cudaDeviceSynchronize());
  std::vector<float> ok(N);
  CHECK(cudaMemcpy(ok.data(), d_ok, N * 4, cudaMemcpyDeviceToHost));
  cudaFree(d_ik);
  cudaFree(d_ok);
  cudaFree(d_iv);
  cudaFree(d_ov);
  bool prefix = std::equal(ref.begin(), ref.end(), ok.begin());
  bool sufinf = true;
  for (int i = valid_items; i < N; ++i)
  {
    sufinf = sufinf && std::isinf(ok[i]);
  }
  if (!prefix)
  {
    ++fails;
  }
  sprintf(buf + strlen(buf), " v=%d:%s%s", valid_items, prefix ? "P" : "!P",
          OVERLOAD == 0 ? (sufinf ? "S" : "-") : "");
}

template <int IPT>
static void run_correct(int& fails)
{
  int vals[4];
  valids_for(kBlock * IPT, vals);
  char bufA[160] = "", bufB[160] = "";
  for (int i = 0; i < 4; ++i)
  {
    check_one<IPT, 0>(vals[i], fails, bufA);
  }
#if FIXED
  for (int i = 0; i < 4; ++i)
  {
    check_one<IPT, 1>(vals[i], fails, bufB);
  }
#endif
  printf("  tile %4d: A(oob):%s%s\n", kBlock * IPT, bufA, FIXED ? "" : "   [B n/a in BASE]");
#if FIXED
  printf("             B(nosent):%s\n", bufB);
#endif
}

template <int IPT, int OVERLOAD, bool PAIRS, int DO>
static double slope_raw(int valid_items)
{
  long long* d;
  CHECK(cudaMalloc(&d, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], y[5];
  for (int i = 0; i < 5; ++i)
  {
    lat_k<IPT, OVERLOAD, PAIRS, DO><<<1, kBlock>>>(12345u, chains[i], valid_items, d);
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
  int vals[4];
  valids_for(kBlock * IPT, vals);
  for (int i = 0; i < 4; ++i)
  {
    const double gen = slope_raw<IPT, 0, PAIRS, 0>(vals[i]);
    const double a   = slope_raw<IPT, 0, PAIRS, 1>(vals[i]) - gen;
#if FIXED
    const double b = slope_raw<IPT, 1, PAIRS, 1>(vals[i]) - gen;
    printf("  %s tile %4d valid %4d:  A=%9.1f   B=%9.1f   (B vs A: %+.1f%%)\n",
           PAIRS ? "pairs" : "keys ", kBlock * IPT, vals[i], a, b, 100.0 * (b - a) / a);
#else
    printf("  %s tile %4d valid %4d:  A=%9.1f\n", PAIRS ? "pairs" : "keys ", kBlock * IPT, vals[i], a);
#endif
  }
}

template <int IPT, int OVERLOAD, bool PAIRS>
static double thr_gelems(int num_SMs, int valid_items)
{
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_k<IPT, OVERLOAD, PAIRS>, kBlock, 0));
  const int grid = kWaves * maxblk * num_SMs;
  thr_k<IPT, OVERLOAD, PAIRS><<<grid, kBlock>>>(kThrIters, valid_items);
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best_ms = 1e30f;
  for (int rep = 0; rep < 5; ++rep)
  {
    cudaEventRecord(e0);
    thr_k<IPT, OVERLOAD, PAIRS><<<grid, kBlock>>>(kThrIters, valid_items);
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best_ms = std::min(best_ms, ms);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  const double elems = (double) grid * kThrIters * (double) valid_items; // useful work only
  return elems / (best_ms * 1e-3) / 1e9;
}

template <int IPT, bool PAIRS>
static void run_thr(int num_SMs)
{
  int vals[4];
  valids_for(kBlock * IPT, vals);
  for (int i = 0; i < 4; ++i)
  {
    const double a = thr_gelems<IPT, 0, PAIRS>(num_SMs, vals[i]);
#if FIXED
    const double b = thr_gelems<IPT, 1, PAIRS>(num_SMs, vals[i]);
    printf("  %s tile %4d valid %4d:  A=%7.1f   B=%7.1f  Gelem/s (valid-normalized)\n",
           PAIRS ? "pairs" : "keys ", kBlock * IPT, vals[i], a, b);
#else
    printf("  %s tile %4d valid %4d:  A=%7.1f  Gelem/s (valid-normalized)\n",
           PAIRS ? "pairs" : "keys ", kBlock * IPT, vals[i], a);
#endif
  }
}

template <int IPT, int OVERLOAD, bool PAIRS>
static void res_row()
{
  cudaFuncAttributes at{};
  CHECK(cudaFuncGetAttributes(&at, thr_k<IPT, OVERLOAD, PAIRS>));
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_k<IPT, OVERLOAD, PAIRS>, kBlock, 0));
  printf("    %s %s: regs=%3d spills=%4zuB smem=%6zuB occ=%d blk/SM\n",
         OVERLOAD == 0 ? "A(oob)   " : "B(nosent)", PAIRS ? "pairs" : "keys ", at.numRegs,
         (size_t) at.localSizeBytes, (size_t) at.sharedSizeBytes, maxblk);
}

template <int IPT>
static void run_res()
{
  printf("  tile %4d:\n", kBlock * IPT);
  res_row<IPT, 0, false>();
  res_row<IPT, 0, true>();
#if FIXED
  res_row<IPT, 1, false>();
  res_row<IPT, 1, true>();
#endif
}

#define FOR_IPT(X) X(1) X(2) X(4) X(8)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d, %d SMs)  build=%s\n", p.name, p.major, p.minor, p.multiProcessorCount,
         FIXED ? "FIX" : "BASE");
  int fails = 0;
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness: sorted-valid-prefix gate (P), oob-suffix==inf info (S/-) ===\n");
#define C(IPT) run_correct<IPT>(fails);
    FOR_IPT(C)
    printf("  prefix failures: %d%s\n", fails, fails ? " <-- INVESTIGATE" : "");
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
    printf("\n=== THROUGHPUT: %d waves, %d iters, Gelem/s of VALID elements ===\n", kWaves, kThrIters);
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
  return fails ? 1 : 0;
}
