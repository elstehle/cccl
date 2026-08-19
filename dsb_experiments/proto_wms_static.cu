// Validation of the productized MERGE_SORT_SEARCH_STATIC switch (statically unrolled MergePath)
// in the real cub headers (shadow tree wms_static/cub/{block,warp}/*.cuh, include-path priority).
//
//   * Warp scope:  cub::WarpMergeSort <float, IPT, 32, NullType, ALGO>, IPT 1,2,3,4,5,6,8,10,12
//   * Block scope: cub::BlockMergeSort<float, 256, IPT, NullType, 1, 1, ALGO>, IPT 1,2,4,8
//
// Methodology per DEVICE_SIDE_BENCHMARKING_ISSUE.md / WARP_MERGE_SORT_RESULTS.md §13:
//   * merge sort is DATA-DEPENDENT -> latency = back-to-back fresh random input, chain serialized
//     on the previous output, generate-only control subtracted; chain-length slope; clock64.
//   * throughput = one occupancy wave x fixed iterations, sink()-guarded, element-normalized.
//   * correctness vs std::sort: full tile, partial tile (valid_items + oob_default), key-value pairs.
//   * res mode: registers / spills / smem / occupancy for both switch settings.
//
// Build (shadow tree FIRST on the include path):
//   nvcc -std=c++17 -arch=sm_100 -O3 -Idsb_experiments/wms_static -Icub -Ilibcudacxx/include -Ithrust \
//        proto_wms_static.cu -o proto_wms_static
// Modes: ./proto_wms_static [correct|lat|thr|res|all]

#include <cub/block/block_merge_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda/ptx>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

constexpr int kWarp   = 32;
constexpr int kBlock  = 256; // block-scope thread count
constexpr auto kDyn   = cub::MERGE_SORT_SEARCH_DYNAMIC;
constexpr auto kStat  = cub::MERGE_SORT_SEARCH_STATIC;

__device__ int g_dcesink[16];

struct CustomLess
{
  template <typename T>
  __device__ __forceinline__ bool operator()(const T& a, const T& b) const
  {
    return a < b;
  }
};

__device__ __forceinline__ float lcg_f(unsigned& s)
{
  s = 1664525u * s + 1013904223u;
  return (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
}

template <int IPT>
__device__ __forceinline__ void sink(float (&v)[IPT])
{
  if (cuda::ptx::get_sreg_smid() == static_cast<uint32_t>(-1))
  {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      sum += v[i];
    }
    *reinterpret_cast<float*>(g_dcesink) += sum;
  }
}

// ------------------------------------------------------------------ latency: warp scope
constexpr int kLatReps = 32;

template <int IPT, cub::MergeSortSearchAlgorithm ALGO, int DO_SORT>
__global__ void __launch_bounds__(kWarp) lat_warp(unsigned seed0, int chain, long long* out)
{
  using WMS = cub::WarpMergeSort<float, IPT, kWarp, cub::NullType, ALGO>;
  __shared__ typename WMS::TempStorage temp;
  const int lane = threadIdx.x & 31;
  unsigned s     = seed0 + (lane + 7) * 2654435761u;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s);
  }
  if (DO_SORT)
  {
    WMS(temp).Sort(k, CustomLess{});
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
      unsigned ss = s ^ __float_as_uint(k[0]); // serialize chain on previous output
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        k[i] = lcg_f(ss);
      }
      if (DO_SORT)
      {
        WMS(temp).Sort(k, CustomLess{});
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

// ------------------------------------------------------------------ latency: block scope
constexpr int kLatRepsB = 24;

template <int IPT, cub::MergeSortSearchAlgorithm ALGO, int DO_SORT>
__global__ void __launch_bounds__(kBlock) lat_block(unsigned seed0, int chain, long long* out)
{
  using BMS = cub::BlockMergeSort<float, kBlock, IPT, cub::NullType, 1, 1, ALGO>;
  __shared__ typename BMS::TempStorage temp;
  unsigned s = seed0 + (threadIdx.x + 7) * 2654435761u;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s);
  }
  if (DO_SORT)
  {
    BMS(temp).Sort(k, CustomLess{});
  }
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int rep = 0; rep < kLatRepsB; ++rep)
  {
    __syncthreads();
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
      if (DO_SORT)
      {
        BMS(temp).Sort(k, CustomLess{});
      }
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      asm volatile("" ::"f"(k[i]));
    }
    asm volatile("" ::: "memory");
    __syncthreads(); // t1 covers the whole block
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    *out = best;
  }
}

// ------------------------------------------------------------------ throughput
constexpr int kThrBlockW = 128; // 4 warps per block for the warp-scope kernel
constexpr int kThrIters  = 200;

template <int IPT, cub::MergeSortSearchAlgorithm ALGO>
__global__ void __launch_bounds__(kThrBlockW) thr_warp(int num_iter)
{
  using WMS = cub::WarpMergeSort<float, IPT, kWarp, cub::NullType, ALGO>;
  __shared__ typename WMS::TempStorage temp[kThrBlockW / kWarp];
  const int warp = threadIdx.x >> 5;
  unsigned s     = 12345u + (threadIdx.x + 7) * 2654435761u;
  float k[IPT];
#pragma unroll 1
  for (int iter = 0; iter < num_iter; ++iter)
  {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = lcg_f(s);
    }
    WMS(temp[warp]).Sort(k, CustomLess{});
    sink(k);
  }
}

template <int IPT, cub::MergeSortSearchAlgorithm ALGO>
__global__ void __launch_bounds__(kBlock) thr_block(int num_iter)
{
  using BMS = cub::BlockMergeSort<float, kBlock, IPT, cub::NullType, 1, 1, ALGO>;
  __shared__ typename BMS::TempStorage temp;
  unsigned s = 12345u + (threadIdx.x + 7) * 2654435761u;
  float k[IPT];
#pragma unroll 1
  for (int iter = 0; iter < num_iter; ++iter)
  {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = lcg_f(s);
    }
    BMS(temp).Sort(k, CustomLess{});
    sink(k);
  }
}

// ------------------------------------------------------------------ correctness
// SCOPE: 0 = warp, 1 = block. valid_items == N selects the full-tile Sort overload.
template <int IPT, cub::MergeSortSearchAlgorithm ALGO, int SCOPE>
__global__ void correct_k(const float* in, float* out, int valid_items)
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  const int tid    = threadIdx.x;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in[tid * IPT + i];
  }
  if constexpr (SCOPE == 0)
  {
    using WMS = cub::WarpMergeSort<float, IPT, kWarp, cub::NullType, ALGO>;
    __shared__ typename WMS::TempStorage temp;
    if (valid_items == NT * IPT)
    {
      WMS(temp).Sort(k, CustomLess{});
    }
    else
    {
      WMS(temp).Sort(k, CustomLess{}, valid_items, INFINITY);
    }
  }
  else
  {
    using BMS = cub::BlockMergeSort<float, kBlock, IPT, cub::NullType, 1, 1, ALGO>;
    __shared__ typename BMS::TempStorage temp;
    if (valid_items == NT * IPT)
    {
      BMS(temp).Sort(k, CustomLess{});
    }
    else
    {
      BMS(temp).Sort(k, CustomLess{}, valid_items, INFINITY);
    }
  }
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out[tid * IPT + i] = k[i];
  }
}

// pairs: values carry the original index; sort is not stable, so check keys sorted + in_key[value] == out_key
template <int IPT, cub::MergeSortSearchAlgorithm ALGO, int SCOPE>
__global__ void correct_pairs_k(const float* in, float* out_k, int* out_v)
{
  const int tid = threadIdx.x;
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in[tid * IPT + i];
    v[i] = tid * IPT + i;
  }
  if constexpr (SCOPE == 0)
  {
    using WMS = cub::WarpMergeSort<float, IPT, kWarp, int, ALGO>;
    __shared__ typename WMS::TempStorage temp;
    WMS(temp).Sort(k, v, CustomLess{});
  }
  else
  {
    using BMS = cub::BlockMergeSort<float, kBlock, IPT, int, 1, 1, ALGO>;
    __shared__ typename BMS::TempStorage temp;
    BMS(temp).Sort(k, v, CustomLess{});
  }
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out_k[tid * IPT + i] = k[i];
    out_v[tid * IPT + i] = v[i];
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

static std::vector<float> rand_input(int n, unsigned seed)
{
  std::vector<float> v(n);
  unsigned s = seed;
  for (int i = 0; i < n; ++i)
  {
    s    = 1664525u * s + 1013904223u;
    v[i] = (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
  }
  return v;
}

template <int IPT, cub::MergeSortSearchAlgorithm ALGO, int SCOPE>
std::vector<float> sort_out(int valid_items)
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  const int N      = NT * IPT;
  auto in          = rand_input(N, 4242u + IPT * 13 + SCOPE * 7); // same input for both algorithms
  float *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CHECK(cudaMemcpy(d_in, in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  correct_k<IPT, ALGO, SCOPE><<<1, NT>>>(d_in, d_out, valid_items);
  CHECK(cudaDeviceSynchronize());
  std::vector<float> out(N);
  CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
  return out;
}

// The guaranteed contract is the sorted VALID PREFIX (what DeviceMergeSort relies on); positions
// beyond valid_items holding oob_default is NOT reliably delivered by stock cub (its oob pre-pass
// skips keys[0] of fully-out-of-range threads and middle rounds run contract-degenerate searches),
// so the suffix is reported informationally and dynamic==static byte-equality is the real oracle.
template <int IPT, int SCOPE>
void check_valid(int valid_items, bool& prefix_d, bool& prefix_s, bool& identical, bool& suffix_inf)
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  const int N      = NT * IPT;
  auto in          = rand_input(N, 4242u + IPT * 13 + SCOPE * 7);
  std::vector<float> ref(in.begin(), in.begin() + valid_items);
  std::sort(ref.begin(), ref.end());
  auto od = sort_out<IPT, kDyn, SCOPE>(valid_items);
  auto os = sort_out<IPT, kStat, SCOPE>(valid_items);
  prefix_d = std::equal(ref.begin(), ref.end(), od.begin());
  prefix_s = std::equal(ref.begin(), ref.end(), os.begin());
  identical = (od == os);
  suffix_inf = true;
  for (int i = valid_items; i < N; ++i)
  {
    suffix_inf = suffix_inf && std::isinf(od[i]);
  }
}

template <int IPT, cub::MergeSortSearchAlgorithm ALGO, int SCOPE>
bool check_pairs()
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  const int N      = NT * IPT;
  auto in          = rand_input(N, 777u + IPT + SCOPE);
  std::vector<float> ref = in;
  std::sort(ref.begin(), ref.end());
  float *d_in, *d_ok;
  int* d_ov;
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK(cudaMalloc(&d_ok, N * sizeof(float)));
  CHECK(cudaMalloc(&d_ov, N * sizeof(int)));
  CHECK(cudaMemcpy(d_in, in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  correct_pairs_k<IPT, ALGO, SCOPE><<<1, NT>>>(d_in, d_ok, d_ov);
  CHECK(cudaDeviceSynchronize());
  std::vector<float> ok(N);
  std::vector<int> ov(N);
  CHECK(cudaMemcpy(ok.data(), d_ok, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ov.data(), d_ov, N * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_ok);
  cudaFree(d_ov);
  std::vector<char> seen(N, 0);
  for (int i = 0; i < N; ++i)
  {
    if (ok[i] != ref[i])
    {
      return false; // keys must equal std::sort order
    }
    if (ov[i] < 0 || ov[i] >= N || seen[ov[i]] || in[ov[i]] != ok[i])
    {
      return false; // values must be a permutation consistent with the keys
    }
    seen[ov[i]] = 1;
  }
  return true;
}

template <int IPT, cub::MergeSortSearchAlgorithm ALGO, int SCOPE, int DO>
double slope_raw()
{
  long long* d;
  CHECK(cudaMalloc(&d, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], y[5];
  for (int i = 0; i < 5; ++i)
  {
    if constexpr (SCOPE == 0)
    {
      lat_warp<IPT, ALGO, DO><<<1, kWarp>>>(12345u, chains[i], d);
    }
    else
    {
      lat_block<IPT, ALGO, DO><<<1, kBlock>>>(12345u, chains[i], d);
    }
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

template <int IPT, cub::MergeSortSearchAlgorithm ALGO, int SCOPE>
double throughput_gelems(int num_SMs)
{
  int maxblk          = 0;
  const int blk       = (SCOPE == 0) ? kThrBlockW : kBlock;
  const void* fn      = (SCOPE == 0) ? (const void*) thr_warp<IPT, ALGO> : (const void*) thr_block<IPT, ALGO>;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, fn, blk, 0));
  const int grid = maxblk * num_SMs;
  auto launch    = [&] {
    if constexpr (SCOPE == 0)
    {
      thr_warp<IPT, ALGO><<<grid, kThrBlockW>>>(kThrIters);
    }
    else
    {
      thr_block<IPT, ALGO><<<grid, kBlock>>>(kThrIters);
    }
  };
  launch(); // warmup
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best_ms = 1e30f;
  for (int rep = 0; rep < 5; ++rep)
  {
    cudaEventRecord(e0);
    launch();
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best_ms = std::min(best_ms, ms);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  const double sorts_per_block = (SCOPE == 0) ? (double) (kThrBlockW / kWarp) : 1.0;
  const double elems_per_sort  = (SCOPE == 0) ? (double) (kWarp * IPT) : (double) (kBlock * IPT);
  const double elems           = (double) grid * sorts_per_block * kThrIters * elems_per_sort;
  return elems / (best_ms * 1e-3) / 1e9;
}

template <int IPT, cub::MergeSortSearchAlgorithm ALGO, int SCOPE>
void print_res(int num_SMs)
{
  cudaFuncAttributes al{}, at{};
  const void* flat = (SCOPE == 0) ? (const void*) lat_warp<IPT, ALGO, 1> : (const void*) lat_block<IPT, ALGO, 1>;
  const void* fthr = (SCOPE == 0) ? (const void*) thr_warp<IPT, ALGO> : (const void*) thr_block<IPT, ALGO>;
  CHECK(cudaFuncGetAttributes(&al, flat));
  CHECK(cudaFuncGetAttributes(&at, fthr));
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, fthr, (SCOPE == 0) ? kThrBlockW : kBlock, 0));
  printf("    %-7s regs(lat)=%3d spills(lat)=%3zuB   regs(thr)=%3d spills(thr)=%3zuB smem=%5zuB occ=%d blk/SM\n",
         ALGO == kDyn ? "dynamic" : "static",
         al.numRegs,
         (size_t) al.localSizeBytes,
         at.numRegs,
         (size_t) at.localSizeBytes,
         (size_t) at.sharedSizeBytes,
         maxblk);
  (void) num_SMs;
}

// ------------------------------------------------------------------ drivers
template <int IPT, int SCOPE>
void run_correct()
{
  constexpr int NT  = (SCOPE == 0) ? kWarp : kBlock;
  const int N       = NT * IPT;
  const int partial = std::min(N, N / 2 + 3);
  bool pd[3], ps[3], id[3], sf[3];
  const int valids[3] = {N, 1, partial};
  check_valid<IPT, SCOPE>(valids[0], pd[0], ps[0], id[0], sf[0]);
  check_valid<IPT, SCOPE>(valids[1], pd[1], ps[1], id[1], sf[1]);
  check_valid<IPT, SCOPE>(valids[2], pd[2], ps[2], id[2], sf[2]);
  bool d_pair = check_pairs<IPT, kDyn, SCOPE>();
  bool s_pair = check_pairs<IPT, kStat, SCOPE>();
  bool all    = d_pair && s_pair;
  bool ident  = true;
  for (int i = 0; i < 3; ++i)
  {
    all   = all && pd[i] && ps[i] && id[i];
    ident = ident && id[i];
  }
  printf("  %s size %4d (IPT %2d): %s  (prefix d/s: full=%d/%d v1=%d/%d v%d=%d/%d  dyn==static=%d  pairs d/s=%d/%d"
         "  [oob-suffix=inf: %d/%d/%d])\n",
         SCOPE == 0 ? "warp " : "block",
         N,
         IPT,
         all ? "PASS" : "FAIL",
         pd[0], ps[0], pd[1], ps[1], partial, pd[2], ps[2], ident, d_pair, s_pair, sf[0], sf[1], sf[2]);
}

template <int IPT, int SCOPE>
void run_lat()
{
  const double gen = slope_raw<IPT, kDyn, SCOPE, 0>(); // generate-only control (algo-independent)
  const double dyn = slope_raw<IPT, kDyn, SCOPE, 1>() - gen;
  const double sta = slope_raw<IPT, kStat, SCOPE, 1>() - gen;
  printf("  %s size %4d (IPT %2d):  dynamic=%8.1f   static=%8.1f   delta=%+7.1f (%+5.1f%%)\n",
         SCOPE == 0 ? "warp " : "block",
         ((SCOPE == 0) ? kWarp : kBlock) * IPT,
         IPT,
         dyn,
         sta,
         sta - dyn,
         100.0 * (sta - dyn) / dyn);
}

template <int IPT, int SCOPE>
void run_thr(int num_SMs)
{
  const double d = throughput_gelems<IPT, kDyn, SCOPE>(num_SMs);
  const double s = throughput_gelems<IPT, kStat, SCOPE>(num_SMs);
  printf("  %s size %4d (IPT %2d):  dynamic=%7.1f   static=%7.1f  Gelem/s   ratio=%.2fx\n",
         SCOPE == 0 ? "warp " : "block",
         ((SCOPE == 0) ? kWarp : kBlock) * IPT,
         IPT,
         d,
         s,
         s / d);
}

template <int IPT, int SCOPE>
void run_res(int num_SMs)
{
  printf("  %s size %4d (IPT %2d):\n", SCOPE == 0 ? "warp " : "block", ((SCOPE == 0) ? kWarp : kBlock) * IPT, IPT);
  print_res<IPT, kDyn, SCOPE>(num_SMs);
  print_res<IPT, kStat, SCOPE>(num_SMs);
}

#define FOR_WARP(X) X(1, 0) X(2, 0) X(3, 0) X(4, 0) X(5, 0) X(6, 0) X(8, 0) X(10, 0) X(12, 0)
#define FOR_BLOCK(X) X(1, 1) X(2, 1) X(4, 1) X(8, 1)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness vs std::sort (full tile, valid_items=1, valid_items=N/2+3, pairs) ===\n");
#define C(IPT, S) run_correct<IPT, S>();
    FOR_WARP(C)
    FOR_BLOCK(C)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== LATENCY: slope cyc/call (random-input chain, generate-only control subtracted) ===\n");
#define L(IPT, S) run_lat<IPT, S>();
    FOR_WARP(L)
    FOR_BLOCK(L)
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== THROUGHPUT: one occupancy wave, fixed workload, Gelem/s ===\n");
#define T(IPT, S) run_thr<IPT, S>(p.multiProcessorCount);
    FOR_WARP(T)
    FOR_BLOCK(T)
  }
  if (mode == "res" || mode == "all")
  {
    printf("\n=== RESOURCES: registers / spills / shared memory / occupancy ===\n");
#define R(IPT, S) run_res<IPT, S>(p.multiProcessorCount);
    FOR_WARP(R)
    FOR_BLOCK(R)
  }
  return 0;
}
