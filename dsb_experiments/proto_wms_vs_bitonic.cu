// WarpMergeSort vs WarpBitonicSort — latency and throughput head-to-head, single warp scope,
// keys-only float, sizes 32..384 (IPT 1,2,3,4,5,6,8,10,12). Node umb-b200-261, sm_100.
//
// Methodology (per DEVICE_SIDE_BENCHMARKING_ISSUE.md):
//   * Latency uses the appropriate critical-path method PER PRIMITIVE:
//       - WarpBitonicSort is DATA-OBLIVIOUS -> an in-place sort->sort dependency chain is a genuine
//         RAW critical path (a light per-iter perturbation keeps it non-idempotent so the compiler
//         cannot collapse the chain). No input generation in the timed region.
//       - WarpMergeSort is DATA-DEPENDENT (less work on sorted input) -> back-to-back random input,
//         chain serialized on the previous output, with a generate-only control subtracted.
//     Both report marginal cyc/call via the chain-length slope; single warp; clock64.
//   * Throughput: identical fixed workload for both (one occupancy wave x NUM_ITER sorts of fresh
//     random register data, sink()-guarded against DCE), element-normalized to Gelem/s.
// The sink() trick (unprovable smid==-1 guard + FP-order-sensitive sum) is copied from
// nvbench_helper/device_side_benchmark.cuh so the oblivious sort's result stays live.
//
// Modes: ./proto_wms_vs_bitonic [correct|lat|thr|all]

#include <cub/warp/warp_bitonic_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda/ptx>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

constexpr int kWarp = 32;

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

// ------------------------------------------------------------------ latency: bitonic (in-place chain)
constexpr int kLatReps = 32;

template <int IPT>
__global__ void __launch_bounds__(kWarp) lat_bitonic(unsigned seed0, int chain, long long* out)
{
  const int lane = threadIdx.x & 31;
  unsigned s     = seed0 + (lane + 7) * 2654435761u;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s);
  }
  cub::detail::WarpBitonicSort<IPT, float>::Sort(k, CustomLess{}); // warmup
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
      k[0] = k[0] + (float) n; // per-iter perturbation: non-idempotent RAW chain, cannot collapse
      cub::detail::WarpBitonicSort<IPT, float>::Sort(k, CustomLess{});
    }
    asm volatile("" ::: "memory");
    sink(k);
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (lane == 0)
  {
    *out = best;
  }
}

// ------------------------------------------------------------------ latency: merge (random input)
template <int IPT, int DO_SORT>
__global__ void __launch_bounds__(kWarp) lat_merge(unsigned seed0, int chain, long long* out)
{
  using WMS = cub::WarpMergeSort<float, IPT, kWarp>;
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
      unsigned ss = s ^ __float_as_uint(k[0]);
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

// ------------------------------------------------------------------ throughput (both), one wave
constexpr int kThrBlock = 128; // 4 warps
constexpr int kThrIters = 200;

// WHICH: 0 = WarpMergeSort, 1 = WarpBitonicSort
template <int IPT, int WHICH>
__global__ void __launch_bounds__(kThrBlock) thr_kernel(int num_iter)
{
  using WMS = cub::WarpMergeSort<float, IPT, kWarp>;
  __shared__ typename WMS::TempStorage temp[kThrBlock / kWarp];
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
    if constexpr (WHICH == 0)
    {
      WMS(temp[warp]).Sort(k, CustomLess{});
    }
    else
    {
      cub::detail::WarpBitonicSort<IPT, float>::Sort(k, CustomLess{});
    }
    sink(k);
  }
}

// ------------------------------------------------------------------ correctness
template <int IPT, int WHICH>
__global__ void __launch_bounds__(kWarp) correct_k(const float* in, float* out)
{
  using WMS = cub::WarpMergeSort<float, IPT, kWarp>;
  __shared__ typename WMS::TempStorage temp;
  const int lane = threadIdx.x & 31;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in[lane * IPT + i];
  }
  if constexpr (WHICH == 0)
  {
    WMS(temp).Sort(k, CustomLess{});
  }
  else
  {
    cub::detail::WarpBitonicSort<IPT, float>::Sort(k, CustomLess{});
  }
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out[lane * IPT + i] = k[i];
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

// bitonic arranges keys STRIPED across the warp (lane l item i -> global l + i*32); merge is blocked
// (lane l item i -> l*IPT + i). Gather accordingly for the reference check.
template <int IPT, int WHICH>
bool check_one()
{
  const int N = kWarp * IPT;
  std::vector<float> in(N), out(N), ref, got(N);
  unsigned s = 4242u + IPT;
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
  correct_k<IPT, WHICH><<<1, kWarp>>>(d_in, d_out);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  // out[lane*IPT+i] as written by the kernel. Reassemble the global sorted order per arrangement.
  for (int lane = 0; lane < kWarp; ++lane)
  {
    for (int i = 0; i < IPT; ++i)
    {
      const int global = (WHICH == 1) ? (lane + i * kWarp) : (lane * IPT + i);
      got[global]      = out[lane * IPT + i];
    }
  }
  bool ok = true;
  for (int i = 0; i < N; ++i)
  {
    if (got[i] != ref[i])
    {
      ok = false;
      break;
    }
  }
  cudaFree(d_in);
  cudaFree(d_out);
  return ok;
}

template <int IPT>
double slope_bitonic()
{
  long long* d;
  CHECK(cudaMalloc(&d, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], y[5];
  for (int i = 0; i < 5; ++i)
  {
    lat_bitonic<IPT><<<1, kWarp>>>(12345u, chains[i], d);
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

template <int IPT, int DO>
double slope_merge_raw()
{
  long long* d;
  CHECK(cudaMalloc(&d, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], y[5];
  for (int i = 0; i < 5; ++i)
  {
    lat_merge<IPT, DO><<<1, kWarp>>>(12345u, chains[i], d);
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

template <int IPT, int WHICH>
double throughput_gelems(int num_SMs)
{
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_kernel<IPT, WHICH>, kThrBlock, 0));
  const int grid = maxblk * num_SMs;
  thr_kernel<IPT, WHICH><<<grid, kThrBlock>>>(kThrIters); // warmup
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best_ms = 1e30f;
  for (int rep = 0; rep < 5; ++rep)
  {
    cudaEventRecord(e0);
    thr_kernel<IPT, WHICH><<<grid, kThrBlock>>>(kThrIters);
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best_ms = std::min(best_ms, ms);
  }
  const double warps  = (double) grid * (kThrBlock / kWarp);
  const double elems  = warps * (double) kThrIters * (kWarp * IPT);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return elems / (best_ms * 1e-3) / 1e9;
}

template <int IPT>
void run_lat()
{
  const double bit  = slope_bitonic<IPT>();
  const double mf   = slope_merge_raw<IPT, 1>();
  const double mg   = slope_merge_raw<IPT, 0>();
  const double merge = mf - mg;
  printf("  size %3d (IPT %2d):  WarpMergeSort=%7.1f   WarpBitonicSort=%7.1f   speedup=%.2fx\n",
         kWarp * IPT,
         IPT,
         merge,
         bit,
         merge / bit);
}

template <int IPT>
void run_thr(int num_SMs)
{
  const double m = throughput_gelems<IPT, 0>(num_SMs);
  const double b = throughput_gelems<IPT, 1>(num_SMs);
  printf("  size %3d (IPT %2d):  WarpMergeSort=%7.1f   WarpBitonicSort=%7.1f  Gelem/s   speedup=%.2fx\n",
         kWarp * IPT,
         IPT,
         m,
         b,
         b / m);
}

template <int IPT>
void run_correct()
{
  const bool m = check_one<IPT, 0>();
  const bool b = check_one<IPT, 1>();
  printf("  size %3d (IPT %2d):  merge %s   bitonic %s\n", kWarp * IPT, IPT, m ? "PASS" : "FAIL", b ? "PASS" : "FAIL");
}

#define FOR_SIZES(X) X(1) X(2) X(3) X(4) X(5) X(6) X(8) X(10) X(12)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness vs std::sort ===\n");
#define C(IPT) run_correct<IPT>();
    FOR_SIZES(C)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== LATENCY: single-warp slope cyc/call (merge=random-input, bitonic=in-place chain) ===\n");
#define L(IPT) run_lat<IPT>();
    FOR_SIZES(L)
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== THROUGHPUT: fixed workload, one wave, Gelem/s ===\n");
#define T(IPT) run_thr<IPT>(p.multiProcessorCount);
    FOR_SIZES(T)
  }
  return 0;
}
