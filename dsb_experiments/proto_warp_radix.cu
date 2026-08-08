// WarpRadixSort (user-provided, ballot-ranked LSD, RadixBits=5 -> 7 passes for float) vs
// WarpMergeSort vs WarpBitonicSort. Latency + throughput, single-warp scope, keys-only float,
// sizes 32..384 (IPT 1..12). Methodology as proto_wms_vs_bitonic.cu: merge & radix use the
// back-to-back random-input chain with a generate-only control subtracted (radix's control flow
// is data-oblivious, but the same harness keeps the numbers comparable and covers its
// store->sort->load register contract); bitonic uses the in-place dependency chain + sink().
// Radix pays its reg->smem store, Sort(), smem->reg load inside the timed region -- that is the
// cost of using it in the same "registers in, registers out" role as the other two.
//
// Modes: ./proto_warp_radix [correct|lat|thr|all]

#include <cub/warp/warp_bitonic_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda/ptx>

#include "warp_radix_sort.cuh"

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

constexpr int kLatReps = 32;

// ------------------------------------------------------------------ latency kernels
template <int IPT, int DO>
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
  if (DO)
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
      if (DO)
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

template <int IPT, int DO>
__global__ void __launch_bounds__(kWarp) lat_radix(unsigned seed0, int chain, long long* out)
{
  __shared__ float buf[kWarp * IPT];
  __shared__ float tmp[kWarp * IPT];
  const int lane = threadIdx.x & 31;
  unsigned s     = seed0 + (lane + 7) * 2654435761u;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s);
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
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          buf[lane * IPT + i] = k[i];
        }
        __syncwarp();
        WarpRadixSort<float>{}.Sort(buf, tmp, kWarp * IPT);
        __syncwarp();
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          k[i] = buf[lane * IPT + i];
        }
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
  cub::detail::WarpBitonicSort<IPT, float>::Sort(k, CustomLess{});
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
      k[0] = k[0] + (float) n;
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

// ------------------------------------------------------------------ throughput (WHICH: 0=merge,1=bitonic,2=radix)
constexpr int kThrBlock = 128;
constexpr int kThrIters = 200;

template <int IPT, int WHICH>
__global__ void __launch_bounds__(kThrBlock) thr_kernel(int num_iter)
{
  using WMS = cub::WarpMergeSort<float, IPT, kWarp>;
  __shared__ typename WMS::TempStorage temp[kThrBlock / kWarp];
  __shared__ float rbuf[kThrBlock / kWarp][kWarp * IPT];
  __shared__ float rtmp[kThrBlock / kWarp][kWarp * IPT];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
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
    else if constexpr (WHICH == 1)
    {
      cub::detail::WarpBitonicSort<IPT, float>::Sort(k, CustomLess{});
    }
    else
    {
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        rbuf[warp][lane * IPT + i] = k[i];
      }
      __syncwarp();
      WarpRadixSort<float>{}.Sort(rbuf[warp], rtmp[warp], kWarp * IPT);
      __syncwarp();
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        k[i] = rbuf[warp][lane * IPT + i];
      }
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
  __shared__ float rbuf[kWarp * IPT];
  __shared__ float rtmp[kWarp * IPT];
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
  else if constexpr (WHICH == 1)
  {
    cub::detail::WarpBitonicSort<IPT, float>::Sort(k, CustomLess{});
  }
  else
  {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      rbuf[lane * IPT + i] = k[i];
    }
    __syncwarp();
    WarpRadixSort<float>{}.Sort(rbuf, rtmp, kWarp * IPT);
    __syncwarp();
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = rbuf[lane * IPT + i];
    }
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

template <int IPT, int WHICH, int DO>
double slope_raw()
{
  long long* d;
  CHECK(cudaMalloc(&d, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], y[5];
  for (int i = 0; i < 5; ++i)
  {
    if constexpr (WHICH == 0)
    {
      lat_merge<IPT, DO><<<1, kWarp>>>(12345u, chains[i], d);
    }
    else if constexpr (WHICH == 1)
    {
      lat_bitonic<IPT><<<1, kWarp>>>(12345u, chains[i], d);
    }
    else
    {
      lat_radix<IPT, DO><<<1, kWarp>>>(12345u, chains[i], d);
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

template <int IPT, int WHICH>
double throughput_gelems(int num_SMs)
{
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_kernel<IPT, WHICH>, kThrBlock, 0));
  const int grid = maxblk * num_SMs;
  thr_kernel<IPT, WHICH><<<grid, kThrBlock>>>(kThrIters);
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
  const double warps = (double) grid * (kThrBlock / kWarp);
  const double elems = warps * (double) kThrIters * (kWarp * IPT);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return elems / (best_ms * 1e-3) / 1e9;
}

template <int IPT>
void run_lat()
{
  const double merge = slope_raw<IPT, 0, 1>() - slope_raw<IPT, 0, 0>();
  const double bit   = slope_raw<IPT, 1, 1>();
  const double radix = slope_raw<IPT, 2, 1>() - slope_raw<IPT, 2, 0>();
  printf("  size %3d (IPT %2d):  merge=%7.1f  bitonic=%7.1f  radix=%7.1f   radix vs best: %.2fx\n",
         kWarp * IPT,
         IPT,
         merge,
         bit,
         radix,
         radix / std::min(merge, bit));
}

template <int IPT>
void run_thr(int num_SMs)
{
  const double m = throughput_gelems<IPT, 0>(num_SMs);
  const double b = throughput_gelems<IPT, 1>(num_SMs);
  const double r = throughput_gelems<IPT, 2>(num_SMs);
  printf("  size %3d (IPT %2d):  merge=%7.1f  bitonic=%7.1f  radix=%7.1f  Gelem/s   radix vs best: %.2fx\n",
         kWarp * IPT,
         IPT,
         m,
         b,
         r,
         r / std::max(m, b));
}

template <int IPT>
void run_correct()
{
  const bool m = check_one<IPT, 0>();
  const bool b = check_one<IPT, 1>();
  const bool r = check_one<IPT, 2>();
  printf("  size %3d (IPT %2d):  merge %s  bitonic %s  radix %s\n",
         kWarp * IPT,
         IPT,
         m ? "PASS" : "FAIL",
         b ? "PASS" : "FAIL",
         r ? "PASS" : "FAIL");
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
    printf("\n=== LATENCY: single-warp slope cyc/call ===\n");
#define L(IPT) run_lat<IPT>();
    FOR_SIZES(L)
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== THROUGHPUT: one wave, Gelem/s ===\n");
#define T(IPT) run_thr<IPT>(p.multiProcessorCount);
    FOR_SIZES(T)
  }
  return 0;
}
