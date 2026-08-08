// Sub-warp WarpBitonicSort (branch exp/sub-warp-bitonic-sort) vs sub-warp WarpMergeSort.
// Small inputs: LogicalWarpThreads LW in {4,8,16} x IPT in {1,2,4} -> segment sizes 4..64.
// One hardware warp runs 32/LW independent sub-sorts (the realistic sub-warp deployment).
// float keys-only, B200. Latency: bitonic = in-place chain + sink (data-oblivious);
// merge = back-to-back random input, generate-only control subtracted. Throughput: one wave,
// 128-thread blocks, all sub-warps sorting, element-normalized.

#include <cub/warp/warp_bitonic_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda/ptx>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <vector>

__device__ int g_dcesink[16];

struct Less
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

template <int IPT, int LW>
__device__ __forceinline__ void bsort(float (&k)[IPT])
{
  using BS = cub::detail::WarpBitonicSort<float, IPT, LW>;
  typename BS::TempStorage ts;
  BS(ts).Sort(k, Less{});
}

constexpr int kLatReps = 32;

// WHICH: 0 = merge, 1 = bitonic. One hw warp = 32/LW concurrent sub-sorts.
template <int LW, int IPT, int WHICH>
__global__ void __launch_bounds__(32) correct_k(const float* in, float* out)
{
  using WMS = cub::WarpMergeSort<float, IPT, LW>;
  __shared__ typename WMS::TempStorage temp[32 / LW];
  const int lane = threadIdx.x & 31;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in[lane * IPT + i]; // harness-blocked; segment mapping handled on host
  }
  if constexpr (WHICH == 0)
  {
    WMS(temp[lane / LW]).Sort(k, Less{});
  }
  else
  {
    bsort<IPT, LW>(k);
  }
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out[lane * IPT + i] = k[i];
  }
}

template <int LW, int IPT, int DO>
__global__ void __launch_bounds__(32) lat_merge(unsigned seed0, int chain, long long* out)
{
  using WMS = cub::WarpMergeSort<float, IPT, LW>;
  __shared__ typename WMS::TempStorage temp[32 / LW];
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
    WMS(temp[lane / LW]).Sort(k, Less{});
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
        WMS(temp[lane / LW]).Sort(k, Less{});
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

template <int LW, int IPT>
__global__ void __launch_bounds__(32) lat_bitonic(unsigned seed0, int chain, long long* out)
{
  const int lane = threadIdx.x & 31;
  unsigned s     = seed0 + (lane + 7) * 2654435761u;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s);
  }
  bsort<IPT, LW>(k);
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
      bsort<IPT, LW>(k);
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

constexpr int kThrBlock = 128;
constexpr int kThrIters = 200;

template <int LW, int IPT, int WHICH>
__global__ void __launch_bounds__(kThrBlock) thr_kernel(int num_iter)
{
  using WMS = cub::WarpMergeSort<float, IPT, LW>;
  __shared__ typename WMS::TempStorage temp[kThrBlock / LW];
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
    if constexpr (WHICH == 0)
    {
      WMS(temp[threadIdx.x / LW]).Sort(k, Less{});
    }
    else
    {
      bsort<IPT, LW>(k);
    }
    sink(k);
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

// per-segment set+sorted check; WHICH==1 (bitonic) output is striped within the logical warp
template <int LW, int IPT, int WHICH>
bool check_one()
{
  const int N = 32 * IPT;
  std::vector<float> in(N), out(N);
  unsigned s = 99u + LW * 7 + IPT;
  for (int i = 0; i < N; ++i)
  {
    s     = 1664525u * s + 1013904223u;
    in[i] = (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
  }
  float *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CHECK(cudaMemcpy(d_in, in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  correct_k<LW, IPT, WHICH><<<1, 32>>>(d_in, d_out);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int seg = 0; seg < 32 / LW; ++seg)
  {
    std::vector<float> want, got;
    for (int l = 0; l < LW; ++l)
    {
      for (int i = 0; i < IPT; ++i)
      {
        const int lane = seg * LW + l;
        want.push_back(in[lane * IPT + i]);
        // logical position within segment: merge = blocked (l*IPT+i); bitonic = striped (l + i*LW)
        const int pos = (WHICH == 1) ? (l + i * LW) : (l * IPT + i);
        (void) pos;
        got.push_back(out[lane * IPT + i]);
      }
    }
    std::sort(want.begin(), want.end());
    // reassemble got in logical order
    std::vector<float> lin(LW * IPT);
    for (int l = 0; l < LW; ++l)
    {
      for (int i = 0; i < IPT; ++i)
      {
        const int pos = (WHICH == 1) ? (l + i * LW) : (l * IPT + i);
        lin[pos]      = out[(seg * LW + l) * IPT + i];
      }
    }
    for (int i = 0; i < LW * IPT; ++i)
    {
      if (lin[i] != want[i])
      {
        ok = false;
      }
    }
  }
  cudaFree(d_in);
  cudaFree(d_out);
  return ok;
}

template <int LW, int IPT, int WHICH, int DO>
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
      lat_merge<LW, IPT, DO><<<1, 32>>>(12345u, chains[i], d);
    }
    else
    {
      lat_bitonic<LW, IPT><<<1, 32>>>(12345u, chains[i], d);
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

template <int LW, int IPT, int WHICH>
double thr_gelems(int num_SMs)
{
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_kernel<LW, IPT, WHICH>, kThrBlock, 0));
  const int grid = maxblk * num_SMs;
  thr_kernel<LW, IPT, WHICH><<<grid, kThrBlock>>>(kThrIters);
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best_ms = 1e30f;
  for (int r = 0; r < 5; ++r)
  {
    cudaEventRecord(e0);
    thr_kernel<LW, IPT, WHICH><<<grid, kThrBlock>>>(kThrIters);
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best_ms = std::min(best_ms, ms);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  const double elems = (double) grid * kThrBlock * IPT * kThrIters;
  return elems / (best_ms * 1e-3) / 1e9;
}

template <int LW, int IPT>
void run_cfg(int num_SMs)
{
  const bool cm      = check_one<LW, IPT, 0>();
  const bool cb      = check_one<LW, IPT, 1>();
  const double merge = slope_raw<LW, IPT, 0, 1>() - slope_raw<LW, IPT, 0, 0>();
  const double bit   = slope_raw<LW, IPT, 1, 1>();
  const double tm    = thr_gelems<LW, IPT, 0>(num_SMs);
  const double tb    = thr_gelems<LW, IPT, 1>(num_SMs);
  printf("  LW=%2d IPT=%d (seg %3d): corr[m=%s b=%s]  LAT merge=%7.1f bitonic=%7.1f (%.2fx)   THR merge=%6.1f bitonic=%6.1f (%.2fx)\n",
         LW,
         IPT,
         LW * IPT,
         cm ? "ok" : "X",
         cb ? "ok" : "X",
         merge,
         bit,
         merge / bit,
         tm,
         tb,
         tb / tm);
}

int main()
{
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  run_cfg<4, 1>(p.multiProcessorCount);
  run_cfg<4, 2>(p.multiProcessorCount);
  run_cfg<4, 4>(p.multiProcessorCount);
  run_cfg<8, 1>(p.multiProcessorCount);
  run_cfg<8, 2>(p.multiProcessorCount);
  run_cfg<8, 4>(p.multiProcessorCount);
  run_cfg<16, 1>(p.multiProcessorCount);
  run_cfg<16, 2>(p.multiProcessorCount);
  run_cfg<16, 4>(p.multiProcessorCount);
  return 0;
}
