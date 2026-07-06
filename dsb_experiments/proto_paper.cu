// Prototype exploring Jia et al. (arXiv:1804.06826) microbenchmarking techniques
// applied to a *composite* device-side primitive (WarpBitonicSort<float>):
//   1. clock64() read-overhead calibration + latency vs. chain-length sweep
//      (separates harness overhead from true per-call latency, like the paper).
//   2. Single-SM throughput vs. warp-level parallelism sweep (paper's ILP/TLP
//      saturation method) -> "how many resident warps hide this primitive?"
//      i.e. Little's law: warps_to_saturate ~= latency * throughput.

#include <cub/warp/warp_bitonic_sort.cuh>

#include <cuda/std/limits>

#include <cstdint>
#include <cstdio>

__device__ int g_sink[4096];

struct CustomLess
{
  template <typename T>
  __device__ bool operator()(const T& a, const T& b) const
  {
    return a < b;
  }
};

template <typename KeyT>
__device__ __forceinline__ KeyT lcg(uint32_t& s)
{
  s = 1664525u * s + 1013904223u;
  return static_cast<KeyT>(s & 0xffffu);
}

constexpr int kReps = 64;

// (1a) pure clock64 read overhead: two reads with only a compiler barrier between.
__global__ void __launch_bounds__(32) clock_overhead(long long* out)
{
  long long best = cuda::std::numeric_limits<long long>::max();
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    __syncwarp();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if ((threadIdx.x & 31) == 0)
  {
    *out = best;
  }
}

// (1b) dependency-chain latency, templated on chain length.
template <int IPT, typename KeyT, int CHAIN>
__global__ void __launch_bounds__(32) lat_dep(long long* out)
{
  const int lane = threadIdx.x & 31;
  uint32_t s     = 12345u + (lane + 7) * 2654435761u;
  KeyT keys[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    keys[i] = lcg<KeyT>(s);
  }
  cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{}); // warmup

  long long best = cuda::std::numeric_limits<long long>::max();
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    __syncwarp();
    long long t0 = clock64();
#pragma unroll
    for (int n = 0; n < CHAIN; ++n)
    {
      cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{});
    }
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
    if (keys[0] == KeyT(-1))
    {
      g_sink[lane] = static_cast<int>(keys[0]);
    }
  }
  if (lane == 0)
  {
    *out = best;
  }
}

// (2) single-SM throughput: BLOCK = 32*W warps on one SM, each warp runs an
// in-place dependency chain of SORTS sorts. Total sorts = W * SORTS. Adding warps
// raises throughput until the SM saturates (latency fully hidden).
template <int IPT, typename KeyT, int SORTS>
__global__ void thr_vs_warps(long long* out)
{
  const int lane = threadIdx.x & 31;
  uint32_t s     = 12345u + (threadIdx.x + 7) * 2654435761u;
  KeyT keys[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    keys[i] = lcg<KeyT>(s);
  }
  cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{}); // warmup
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 4
  for (int n = 0; n < SORTS; ++n)
  {
    cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{});
  }
  long long t1 = clock64();
  if (keys[0] == KeyT(-1))
  {
    g_sink[threadIdx.x & 4095] = static_cast<int>(keys[0]);
  }
  // one representative timing (thread 0); all warps run concurrently on 1 SM
  if (threadIdx.x == 0)
  {
    *out = t1 - t0;
  }
}

template <class K>
long long measure(K kernel, int block)
{
  long long* d;
  cudaMalloc(&d, sizeof(long long));
  kernel<<<1, block>>>(d);
  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    printf("ERROR %s\n", cudaGetErrorString(cudaGetLastError()));
    return -1;
  }
  long long v = 0;
  cudaMemcpy(&v, d, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(d);
  return v;
}

int main()
{
  constexpr int IPT = 4; // len=128 float
  const int len     = IPT * 32;

  printf("=== (1) clock64 overhead + latency vs. chain length (WarpBitonicSort float len=%d) ===\n", len);
  long long ovh = measure(clock_overhead, 32);
  printf("  clock64 read overhead: %lld cyc\n", ovh);
  long long c1  = measure(lat_dep<IPT, float, 1>, 32);
  long long c2  = measure(lat_dep<IPT, float, 2>, 32);
  long long c4  = measure(lat_dep<IPT, float, 4>, 32);
  long long c8  = measure(lat_dep<IPT, float, 8>, 32);
  long long c16 = measure(lat_dep<IPT, float, 16>, 32);
  long long c32 = measure(lat_dep<IPT, float, 32>, 32);
  long long c64 = measure(lat_dep<IPT, float, 64>, 32);
  printf("  chain=1  total=%-7lld  naive/call=%7.1f  overhead-subtracted/call=%7.1f\n", c1, (double) c1, (double) (c1 - ovh));
  printf("  chain=2  total=%-7lld  naive/call=%7.1f\n", c2, c2 / 2.0);
  printf("  chain=4  total=%-7lld  naive/call=%7.1f\n", c4, c4 / 4.0);
  printf("  chain=8  total=%-7lld  naive/call=%7.1f\n", c8, c8 / 8.0);
  printf("  chain=16 total=%-7lld  naive/call=%7.1f\n", c16, c16 / 16.0);
  printf("  chain=32 total=%-7lld  naive/call=%7.1f\n", c32, c32 / 32.0);
  printf("  chain=64 total=%-7lld  naive/call=%7.1f\n", c64, c64 / 64.0);
  double slope = (double) (c64 - c1) / (64 - 1); // overhead-independent per-call latency
  printf("  -> slope (c64-c1)/63 = %.1f cyc/call  == true latency, independent of harness overhead\n", slope);

  printf("\n=== (2) single-SM throughput vs. warp-parallelism (Little's law saturation) ===\n");
  constexpr int SORTS = 512;
  printf("  each warp: %d in-place sorts; total sorts = W*%d\n", SORTS, SORTS);
  double peak = 0;
  int Ws[]    = {1, 2, 4, 8, 16, 32, 48, 64};
  for (int wi = 0; wi < 8; ++wi)
  {
    int W       = Ws[wi];
    long long cyc;
    switch (W)
    {
      case 1:  cyc = measure(thr_vs_warps<IPT, float, SORTS>, 32);   break;
      case 2:  cyc = measure(thr_vs_warps<IPT, float, SORTS>, 64);   break;
      case 4:  cyc = measure(thr_vs_warps<IPT, float, SORTS>, 128);  break;
      case 8:  cyc = measure(thr_vs_warps<IPT, float, SORTS>, 256);  break;
      case 16: cyc = measure(thr_vs_warps<IPT, float, SORTS>, 512);  break;
      case 32: cyc = measure(thr_vs_warps<IPT, float, SORTS>, 1024); break;
      case 48: cyc = measure(thr_vs_warps<IPT, float, SORTS>, 1536); break;
      default: cyc = measure(thr_vs_warps<IPT, float, SORTS>, 2048); break;
    }
    double sorts   = double(W) * SORTS;
    double per_cyc = sorts / double(cyc);
    if (per_cyc > peak)
    {
      peak = per_cyc;
    }
    printf("  W=%-2d warps  cyc=%-8lld  %.4f sorts/cyc  (%.1f%% of peak)\n", W, cyc, per_cyc, 100.0 * per_cyc / peak);
  }
  printf("  -> single-call latency ~= %.0f cyc; peak throughput ~= %.4f sorts/cyc\n", slope, peak);
  printf("  -> Little's law warps-to-saturate ~= latency*throughput = %.1f warps\n", slope * peak);
  return 0;
}
