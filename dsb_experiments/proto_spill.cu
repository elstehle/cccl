// Prototype: does register pressure raise single-warp latency only via spilling?
//
// Same WarpBitonicSort<float> len=128 dependency-chain latency kernel, compiled
// repeatedly under different register budgets (-maxrregcount). For each budget we
// print numRegs, spill/local bytes (cudaFuncGetAttributes.localSizeBytes) and the
// measured per-call latency. Expectation: reducing the budget lowers numRegs with
// FLAT latency until it forces spills, at which point latency jumps.

#include <cub/warp/warp_bitonic_sort.cuh>

#include <cuda/std/limits>

#include <cstdint>
#include <cstdio>

__device__ int g_sink[64];

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

constexpr int kReps = 128;

template <int IPT, typename KeyT, int N>
__global__ void chain(long long* out)
{
  const int lane = threadIdx.x & 31;
  uint32_t s     = 12345u + (lane + 7) * 2654435761u;
  KeyT keys[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    keys[i] = lcg<KeyT>(s);
  }
  cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{});

  long long best = cuda::std::numeric_limits<long long>::max();
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    __syncwarp();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll
    for (int n = 0; n < N; ++n)
    {
      cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{});
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      asm volatile("" : : "f"(keys[i]));
    }
    asm volatile("" ::: "memory");
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

int main()
{
  constexpr int IPT = 16; // len=512: heavy enough that a low register budget forces spills
  constexpr int N1  = 1;
  constexpr int N2  = 64;
  auto k1           = chain<IPT, float, N1>;
  auto k2           = chain<IPT, float, N2>;

  long long* d;
  cudaMalloc(&d, sizeof(long long));
  k1<<<1, 32>>>(d);
  cudaDeviceSynchronize();
  long long c1 = 0;
  cudaMemcpy(&c1, d, sizeof(long long), cudaMemcpyDeviceToHost);
  k2<<<1, 32>>>(d);
  cudaDeviceSynchronize();
  long long c2 = 0;
  cudaMemcpy(&c2, d, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(d);

  cudaFuncAttributes a{};
  cudaFuncGetAttributes(&a, k2);
  double slope = double(c2 - c1) / (N2 - N1);
  printf("regs=%-3d  local(spill)=%-4zu B  slope_latency=%7.1f cyc/call  cold(N=1)=%lld\n",
         a.numRegs,
         a.localSizeBytes,
         slope,
         c1);
  return 0;
}
