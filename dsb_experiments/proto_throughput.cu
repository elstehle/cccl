// Prototype: THROUGHPUT measurement for device-side primitives.
//
// Demonstrates the two properties a robust throughput benchmark needs:
//   (T1) FIXED workload (independent of compiler-chosen occupancy) so the amount
//        of work does not silently change between toolkit/library versions.
//   (T2) ELEMENT-NORMALIZED result (elem/s), which stays stable even if occupancy
//        changes, unlike a raw kernel-time metric.
//
// It also prints the occupancy-derived "one wave" grid to show how a raw-time,
// occupancy-sized workload (what the current WarpReduce bench uses) would move
// around with occupancy while elem/s does not.

#include <cub/warp/warp_bitonic_sort.cuh>

#include <cuda/ptx>

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

// One warp sorts LEN keys; grid*block warps sort (grid*block/32)*LEN keys total.
// UNROLL sorts per thread amortize loop overhead (data-oblivious => safe, no chain).
template <int IPT, typename KeyT, int UNROLL>
__global__ void sort_throughput(int outer)
{
  const int lane = threadIdx.x & 31;
  uint32_t s     = 12345u + (blockIdx.x * blockDim.x + threadIdx.x + 7) * 2654435761u;
  KeyT acc       = KeyT{};
  for (int o = 0; o < outer; ++o)
  {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
    {
      KeyT keys[IPT];
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        keys[i] = lcg<KeyT>(s);
      }
      cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{});
      acc += keys[0] + keys[IPT - 1];
    }
  }
  if (cuda::ptx::get_sreg_smid() == static_cast<uint32_t>(-1))
  {
    g_sink[lane] = static_cast<int>(acc);
  }
}

template <int IPT, int UNROLL, typename KeyT>
void bench(const char* label, int len, int num_sms)
{
  constexpr int block = 128;
  auto kernel         = sort_throughput<IPT, KeyT, UNROLL>;

  int max_blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block, 0);
  const int wave_grid = max_blocks * num_sms; // "one wave" (occupancy-sized) grid

  // (T1) FIXED workload: target a fixed number of warp-sorts regardless of occupancy.
  const long long target_sorts = 1LL << 26; // 67M warp-level sorts
  const int warps_per_block    = block / 32;
  // choose grid so grid*warps_per_block*UNROLL*outer == target_sorts, with a modest grid
  const int grid  = wave_grid; // launch one wave, use 'outer' to reach fixed workload
  const long long sorts_per_launch_per_outer = (long long) grid * warps_per_block * UNROLL;
  const int outer = (int) ((target_sorts + sorts_per_launch_per_outer - 1) / sorts_per_launch_per_outer);
  const long long total_sorts = sorts_per_launch_per_outer * outer;
  const long long total_elems = total_sorts * len;

  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  sort_throughput<IPT, KeyT, UNROLL><<<grid, block>>>(outer); // warmup
  cudaDeviceSynchronize();

  float best_ms = 1e30f;
  for (int rep = 0; rep < 7; ++rep)
  {
    cudaEventRecord(a);
    sort_throughput<IPT, KeyT, UNROLL><<<grid, block>>>(outer);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    if (ms < best_ms)
    {
      best_ms = ms;
    }
  }
  double elems_per_s = total_elems / (best_ms * 1e-3);
  cudaFuncAttributes attr{};
  cudaFuncGetAttributes(&attr, kernel);
  printf("  %-14s len=%-4d regs=%-3d maxBlk/SM=%-2d waveGrid=%-6d | fixed workload=%.0fM elems  time=%7.3f ms  %7.1f G elem/s\n",
         label,
         len,
         attr.numRegs,
         max_blocks,
         wave_grid,
         total_elems / 1e6,
         best_ms,
         elems_per_s / 1e9);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
}

int main()
{
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);
  printf("=== WarpBitonicSort<float> throughput: FIXED workload, element-normalized ===\n");
  printf("device: %s, %d SMs\n", p.name, p.multiProcessorCount);
  for (int rep = 0; rep < 2; ++rep)
  {
    printf("--- pass %d ---\n", rep);
    bench<2, 8, float>("len=64", 64, p.multiProcessorCount);
    bench<4, 4, float>("len=128", 128, p.multiProcessorCount);
    bench<8, 2, float>("len=256", 256, p.multiProcessorCount);
  }
  return 0;
}
