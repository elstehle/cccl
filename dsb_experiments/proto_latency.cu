// Prototype: precise device-side LATENCY measurement for warp/block primitives.
//
// Goals demonstrated here:
//   1. Measure per-call latency in *clock cycles* via clock64() inside a single warp,
//      eliminating kernel-launch overhead and GPU-clock-rate variability.
//   2. Use a true data-dependency chain so a *statically unrolled* body cannot overlap
//      calls (no cross-iteration ILP) -> exposes the real critical-path latency.
//   3. Quantify how much cross-iteration ILP would hide latency (INDEP variant), and how
//      much input generation contaminates a timed region (GENONLY variant).
//
// Build: nvcc -std=c++17 -arch=sm_100 -O3 <includes> proto_latency.cu -o proto_latency

#include <cub/warp/warp_bitonic_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda/std/limits>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

__device__ int g_sink[256];

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

constexpr int kReps = 64; // outer (untimed) repetitions; we keep the min

// ---------------------------------------------------------------------------
// DEP: dependency chain. Each Sort is in-place, so call n reads call n-1's
// output. Bitonic sort is data-oblivious (fixed network) => constant work,
// and repeated sorting of sorted data cannot be optimized away (uses shfl).
// ---------------------------------------------------------------------------
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
  cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{}); // warmup, untimed

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

// ---------------------------------------------------------------------------
// INDEP: ILP independent. Cycle through ILP separate arrays; consecutive Sorts
// touch different arrays, so the hardware can overlap them. Reveals how much
// latency cross-iteration ILP hides. Inputs prepared outside the timed region.
// ---------------------------------------------------------------------------
template <int IPT, typename KeyT, int CHAIN, int ILP>
__global__ void __launch_bounds__(32) lat_indep(long long* out)
{
  const int lane = threadIdx.x & 31;
  uint32_t s     = 12345u + (lane + 7) * 2654435761u;
  KeyT keys[ILP][IPT];
#pragma unroll
  for (int j = 0; j < ILP; ++j)
  {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      keys[j][i] = lcg<KeyT>(s);
    }
    cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys[j], CustomLess{}); // warmup
  }

  long long best = cuda::std::numeric_limits<long long>::max();
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    __syncwarp();
    long long t0 = clock64();
#pragma unroll
    for (int n = 0; n < CHAIN; ++n)
    {
      cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys[n % ILP], CustomLess{});
    }
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      if (keys[j][0] == KeyT(-1))
      {
        g_sink[lane] = static_cast<int>(keys[j][0]);
      }
    }
  }
  if (lane == 0)
  {
    *out = best;
  }
}

// ---------------------------------------------------------------------------
// PROD: replicates the *current* production methodology (device_side_benchmark.cuh):
// fresh data generated INSIDE the timed loop, sort, then feed output to a sink,
// with the loop kept rolled (#pragma unroll 1). Timed here with clock64 so we can
// compare, in cycles, what the production harness actually measures vs. the
// gold-standard dependency chain (DEP).
// ---------------------------------------------------------------------------
template <int IPT, typename KeyT, int CHAIN>
__global__ void __launch_bounds__(32) lat_prod(long long* out)
{
  const int lane = threadIdx.x & 31;
  uint32_t s     = 12345u + (lane + 7) * 2654435761u;
  KeyT keys[IPT];
  long long acc  = 0;
  long long best = cuda::std::numeric_limits<long long>::max();
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    __syncwarp();
    long long t0 = clock64();
#pragma unroll 1
    for (int n = 0; n < CHAIN; ++n)
    {
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        keys[i] = lcg<KeyT>(s);
      }
      cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{});
      acc += static_cast<long long>(keys[0]); // keep sort output live (sink-like)
    }
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (lane == 0)
  {
    g_sink[0] = static_cast<int>(acc);
    *out      = best;
  }
}

// ---------------------------------------------------------------------------
// Fair comparison harness: WarpMergeSort (comparison-based, data-DEPENDENT) with
// the same PROD-style methodology (fresh random input each call). Because merge
// sort's work depends on the input, we must feed varied input each call rather
// than reuse a dependency chain (which would measure the already-sorted best case).
// ---------------------------------------------------------------------------
template <int IPT, typename KeyT, int CHAIN>
__global__ void __launch_bounds__(32) lat_prod_merge(long long* out)
{
  using WMS      = cub::WarpMergeSort<KeyT, IPT, 32>;
  __shared__ typename WMS::TempStorage temp;
  const int lane = threadIdx.x & 31;
  uint32_t s     = 12345u + (lane + 7) * 2654435761u;
  KeyT keys[IPT];
  long long acc  = 0;
  long long best = cuda::std::numeric_limits<long long>::max();
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    __syncwarp();
    long long t0 = clock64();
#pragma unroll 1
    for (int n = 0; n < CHAIN; ++n)
    {
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        keys[i] = lcg<KeyT>(s);
      }
      WMS(temp).Sort(keys, CustomLess{});
      acc += static_cast<long long>(keys[0]);
    }
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (lane == 0)
  {
    g_sink[0] = static_cast<int>(acc);
    *out      = best;
  }
}

template <typename KernelT>
double run(KernelT kernel, int chain, const char* label, int len)
{
  long long* d_out;
  cudaMalloc(&d_out, sizeof(long long));
  kernel<<<1, 32>>>(d_out);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
  {
    printf("  %-28s len=%-4d ERROR: %s\n", label, len, cudaGetErrorString(e));
    cudaFree(d_out);
    return 0.0;
  }
  long long cyc = 0;
  cudaMemcpy(&cyc, d_out, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFuncAttributes attr{};
  cudaFuncGetAttributes(&attr, kernel);
  double per_call = double(cyc) / chain;
  printf("  %-28s len=%-4d chain=%-3d  %9.1f cyc total  %8.2f cyc/call   regs=%d smem=%zu spill=%zu\n",
         label,
         len,
         chain,
         double(cyc),
         per_call,
         attr.numRegs,
         attr.sharedSizeBytes,
         attr.localSizeBytes);
  cudaFree(d_out);
  return per_call;
}

int main()
{
  constexpr int CHAIN = 32;
  printf("=== WarpBitonicSort float latency (cycles), single warp, clock64 ===\n");

  for (int pass = 0; pass < 2; ++pass)
  {
    printf("--- pass %d (check reproducibility) ---\n", pass);
    printf("[len=128, IPT=4]\n");
    run(lat_dep<4, float, CHAIN>, CHAIN, "DEP (chain, true latency)", 128);
    run(lat_indep<4, float, CHAIN, 4>, CHAIN, "INDEP ILP=4 (overlap)", 128);
    run(lat_indep<4, float, CHAIN, 8>, CHAIN, "INDEP ILP=8 (overlap)", 128);
    run(lat_prod<4, float, CHAIN>, CHAIN, "PROD (gen-in-loop, rolled)", 128);

    printf("[len=256, IPT=8]\n");
    run(lat_dep<8, float, CHAIN>, CHAIN, "DEP (chain, true latency)", 256);
    run(lat_indep<8, float, CHAIN, 4>, CHAIN, "INDEP ILP=4 (overlap)", 256);
    run(lat_prod<8, float, CHAIN>, CHAIN, "PROD (gen-in-loop, rolled)", 256);

    printf("[len=64, IPT=2]\n");
    run(lat_dep<2, float, CHAIN>, CHAIN, "DEP (chain, true latency)", 64);
    run(lat_indep<2, float, CHAIN, 4>, CHAIN, "INDEP ILP=4 (overlap)", 64);
    run(lat_prod<2, float, CHAIN>, CHAIN, "PROD (gen-in-loop, rolled)", 64);
  }

  printf("\n=== Comparison: WarpBitonicSort vs WarpMergeSort (float, PROD-style, random input) ===\n");
  printf("[len=128]\n");
  run(lat_prod<4, float, CHAIN>, CHAIN, "BitonicSort", 128);
  run(lat_prod_merge<4, float, CHAIN>, CHAIN, "MergeSort", 128);
  printf("[len=256]\n");
  run(lat_prod<8, float, CHAIN>, CHAIN, "BitonicSort", 256);
  run(lat_prod_merge<8, float, CHAIN>, CHAIN, "MergeSort", 256);
  printf("[len=64]\n");
  run(lat_prod<2, float, CHAIN>, CHAIN, "BitonicSort", 64);
  run(lat_prod_merge<2, float, CHAIN>, CHAIN, "MergeSort", 64);
  return 0;
}
