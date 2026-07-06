// Prototype for two latency-regime correctness challenges:
//   (1) Timer-boundary fencing: stop the compiler/HW from moving primitive work
//       out of the [t0, t1] window (start-hoist) or reading the end clock before
//       the primitive's long-latency ops COMPLETE (end-undershoot).
//   (2) Block-level timing when warps arrive at the barrier at different times.

#include <cub/block/block_topk.cuh>
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

constexpr int kReps = 128;

// ============================ (1) timer fencing ============================

// UNFENCED: t1 does not depend on the sort output; nothing prevents the end
// clock read from being issued before the final sort's shuffles complete.
template <int IPT, typename KeyT, int CHAIN>
__global__ void __launch_bounds__(32) lat_unfenced(long long* out)
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

// FENCED: compiler barrier after t0 (block start-hoist), and force every output
// register to be consumed before t1 (scoreboard wait => end clock waits for the
// last shuffle to COMPLETE, not just issue).
template <int IPT, typename KeyT, int CHAIN>
__global__ void __launch_bounds__(32) lat_fenced(long long* out)
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
    asm volatile("" ::: "memory"); // start fence: no sort op may hoist above t0
#pragma unroll
    for (int n = 0; n < CHAIN; ++n)
    {
      cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{});
    }
    // end fence: consume all outputs so t1 waits for the last shuffle to retire
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      asm volatile("" ::: "memory");
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

// ======================= (2) block-level timing =======================
// Block-level primitive: all warps share one SM => one consistent clock64.
// Bracket the primitive with __syncthreads() so t0 is a common (aligned) start
// and t1 is read only after the SLOWEST warp finishes. Record per-warp arrival
// (pre-barrier), start and end so we can see the skew and aggregate correctly.
template <int BLOCK, int IPT, typename KeyT, int K, int CHAIN>
__global__ void __launch_bounds__(BLOCK)
block_topk_lat(long long* d_arrive, long long* d_t0, long long* d_t1, long long* d_thread0)
{
  using topk_t = cub::detail::block_topk<KeyT, BLOCK, IPT>;
  __shared__ typename topk_t::TempStorage temp;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  uint32_t s     = 12345u + (threadIdx.x + 7) * 2654435761u;
  KeyT keys[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    keys[i] = lcg<KeyT>(s);
  }

  long long tarrive = clock64(); // natural per-warp arrival skew (no barrier yet)

  __syncthreads(); // align all warps
  long long t0 = clock64();
  asm volatile("" ::: "memory");
#pragma unroll 1
  for (int n = 0; n < CHAIN; ++n)
  {
    topk_t(temp).template max_keys<true>(keys, K, BLOCK * IPT);
  }
  asm volatile("" ::: "memory");
  __syncthreads(); // wait for the slowest warp to finish the primitive
  long long t1 = clock64();

  if (lane == 0)
  {
    d_arrive[warp] = tarrive;
    d_t0[warp]     = t0;
    d_t1[warp]     = t1;
  }
  if (threadIdx.x == 0)
  {
    *d_thread0 = t1 - t0;
  }
  if (keys[0] == KeyT(-1))
  {
    g_sink[threadIdx.x & 4095] = static_cast<int>(keys[0]);
  }
}

template <class K>
long long measure(K kernel, int block)
{
  long long* d;
  cudaMalloc(&d, sizeof(long long));
  kernel<<<1, block>>>(d);
  cudaDeviceSynchronize();
  long long v = 0;
  cudaMemcpy(&v, d, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(d);
  return v;
}

int main()
{
  constexpr int IPT = 4; // len=128 float
  printf("=== (1) End-timer fencing: WarpBitonicSort float len=128 (min cyc) ===\n");
  long long u1 = measure(lat_unfenced<IPT, float, 1>, 32);
  long long f1 = measure(lat_fenced<IPT, float, 1>, 32);
  long long u32 = measure(lat_unfenced<IPT, float, 32>, 32);
  long long f32 = measure(lat_fenced<IPT, float, 32>, 32);
  printf("  chain=1   unfenced=%-6lld  fenced=%-6lld  (fence adds %lld cyc = exposed tail drain)\n", u1, f1, f1 - u1);
  printf("  chain=32  unfenced=%-6lld  fenced=%-6lld  per-call unf=%.1f fen=%.1f\n", u32, f32, u32 / 32.0, f32 / 32.0);
  printf("  slope unfenced=%.1f  slope fenced=%.1f  (marginal latency is fence-insensitive)\n",
         (u32 - u1) / 31.0,
         (f32 - f1) / 31.0);

  printf("\n=== (2) Block-level timing: block_topk<float> BLOCK=256 (8 warps), one SM ===\n");
  constexpr int BLOCK = 256;
  constexpr int NW    = BLOCK / 32;
  long long *d_arrive, *d_t0, *d_t1, *d_thr0;
  cudaMalloc(&d_arrive, NW * sizeof(long long));
  cudaMalloc(&d_t0, NW * sizeof(long long));
  cudaMalloc(&d_t1, NW * sizeof(long long));
  cudaMalloc(&d_thr0, sizeof(long long));
  block_topk_lat<BLOCK, IPT, float, 32, 8><<<1, BLOCK>>>(d_arrive, d_t0, d_t1, d_thr0);
  cudaDeviceSynchronize();
  long long arrive[NW], t0[NW], t1[NW], thr0;
  cudaMemcpy(arrive, d_arrive, NW * sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(t0, d_t0, NW * sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(t1, d_t1, NW * sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&thr0, d_thr0, sizeof(long long), cudaMemcpyDeviceToHost);
  long long a0 = arrive[0], amin = arrive[0], amax = arrive[0];
  long long t0min = t0[0], t0max = t0[0], t1min = t1[0], t1max = t1[0];
  for (int w = 0; w < NW; ++w)
  {
    amin  = ::min(amin, arrive[w]);
    amax  = ::max(amax, arrive[w]);
    t0min = ::min(t0min, t0[w]);
    t0max = ::max(t0max, t0[w]);
    t1min = ::min(t1min, t1[w]);
    t1max = ::max(t1max, t1[w]);
  }
  printf("  pre-barrier arrival skew across warps:  max-min = %lld cyc\n", amax - amin);
  printf("  post-barrier start (t0) skew:           max-min = %lld cyc  (barrier aligned)\n", t0max - t0min);
  printf("  post-work end (t1) skew:                max-min = %lld cyc  (closing barrier aligned)\n", t1max - t1min);
  printf("  block latency = max(t1)-min(t0) = %lld cyc\n", t1max - t0min);
  printf("  thread0 (t1-t0)                 = %lld cyc  (agrees since barriers align warps)\n", thr0);
  return 0;
}
