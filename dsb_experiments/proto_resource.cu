// Prototype: RESOURCE-USAGE measurement for device-side primitives.
//
// Explores 3 complementary regimes for "how expensive is this primitive to embed?":
//   (R1) Standalone footprint: regs / static smem / spill bytes of a canonical
//        kernel that does only the primitive (the "ballpark" number).
//   (R2) Carrier delta: a user-supplied carrier kernel instantiated WITH the
//        primitive and with a NoOp; the register/smem *delta* attributes the cost
//        to the primitive in a realistic surrounding kernel.
//   (R3) Occupancy & pressure: theoretical max occupancy for a block size, plus a
//        launch-bounds sweep (min-blocks/SM) to see when the primitive starts to
//        spill -- i.e. "if a user needs occupancy X, what does it cost / does it fit".
//
// All numbers come from cudaFuncGetAttributes (runtime, no external tools needed).

#include <cub/block/block_topk.cuh>
#include <cub/warp/warp_bitonic_sort.cuh>

#include <cuda/ptx>
#include <cuda/std/array>

#include <cstdint>
#include <cstdio>

__device__ int g_sink[64];

// Unprovable guard: smid is never (uint32_t)-1, but the compiler cannot prove it,
// so the value-producing code that feeds the sink is retained (no DCE).
template <typename KeyT, int IPT>
__device__ __forceinline__ void keep(KeyT (&keys)[IPT])
{
  if (cuda::ptx::get_sreg_smid() == static_cast<uint32_t>(-1))
  {
    KeyT acc = KeyT{};
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      acc += keys[i];
    }
    g_sink[threadIdx.x & 63] = static_cast<int>(acc);
  }
}

struct CustomLess
{
  template <typename T>
  __device__ bool operator()(const T& a, const T& b) const
  {
    return a < b;
  }
};

struct NoOpSort
{
  template <int IPT, typename KeyT>
  __device__ __forceinline__ void operator()(KeyT (&)[IPT]) const
  {}
};

struct BitonicSortOp
{
  template <int IPT, typename KeyT>
  __device__ __forceinline__ void operator()(KeyT (&keys)[IPT]) const
  {
    cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{});
  }
};

// (R1)+(R2) A realistic *carrier* kernel: global load -> some surrounding ALU that
// keeps values live across the primitive -> primitive (or NoOp) -> global store.
template <int IPT, typename KeyT, class Op>
__global__ void carrier(const KeyT* __restrict__ in, KeyT* __restrict__ out, int n, Op op)
{
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * IPT;
  KeyT keys[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    keys[i] = (base + i < n) ? in[base + i] : KeyT{};
  }
  // surrounding work that must stay live across the primitive call
  KeyT tally = KeyT{};
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    tally += keys[i] * KeyT(i + 1);
  }
  op(keys);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out[base + i] = keys[i] + tally;
  }
}

// (R1) minimal standalone footprint kernel (no global mem, register-resident).
template <int IPT, typename KeyT, class Op>
__global__ void standalone(long long seed, KeyT* out, Op op)
{
  KeyT keys[IPT];
  uint32_t s = static_cast<uint32_t>(seed) + threadIdx.x;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    s       = 1664525u * s + 1013904223u;
    keys[i] = static_cast<KeyT>(s & 0xffffu);
  }
  op(keys);
  keep(keys);
  (void) out;
}

// (R3) launch-bounds-constrained standalone: force at least MINB blocks/SM.
template <int BLOCK, int MINB, int IPT, typename KeyT, class Op>
__global__ void __launch_bounds__(BLOCK, MINB) standalone_lb(long long seed, KeyT* out, Op op)
{
  KeyT keys[IPT];
  uint32_t s = static_cast<uint32_t>(seed) + threadIdx.x;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    s       = 1664525u * s + 1013904223u;
    keys[i] = static_cast<KeyT>(s & 0xffffu);
  }
  op(keys);
  keep(keys);
  (void) out;
}

// Block-level primitive that uses shared memory (radix top-k) -- contrasts with the
// register-only, smem-free bitonic sort.
template <int BlockDimX, int IPT, typename KeyT, int K>
__global__ void block_topk_standalone(long long seed, KeyT* out)
{
  using topk_t = cub::detail::block_topk<KeyT, BlockDimX, IPT>;
  __shared__ typename topk_t::TempStorage temp;
  KeyT keys[IPT];
  uint32_t s = static_cast<uint32_t>(seed) + threadIdx.x;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    s       = 1664525u * s + 1013904223u;
    keys[i] = static_cast<KeyT>(s & 0xffffu);
  }
  topk_t(temp).template max_keys<true>(keys, K, BlockDimX * IPT);
  keep(keys);
  (void) out;
}

template <int BlockDimX, int MINB, int IPT, typename KeyT, int K>
__global__ void __launch_bounds__(BlockDimX, MINB) block_topk_standalone_lb(long long seed, KeyT* out)
{
  using topk_t = cub::detail::block_topk<KeyT, BlockDimX, IPT>;
  __shared__ typename topk_t::TempStorage temp;
  KeyT keys[IPT];
  uint32_t s = static_cast<uint32_t>(seed) + threadIdx.x;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    s       = 1664525u * s + 1013904223u;
    keys[i] = static_cast<KeyT>(s & 0xffffu);
  }
  topk_t(temp).template max_keys<true>(keys, K, BlockDimX * IPT);
  keep(keys);
  (void) out;
}

template <class Kernel>
void report(const char* label, Kernel k, int block)
{
  cudaFuncAttributes a{};
  cudaFuncGetAttributes(&a, k);
  int max_blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, k, block, a.sharedSizeBytes);
  int warps_per_block = block / 32;
  int active_warps    = max_blocks * warps_per_block;
  double occ          = active_warps / 64.0; // B200: 64 warps/SM peak
  printf("  %-38s regs=%-3d smem=%-4zu spill=%-4zu | block=%-4d maxBlk/SM=%-2d occ=%4.0f%%\n",
         label,
         a.numRegs,
         a.sharedSizeBytes,
         a.localSizeBytes,
         block,
         max_blocks,
         100.0 * occ);
}

int main()
{
  printf("=== (R1) Standalone footprint: WarpBitonicSort<float> ===\n");
  report("len=64  (IPT=2) sort-only", standalone<2, float, BitonicSortOp>, 256);
  report("len=128 (IPT=4) sort-only", standalone<4, float, BitonicSortOp>, 256);
  report("len=256 (IPT=8) sort-only", standalone<8, float, BitonicSortOp>, 256);

  printf("\n=== (R2) Carrier delta (regs/smem attributable to the primitive) ===\n");
  printf("  len=128, block=256\n");
  report("carrier + NoOp     (baseline)", carrier<4, float, NoOpSort>, 256);
  report("carrier + Bitonic  (with prim)", carrier<4, float, BitonicSortOp>, 256);
  printf("  len=256, block=256\n");
  report("carrier + NoOp     (baseline)", carrier<8, float, NoOpSort>, 256);
  report("carrier + Bitonic  (with prim)", carrier<8, float, BitonicSortOp>, 256);

  printf("\n=== (R3) Register-pressure / launch-bounds sweep: len=256 sort-only, block=256 ===\n");
  report("no launch bound",           standalone<8, float, BitonicSortOp>, 256);
  report("__launch_bounds__(256, 2)", standalone_lb<256, 2, 8, float, BitonicSortOp>, 256);
  report("__launch_bounds__(256, 4)", standalone_lb<256, 4, 8, float, BitonicSortOp>, 256);
  report("__launch_bounds__(256, 6)", standalone_lb<256, 6, 8, float, BitonicSortOp>, 256);
  report("__launch_bounds__(256, 8)", standalone_lb<256, 8, 8, float, BitonicSortOp>, 256);

  printf("\n=== (R1') Standalone footprint: block_topk<float> (uses shared memory) ===\n");
  report("block=128 IPT=4 K=32", block_topk_standalone<128, 4, float, 32>, 128);
  report("block=256 IPT=4 K=32", block_topk_standalone<256, 4, float, 32>, 256);
  report("block=256 IPT=8 K=32", block_topk_standalone<256, 8, float, 32>, 256);

  printf("\n=== (R3') Pressure sweep: block_topk block=256 IPT=8 K=32 ===\n");
  report("no launch bound",            block_topk_standalone<256, 8, float, 32>, 256);
  report("__launch_bounds__(256, 4)",  block_topk_standalone_lb<256, 4, 8, float, 32>, 256);
  report("__launch_bounds__(256, 6)",  block_topk_standalone_lb<256, 6, 8, float, 32>, 256);
  report("__launch_bounds__(256, 8)",  block_topk_standalone_lb<256, 8, 8, float, 32>, 256);
  return 0;
}
