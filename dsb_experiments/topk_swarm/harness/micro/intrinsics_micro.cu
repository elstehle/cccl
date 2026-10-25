// Chain-slope microbenchmark of selection primitives on sm_100 class GPUs.
//
// Extends reference/proto_topk_micro.cu (July 2026). Methodology (BRIEF.md section 4): one warp
// (<<<1,32>>>) or one block (<<<1,256>>>), clock64 cycles, chain lengths 1, 2, 4, 8, 16, least
// squares slope = marginal cycles per call, min over kReps repetitions, barrier bracketing for
// blocks, memory clobber before the end clock, outputs consumed by sink() (an smid guard the
// compiler cannot prove false). Data-oblivious primitives run an in-place dependency chain with a
// per-iteration perturbation. Data-dependent primitives (match.any) run with a fixed class
// structure that the chain preserves.
//
// Output: one JSON line per measurement. Fields: name, cycles (slope), intercept, r2, raw (cycles
// per chain length), block, arch, cuda, nvcc, variant, node, date, file. Numbers never go into
// comments, they go into cards and TECHNIQUES.md.
//
// Build (both x86 B200 and aarch64 GB200):
//   nvcc -arch=sm_100  -O3 -std=c++17 -o intrinsics_micro_sm100  intrinsics_micro.cu
//   nvcc -arch=sm_100f -O3 -std=c++17 -o intrinsics_micro_sm100f intrinsics_micro.cu
// The sm_100f binary additionally measures the family-specific primitives (redux.sync f32).
//
// Run:  ./intrinsics_micro_sm100 --node <name> --date <yyyy-mm-dd> --variant sm_100 [--only <substr>]

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

__device__ int g_sink;

constexpr int kReps     = 64;
constexpr int kMaxChain = 16;
constexpr unsigned kFull = 0xffffffffu;

// Family-specific or arch-specific sm_100 features (redux.sync .f32 needs sm_100a or sm_100f).
// The device-side macros below exist only in the device compilation pass, so the host decides
// with MICRO_FAMILY (passed as -DMICRO_FAMILY=1 by run_micro.sh for the sm_100f build).
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) \
  || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 \
      && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100)
#  define MICRO_HAS_SM100_FAMILY 1
#else
#  define MICRO_HAS_SM100_FAMILY 0
#endif
#ifndef MICRO_FAMILY
#  define MICRO_FAMILY 0
#endif

// ---------------------------------------------------------------- anti-DCE sink
__device__ __forceinline__ unsigned smid()
{
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

// The guard is never true (smid is a small number) but the compiler cannot prove it, so v stays live.
// The smid read (S2R, about 80 cycles) is taken once before the timed region: inside it, the read
// would overlap the chain and hide any chain shorter than itself (seen with dependent IMAD chains).
template <typename T>
__device__ __forceinline__ void sink(T v, unsigned sm)
{
  if (sm == 0xffffffffu)
  {
    g_sink = static_cast<int>(v);
  }
}

__device__ __forceinline__ unsigned lane_id()
{
  return threadIdx.x & 31;
}

// ---------------------------------------------------------------- inline PTX wrappers
__device__ __forceinline__ unsigned redux_max_u32(unsigned v)
{
  return __reduce_max_sync(kFull, v);
}

__device__ __forceinline__ float redux_max_f32(float v)
{
  float r;
#if MICRO_HAS_SM100_FAMILY
  asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(v));
#else
  __trap(); // only reachable when the host was built with MICRO_FAMILY but the device pass is not family specific
  r = v;
#endif
  return r;
}
__device__ __forceinline__ float redux_max_abs_nan_f32(float v)
{
  float r;
#if MICRO_HAS_SM100_FAMILY
  asm volatile("redux.sync.max.abs.NaN.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(v));
#else
  __trap();
  r = v;
#endif
  return r;
}

__device__ __forceinline__ unsigned max_bf16x2(unsigned a, unsigned b)
{
  unsigned r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ unsigned max_f16x2(unsigned a, unsigned b)
{
  unsigned r;
  asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ unsigned max_u16x2_ptx(unsigned a, unsigned b)
{
  unsigned r;
  asm("max.u16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float max3_f32_ptx(float a, float b, float c)
{
  float r;
  asm("max.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ unsigned bmsk_clamp(unsigned pos, unsigned width)
{
  unsigned r;
  asm("bmsk.clamp.b32 %0, %1, %2;" : "=r"(r) : "r"(pos), "r"(width));
  return r;
}
__device__ __forceinline__ unsigned elect_one(unsigned mask, int& is_leader)
{
  unsigned lane;
  int pred;
  asm volatile(
    "{\n\t.reg .pred p;\n\t"
    "elect.sync %0|p, %2;\n\t"
    "selp.s32 %1, 1, 0, p;\n\t}"
    : "=r"(lane), "=r"(pred)
    : "r"(mask));
  is_leader = pred;
  return lane;
}
__device__ __forceinline__ unsigned match_any_u32(unsigned v)
{
  return __match_any_sync(kFull, v);
}

__device__ __forceinline__ unsigned smem_u32addr(const void* p)
{
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3)
{
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_trans(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3)
{
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void stsm_x4(unsigned addr, unsigned r0, unsigned r1, unsigned r2, unsigned r3)
{
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(addr),
               "r"(r0),
               "r"(r1),
               "r"(r2),
               "r"(r3)
               : "memory");
}

__device__ __forceinline__ unsigned movm_trans(unsigned a)
{
  unsigned d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}

// mbarrier and bulk copy (sm_90+)
__device__ __forceinline__ void mbar_init(unsigned addr, unsigned count)
{
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init()
{
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_smem()
{
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void mbar_inval(unsigned addr)
{
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(addr) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(unsigned addr, unsigned bytes)
{
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(unsigned dst, const void* src, unsigned bytes, unsigned mbar)
{
  asm volatile(
    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
    "l"(src),
    "r"(bytes),
    "r"(mbar)
    : "memory");
}
__device__ __forceinline__ void mbar_wait_parity(unsigned addr, unsigned parity)
{
  // labels are scoped by the braces, so repeated inlining is fine
  asm volatile(
    "{\n\t.reg .pred p;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
    "@p bra LAB_DONE;\n\t"
    "bra LAB_WAIT;\n\t"
    "LAB_DONE:\n\t}" ::"r"(addr),
    "r"(parity)
    : "memory");
}

// ---------------------------------------------------------------- helpers shared by experiments
__device__ __forceinline__ unsigned warp_max_shfl(unsigned x)
{
#pragma unroll
  for (int d = 16; d; d >>= 1)
  {
    x = ::max(x, __shfl_xor_sync(kFull, x, d));
  }
  return x;
}
__device__ __forceinline__ unsigned long long warp_max_shfl(unsigned long long x)
{
#pragma unroll
  for (int d = 16; d; d >>= 1)
  {
    x = ::max(x, __shfl_xor_sync(kFull, x, d));
  }
  return x;
}

struct NoSmem
{
  int dummy;
};

// Every experiment: value_t, Smem, block, init(), reset(sh, n), step(v, n, sh). The chain is
// v = step(v, n, sh) for n in [0, N).

// ================================================================ A. ALU and SIMD operations (1 warp)
// ptxas reassociates chains of a single associative operation (max(max(v, a), b) becomes one
// VIMNMX3, adds fold, prmt composes), and a bare asm barrier does not stop it. Each step therefore
// interleaves kAluOps dependent operations with kAluOps dependent IMADs (v = v * 3 + c), which ptxas
// does not reassociate with the operation under test. The per-operation cost is
// (slope of the pair chain minus slope of the IMAD-only chain) / kAluOps.
constexpr int kAluOps = 8;

template <class Op>
struct AluPair
{
  using value_t = typename Op::value_t;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return Op::init();
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
#pragma unroll
    for (int i = 0; i < kAluOps; ++i)
    {
      v = Op::apply(v, i);
      v = Op::mix(v, i);
    }
    return v;
  }
};

template <class Op>
struct AluBase // the IMAD-only chain of AluPair<Op>
{
  using value_t = typename Op::value_t;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return Op::init();
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
#pragma unroll
    for (int i = 0; i < kAluOps; ++i)
    {
      v = Op::mix(v, i);
    }
    return v;
  }
};

struct OpU32
{
  using value_t = unsigned;
  static __device__ value_t init()
  {
    return threadIdx.x * 2654435761u;
  }
  static __device__ value_t mix(value_t v, int i)
  {
    return v * 3u + (threadIdx.x + static_cast<unsigned>(i) * 7u);
  }
  static __device__ unsigned c1(int i)
  {
    return threadIdx.x * 77u + static_cast<unsigned>(i) * 1000003u;
  }
  static __device__ unsigned c2(int i)
  {
    return threadIdx.x * 131u + static_cast<unsigned>(i) * 998244353u;
  }
};
struct OpIAdd : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return v + c1(i);
  }
};
struct OpIMax : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return ::max(v, c1(i));
  }
};
struct OpIMax3 : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return __vimax3_u32(v, c1(i), c2(i));
  }
};
struct OpIMaxMax : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return ::max(::max(v, c1(i)), c2(i));
  }
};
struct OpIAddMax : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return ::max(v + c1(i), c2(i));
  }
};
struct OpIMax3S : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return static_cast<unsigned>(
      __vimax3_s32(static_cast<int>(v), static_cast<int>(c1(i)), static_cast<int>(c2(i))));
  }
};
struct OpPrmt : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return __byte_perm(v, c1(i), 0x3421u);
  }
};
struct OpBmsk : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return bmsk_clamp(v & 31u, c1(i) & 31u) + v;
  }
};
struct OpVMaxU2 : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return __vmaxu2(v, c1(i));
  }
};
struct OpVMaxU2Ptx : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return max_u16x2_ptx(v, c1(i));
  }
};
struct OpVMaxS2 : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return __vmaxs2(v, c1(i));
  }
};
struct OpVMax3U16x2 : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return __vimax3_u16x2(v, c1(i), c2(i));
  }
};
struct OpVMaxU4 : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return __vmaxu4(v, c1(i));
  }
};
struct OpHMax2Bf16 : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    unsigned cb      = c1(i);
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&v);
    __nv_bfloat162 c = *reinterpret_cast<__nv_bfloat162*>(&cb);
    __nv_bfloat162 r = __hmax2(a, c);
    return *reinterpret_cast<unsigned*>(&r);
  }
};
struct OpMaxBf16x2Ptx : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return max_bf16x2(v, c1(i));
  }
};
struct OpMinBf16x2Ptx : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    unsigned r;
    asm("min.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(c1(i)));
    return r;
  }
};
struct OpHgt2MaskBf16 : OpU32 // per-half 0xffff masks from a bf16x2 compare
{
  static __device__ value_t apply(value_t v, int i)
  {
    unsigned cb      = c1(i);
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&v);
    __nv_bfloat162 c = *reinterpret_cast<__nv_bfloat162*>(&cb);
    return __hgt2_mask(a, c);
  }
};
struct OpSetGtBf16x2Ptx : OpU32 // set.gt.u32.bf16x2: all-ones word when both halves compare true
{
  static __device__ value_t apply(value_t v, int i)
  {
    unsigned r;
    asm("set.gt.u32.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(c1(i)));
    return r;
  }
};
struct OpVcmpgtu2 : OpU32 // per-half 0xffff masks from a u16x2 compare
{
  static __device__ value_t apply(value_t v, int i)
  {
    return __vcmpgtu2(v, c1(i));
  }
};
struct OpVcmpgtu4 : OpU32 // per-byte 0xff masks from a u8x4 compare
{
  static __device__ value_t apply(value_t v, int i)
  {
    return __vcmpgtu4(v, c1(i));
  }
};
struct OpMaxF16x2Ptx : OpU32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return max_f16x2(v, c1(i));
  }
};

struct OpF32
{
  using value_t = float;
  static __device__ value_t init()
  {
    return static_cast<float>(threadIdx.x) * 0.37f;
  }
  static __device__ value_t mix(value_t v, int i)
  {
    return fmaf(v, 1.0001f, static_cast<float>(threadIdx.x + static_cast<unsigned>(i)));
  }
  static __device__ float c1(int i)
  {
    return static_cast<float>(threadIdx.x) * 1.5f + static_cast<float>(i);
  }
  static __device__ float c2(int i)
  {
    return static_cast<float>(threadIdx.x) * 2.5f + static_cast<float>(i);
  }
};
struct OpFMax : OpF32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return fmaxf(v, c1(i));
  }
};
struct OpFMaxMax : OpF32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return fmaxf(fmaxf(v, c1(i)), c2(i));
  }
};
struct OpFMax3Ptx : OpF32
{
  static __device__ value_t apply(value_t v, int i)
  {
    return max3_f32_ptx(v, c1(i), c2(i));
  }
};

// Variable-latency integer ops run as plain dependent chains (they are long enough not to be hidden).
struct FfsChain // __ffs: BREV + FLO
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u | 1u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return v + static_cast<unsigned>(__ffs(static_cast<int>(v)));
  }
};

struct PopcChain
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return v * 3u + static_cast<unsigned>(__popc(v));
  }
};

struct FnsChain // __fns(mask, base, offset): the n-th set bit
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u | 0x80000001u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned pos = __fns(v, 0, 2);
    return (v * 3u) | (1u << (pos & 31)) | 1u;
  }
};

// ================================================================ B. warp collectives (1 warp)
struct ReduxU32
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return redux_max_u32(v) + threadIdx.x;
  }
};

struct ReduxF32
{
  using value_t = float;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return static_cast<float>(threadIdx.x) * 0.37f;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return redux_max_f32(v) + static_cast<float>(threadIdx.x);
  }
};
struct ReduxF32AbsNaN
{
  using value_t = float;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return static_cast<float>(threadIdx.x) * 0.37f;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return redux_max_abs_nan_f32(v) + static_cast<float>(threadIdx.x);
  }
};

// Control for fp32 keys without redux.f32: twiddle to u32, redux.max.u32, untwiddle.
struct ReduxF32ViaTwiddle
{
  using value_t = float;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return static_cast<float>(threadIdx.x) * 0.37f;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned b = __float_as_uint(v);
    unsigned m = 0u - (b >> 31);
    unsigned t = b ^ (m | 0x80000000u);
    unsigned r = redux_max_u32(t);
    unsigned m2 = ((r >> 31) - 1u) | 0x80000000u;
    return __uint_as_float(r ^ m2) + static_cast<float>(threadIdx.x);
  }
};

struct ShflXor1
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return __shfl_xor_sync(kFull, v, 1) + 1;
  }
};

struct ShflIdxDep // shfl.idx with a data-dependent source lane
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return __shfl_sync(kFull, v, v & 31u) + 1;
  }
};

struct WarpMaxShflU32
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return warp_max_shfl(v) + threadIdx.x;
  }
};

struct WarpMaxShflU64
{
  using value_t = unsigned long long;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761ull;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return warp_max_shfl(v) + threadIdx.x;
  }
};

struct WarpMaxShflF32 // fmaxf shuffle tree, the portable fp32 warp max
{
  using value_t = float;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return static_cast<float>(threadIdx.x) * 0.37f;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    float x = v;
#pragma unroll
    for (int d = 16; d; d >>= 1)
    {
      x = fmaxf(x, __shfl_xor_sync(kFull, x, d));
    }
    return x + static_cast<float>(threadIdx.x);
  }
};

// K independent warp collectives per step (all depend on v, none on each other), merged into the next
// input: (slope minus the dependent-chain slope of one op) / (K - 1) is the issue cost of each further
// op in flight, the pipelining assumption of N1 and W3.
template <int K, int BLOCK = 32>
struct BallotIndep
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned acc = 0;
#pragma unroll
    for (int i = 0; i < K; ++i)
    {
      acc += __ballot_sync(kFull, (v >> ((threadIdx.x + i) & 31)) & 1);
    }
    return acc * 3u + threadIdx.x;
  }
};

template <int K, int BLOCK = 32>
struct ReduxIndep
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned acc = 0;
#pragma unroll
    for (int i = 0; i < K; ++i)
    {
      acc += redux_max_u32(v ^ (static_cast<unsigned>(i) * 0x9e3779b9u));
    }
    return acc * 3u + threadIdx.x;
  }
};

template <int K, int BLOCK = 32>
struct ShflIndep
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned acc = 0;
#pragma unroll
    for (int i = 0; i < K; ++i)
    {
      acc += __shfl_xor_sync(kFull, v + static_cast<unsigned>(i), 1 << (i & 4 ? 4 : i & 3));
    }
    return acc * 3u + threadIdx.x;
  }
};

struct Ballot
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return __ballot_sync(kFull, (v >> (threadIdx.x & 31)) & 1) + threadIdx.x;
  }
};

struct BallotFfsElect // ballot + ffs + "am I the winner" select: the index-resolution idiom
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned b = __ballot_sync(kFull, (v >> (threadIdx.x & 31)) & 1);
    int winner = __ffs(b) - 1;
    return v + (winner == static_cast<int>(threadIdx.x) ? 1u : 2u);
  }
};

struct ReduxBallotRound // W2 extraction round for unpacked keys: redux max, ballot(v == max), ffs, winner retires
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned m = redux_max_u32(v);
    unsigned b = __ballot_sync(kFull, v == m);
    int winner = __ffs(b) - 1;
    return (static_cast<int>(threadIdx.x) == winner) ? (v >> 1) : (v + m);
  }
};

struct ReduxPackedRound // W2 round for packed (key << 16 | index): redux max is the arg max, winner by compare
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return (threadIdx.x * 2654435761u) | threadIdx.x;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned m = redux_max_u32(v);
    return (v == m) ? (v >> 1) : (v + (m & 0xffff0000u));
  }
};

// Packed W2 extraction round with a per-lane sorted list of IPT unique words (key high, inverted
// index low): redux max of the heads, the winner (head == max) shifts its list down by one with
// predicated selects, the next round's redux reads the new head. Critical path: redux, compare,
// select of the head. The other IPT - 1 selects are off the path but take issue slots.
__device__ __forceinline__ unsigned packed_word(unsigned lane, unsigned i, unsigned ipt)
{
  unsigned h = (lane * 2654435761u + i * 40503u + 12345u) >> 7;
  return ((h & 0xffffu) << 16) | (0xffffu - (lane * ipt + i)); // unique: the low half is a unique inverted index
}

template <int IPT, bool BRANCHY>
struct W2PackedList
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  unsigned list[IPT];
  __device__ void fill()
  {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      list[i] = packed_word(threadIdx.x, static_cast<unsigned>(i), IPT);
    }
    // sort descending (insertion sort, static indices, outside the timed region)
#pragma unroll
    for (int i = 1; i < IPT; ++i)
    {
#pragma unroll
      for (int j = i; j > 0; --j)
      {
        unsigned a = list[j - 1], b = list[j];
        list[j - 1] = ::max(a, b);
        list[j]     = ::min(a, b);
      }
    }
  }
  __device__ value_t init(Smem&)
  {
    fill();
    return 0;
  }
  __device__ void reset(Smem&, int)
  {
    fill();
  }
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned m = redux_max_u32(list[0]);
    bool win   = (list[0] == m);
    if constexpr (BRANCHY)
    {
      if (win)
      {
#pragma unroll
        for (int i = 0; i < IPT - 1; ++i)
        {
          list[i] = list[i + 1];
        }
        list[IPT - 1] = 0u;
      }
    }
    else
    {
#pragma unroll
      for (int i = 0; i < IPT - 1; ++i)
      {
        list[i] = win ? list[i + 1] : list[i];
      }
      list[IPT - 1] = win ? 0u : list[IPT - 1];
    }
    return v * 3u + m;
  }
  __device__ value_t finish(value_t v) // keeps every list entry live, so no shift can be dropped as dead
  {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      v = v * 3u + list[i];
    }
    return v;
  }
};

// ================================================================ pipe throughput (for opt-N5)
// K independent accumulators per lane, one operation each per step, as inline PTX so that nothing
// folds across steps. Run by 1, 4, 8 and 16 warps per block. cycles per step divided by the SASS
// instruction count per step (read from the dump, fused three-input forms count as one) is the
// cycles per warp instruction, and warps per scheduler times instructions per step over cycles per
// step is the sustained warp instructions per cycle per scheduler.
struct TpMaxBf16x2
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    asm volatile("max.bf16x2 %0, %0, %1;" : "+r"(v) : "r"(c));
  }
};
struct TpMaxU32
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    asm volatile("max.u32 %0, %0, %1;" : "+r"(v) : "r"(c));
  }
};
struct TpMaxU16x2
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    asm volatile("max.u16x2 %0, %0, %1;" : "+r"(v) : "r"(c));
  }
};
struct TpSetBf16x2
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    asm volatile("set.gt.u32.bf16x2 %0, %0, %1;" : "+r"(v) : "r"(c));
  }
};
struct TpLop3
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v) : "r"(c), "r"(c ^ 0x5555u));
  }
};
struct TpImad
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    asm volatile("mad.lo.u32 %0, %0, 3, %1;" : "+r"(v) : "r"(c));
  }
};
struct TpMaxF32
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    float f = __uint_as_float(v);
    asm volatile("max.f32 %0, %0, %1;" : "+f"(f) : "f"(__uint_as_float(c)));
    v = __float_as_uint(f);
  }
};
struct TpSel
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.gt.u32 p, %0, %1;\n\tselp.b32 %0, %1, %0, p;\n\t}" : "+r"(v) : "r"(c));
  }
};
// alternating max and min on the same accumulator (step parity): adjacent steps cannot fuse into a
// three-input form, so the SASS count per step equals the operation count
struct TpAltBf16x2
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c, int n)
  {
    if (n & 1)
    {
      asm volatile("min.bf16x2 %0, %0, %1;" : "+r"(v) : "r"(c));
    }
    else
    {
      asm volatile("max.bf16x2 %0, %0, %1;" : "+r"(v) : "r"(c));
    }
  }
};
struct TpAltU32
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c, int n)
  {
    if (n & 1)
    {
      asm volatile("min.u32 %0, %0, %1;" : "+r"(v) : "r"(c));
    }
    else
    {
      asm volatile("max.u32 %0, %0, %1;" : "+r"(v) : "r"(c));
    }
  }
};
struct TpAltU16x2
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c, int n)
  {
    if (n & 1)
    {
      asm volatile("min.u16x2 %0, %0, %1;" : "+r"(v) : "r"(c));
    }
    else
    {
      asm volatile("max.u16x2 %0, %0, %1;" : "+r"(v) : "r"(c));
    }
  }
};
struct TpAltF32
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c, int n)
  {
    float f = __uint_as_float(v);
    if (n & 1)
    {
      asm volatile("min.f32 %0, %0, %1;" : "+f"(f) : "f"(__uint_as_float(c)));
    }
    else
    {
      asm volatile("max.f32 %0, %0, %1;" : "+f"(f) : "f"(__uint_as_float(c)));
    }
    v = __float_as_uint(f);
  }
};
struct TpShfl
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c, int n)
  {
    v = __shfl_xor_sync(kFull, v, 1 + (n & 3));
  }
};

// one bitonic stage on packed bf16x2: shuffle partner, max and min, keep one by lane direction
struct TpStageBf16x2
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    unsigned t = __shfl_xor_sync(kFull, v, 1);
    unsigned a, b;
    asm volatile("max.bf16x2 %0, %1, %2;" : "=r"(a) : "r"(v), "r"(t));
    asm volatile("min.bf16x2 %0, %1, %2;" : "=r"(b) : "r"(v), "r"(t));
    v = (threadIdx.x & 1) ? b : a;
  }
};
struct TpStageU16x2 // the same stage on twiddled u16x2 words
{
  static __device__ __forceinline__ void op(unsigned& v, unsigned c)
  {
    unsigned t = __shfl_xor_sync(kFull, v, 1);
    unsigned a, b;
    asm volatile("max.u16x2 %0, %1, %2;" : "=r"(a) : "r"(v), "r"(t));
    asm volatile("min.u16x2 %0, %1, %2;" : "=r"(b) : "r"(v), "r"(t));
    v = (threadIdx.x & 1) ? b : a;
  }
};

template <class Op>
__device__ __forceinline__ auto call_op(unsigned& v, unsigned c, int n, int) -> decltype(Op::op(v, c, n), void())
{
  Op::op(v, c, n);
}
template <class Op>
__device__ __forceinline__ void call_op(unsigned& v, unsigned c, int, long)
{
  Op::op(v, c);
}

template <class Op, int K, int BLOCK>
struct Tput
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = BLOCK;
  unsigned acc[K];
  __device__ value_t init(Smem&)
  {
#pragma unroll
    for (int i = 0; i < K; ++i)
    {
      acc[i] = (threadIdx.x * 2654435761u) ^ (static_cast<unsigned>(i) * 0x9e3779b9u) ^ 0x3f803f80u;
    }
    return 0;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int n, Smem&)
  {
    // one operand per step for all K accumulators (one IADD per step, not K), it changes with n so
    // that nothing is loop invariant
    unsigned c = (threadIdx.x * 77u + static_cast<unsigned>(n) * 0x01010101u) ^ 0x3f003f00u;
#pragma unroll
    for (int i = 0; i < K; ++i)
    {
      call_op<Op>(acc[i], c, n, 0);
    }
    return v + 1u;
  }
  __device__ value_t finish(value_t v)
  {
#pragma unroll
    for (int i = 0; i < K; ++i)
    {
      v = v * 3u + acc[i];
    }
    return v;
  }
};

struct ReduxElect // redux chain with an elect.sync per stepstruct ReduxElect // redux chain with an elect.sync per step feeding the next input: incremental cost of elect
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned m = redux_max_u32(v);
    int leader;
    unsigned l = elect_one(kFull, leader);
    return m + threadIdx.x + l + (leader ? 1u : 0u);
  }
};

struct ReduxFfsLeader // same chain with ffs(activemask) leader election instead of elect.sync
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned m = redux_max_u32(v);
    unsigned am = __activemask();
    unsigned l  = static_cast<unsigned>(__ffs(am) - 1);
    return m + threadIdx.x + l + ((threadIdx.x == l) ? 1u : 0u);
  }
};

// match.any with U value classes per warp (cost depends on the number of distinct values).
template <int U>
struct MatchAnyClasses
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x % U;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned mask = match_any_u32(v);
    // all lanes of a class see the same popc, so the class structure is preserved along the chain
    return (threadIdx.x % U) + 32u * static_cast<unsigned>(__popc(mask));
  }
};

struct SyncWarp
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    __syncwarp();
    return v + 1;
  }
};

struct LdsSts
{
  using value_t = unsigned;
  struct Smem
  {
    unsigned buf[32];
  };
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    volatile unsigned* p = sh.buf;
    p[threadIdx.x]       = v;
    return p[threadIdx.x] + 1;
  }
};

// ---------------------------------------------------------------- register transposes through smem
// 4 x 32 x 32-bit words (one 8x8 b16 matrix quad) written and read back transposed.
struct TransposeStsLds // control: 4 STS, syncwarp, 4 LDS with a transposed address pattern
{
  using value_t = unsigned;
  struct Smem
  {
    __align__(16) unsigned buf[128];
  };
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    unsigned lane = threadIdx.x;
    sh.buf[lane * 4 + 0] = v;
    sh.buf[lane * 4 + 1] = v + 1;
    sh.buf[lane * 4 + 2] = v + 2;
    sh.buf[lane * 4 + 3] = v + 3;
    __syncwarp();
    unsigned r0 = sh.buf[(lane >> 2) + 32 * (lane & 3)];
    unsigned r1 = sh.buf[(lane >> 2) + 32 * (lane & 3) + 8];
    unsigned r2 = sh.buf[(lane >> 2) + 32 * (lane & 3) + 16];
    unsigned r3 = sh.buf[(lane >> 2) + 32 * (lane & 3) + 24];
    __syncwarp();
    return r0 + r1 + r2 + r3 + 1u;
  }
};

struct TransposeStsmLdsm // stmatrix.x4 then ldmatrix.x4.trans
{
  using value_t = unsigned;
  struct Smem
  {
    __align__(128) unsigned buf[128];
  };
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    unsigned lane = threadIdx.x;
    unsigned base = smem_u32addr(sh.buf);
    // lane i supplies the address of row (i & 7) of matrix (i >> 3), rows are 16 bytes
    unsigned addr = base + (lane >> 3) * 128u + (lane & 7) * 16u;
    stsm_x4(addr, v, v + 1, v + 2, v + 3);
    __syncwarp();
    unsigned r0, r1, r2, r3;
    ldsm_x4_trans(addr, r0, r1, r2, r3);
    __syncwarp();
    return r0 + r1 + r2 + r3 + 1u;
  }
};

struct TransposeMovm // four movmatrix.trans, one per register, no shared memory
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned r0 = movm_trans(v);
    unsigned r1 = movm_trans(v + 1);
    unsigned r2 = movm_trans(v + 2);
    unsigned r3 = movm_trans(v + 3);
    return r0 + r1 + r2 + r3 + 1u;
  }
};

struct LdsmOnly // ldmatrix.x4 round trip through a lane-dependent address (STS of one word, then ldsm)
{
  using value_t = unsigned;
  struct Smem
  {
    __align__(128) unsigned buf[128];
  };
  static constexpr int block = 32;
  __device__ value_t init(Smem& sh)
  {
    for (int i = threadIdx.x; i < 128; i += 32)
    {
      sh.buf[i] = i;
    }
    __syncwarp();
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    unsigned lane = threadIdx.x;
    unsigned base = smem_u32addr(sh.buf);
    unsigned addr = base + ((lane + (v & 7u)) & 31u) * 16u;
    unsigned r0, r1, r2, r3;
    ldsm_x4(addr, r0, r1, r2, r3);
    return r0 + r1 + r2 + r3 + v;
  }
};

// ================================================================ C. shared atomics
template <typename T>
struct AtomSmem
{
  T word;
  T slots[kMaxChain];
  T priv[256];
};

// Single-thread dependent chain: pure round-trip latency of one shared atomicMax with return.
template <typename T>
struct AtomUncontended
{
  using value_t = T;
  using Smem    = AtomSmem<T>;
  static constexpr int block = 32;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      sh.word = 0;
    }
    return threadIdx.x + 1;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      v = atomicMax(&sh.word, v + 1) + 1;
    }
    return v;
  }
};

// C threads hammer one word, each with a self-dependent chain (with return): per-op cost under contention.
template <typename T, int C, int BLOCK>
struct AtomContended
{
  using value_t = T;
  using Smem    = AtomSmem<T>;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      sh.word = 0;
    }
    return threadIdx.x + 1;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    if (threadIdx.x < C)
    {
      v = atomicMax(&sh.word, v + 2) + 2;
    }
    return v;
  }
};

// Fire-and-forget (no return) atomicMax, kAluOps per step interleaved with the IMAD chain of
// AluBase<OpU32>. Threads below C hit one shared word, the others hit a private word each, so no
// thread branches around the atomic (a divergent branch per atomic costs more than the atomic).
// (slope minus the IMAD-only slope) / kAluOps is the issue cost of one such atomic in that idiom.
template <typename T, int C, int BLOCK>
struct AtomNoRet
{
  using value_t = T;
  using Smem    = AtomSmem<T>;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      sh.word = 0;
    }
    sh.priv[threadIdx.x] = 0;
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    T* target = (threadIdx.x < C) ? &sh.word : &sh.priv[threadIdx.x];
#pragma unroll
    for (int i = 0; i < kAluOps; ++i)
    {
      atomicMax(target, v);
      v = v * 3 + (threadIdx.x + static_cast<unsigned>(i) * 7u);
    }
    return v;
  }
};

template <int BLOCK>
struct AluBase256 // IMAD-only chain at BLOCK threads, the base for the fire-and-forget rows at that block size
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
#pragma unroll
    for (int i = 0; i < kAluOps; ++i)
    {
      v = v * 3u + (threadIdx.x + static_cast<unsigned>(i) * 7u);
    }
    return v;
  }
};

// One-atomic-per-warp (lane 0 of all 8 warps): the hybrid combine's contention level.
template <typename T>
struct AtomWarpLane0
{
  using value_t = T;
  using Smem    = AtomSmem<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      sh.word = 0;
    }
    return threadIdx.x + 1;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    if ((threadIdx.x & 31) == 0)
    {
      v = atomicMax(&sh.word, v + 2) + 2;
    }
    return v;
  }
};

// 256 threads across 32 banked words (tid%32): contention 8 per word, bank-parallel.
struct AtomBanked
{
  using value_t = unsigned;
  struct Smem
  {
    unsigned words[32];
  };
  static constexpr int block = 256;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x < 32)
    {
      sh.words[threadIdx.x] = 0;
    }
    return threadIdx.x + 1;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    return atomicMax(&sh.words[threadIdx.x & 31], v + 2) + 2;
  }
};

// Warp-scope atomic max round: 32 lanes fire-and-forget into slot[n], syncwarp, read back.
struct WarpMaxAtomic32
{
  using value_t = unsigned;
  using Smem    = AtomSmem<unsigned>;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem& sh, int)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += 32)
    {
      sh.slots[i] = 0;
    }
    __syncwarp();
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    atomicMax(&sh.slots[n], v);
    __syncwarp();
    return *reinterpret_cast<volatile unsigned*>(&sh.slots[n]) + threadIdx.x;
  }
};

// ================================================================ D. barriers
template <int BLOCK>
struct SyncThreads
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    __syncthreads();
    return v + 1;
  }
};

// ================================================================ E. composed block-wide max rounds (256 threads)
template <typename T>
struct BlockMaxSmem
{
  T partial[8];
  T slots[kMaxChain];
};

// classic: warp shfl-max -> smem partials -> 1 barrier -> every warp redundantly reduces the 8 partials.
template <typename T>
struct BlockMaxClassic
{
  using value_t = T;
  using Smem    = BlockMaxSmem<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    T x = warp_max_shfl(v);
    if ((threadIdx.x & 31) == 0)
    {
      sh.partial[threadIdx.x >> 5] = x;
    }
    __syncthreads();
    T y = sh.partial[threadIdx.x & 7];
#pragma unroll
    for (int d = 4; d; d >>= 1)
    {
      y = ::max(y, __shfl_xor_sync(kFull, y, d));
    }
    return y + threadIdx.x;
  }
};

// classic with redux.sync for the warp phase and the 8-way combine (u32 only).
struct BlockMaxClassicRedux
{
  using value_t = unsigned;
  using Smem    = BlockMaxSmem<unsigned>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    unsigned x = redux_max_u32(v);
    if ((threadIdx.x & 31) == 0)
    {
      sh.partial[threadIdx.x >> 5] = x;
    }
    __syncthreads();
    unsigned y = sh.partial[threadIdx.x & 7];
    y          = redux_max_u32(y);
    return y + threadIdx.x;
  }
};

// atomic256: all 256 threads atomicMax into slot[n] -> 1 barrier -> read.
template <typename T>
struct BlockMaxAtomic256
{
  using value_t = T;
  using Smem    = BlockMaxSmem<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem& sh, int)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += block)
    {
      sh.slots[i] = 0;
    }
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    atomicMax(&sh.slots[n], v);
    __syncthreads();
    return sh.slots[n] + threadIdx.x;
  }
};

// atomic256 with the barrier replaced by waiting on each thread's own returned value, then barrier, then read.
template <typename T>
struct BlockMaxAtomic256Ret
{
  using value_t = T;
  using Smem    = BlockMaxSmem<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem& sh, int)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += block)
    {
      sh.slots[i] = 0;
    }
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    T old = atomicMax(&sh.slots[n], v);
    __syncthreads();
    return sh.slots[n] + threadIdx.x + (old & 1);
  }
};

// Completion check for the fire-and-forget round: does the value read after the barrier equal the
// true block max? The chain carries the true max (so the timing of this row is not a round cost),
// report() returns the number of mismatching (round, thread) pairs over all repetitions.
template <typename T>
struct BlockMaxSmemCheck
{
  T partial[8];
  T slots[kMaxChain];
  unsigned mismatches;
};
template <typename T>
struct BlockMaxAtomic256Check
{
  using value_t = T;
  using Smem    = BlockMaxSmemCheck<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      sh.mismatches = 0;
    }
    return threadIdx.x * 2654435761u + static_cast<T>(threadIdx.x) * 977u;
  }
  __device__ void reset(Smem& sh, int)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += block)
    {
      sh.slots[i] = 0;
    }
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    atomicMax(&sh.slots[n], v);
    __syncthreads();
    T a = sh.slots[n];
    T x = warp_max_shfl(v);
    if ((threadIdx.x & 31) == 0)
    {
      sh.partial[threadIdx.x >> 5] = x;
    }
    __syncthreads();
    T c = sh.partial[0];
#pragma unroll
    for (int w = 1; w < 8; ++w)
    {
      c = ::max(c, sh.partial[w]);
    }
    if (a != c)
    {
      atomicAdd(&sh.mismatches, 1u);
    }
    __syncthreads();
    return c * 2654435761u + threadIdx.x;
  }
  __device__ unsigned report(Smem& sh)
  {
    return sh.mismatches;
  }
};

// Eight independent fire-and-forget atomics per step (values ready in advance), then one dependent IMAD:
// the MIO throughput of same-address bursts without the issue dependency of AtomNoRet.
template <typename T, int C, int BLOCK>
struct AtomNoRetIndep
{
  using value_t = T;
  using Smem    = AtomSmem<T>;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      sh.word = 0;
    }
    sh.priv[threadIdx.x] = 0;
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    T* target = (threadIdx.x < C) ? &sh.word : &sh.priv[threadIdx.x];
#pragma unroll
    for (int i = 0; i < kAluOps; ++i)
    {
      atomicMax(target, v + static_cast<T>(i) * 3u);
    }
    return v * 3 + threadIdx.x;
  }
};

// Block size study (for opt-N8): the all-threads aggregated atomic round and the classic redux round
// at 2, 4, 8 and 16 warps per block, same IPT-free body as atomic256 and classic-redux above.
template <int BLOCK>
struct BlockMaxAtomicB
{
  using value_t = unsigned;
  using Smem    = AtomSmem<unsigned>;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem& sh, int)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += BLOCK)
    {
      sh.slots[i] = 0;
    }
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    atomicMax(&sh.slots[n], v);
    __syncthreads();
    return sh.slots[n] + threadIdx.x;
  }
};

template <int BLOCK>
struct BlockMaxClassicReduxB
{
  using value_t = unsigned;
  struct Smem
  {
    unsigned partial[BLOCK / 32];
  };
  static constexpr int block = BLOCK;
  static constexpr int warps = BLOCK / 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    unsigned x = redux_max_u32(v);
    if ((threadIdx.x & 31) == 0)
    {
      sh.partial[threadIdx.x >> 5] = x;
    }
    __syncthreads();
    unsigned y = sh.partial[(threadIdx.x & 31) % warps];
    y          = redux_max_u32(y);
    return y + threadIdx.x;
  }
};

// hybrid8: warp shfl-max -> lane0-of-each-warp atomicMax into slot[n] -> 1 barrier -> read.
template <typename T>
struct BlockMaxHybrid8
{
  using value_t = T;
  using Smem    = BlockMaxSmem<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem& sh, int)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += block)
    {
      sh.slots[i] = 0;
    }
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    T x = warp_max_shfl(v);
    if ((threadIdx.x & 31) == 0)
    {
      atomicMax(&sh.slots[n], x);
    }
    __syncthreads();
    return sh.slots[n] + threadIdx.x;
  }
};

// hybrid8 with redux instead of shuffle tree (u32 only).
struct BlockMaxHybrid8Redux
{
  using value_t = unsigned;
  using Smem    = BlockMaxSmem<unsigned>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem& sh, int)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += block)
    {
      sh.slots[i] = 0;
    }
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    unsigned x = redux_max_u32(v);
    if ((threadIdx.x & 31) == 0)
    {
      atomicMax(&sh.slots[n], x);
    }
    __syncthreads();
    return sh.slots[n] + threadIdx.x;
  }
};

// ================================================================ F. shared memory fills from global (256 threads)
// A pointer chase over kFillChunks chunks spaced kFillStride bytes apart in global memory (4 MB
// footprint visited in one cycle of length kFillChunks, larger than L1, so the loads come from L2): word 0 of a chunk holds the byte offset of
// the next chunk, so step n+1 cannot start before step n has landed in shared memory.
constexpr int kFillChunks = 256;
constexpr int kFillStride = 16384;
__device__ unsigned char* g_fill_buf = nullptr;

template <int BYTES>
struct FillSmem
{
  __align__(128) unsigned char buf[BYTES];
  __align__(8) unsigned long long mbar;
  unsigned next;
};

// control: every thread loads its 16-byte vectors into registers, stores them to smem, barrier, read word 0.
template <int BYTES>
struct FillLdgSts
{
  using value_t = unsigned;
  using Smem    = FillSmem<BYTES>;
  static constexpr int block      = 256;
  static constexpr int per_thread = BYTES / 16 / block;
  static_assert(BYTES % (16 * block) == 0, "BYTES must be a multiple of 16 * block");
  __device__ value_t init(Smem&)
  {
    return 0;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t off, int, Smem& sh)
  {
    const uint4* src = reinterpret_cast<const uint4*>(g_fill_buf + off);
    uint4* dst       = reinterpret_cast<uint4*>(sh.buf);
    uint4 r[per_thread];
#pragma unroll
    for (int i = 0; i < per_thread; ++i)
    {
      r[i] = __ldg(&src[threadIdx.x + i * block]); // LDG, not a generic LD
    }
#pragma unroll
    for (int i = 0; i < per_thread; ++i)
    {
      dst[threadIdx.x + i * block] = r[i];
    }
    __syncthreads();
    unsigned next = *reinterpret_cast<volatile unsigned*>(sh.buf);
    __syncthreads();
    return next;
  }
};

// cp.async.bulk global -> shared with mbarrier completion, one elected thread issues, all wait.
template <int BYTES>
struct FillBulk
{
  using value_t = unsigned;
  using Smem    = FillSmem<BYTES>;
  static constexpr int block = 256;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      mbar_init(smem_u32addr(&sh.mbar), 1);
      fence_mbar_init();
    }
    __syncthreads();
    return 0;
  }
  // Re-initialize per repetition so that step n always waits on phase n (parity n & 1). Without this
  // an odd chain length would leave the barrier one phase ahead at the start of the next repetition.
  __device__ void reset(Smem& sh, int)
  {
    if (threadIdx.x == 0)
    {
      unsigned mb = smem_u32addr(&sh.mbar);
      mbar_inval(mb);
      mbar_init(mb, 1);
      fence_mbar_init();
    }
  }
  __device__ value_t step(value_t off, int n, Smem& sh)
  {
    unsigned mb = smem_u32addr(&sh.mbar);
    if (threadIdx.x == 0)
    {
      fence_proxy_async_smem(); // generic-proxy reads of buf (previous step) before the async-proxy write
      mbar_arrive_expect_tx(mb, BYTES);
      bulk_g2s(smem_u32addr(sh.buf), g_fill_buf + off, BYTES, mb);
    }
    mbar_wait_parity(mb, static_cast<unsigned>(n & 1));
    unsigned next = *reinterpret_cast<volatile unsigned*>(sh.buf);
    __syncthreads();
    return next;
  }
};

// ================================================================ harness
// An experiment may define `unsigned report(Smem&)` to put a verification count into the JSON
// "check" field instead of the chain value (used by the atomics completion check).
template <class E, class = void>
struct has_report : std::false_type
{};
template <class E>
struct has_report<E, std::void_t<decltype(std::declval<E&>().report(std::declval<typename E::Smem&>()))>>
    : std::true_type
{};
// An experiment may define `value_t finish(value_t)` to fold its remaining state into the chain value
// inside the timed region (lands in the intercept, keeps that state live so the compiler cannot drop
// the work that produced it).
template <class E, class = void>
struct has_finish : std::false_type
{};
template <class E>
struct has_finish<E, std::void_t<decltype(std::declval<E&>().finish(std::declval<typename E::value_t>()))>>
    : std::true_type
{};

template <class Exp, int N>
__global__ void __launch_bounds__(Exp::block) bench(long long* out, unsigned* check)
{
  __shared__ typename Exp::Smem sh;
  Exp e;
  auto v          = e.init(sh);
  long long best  = LLONG_MAX;
  unsigned sm     = smid(); // hoisted out of the timed region, see sink()
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    e.reset(sh, N);
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll
    for (int n = 0; n < N; ++n)
    {
      v = e.step(v, n, sh);
    }
    if constexpr (has_finish<Exp>::value)
    {
      v = e.finish(v);
    }
    sink(v, sm); // scoreboard wait on the chain result before the closing barrier
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    *out = best;
    if constexpr (has_report<Exp>::value)
    {
      *check = e.report(sh);
    }
    else
    {
      *check = static_cast<unsigned>(v);
    }
  }
}

template <class Exp, int N>
bool go(long long* d, unsigned* dc, double& cyc, unsigned& chk)
{
  bench<Exp, N><<<1, Exp::block>>>(d, dc);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "KERNEL ERROR: %s\n", cudaGetErrorString(err));
    cudaGetLastError();
    return false;
  }
  long long v = 0;
  cudaMemcpy(&v, d, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&chk, dc, sizeof(unsigned), cudaMemcpyDeviceToHost);
  cyc = static_cast<double>(v);
  return true;
}

void fit(const double* x, const double* y, int m, double& a, double& b, double& r2)
{
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = 0; i < m; ++i)
  {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  b            = (m * sxy - sx * sy) / (m * sxx - sx * sx);
  a            = (sy - b * sx) / m;
  double ybar  = sy / m;
  double ssres = 0, sstot = 0;
  for (int i = 0; i < m; ++i)
  {
    double f = a + b * x[i];
    ssres += (y[i] - f) * (y[i] - f);
    sstot += (y[i] - ybar) * (y[i] - ybar);
  }
  r2 = (sstot > 0) ? 1.0 - ssres / sstot : 1.0;
}

struct Meta
{
  std::string arch, cuda, nvcc, variant, node, date, only, device;
};

static Meta g_meta;

template <class Exp>
void run(const char* name, const char* group)
{
  if (!g_meta.only.empty() && std::string(name).find(g_meta.only) == std::string::npos)
  {
    return;
  }
  long long* d;
  unsigned* dc;
  cudaMalloc(&d, sizeof(long long));
  cudaMalloc(&dc, sizeof(unsigned));
  double x[5] = {1, 2, 4, 8, 16}, y[5];
  unsigned chk[5];
  bool ok = true;
  ok &= go<Exp, 1>(d, dc, y[0], chk[0]);
  ok &= go<Exp, 2>(d, dc, y[1], chk[1]);
  ok &= go<Exp, 4>(d, dc, y[2], chk[2]);
  ok &= go<Exp, 8>(d, dc, y[3], chk[3]);
  ok &= go<Exp, 16>(d, dc, y[4], chk[4]);
  cudaFree(d);
  cudaFree(dc);
  double a = 0, b = 0, r2 = 0;
  if (ok)
  {
    fit(x, y, 5, a, b, r2);
  }
  printf("{\"name\":\"%s\",\"group\":\"%s\",\"cycles\":%.2f,\"intercept\":%.1f,\"r2\":%.5f,"
         "\"raw\":[%.0f,%.0f,%.0f,%.0f,%.0f],\"check\":[%u,%u,%u,%u,%u],\"ok\":%s,\"block\":%d,"
         "\"arch\":\"%s\",\"device\":\"%s\",\"cuda\":\"%s\",\"nvcc\":\"%s\",\"variant\":\"%s\",\"node\":\"%s\","
         "\"date\":\"%s\",\"file\":\"harness/micro/intrinsics_micro.cu\",\"method\":\"chain-slope min of %d reps\"}\n",
         name,
         group,
         b,
         a,
         r2,
         y[0],
         y[1],
         y[2],
         y[3],
         y[4],
         chk[0],
         chk[1],
         chk[2],
         chk[3],
         chk[4],
         ok ? "true" : "false",
         Exp::block,
         g_meta.arch.c_str(),
         g_meta.device.c_str(),
         g_meta.cuda.c_str(),
         g_meta.nvcc.c_str(),
         g_meta.variant.c_str(),
         g_meta.node.c_str(),
         g_meta.date.c_str(),
         kReps);
  fflush(stdout);
}

// kFillChunks chunks kFillStride bytes apart, word 0 of chunk i holds the byte offset of chunk (i + 101) mod kFillChunks
// (101 is coprime to 256, so the chase visits every chunk once per cycle).
static void prepare_fill_buffer()
{
  std::vector<unsigned> h(static_cast<size_t>(kFillChunks) * kFillStride / 4, 0x12345678u);
  for (int i = 0; i < kFillChunks; ++i)
  {
    h[static_cast<size_t>(i) * (kFillStride / 4)] = ((static_cast<unsigned>(i) + 101u) % kFillChunks) * kFillStride;
  }
  unsigned char* dbuf = nullptr;
  cudaMalloc(&dbuf, h.size() * 4);
  cudaMemcpy(dbuf, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(g_fill_buf, &dbuf, sizeof(dbuf));
}

int main(int argc, char** argv)
{
  g_meta.variant = "unknown";
  g_meta.node    = "unknown";
  g_meta.date    = "unknown";
  for (int i = 1; i + 1 < argc; i += 2)
  {
    if (!strcmp(argv[i], "--node"))
    {
      g_meta.node = argv[i + 1];
    }
    else if (!strcmp(argv[i], "--date"))
    {
      g_meta.date = argv[i + 1];
    }
    else if (!strcmp(argv[i], "--variant"))
    {
      g_meta.variant = argv[i + 1];
    }
    else if (!strcmp(argv[i], "--only"))
    {
      g_meta.only = argv[i + 1];
    }
  }
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  char buf[64];
  snprintf(buf, sizeof(buf), "sm_%d%d", p.major, p.minor);
  g_meta.arch   = buf;
  g_meta.device = p.name;
  int rt        = 0;
  cudaRuntimeGetVersion(&rt);
  snprintf(buf, sizeof(buf), "%d.%d", rt / 1000, (rt % 1000) / 10);
  g_meta.cuda = buf;
  snprintf(buf, sizeof(buf), "%d.%d.%d", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
  g_meta.nvcc = buf;
  fprintf(stderr, "device: %s (%s, %d SMs) runtime %s nvcc %s variant %s\n",
          p.name, g_meta.arch.c_str(), p.multiProcessorCount, g_meta.cuda.c_str(), g_meta.nvcc.c_str(),
          g_meta.variant.c_str());

  // A. ALU and SIMD operations: pair chains against the IMAD-only base chain, kAluOps per step
  run<AluBase<OpU32>>("alu base: 8 x imad u32 per step", "alu");
  run<AluPair<OpIAdd>>("alu pair: 8 x (iadd u32 + imad)", "alu");
  run<AluPair<OpIMax>>("alu pair: 8 x (max u32 + imad)", "alu");
  run<AluPair<OpIMax3>>("alu pair: 8 x (__vimax3_u32 + imad)", "alu");
  run<AluPair<OpIMax3S>>("alu pair: 8 x (__vimax3_s32 + imad)", "alu");
  run<AluPair<OpIMaxMax>>("alu pair: 8 x (max(max(v,a),b) u32 + imad)", "alu");
  run<AluPair<OpIAddMax>>("alu pair: 8 x (max(v+a,b) u32 + imad)", "alu");
  run<AluPair<OpPrmt>>("alu pair: 8 x (prmt + imad)", "alu");
  run<AluPair<OpBmsk>>("alu pair: 8 x (bmsk.clamp + iadd + imad)", "alu");
  run<AluPair<OpVMaxU2>>("simd pair: 8 x (__vmaxu2 u16x2 + imad)", "simd");
  run<AluPair<OpVMaxU2Ptx>>("simd pair: 8 x (max.u16x2 ptx + imad)", "simd");
  run<AluPair<OpVMaxS2>>("simd pair: 8 x (__vmaxs2 s16x2 + imad)", "simd");
  run<AluPair<OpVMax3U16x2>>("simd pair: 8 x (__vimax3_u16x2 + imad)", "simd");
  run<AluPair<OpVMaxU4>>("simd pair: 8 x (__vmaxu4 u8x4 + imad)", "simd");
  run<AluPair<OpHMax2Bf16>>("simd pair: 8 x (__hmax2 bf16x2 + imad)", "simd");
  run<AluPair<OpMaxBf16x2Ptx>>("simd pair: 8 x (max.bf16x2 ptx + imad)", "simd");
  run<AluPair<OpMaxF16x2Ptx>>("simd pair: 8 x (max.f16x2 ptx + imad)", "simd");
  run<AluPair<OpMinBf16x2Ptx>>("simd pair: 8 x (min.bf16x2 ptx + imad)", "simd");
  run<AluPair<OpHgt2MaskBf16>>("simd pair: 8 x (__hgt2_mask bf16x2 per-half mask + imad)", "simd");
  run<AluPair<OpSetGtBf16x2Ptx>>("simd pair: 8 x (set.gt.u32.bf16x2 ptx + imad)", "simd");
  run<AluPair<OpVcmpgtu2>>("simd pair: 8 x (__vcmpgtu2 u16x2 per-half mask + imad)", "simd");
  run<AluPair<OpVcmpgtu4>>("simd pair: 8 x (__vcmpgtu4 u8x4 per-byte mask + imad)", "simd");
  run<AluBase<OpF32>>("alu base: 8 x ffma f32 per step", "alu");
  run<AluPair<OpFMax>>("alu pair: 8 x (fmaxf + ffma)", "alu");
  run<AluPair<OpFMaxMax>>("alu pair: 8 x (fmaxf(fmaxf(v,a),b) + ffma)", "alu");
  run<AluPair<OpFMax3Ptx>>("alu pair: 8 x (max.f32 3-input ptx + ffma)", "alu");
  run<FfsChain>("alu __ffs", "alu");
  run<PopcChain>("alu __popc", "alu");
  run<FnsChain>("alu __fns", "alu");

  // B. warp collectives
  run<ReduxU32>("redux.sync.max.u32", "warp");
#if MICRO_FAMILY
  run<ReduxF32>("redux.sync.max.f32 (sm_100f)", "warp");
  run<ReduxF32AbsNaN>("redux.sync.max.abs.NaN.f32 (sm_100f)", "warp");
#endif
  run<ReduxF32ViaTwiddle>("redux.sync.max.u32 on twiddled f32 (twiddle+redux+untwiddle)", "warp");
  run<ShflXor1>("shfl_xor (single)", "warp");
  run<ShflIdxDep>("shfl_idx data-dependent lane", "warp");
  run<WarpMaxShflU32>("warp max 5-shfl tree u32", "warp");
  run<WarpMaxShflU64>("warp max 5-shfl tree u64", "warp");
  run<WarpMaxShflF32>("warp max 5-shfl tree f32 (fmaxf)", "warp");
  run<Ballot>("ballot_sync", "warp");
  run<BallotIndep<1>>("ballot indep: 1 per step + merge", "warp");
  run<BallotIndep<4>>("ballot indep: 4 independent per step + merge", "warp");
  run<BallotIndep<8>>("ballot indep: 8 independent per step + merge", "warp");
  run<BallotIndep<16>>("ballot indep: 16 independent per step + merge", "warp");
  run<ReduxIndep<1>>("redux indep: 1 per step + merge", "warp");
  run<ReduxIndep<8>>("redux indep: 8 independent per step + merge", "warp");
  run<ShflIndep<1>>("shfl indep: 1 per step + merge", "warp");
  run<ShflIndep<8>>("shfl indep: 8 independent per step + merge", "warp");
  // pipe throughput: 8 independent accumulators per lane, 1, 4, 8, 16 warps per block
#define TPUT_ROWS(OP, LABEL)                                                              \
  run<Tput<OP, 8, 32>>("tput " LABEL " x8 indep, 1 warp", "tput");                      \
  run<Tput<OP, 8, 128>>("tput " LABEL " x8 indep, 4 warps (128 thr)", "tput");         \
  run<Tput<OP, 8, 256>>("tput " LABEL " x8 indep, 8 warps (256 thr)", "tput");         \
  run<Tput<OP, 8, 512>>("tput " LABEL " x8 indep, 16 warps (512 thr)", "tput");
  TPUT_ROWS(TpMaxBf16x2, "max.bf16x2 HMNMX2")
  TPUT_ROWS(TpMaxU32, "max.u32 VIMNMX")
  TPUT_ROWS(TpMaxU16x2, "max.u16x2 VIMNMX.U16x2")
  TPUT_ROWS(TpSetBf16x2, "set.gt.u32.bf16x2 HSET2")
  TPUT_ROWS(TpLop3, "lop3 LOP3")
  TPUT_ROWS(TpImad, "mad.lo.u32 IMAD")
  TPUT_ROWS(TpMaxF32, "max.f32 FMNMX")
  TPUT_ROWS(TpSel, "setp+selp ISETP+SEL")
  TPUT_ROWS(TpStageBf16x2, "stage bf16x2: shfl + max + min + sel")
  TPUT_ROWS(TpStageU16x2, "stage u16x2: shfl + max + min + sel")
  TPUT_ROWS(TpAltBf16x2, "alt max/min bf16x2 HMNMX2 unfused")
  TPUT_ROWS(TpAltU32, "alt max/min u32 VIMNMX unfused")
  TPUT_ROWS(TpAltU16x2, "alt max/min u16x2 VIMNMX.U16x2 unfused")
  TPUT_ROWS(TpAltF32, "alt max/min f32 FMNMX unfused")
  TPUT_ROWS(TpShfl, "shfl.bfly SHFL")
#undef TPUT_ROWS
  // the same chains run by every warp of a 128, 256 and 512 thread block at once: per-scheduler or per-SM limits
  run<ReduxIndep<1, 128>>("redux indep x4 warps: 1 per step + merge (128 thr)", "warp");
  run<ReduxIndep<4, 128>>("redux indep x4 warps: 4 independent per step + merge (128 thr)", "warp");
  run<ReduxIndep<8, 128>>("redux indep x4 warps: 8 independent per step + merge (128 thr)", "warp");
  run<ReduxIndep<1, 256>>("redux indep x8 warps: 1 per step + merge (256 thr)", "warp");
  run<ReduxIndep<4, 256>>("redux indep x8 warps: 4 independent per step + merge (256 thr)", "warp");
  run<ReduxIndep<8, 256>>("redux indep x8 warps: 8 independent per step + merge (256 thr)", "warp");
  run<ReduxIndep<1, 512>>("redux indep x16 warps: 1 per step + merge (512 thr)", "warp");
  run<ReduxIndep<4, 512>>("redux indep x16 warps: 4 independent per step + merge (512 thr)", "warp");
  run<ReduxIndep<8, 512>>("redux indep x16 warps: 8 independent per step + merge (512 thr)", "warp");
  run<BallotIndep<8, 128>>("ballot indep x4 warps: 8 independent per step + merge (128 thr)", "warp");
  run<BallotIndep<8, 256>>("ballot indep x8 warps: 8 independent per step + merge (256 thr)", "warp");
  run<BallotIndep<8, 512>>("ballot indep x16 warps: 8 independent per step + merge (512 thr)", "warp");
  run<ShflIndep<8, 128>>("shfl indep x4 warps: 8 independent per step + merge (128 thr)", "warp");
  run<ShflIndep<8, 256>>("shfl indep x8 warps: 8 independent per step + merge (256 thr)", "warp");
  run<ShflIndep<8, 512>>("shfl indep x16 warps: 8 independent per step + merge (512 thr)", "warp");
  run<BallotFfsElect>("ballot + ffs + winner select", "warp");
  run<ReduxBallotRound>("W2 round unpacked: redux + ballot(v==max) + ffs + select", "warp");
  run<ReduxPackedRound>("W2 round packed key|index: redux + compare select", "warp");
  run<W2PackedList<1, false>>("W2 packed list round IPT 1: redux + head compare + predicated shift", "warp");
  run<W2PackedList<4, false>>("W2 packed list round IPT 4: redux + head compare + predicated shift", "warp");
  run<W2PackedList<8, false>>("W2 packed list round IPT 8: redux + head compare + predicated shift", "warp");
  run<W2PackedList<16, false>>("W2 packed list round IPT 16: redux + head compare + predicated shift", "warp");
  run<W2PackedList<32, false>>("W2 packed list round IPT 32: redux + head compare + predicated shift", "warp");
  run<W2PackedList<8, true>>("W2 packed list round IPT 8: redux + head compare + branchy shift", "warp");
  run<W2PackedList<32, true>>("W2 packed list round IPT 32: redux + head compare + branchy shift", "warp");
  run<ReduxElect>("redux chain + elect.sync per step", "warp");
  run<ReduxFfsLeader>("redux chain + ffs(activemask) leader per step", "warp");
  run<MatchAnyClasses<1>>("match_any 1 class (all equal)", "warp");
  run<MatchAnyClasses<2>>("match_any 2 classes", "warp");
  run<MatchAnyClasses<4>>("match_any 4 classes", "warp");
  run<MatchAnyClasses<8>>("match_any 8 classes", "warp");
  run<MatchAnyClasses<16>>("match_any 16 classes", "warp");
  run<MatchAnyClasses<32>>("match_any 32 classes (all distinct)", "warp");
  run<SyncWarp>("syncwarp", "warp");
  run<LdsSts>("STS+LDS round trip", "smem");
  run<TransposeStsLds>("transpose 4 words: 4 STS + syncwarp + 4 LDS", "smem");
  run<TransposeStsmLdsm>("transpose 4 words: stmatrix.x4 + syncwarp + ldmatrix.x4.trans", "smem");
  run<TransposeMovm>("transpose 4 words: 4 x movmatrix.trans (registers only)", "smem");
  run<LdsmOnly>("ldmatrix.x4 (address-dependent chain)", "smem");

  // C. shared atomics
  run<AtomUncontended<unsigned>>("atomicMax smem u32 with return, uncontended", "atomics");
  run<AtomUncontended<unsigned long long>>("atomicMax smem u64 with return, uncontended", "atomics");
  run<AtomContended<unsigned, 2, 32>>("atomicMax smem u32 with return, contention 2", "atomics");
  run<AtomContended<unsigned, 8, 32>>("atomicMax smem u32 with return, contention 8", "atomics");
  run<AtomContended<unsigned, 32, 32>>("atomicMax smem u32 with return, contention 32", "atomics");
  run<AtomContended<unsigned, 256, 256>>("atomicMax smem u32 with return, contention 256", "atomics");
  run<AtomContended<unsigned long long, 32, 32>>("atomicMax smem u64 with return, contention 32", "atomics");
  run<AtomContended<unsigned long long, 256, 256>>("atomicMax smem u64 with return, contention 256", "atomics");
  run<AtomWarpLane0<unsigned>>("atomicMax smem u32 with return, 8 warps lane0", "atomics");
  run<AtomWarpLane0<unsigned long long>>("atomicMax smem u64 with return, 8 warps lane0", "atomics");
  run<AtomBanked>("atomicMax smem u32 with return, 256thr/32banks", "atomics");
  run<AluBase256<256>>("alu base: 8 x imad u32 per step (256 thr)", "alu");
  run<AtomNoRet<unsigned, 1, 32>>("atomics pair: 8 x (atomicMax smem u32 fire-and-forget, 1 on the slot, 31 private + imad)", "atomics");
  run<AtomNoRet<unsigned, 32, 32>>("atomics pair: 8 x (atomicMax smem u32 fire-and-forget, 32 on one slot + imad)", "atomics");
  run<AtomNoRet<unsigned, 256, 256>>("atomics pair: 8 x (atomicMax smem u32 fire-and-forget, 256 on one slot + imad)", "atomics");
  run<AtomNoRet<unsigned long long, 256, 256>>("atomics pair: 8 x (atomicMax smem u64 fire-and-forget, 256 on one slot + imad)", "atomics");
  run<AtomNoRetIndep<unsigned, 32, 32>>("atomics burst: 8 independent atomicMax smem u32 fire-and-forget, 32 on one slot + imad", "atomics");
  run<AtomNoRetIndep<unsigned, 256, 256>>("atomics burst: 8 independent atomicMax smem u32 fire-and-forget, 256 on one slot + imad", "atomics");
  run<AtomNoRetIndep<unsigned, 1, 256>>("atomics burst: 8 independent atomicMax smem u32 fire-and-forget, 256 private words + imad", "atomics");
  run<WarpMaxAtomic32>("warp max atomic32: 32 atomicMax one slot + syncwarp + LDS", "atomics");

  // D. barriers
  run<SyncThreads<128>>("__syncthreads (128 thr)", "barrier");
  run<SyncThreads<256>>("__syncthreads (256 thr)", "barrier");
  run<SyncThreads<512>>("__syncthreads (512 thr)", "barrier");

  // E. composed block-wide max (256 threads)
  run<BlockMaxClassic<unsigned>>("blockmax classic u32 (shfl+smem+bar+8-way shfl)", "blockmax");
  run<BlockMaxClassic<unsigned long long>>("blockmax classic u64 (shfl+smem+bar+8-way shfl)", "blockmax");
  run<BlockMaxClassicRedux>("blockmax classic u32 (redux+smem+bar+redux)", "blockmax");
  run<BlockMaxAtomic256<unsigned>>("blockmax atomic256 u32", "blockmax");
  run<BlockMaxAtomic256<unsigned long long>>("blockmax atomic256 u64", "blockmax");
  run<BlockMaxAtomic256Ret<unsigned>>("blockmax atomic256 u32 with return (each thread waits) + bar", "blockmax");
  run<BlockMaxAtomic256Check<unsigned>>("blockmax atomic256 u32 completion check (check = mismatches, not a round cost)", "blockmax");
  run<BlockMaxHybrid8<unsigned>>("blockmax hybrid8 u32 (shfl+atomic)", "blockmax");
  run<BlockMaxHybrid8<unsigned long long>>("blockmax hybrid8 u64 (shfl+atomic)", "blockmax");
  run<BlockMaxHybrid8Redux>("blockmax hybrid8 u32 (redux+atomic)", "blockmax");
  // block size study: barrier, all-threads aggregated atomic round and classic redux round against warps per block
  run<SyncThreads<64>>("__syncthreads, 2 warps (64 thr)", "barrier");
  run<SyncThreads<128>>("__syncthreads, 4 warps (128 thr)", "barrier");
  run<SyncThreads<256>>("__syncthreads, 8 warps (256 thr)", "barrier");
  run<SyncThreads<512>>("__syncthreads, 16 warps (512 thr)", "barrier");
  run<BlockMaxAtomicB<64>>("blockmax atomic all-threads u32, 2 warps (64 thr)", "blockmax");
  run<BlockMaxAtomicB<128>>("blockmax atomic all-threads u32, 4 warps (128 thr)", "blockmax");
  run<BlockMaxAtomicB<256>>("blockmax atomic all-threads u32, 8 warps (256 thr)", "blockmax");
  run<BlockMaxAtomicB<512>>("blockmax atomic all-threads u32, 16 warps (512 thr)", "blockmax");
  run<BlockMaxClassicReduxB<64>>("blockmax classic redux u32, 2 warps (64 thr)", "blockmax");
  run<BlockMaxClassicReduxB<128>>("blockmax classic redux u32, 4 warps (128 thr)", "blockmax");
  run<BlockMaxClassicReduxB<256>>("blockmax classic redux u32, 8 warps (256 thr)", "blockmax");
  run<BlockMaxClassicReduxB<512>>("blockmax classic redux u32, 16 warps (512 thr)", "blockmax");

  // F. shared memory fills (256 threads), pointer-chased through a 4 MB footprint (L2 resident, beyond L1)
  prepare_fill_buffer();
  run<FillLdgSts<4096>>("fill smem 4KB: LDG.128 + STS + bar (control)", "fill");
  run<FillBulk<4096>>("fill smem 4KB: cp.async.bulk + mbarrier", "fill");
  run<FillLdgSts<16384>>("fill smem 16KB: LDG.128 x4 + STS + bar (control)", "fill");
  run<FillBulk<16384>>("fill smem 16KB: cp.async.bulk + mbarrier", "fill");
  return 0;
}
