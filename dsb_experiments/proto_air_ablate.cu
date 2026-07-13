// Granular ablation of the block_topk_air latency optimizations, in integration order, across a
// key/value type matrix — the basis for deciding which changes to upstream and for which type
// combinations (see BLOCK_TOPK_AIR_OPT_RESULTS.md).
//
// AirAblate<KeyT, ValueT, LVL> applies changes CUMULATIVELY:
//   L0  faithful reimplementation of today's block_topk_air (runtime pass loop, per-pass
//       histogram init, scan + smem writeback + separate choose, setup phase + 2-class ticket
//       scatter, keys / values in separate exchange trips, -0.0 normalize/flip/restore)
//   L1  + fused scan+choose: crossing test on the scan's register results
//       (exclusive = inclusive - count); no histogram writeback, no choose phase, -1 barrier/pass
//   L2  + double-buffered histograms: one prologue init; next-pass buffer zeroed during the
//       histogram phase; the per-pass init phase and its barrier disappear (+1 KB smem)
//   L3  + preset scatter counters + computed class-1 base (tied position = total_selected +
//       zero-based ticket): the scatter setup phase, its barrier, and the post-loop barrier go
//   L4  + pair scatter: keys and values scattered together, gathered once (-2 barriers, -1 item
//       pass; exchange tile*max(sizeof) -> tile*sizeof(pair)). PAIRS ONLY (keys-only: alias of L3)
//   L5  + original-value scatter: keep a register copy, scatter it, drop untwiddle + the whole
//       -0.0 normalize/flip machinery (fp keys; for integer keys only the un-conversion goes)
//   L6  + compile-time unrolled pass loop (immediate shifts/masks; needs compile-time bit range)
//   L7  + packed single-word pass state (DIAGNOSTIC: regressed inside PR #9066, neutral here —
//       measured per type to settle applicability)
//
// Type matrix instantiated: f32+i32 (all levels), f32 keys-only, u32+i32 (all levels),
// f16+i32 (L0/L6: 2-pass structure), f64+i32 (L0/L6: 8-pass, 64-bit ord), f32+i64 (L3/L4/L6:
// 8-byte values stress the pair-exchange smem trade).
//
// Modes: ./proto_air_ablate [correct|lat|thr|res|all]. Methodology as in the other suites.

#include <cub/block/block_scan.cuh>

#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

constexpr unsigned kFull = 0xffffffffu;
constexpr int kBlock     = 256;
constexpr int kIpt       = 4;
constexpr int kN         = 1024;
constexpr int kK         = 16;

__device__ int g_sink;

using block_scan_t = cub::BlockScan<unsigned, kBlock, cub::BLOCK_SCAN_WARP_SCANS>;

struct KeysOnly
{};

// ------------------------------------------------------------------ key traits
template <typename T>
struct AKey;

template <>
struct AKey<float>
{
  using ord_t                   = unsigned;
  static constexpr int key_bits = 32;
  static constexpr bool is_fp   = true;
  static __device__ __forceinline__ ord_t to_ord(float f)
  {
    unsigned u = __float_as_uint(f);
    return u ^ (((unsigned) ((int) u >> 31)) | 0x80000000u);
  }
  static __device__ __forceinline__ float from_ord(ord_t t)
  {
    unsigned m = (~((unsigned) ((int) t >> 31))) | 0x80000000u;
    return __uint_as_float(t ^ m);
  }
  static __device__ __forceinline__ ord_t ord_mzero()
  {
    return 0x7fffffffu;
  }
  static __device__ __forceinline__ ord_t ord_pzero()
  {
    return 0x80000000u;
  }
  static __device__ __forceinline__ float neg_zero()
  {
    return -0.0f;
  }
};

template <>
struct AKey<unsigned>
{
  using ord_t                   = unsigned;
  static constexpr int key_bits = 32;
  static constexpr bool is_fp   = false;
  static __device__ __forceinline__ ord_t to_ord(unsigned u)
  {
    return u;
  }
  static __device__ __forceinline__ unsigned from_ord(ord_t t)
  {
    return t;
  }
  static __device__ __forceinline__ ord_t ord_mzero()
  {
    return 0;
  }
  static __device__ __forceinline__ ord_t ord_pzero()
  {
    return 0;
  }
  static __device__ __forceinline__ unsigned neg_zero()
  {
    return 0;
  }
};

template <>
struct AKey<__half>
{
  using ord_t                   = unsigned short;
  static constexpr int key_bits = 16;
  static constexpr bool is_fp   = true;
  static __device__ __forceinline__ ord_t to_ord(__half h)
  {
    unsigned short u = __half_as_ushort(h);
    return (unsigned short) (u ^ ((u >> 15) ? 0xffffu : 0x8000u));
  }
  static __device__ __forceinline__ __half from_ord(ord_t t)
  {
    unsigned short u = (unsigned short) (t ^ ((t >> 15) ? 0x8000u : 0xffffu));
    return __ushort_as_half(u);
  }
  static __device__ __forceinline__ ord_t ord_mzero()
  {
    return 0x7fffu;
  }
  static __device__ __forceinline__ ord_t ord_pzero()
  {
    return 0x8000u;
  }
  static __device__ __forceinline__ __half neg_zero()
  {
    return __ushort_as_half(0x8000u);
  }
};

template <>
struct AKey<double>
{
  using ord_t                   = unsigned long long;
  static constexpr int key_bits = 64;
  static constexpr bool is_fp   = true;
  static __device__ __forceinline__ ord_t to_ord(double f)
  {
    unsigned long long u = (unsigned long long) __double_as_longlong(f);
    return u ^ (((unsigned long long) ((long long) u >> 63)) | 0x8000000000000000ull);
  }
  static __device__ __forceinline__ double from_ord(ord_t t)
  {
    unsigned long long m = (~((unsigned long long) ((long long) t >> 63))) | 0x8000000000000000ull;
    return __longlong_as_double((long long) (t ^ m));
  }
  static __device__ __forceinline__ ord_t ord_mzero()
  {
    return 0x7fffffffffffffffull;
  }
  static __device__ __forceinline__ ord_t ord_pzero()
  {
    return 0x8000000000000000ull;
  }
  static __device__ __forceinline__ double neg_zero()
  {
    return -0.0;
  }
};

// ------------------------------------------------------------------ the leveled template
template <typename KeyT, typename ValueT, int LVL, bool ORIG = (LVL >= 5)>
struct AirAblate
{
  using key_t                     = KeyT;
  using AK                        = AKey<KeyT>;
  using ord_t                     = typename AK::ord_t;
  static constexpr bool keys_only = ::std::is_same_v<ValueT, KeysOnly>;
  using value_t                   = ::std::conditional_t<keys_only, int, ValueT>;
  static constexpr int NPASS      = AK::key_bits / 8;
  static constexpr int NBUF       = (LVL >= 2) ? 2 : 1;
  static constexpr bool use_pair  = (LVL >= 4) && !keys_only;

  struct pair_t
  {
    KeyT k;
    value_t v;
  };
  struct ExchKeys
  {
    KeyT k[kN];
  };
  struct ExchClassic
  {
    union
    {
      KeyT k[kN];
      value_t v[kN];
    } u;
  };
  struct ExchPair
  {
    pair_t p[kN];
  };
  using exch_t =
    ::std::conditional_t<keys_only, ExchKeys, ::std::conditional_t<use_pair, ExchPair, ExchClassic>>;

  struct PassStage
  {
    unsigned hist[NBUF][256];
    typename block_scan_t::TempStorage scan_temp;
  };
  struct SelectStage
  {
    unsigned sel_off[2]; // L<3 ticket counters (aliased region, reset in the setup phase)
    exch_t exch;
  };
  struct Smem
  {
    union
    {
      PassStage passes;
      SelectStage select;
    } stage;
    // pass state outside the union so the last state read never races the exchange writes
    unsigned st_packed;
    unsigned st_sel, st_cands, st_bucket;
    unsigned cntA, cntB; // L>=3 preset ticket counters (never aliased)
  };

  // One radix pass; returns true when all remaining candidates are selected (early exit).
  __device__ __forceinline__ static bool pass_step(
    Smem& sh,
    const ord_t (&uk)[4],
    int& k,
    int& total_selected,
    int& num_candidates,
    ord_t& kth_prefix,
    ord_t& pmask,
    int pass,
    int pass_begin)
  {
    unsigned(&cur)[256] = sh.stage.passes.hist[(LVL >= 2) ? (pass & 1) : 0];
    if constexpr (LVL < 2)
    {
      cur[threadIdx.x] = 0;
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if ((uk[i] & pmask) == kth_prefix)
      {
        const unsigned d = (unsigned) ((uk[i] >> pass_begin) & 0xff);
        atomicAdd(&cur[255 - d], 1u);
      }
    }
    if constexpr (LVL >= 2)
    {
      if (pass > 0 && pass + 1 < NPASS)
      {
        sh.stage.passes.hist[(pass + 1) & 1][threadIdx.x] = 0;
      }
    }
    __syncthreads();

    if constexpr (LVL >= 1)
    {
      // fused scan + choose
      const unsigned cnt = cur[threadIdx.x];
      unsigned incl;
      block_scan_t(sh.stage.passes.scan_temp).InclusiveSum(cnt, incl);
      const unsigned excl = incl - cnt;
      if (excl < (unsigned) k && incl >= (unsigned) k)
      {
        if constexpr (LVL >= 7)
        {
          sh.st_packed = (threadIdx.x << 16) | ((incl - excl) << 5) | excl;
        }
        else
        {
          sh.st_bucket = threadIdx.x;
          sh.st_cands  = incl - excl;
          sh.st_sel    = excl;
        }
      }
      __syncthreads();
    }
    else
    {
      // faithful: scan + writeback, barrier, separate choose, barrier
      unsigned tb = cur[threadIdx.x];
      block_scan_t(sh.stage.passes.scan_temp).InclusiveSum(tb, tb);
      cur[threadIdx.x] = tb;
      __syncthreads();
      {
        const unsigned prev = (threadIdx.x == 0) ? 0 : cur[threadIdx.x - 1];
        const unsigned c    = cur[threadIdx.x];
        if (prev < (unsigned) k && c >= (unsigned) k)
        {
          sh.st_bucket = threadIdx.x;
          sh.st_cands  = c - prev;
          sh.st_sel    = prev;
        }
      }
      __syncthreads();
    }

    unsigned sel, cands, bucket;
    if constexpr (LVL >= 7)
    {
      const unsigned st = sh.st_packed;
      sel               = st & 0x1fu;
      cands             = (st >> 5) & 0x7ffu;
      bucket            = st >> 16;
    }
    else
    {
      sel    = sh.st_sel;
      cands  = sh.st_cands;
      bucket = sh.st_bucket;
    }
    k -= (int) sel;
    num_candidates = (int) cands;
    total_selected += (int) sel;
    const unsigned kth_digit = 255u - bucket;
    kth_prefix |= ord_t(kth_digit) << pass_begin;
    pmask |= ord_t(0xff) << pass_begin;
    return num_candidates == k;
  }

  __device__ __forceinline__ static void
  run(const KeyT (&v)[4], const int (&idx)[4], Smem& sh, KeyT* out_v, int* out_i)
  {
    value_t vals[4];
    if constexpr (!keys_only)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        vals[i] = (value_t) idx[i];
      }
    }
    ord_t uk[4];
    unsigned flip = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      uk[i] = AK::to_ord(v[i]);
    }
    if constexpr (AK::is_fp && !ORIG)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if (uk[i] == AK::ord_mzero())
        {
          flip |= 1u << i;
          uk[i] = AK::ord_pzero();
        }
      }
    }
    if constexpr (LVL >= 3)
    {
      if (threadIdx.x == kBlock - 1)
      {
        sh.cntA = 0;
        sh.cntB = 0;
      }
    }
    if constexpr (LVL >= 2)
    {
      sh.stage.passes.hist[0][threadIdx.x] = 0;
      sh.stage.passes.hist[1][threadIdx.x] = 0;
      __syncthreads();
    }

    ord_t kth_prefix = 0, pmask = 0;
    int k              = kK;
    int total_selected = 0;
    int num_candidates = kN;
    if constexpr (LVL >= 6)
    {
#pragma unroll
      for (int pass = 0; pass < NPASS; ++pass)
      {
        if (pass_step(sh, uk, k, total_selected, num_candidates, kth_prefix, pmask, pass, AK::key_bits - 8 * (pass + 1)))
        {
          break;
        }
      }
    }
    else
    {
#pragma unroll 1
      for (int pass = 0; pass < NPASS; ++pass)
      {
        const int pass_begin = AK::key_bits - 8 * (pass + 1);
        if (pass_step(sh, uk, k, total_selected, num_candidates, kth_prefix, pmask, pass, pass_begin))
        {
          break;
        }
      }
    }

    // epilogue
    const bool select_all = (num_candidates + total_selected == kK);
    if constexpr (LVL < 3)
    {
      __syncthreads(); // repurpose shared memory (post-loop)
      if (threadIdx.x == 0)
      {
        sh.stage.select.sel_off[0] = 0;
        sh.stage.select.sel_off[1] = (unsigned) total_selected;
      }
      __syncthreads(); // setup
    }
    [[maybe_unused]] int scatter_idx[4] = {-1, -1, -1, -1};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const ord_t kp     = uk[i] & pmask;
      const bool is_sel  = kp > kth_prefix;
      const bool is_cand = kp == kth_prefix;
      const bool cls1    = !select_all && is_cand;
      if (is_sel || is_cand)
      {
        unsigned off;
        if constexpr (LVL >= 3)
        {
          const unsigned t = atomicAdd(cls1 ? &sh.cntB : &sh.cntA, 1u);
          off              = cls1 ? (unsigned) total_selected + t : t;
        }
        else
        {
          off = atomicAdd(&sh.stage.select.sel_off[cls1 ? 1 : 0], 1u);
        }
        KeyT kv;
        if constexpr (ORIG)
        {
          kv = v[i];
        }
        else if constexpr (AK::is_fp)
        {
          kv = ((flip >> i) & 1) ? AK::neg_zero() : AK::from_ord(uk[i]);
        }
        else
        {
          kv = AK::from_ord(uk[i]);
        }
        if constexpr (use_pair)
        {
          sh.stage.select.exch.p[off] = pair_t{kv, vals[i]};
        }
        else if constexpr (keys_only)
        {
          sh.stage.select.exch.k[off] = kv;
        }
        else
        {
          sh.stage.select.exch.u.k[off] = kv;
          scatter_idx[i]                = (int) off;
        }
      }
    }
    __syncthreads();
    KeyT outk[4]     = {};
    value_t outv[4]  = {};
    if constexpr (use_pair)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        const int bi = threadIdx.x * 4 + i;
        if (bi < kK)
        {
          const pair_t p = sh.stage.select.exch.p[bi];
          outk[i]        = p.k;
          outv[i]        = p.v;
        }
      }
    }
    else if constexpr (keys_only)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        const int bi = threadIdx.x * 4 + i;
        if (bi < kK)
        {
          outk[i] = sh.stage.select.exch.k[bi];
        }
      }
    }
    else
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        const int bi = threadIdx.x * 4 + i;
        if (bi < kK)
        {
          outk[i] = sh.stage.select.exch.u.k[bi];
        }
      }
      __syncthreads(); // repurpose exchange for values
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if (scatter_idx[i] >= 0)
        {
          sh.stage.select.exch.u.v[scatter_idx[i]] = vals[i];
        }
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        const int bi = threadIdx.x * 4 + i;
        if (bi < kK)
        {
          outv[i] = sh.stage.select.exch.u.v[bi];
        }
      }
    }
    if (threadIdx.x < kK / kIpt)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        out_v[threadIdx.x * 4 + i] = outk[i];
        out_i[threadIdx.x * 4 + i] = keys_only ? -2 : (int) outv[i];
      }
    }
    __syncthreads();
  }
};

// ------------------------------------------------------------------ instantiations
#define AB(NAME, K, V, L, STR)                    \
  struct NAME : AirAblate<K, V, L>                \
  {                                               \
    static constexpr const char* name = STR;      \
  };

AB(F32P_L0, float, int, 0, "f32+i32  L0 faithful air")
AB(F32P_L1, float, int, 1, "f32+i32  L1 +fused scan/choose")
AB(F32P_L2, float, int, 2, "f32+i32  L2 +double-buffered hist")
AB(F32P_L3, float, int, 3, "f32+i32  L3 +preset counters")
AB(F32P_L4, float, int, 4, "f32+i32  L4 +pair scatter")
AB(F32P_L5, float, int, 5, "f32+i32  L5 +orig-value scatter")
AB(F32P_L6, float, int, 6, "f32+i32  L6 +unrolled passes")
AB(F32P_L7, float, int, 7, "f32+i32  L7 +packed state (diag)")

AB(F32K_L0, float, KeysOnly, 0, "f32 keys L0 faithful air")
AB(F32K_L1, float, KeysOnly, 1, "f32 keys L1 +fused scan/choose")
AB(F32K_L2, float, KeysOnly, 2, "f32 keys L2 +double-buffered hist")
AB(F32K_L3, float, KeysOnly, 3, "f32 keys L3 +preset counters")
AB(F32K_L5, float, KeysOnly, 5, "f32 keys L5 +orig-value scatter")
AB(F32K_L6, float, KeysOnly, 6, "f32 keys L6 +unrolled passes")
struct F32K_L6NO : AirAblate<float, KeysOnly, 6, false>
{
  static constexpr const char* name = "f32 keys L6 w/o orig-scatter (ctrl)";
};

AB(U32P_L0, unsigned, int, 0, "u32+i32  L0 faithful air")
AB(U32P_L1, unsigned, int, 1, "u32+i32  L1 +fused scan/choose")
AB(U32P_L2, unsigned, int, 2, "u32+i32  L2 +double-buffered hist")
AB(U32P_L3, unsigned, int, 3, "u32+i32  L3 +preset counters")
AB(U32P_L4, unsigned, int, 4, "u32+i32  L4 +pair scatter")
AB(U32P_L5, unsigned, int, 5, "u32+i32  L5 +orig-value scatter")
AB(U32P_L6, unsigned, int, 6, "u32+i32  L6 +unrolled passes")
AB(U32P_L7, unsigned, int, 7, "u32+i32  L7 +packed state (diag)")

AB(F16P_L0, __half, int, 0, "f16+i32  L0 faithful air (2 passes)")
AB(F16P_L6, __half, int, 6, "f16+i32  L6 all changes")

AB(F64P_L0, double, int, 0, "f64+i32  L0 faithful air (8 passes)")
AB(F64P_L6, double, int, 6, "f64+i32  L6 all changes")

AB(F32LLP_L3, float, long long, 3, "f32+i64  L3 preset (classic exch)")
AB(F32LLP_L4, float, long long, 4, "f32+i64  L4 +pair scatter (16B pair)")
AB(F32LLP_L6, float, long long, 6, "f32+i64  L6 all changes")

// ------------------------------------------------------------------ harness
template <class P>
struct Box
{
  typename P::Smem s;
  typename P::key_t out_v[kK];
  int out_i[kK];
};

template <class P>
__global__ void __launch_bounds__(kBlock) correct_kernel(const typename P::key_t* in, typename P::key_t* out_v, int* out_i)
{
  __shared__ Box<P> box;
  typename P::key_t v[4];
  int idx[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    idx[i] = threadIdx.x * 4 + i;
    v[i]   = in[idx[i]];
  }
  P::run(v, idx, box.s, box.out_v, box.out_i);
  if (threadIdx.x < kK)
  {
    out_v[threadIdx.x] = box.out_v[threadIdx.x];
    out_i[threadIdx.x] = box.out_i[threadIdx.x];
  }
}

constexpr int kLatReps = 24;

__device__ __forceinline__ float chain_link(float base, float out0)
{
  return fmaf(out0, 0.0f, base);
}
__device__ __forceinline__ double chain_link(double base, double out0)
{
  return fma(out0, 0.0, base);
}
__device__ __forceinline__ unsigned chain_link(unsigned base, unsigned out0)
{
  unsigned d1 = out0, d2 = out0;
  asm volatile("" : "+r"(d1));
  asm volatile("" : "+r"(d2));
  return base ^ d1 ^ d2;
}
__device__ __forceinline__ __half chain_link(__half base, __half out0)
{
  unsigned short d1 = __half_as_ushort(out0), d2 = d1;
  asm volatile("" : "+h"(d1));
  asm volatile("" : "+h"(d2));
  return __ushort_as_half((unsigned short) (__half_as_ushort(base) ^ d1 ^ d2));
}

template <class P>
__global__ void __launch_bounds__(kBlock) lat_kernel(const typename P::key_t* in, int chain, long long* out)
{
  using K = typename P::key_t;
  __shared__ Box<P> box;
  K v0[4];
  int idx[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    idx[i] = threadIdx.x * 4 + i;
    v0[i]  = in[idx[i]];
  }
  P::run(v0, idx, box.s, box.out_v, box.out_i);
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int rep = 0; rep < kLatReps; ++rep)
  {
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll 1
    for (int n = 0; n < chain; ++n)
    {
      K v[4];
      v[0] = chain_link(v0[0], box.out_v[0]);
      v[1] = v0[1];
      v[2] = v0[2];
      v[3] = v0[3];
      P::run(v, idx, box.s, box.out_v, box.out_i);
    }
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    *out = best;
  }
  if (box.out_i[0] == -12345)
  {
    g_sink = box.out_i[0];
  }
}

template <class P>
__global__ void __launch_bounds__(kBlock) thr_kernel(const typename P::key_t* in, int outer, typename P::key_t* sink)
{
  using K = typename P::key_t;
  __shared__ Box<P> box;
  K v0[4];
  int idx[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    idx[i] = threadIdx.x * 4 + i;
    v0[i]  = in[idx[i]];
  }
  P::run(v0, idx, box.s, box.out_v, box.out_i);
#pragma unroll 1
  for (int n = 0; n < outer; ++n)
  {
    K v[4];
    v[0] = chain_link(v0[0], box.out_v[0]);
    v[1] = v0[1];
    v[2] = v0[2];
    v[3] = v0[3];
    P::run(v, idx, box.s, box.out_v, box.out_i);
  }
  if (threadIdx.x < kK)
  {
    sink[blockIdx.x * kK + threadIdx.x] = box.out_v[threadIdx.x];
  }
}

// ------------------------------------------------------------------ host: patterns / validation
struct Lcg
{
  unsigned s;
  explicit Lcg(unsigned seed)
      : s(seed)
  {}
  unsigned next()
  {
    s = 1664525u * s + 1013904223u;
    return s;
  }
  float uniform()
  {
    return (next() >> 8) * (1.0f / 16777216.0f);
  }
};

static std::vector<float> gen_f(const std::string& p, unsigned seed)
{
  std::vector<float> v(kN);
  Lcg rng(seed * 2654435761u + 12345u);
  auto normal = [&]() {
    float u1 = std::max(rng.uniform(), 1e-7f);
    float u2 = rng.uniform();
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.28318530718f * u2);
  };
  if (p == "random")
  {
    for (auto& x : v)
    {
      x = normal();
    }
  }
  else if (p == "tie_heavy")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = (float) (j % 64) / 64.f;
    }
  }
  else if (p == "pivot_tie40")
  {
    for (auto& x : v)
    {
      x = 1.f;
    }
    int placed = 0;
    while (placed < 40)
    {
      unsigned pos = rng.next() % kN;
      if (v[pos] != 2.f)
      {
        v[pos] = 2.f;
        ++placed;
      }
    }
  }
  else if (p == "sorted_asc")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = (float) j / kN;
    }
  }
  else if (p == "all_equal")
  {
    for (auto& x : v)
    {
      x = 1.f;
    }
  }
  else if (p == "neg_zero")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = -1.f - (float) j;
    }
    for (int j = 0; j < 8; ++j)
    {
      v[100 + j] = -0.0f;
      v[200 + j] = +0.0f;
    }
  }
  else
  {
    printf("unknown pattern %s\n", p.c_str());
    exit(1);
  }
  return v;
}

static unsigned h_twiddle(float f)
{
  unsigned u;
  std::memcpy(&u, &f, 4);
  return u ^ (((unsigned) ((int) u >> 31)) | 0x80000000u);
}

template <typename K>
static std::vector<K> gen_pattern(const std::string& p, unsigned seed);
template <>
std::vector<float> gen_pattern<float>(const std::string& p, unsigned seed)
{
  return gen_f(p, seed);
}
template <>
std::vector<unsigned> gen_pattern<unsigned>(const std::string& p, unsigned seed)
{
  auto f = gen_f(p == "neg_zero" ? "random" : p, seed);
  std::vector<unsigned> v(kN);
  for (int i = 0; i < kN; ++i)
  {
    v[i] = h_twiddle(f[i]);
  }
  return v;
}
template <>
std::vector<__half> gen_pattern<__half>(const std::string& p, unsigned seed)
{
  auto f = gen_f(p, seed);
  std::vector<__half> v(kN);
  for (int i = 0; i < kN; ++i)
  {
    v[i] = __float2half(f[i]);
  }
  return v;
}
template <>
std::vector<double> gen_pattern<double>(const std::string& p, unsigned seed)
{
  auto f = gen_f(p, seed);
  std::vector<double> v(kN);
  for (int i = 0; i < kN; ++i)
  {
    v[i] = (double) f[i];
  }
  return v;
}

static double to_cmp(float x)
{
  return (double) x;
}
static double to_cmp(unsigned x)
{
  return (double) x;
}
static double to_cmp(double x)
{
  return x;
}
static double to_cmp(__half x)
{
  return (double) __half2float(x);
}

template <typename K>
static bool validate(const std::vector<K>& in, const K* ov, const int* oi, bool keys_only, std::string& why)
{
  std::vector<double> ref(kN);
  for (int i = 0; i < kN; ++i)
  {
    ref[i] = to_cmp(in[i]);
  }
  std::vector<double> sorted = ref;
  std::sort(sorted.begin(), sorted.end(), [](double a, double b) {
    return a > b;
  });
  std::vector<double> want(sorted.begin(), sorted.begin() + kK);
  std::vector<double> got(kK);
  for (int i = 0; i < kK; ++i)
  {
    got[i] = to_cmp(ov[i]);
  }
  std::sort(want.begin(), want.end());
  std::sort(got.begin(), got.end());
  for (int i = 0; i < kK; ++i)
  {
    if (!(want[i] == got[i]))
    {
      char b[128];
      snprintf(b, sizeof b, "value multiset mismatch at %d: want %g got %g", i, want[i], got[i]);
      why = b;
      return false;
    }
  }
  if (keys_only)
  {
    return true;
  }
  bool used[kN] = {};
  for (int i = 0; i < kK; ++i)
  {
    if (oi[i] < 0 || oi[i] >= kN)
    {
      why = "index out of range";
      return false;
    }
    if (used[oi[i]])
    {
      why = "duplicate index " + std::to_string(oi[i]);
      return false;
    }
    used[oi[i]] = true;
    if (!(ref[oi[i]] == to_cmp(ov[i])))
    {
      why = "index/value mismatch at " + std::to_string(i);
      return false;
    }
  }
  return true;
}

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

template <class P>
void run_correct()
{
  using K = typename P::key_t;
  std::vector<std::string> pats = {"random", "tie_heavy", "pivot_tie40", "sorted_asc", "all_equal"};
  if (AKey<K>::is_fp)
  {
    pats.push_back("neg_zero");
  }
  K *d_in, *d_ov;
  int* d_oi;
  CHECK(cudaMalloc(&d_in, kN * sizeof(K)));
  CHECK(cudaMalloc(&d_ov, kK * sizeof(K)));
  CHECK(cudaMalloc(&d_oi, kK * sizeof(int)));
  int fails = 0, runs = 0;
  for (const auto& pat : pats)
  {
    const int seeds = (pat == "random") ? 6 : 2;
    for (int seed = 0; seed < seeds; ++seed)
    {
      auto in = gen_pattern<K>(pat, seed);
      CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(K), cudaMemcpyHostToDevice));
      correct_kernel<P><<<1, kBlock>>>(d_in, d_ov, d_oi);
      CHECK(cudaDeviceSynchronize());
      K ov[kK];
      int oi[kK];
      CHECK(cudaMemcpy(ov, d_ov, sizeof ov, cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(oi, d_oi, sizeof oi, cudaMemcpyDeviceToHost));
      std::string why;
      ++runs;
      if (!validate(in, ov, oi, P::keys_only, why))
      {
        ++fails;
        printf("    FAIL %-16s seed %d: %s\n", pat.c_str(), seed, why.c_str());
      }
    }
  }
  printf("  %-40s %s (%d runs)\n", P::name, fails ? "FAIL" : "PASS", runs);
  cudaFree(d_in);
  cudaFree(d_ov);
  cudaFree(d_oi);
}

template <class P>
void run_lat()
{
  using K            = typename P::key_t;
  const char* pats[] = {"random", "tie_heavy", "pivot_tie40", "sorted_asc"};
  K* d_in;
  long long* d_out;
  CHECK(cudaMalloc(&d_in, kN * sizeof(K)));
  CHECK(cudaMalloc(&d_out, sizeof(long long)));
  printf("  %-40s", P::name);
  for (const char* pat : pats)
  {
    auto in = gen_pattern<K>(pat, 0);
    CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(K), cudaMemcpyHostToDevice));
    const int chains[] = {1, 2, 4, 8, 16};
    double x[5], y[5];
    for (int i = 0; i < 5; ++i)
    {
      lat_kernel<P><<<1, kBlock>>>(d_in, chains[i], d_out);
      CHECK(cudaDeviceSynchronize());
      long long c;
      CHECK(cudaMemcpy(&c, d_out, sizeof c, cudaMemcpyDeviceToHost));
      x[i] = chains[i];
      y[i] = (double) c;
    }
    double a, b;
    fit(x, y, 5, a, b);
    printf("  %s=%6.0f", pat, b);
  }
  printf("   cyc/call (slope)\n");
  cudaFree(d_in);
  cudaFree(d_out);
}

template <class P>
void run_thr()
{
  using K            = typename P::key_t;
  constexpr int grid = 2048, outer = 32;
  K *d_in, *d_sink;
  CHECK(cudaMalloc(&d_in, kN * sizeof(K)));
  CHECK(cudaMalloc(&d_sink, (size_t) grid * kK * sizeof(K)));
  auto in = gen_pattern<K>("random", 0);
  CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(K), cudaMemcpyHostToDevice));
  thr_kernel<P><<<grid, kBlock>>>(d_in, outer, d_sink);
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best_ms = 1e30f;
  for (int rep = 0; rep < 5; ++rep)
  {
    cudaEventRecord(e0);
    thr_kernel<P><<<grid, kBlock>>>(d_in, outer, d_sink);
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best_ms = std::min(best_ms, ms);
  }
  const double calls  = (double) grid * (outer + 1);
  const double gelems = calls * kN / (best_ms * 1e-3) / 1e9;
  printf("  %-40s %8.1f G elem/s   (%.3f ms)\n", P::name, gelems, best_ms);
  cudaFree(d_in);
  cudaFree(d_sink);
}

template <class P>
void run_res()
{
  cudaFuncAttributes a;
  CHECK(cudaFuncGetAttributes(&a, (const void*) thr_kernel<P>));
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*) thr_kernel<P>, kBlock, 0);
  printf("  %-40s regs=%3d  smem=%5zu B  local(spill)=%4zu B  maxblk/SM=%d\n",
         P::name,
         a.numRegs,
         a.sharedSizeBytes,
         a.localSizeBytes,
         occ);
}

#define FOREACH_PROTO(X) \
  X(F32P_L0)             \
  X(F32P_L1)             \
  X(F32P_L2)             \
  X(F32P_L3)             \
  X(F32P_L4)             \
  X(F32P_L5)             \
  X(F32P_L6)             \
  X(F32P_L7)             \
  X(F32K_L0)             \
  X(F32K_L1)             \
  X(F32K_L2)             \
  X(F32K_L3)             \
  X(F32K_L5)             \
  X(F32K_L6)             \
  X(F32K_L6NO)           \
  X(U32P_L0)             \
  X(U32P_L1)             \
  X(U32P_L2)             \
  X(U32P_L3)             \
  X(U32P_L4)             \
  X(U32P_L5)             \
  X(U32P_L6)             \
  X(U32P_L7)             \
  X(F16P_L0)             \
  X(F16P_L6)             \
  X(F64P_L0)             \
  X(F64P_L6)             \
  X(F32LLP_L3)           \
  X(F32LLP_L4)           \
  X(F32LLP_L6)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness ===\n");
#define RUNC(P) run_correct<P>();
    FOREACH_PROTO(RUNC)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== latency: slope cyc/call ===\n");
#define RUNL(P) run_lat<P>();
    FOREACH_PROTO(RUNL)
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== throughput (random input) ===\n");
#define RUNT(P) run_thr<P>();
    FOREACH_PROTO(RUNT)
  }
  if (mode == "res" || mode == "all")
  {
    printf("\n=== resources ===\n");
#define RUNR(P) run_res<P>();
    FOREACH_PROTO(RUNR)
  }
  return 0;
}
