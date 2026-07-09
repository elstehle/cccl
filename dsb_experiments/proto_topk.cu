// Prototype suite: latency-optimized block-wide top-K at (BLOCK=256, N=1024, K=16, float, sm_100).
// See BLOCK_TOPK_LATENCY_SPEC.md. Measurement methodology per DEVICE_SIDE_BENCHMARKING_ISSUE.md:
// chain-length slope latency (data-dependent chaining: one input element derived from the previous
// call's output), single-block __syncthreads bracketing, min-of-reps, cycles via clock64.
//
// Prototypes (genuinely different strategies):
//   air          : cub::detail::block_topk_air radix-select (incumbent baseline)
//   bitonic_hier : per-warp WarpBitonicTopK(128->16), 1 barrier, warp0 WarpBitonicTopK(128->16)
//   redux_iter   : pure shuffle+smem block-wide iterative extraction on packed u64 (approach A)
//   atomic_iter  : <=16 rounds of fire-and-forget smem atomicMax(u32 twiddled value) + 1 barrier
//                  each; rounds carry values only (shift-down removal, all register indices static);
//                  in-loop count tracking + break; 2-tier (strict/boundary) value scatter
//   atomic_adaptive: the recommended float design. Fully branchless rounds (no in-loop bookkeeping),
//                  early flood check after round 2 (count vs slot[1]) diverting tie floods to a
//                  short 2-slot resolution; common case: fast-path scatter when exactly K items
//                  sit on/above slot[15]; otherwise post-hoc M* histogram + single-pass 2-tier
//   hybrid       : atomic_iter with redux.sync warp pre-reduction before the atomic (spec §7)
//   hier_extract : barrier-free per-warp redux iterative extraction (16 rounds) -> 1 barrier ->
//                  warp0 redux iterative extraction over the 8x16 candidates (approach E)
//   hist_narrow  : latency-tuned radix narrowing: 256-bin smem atomic histogram over twiddled bits,
//                  single-warp suffix scan to locate the k-th bucket, early exit, 2-tier scatter
//   atomic_iter_h: __half secondary dtype; exact (value,index) packed into u32 -> every round is a
//                  full argmax; single-winner extraction (no tie machinery needed)
//
// Modes: ./proto_topk [correct|lat|thr|res|all]
//
// Correctness (§8): set semantics. multiset(out values) == multiset(ref top-16); indices distinct;
// in[idx] == value. Patterns: random(8 seeds), quantized_random, relu_quantized, tie_heavy,
// pivot_tie(4 and 40 tied 2.0s at random positions), all_equal, sorted_asc.

#include <cub/block/specializations/block_topk_air.cuh>
#include <cub/warp/warp_bitonic_topk.cuh>

#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

constexpr unsigned kFull = 0xffffffffu;
constexpr int kBlock     = 256;
constexpr int kIpt       = 4;
constexpr int kN         = 1024;
constexpr int kK         = 16;

__device__ int g_sink;

// ------------------------------------------------------------------ utilities
__device__ __forceinline__ unsigned twiddle(float f)
{
  unsigned u = __float_as_uint(f);
  return u ^ (((unsigned) ((int) u >> 31)) | 0x80000000u);
}

__device__ __forceinline__ float untwiddle(unsigned t)
{
  unsigned m = (~((unsigned) ((int) t >> 31))) | 0x80000000u;
  return __uint_as_float(t ^ m);
}

__device__ __forceinline__ unsigned twiddle16(__half h)
{
  unsigned short u = __half_as_ushort(h);
  return (unsigned) (u ^ ((u >> 15) ? 0xffffu : 0x8000u)) & 0xffffu;
}

__device__ __forceinline__ __half untwiddle16(unsigned t)
{
  unsigned short u = (unsigned short) (t & 0xffffu);
  u                = u ^ ((u >> 15) ? 0x8000u : 0xffffu);
  return __ushort_as_half(u);
}

// values-only descending sorting network (static indices only)
template <typename T>
__device__ __forceinline__ void cswap_desc(T& a, T& b)
{
  T lo = ::min(a, b);
  a    = ::max(a, b);
  b    = lo;
}

template <typename T>
__device__ __forceinline__ void sort4_desc(T (&v)[4])
{
  cswap_desc(v[0], v[1]);
  cswap_desc(v[2], v[3]);
  cswap_desc(v[0], v[2]);
  cswap_desc(v[1], v[3]);
  cswap_desc(v[1], v[2]);
}

// (value,index) descending sorting network
template <typename T>
__device__ __forceinline__ void cswap_desc_kv(T& a, T& b, int& ia, int& ib)
{
  if (a < b)
  {
    T t = a;
    a   = b;
    b   = t;
    int ti = ia;
    ia     = ib;
    ib     = ti;
  }
}

template <typename T>
__device__ __forceinline__ void sort4_desc_kv(T (&v)[4], int (&x)[4])
{
  cswap_desc_kv(v[0], v[1], x[0], x[1]);
  cswap_desc_kv(v[2], v[3], x[2], x[3]);
  cswap_desc_kv(v[0], v[2], x[0], x[2]);
  cswap_desc_kv(v[1], v[3], x[1], x[3]);
  cswap_desc_kv(v[1], v[2], x[1], x[2]);
}

// branchless shift-down removal: drop the front element, keep the list sorted (static indices)
template <typename T>
__device__ __forceinline__ void shift1(T (&s)[4])
{
  s[0] = s[1];
  s[1] = s[2];
  s[2] = s[3];
  s[3] = T(0);
}

template <typename T>
__device__ __forceinline__ void shift1_kv(T (&s)[4], int (&x)[4])
{
  s[0] = s[1];
  s[1] = s[2];
  s[2] = s[3];
  s[3] = T(0);
  x[0] = x[1];
  x[1] = x[2];
  x[2] = x[3];
  x[3] = -1;
}

__device__ __forceinline__ unsigned long long warp_max_u64(unsigned long long x)
{
#pragma unroll
  for (int d = 16; d; d >>= 1)
  {
    x = ::max(x, __shfl_xor_sync(kFull, x, d));
  }
  return x;
}

struct Greater
{
  template <typename T>
  __device__ bool operator()(const T& a, const T& b) const
  {
    return a > b;
  }
};

// ------------------------------------------------------------------ prototypes
// Uniform contract: run(v, idx, smem, out_v, out_i). All 256 threads call it; v/idx are the
// caller's 4 blocked elements. Results land in out_v[16]/out_i[16] (shared). run() ends with
// __syncthreads() so shared state may be reused by the next call.

struct ProtoAir
{
  static constexpr const char* name = "air (radix-select baseline)";
  static constexpr bool is_half     = false;
  using topk_t                      = cub::detail::block_topk_air<float, kBlock, kIpt, int>;
  struct Smem
  {
    typename topk_t::TempStorage ts;
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    float k[4];
    int val[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      k[i]   = v[i];
      val[i] = idx[i];
    }
    topk_t(sh.ts).template select_pairs<cub::detail::topk::select::max, true>(k, val, kK, kN);
    if (threadIdx.x < kK / kIpt)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        out_v[threadIdx.x * 4 + i] = k[i];
        out_i[threadIdx.x * 4 + i] = val[i];
      }
    }
    __syncthreads();
  }
};

struct ProtoBitonicHier
{
  static constexpr const char* name = "bitonic_hier (warp bitonic topk x2)";
  static constexpr bool is_half     = false;
  using wtopk                       = cub::detail::WarpBitonicTopK<32, float, int>;
  struct Smem
  {
    float cv[128];
    int ci[128];
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float k[4];
    int val[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      k[i]   = v[i];
      val[i] = idx[i];
    }
    wtopk{}.TopK(k, val, Greater{}, kK);
    // top-16 sits in the first 16 striped positions: item 0 of lanes 0..15
    if (lane < kK)
    {
      sh.cv[warp * kK + lane] = k[0];
      sh.ci[warp * kK + lane] = val[0];
    }
    __syncthreads();
    if (warp == 0)
    {
      float c[4];
      int ci2[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        c[i]   = sh.cv[lane + 32 * i];
        ci2[i] = sh.ci[lane + 32 * i];
      }
      wtopk{}.TopK(c, ci2, Greater{}, kK);
      if (lane < kK)
      {
        out_v[lane] = c[0];
        out_i[lane] = ci2[0];
      }
    }
    __syncthreads();
  }
};

struct ProtoReduxIter
{
  static constexpr const char* name = "redux_iter (shuffle+smem, packed u64)";
  static constexpr bool is_half     = false;
  struct Smem
  {
    unsigned long long part[2][8];
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned long long p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      p[i] = ((unsigned long long) twiddle(v[i]) << 32) | (unsigned) idx[i];
    }
    sort4_desc(p);
#pragma unroll
    for (int r = 0; r < kK; ++r)
    {
      unsigned long long w = warp_max_u64(p[0]);
      if (lane == 0)
      {
        sh.part[r & 1][warp] = w;
      }
      __syncthreads();
      unsigned long long y = sh.part[r & 1][threadIdx.x & 7];
#pragma unroll
      for (int d = 4; d; d >>= 1)
      {
        y = ::max(y, __shfl_xor_sync(kFull, y, d));
      }
      if (p[0] == y) // packed word is unique (index in low bits) => unique owner
      {
        out_v[r] = untwiddle((unsigned) (y >> 32));
        out_i[r] = (int) (y & 0xffffffffu);
        shift1(p);
      }
    }
    __syncthreads();
  }
};

// Shared implementation of the atomic iterative extraction; Hybrid toggles the warp pre-reduction.
// Rounds carry values only. Removal is a branchless shift-down. Extraction membership is purely
// value-based (an item is extracted iff value > M*, boundary-tied iff value == M*), so no marking
// state is needed in the loop; indices are resolved once at the end from the original registers.
template <bool UseRedux>
struct AtomicIterImpl
{
  struct Smem
  {
    unsigned slot[kK];
    unsigned cnt, cntA, cntB;
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    // reset scratch
    if (threadIdx.x < kK)
    {
      sh.slot[threadIdx.x] = 0;
    }
    if (threadIdx.x == kBlock - 1)
    {
      sh.cnt  = 0;
      sh.cntA = 0;
      sh.cntB = 0;
    }
    unsigned tw[4]; // original order, for the final scatter
    unsigned s[4]; // working copy, sorted descending, consumed by shift-down
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      tw[i] = twiddle(v[i]);
      s[i]  = tw[i];
    }
    sort4_desc(s);
    __syncthreads();

    unsigned M = 0, Mprev = 0;
#pragma unroll
    for (int r = 0; r < kK; ++r)
    {
      if constexpr (UseRedux)
      {
        unsigned wmax = __reduce_max_sync(kFull, s[0]);
        if ((threadIdx.x & 31) == 0)
        {
          atomicMax(&sh.slot[r], wmax);
        }
      }
      else
      {
        atomicMax(&sh.slot[r], s[0]);
      }
      __syncthreads();
      Mprev            = M;
      M                = sh.slot[r];
      const unsigned c = sh.cnt; // extractions from rounds < r (adds completed pre-barrier)
      if (c >= kK)
      {
        M = Mprev; // crossing happened at round r-1; M* is that round's max
        break;
      }
      // branchless multiplicity extraction: shift past ALL of my items equal to the round max
      unsigned nm = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j)
      {
        const bool w = (s[0] == M) && (M != 0);
        nm += w;
        s[0] = w ? s[1] : s[0];
        s[1] = w ? s[2] : s[1];
        s[2] = w ? s[3] : s[2];
        s[3] = w ? 0u : s[3];
      }
      atomicAdd(&sh.cnt, nm); // unconditional fire-and-forget (a divergent branch costs more)
    }
    const unsigned Mstar = M;
    // tier A: items strictly above the boundary value (guaranteed top-k, count < 16)
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (tw[i] > Mstar)
      {
        unsigned t = atomicAdd(&sh.cntA, 1u);
        out_v[t]   = v[i];
        out_i[t]   = idx[i];
      }
    }
    __syncthreads();
    const unsigned nA = sh.cntA;
    // tier B: boundary ties fill the remaining slots from the back; surplus ties are dropped
    // (any subset of boundary ties is a valid top-16 under the relaxed semantics)
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (tw[i] == Mstar)
      {
        unsigned t = atomicAdd(&sh.cntB, 1u);
        int pos    = kK - 1 - (int) t;
        if (pos >= (int) nA)
        {
          out_v[pos] = v[i];
          out_i[pos] = idx[i];
        }
      }
    }
    __syncthreads();
  }
};

struct ProtoAtomicIter : AtomicIterImpl<false>
{
  static constexpr const char* name = "atomic_iter (smem atomicMax rounds)";
  static constexpr bool is_half     = false;
};

struct ProtoHybrid : AtomicIterImpl<true>
{
  static constexpr const char* name = "hybrid (redux + atomicMax rounds)";
  static constexpr bool is_half     = false;
};

// atomic_iter without any in-loop bookkeeping: rounds are half-style branchless (post + barrier +
// predicated shift), always 16 of them. Afterwards, one counting pass over slot[15] decides:
//   fast path (no boundary multiplicity, the overwhelmingly common case): the 16 items >= slot[15]
//     ARE the top-16 -> single-ticket scatter, no histogram, no tiers.
//   flood path: reconstruct the boundary value M* post-hoc: items >= slot[15] vote into a tiny
//     per-round histogram; a 16-lane prefix scan finds where the cumulative count crosses K, which
//     also yields tier A's exact size, so both tiers scatter in a single pass.
struct ProtoAtomicAdaptive
{
  static constexpr const char* name = "atomic_adaptive (branchless rounds)";
  static constexpr bool is_half     = false;
  struct Smem
  {
    unsigned slot[kK];
    unsigned rh[kK];
    unsigned cnt, cnt2;
    unsigned mstar, nAstar;
    unsigned cntA, cntB;
  };

  // Resolve the output once a count c of items >= slX (= slot[R-1]) is known, c >= K.
  //   c == K: those items ARE the top-16 -> single-ticket scatter.
  //   c >  K: post-hoc boundary: participating items vote into a per-round histogram over the
  //           first R slots; a prefix scan finds where the cumulative count crosses K, yielding
  //           M* and tier A's exact size, so both tiers scatter in one pass.
  template <int R>
  __device__ __forceinline__ static void resolve(
    const float (&v)[4],
    const int (&idx)[4],
    const unsigned (&tw)[4],
    Smem& sh,
    unsigned slX,
    unsigned c,
    unsigned myc,
    float* out_v,
    int* out_i)
  {
    if (c == kK)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if (tw[i] >= slX)
        {
          unsigned t = atomicAdd(&sh.cntA, 1u);
          out_v[t]   = v[i];
          out_i[t]   = idx[i];
        }
      }
      return;
    }
    if (myc) // only items >= slX participate (warp-uniform skip elsewhere)
    {
      unsigned sl[R];
#pragma unroll
      for (int j = 0; j < R; ++j)
      {
        sl[j] = sh.slot[j];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if (tw[i] >= slX)
        {
          unsigned jj = 0;
          bool eq     = false;
#pragma unroll
          for (int j = 0; j < R; ++j)
          {
            jj += (sl[j] > tw[i]);
            eq |= (sl[j] == tw[i]) && (sl[j] != 0);
          }
          if (eq)
          {
            atomicAdd(&sh.rh[jj], 1u);
          }
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
      const int lane = threadIdx.x;
      unsigned r     = (lane < R) ? sh.rh[lane] : 0;
      unsigned cc    = r; // inclusive prefix over rounds = #items >= that round's value
#pragma unroll
      for (int d = 1; d < kK; d <<= 1)
      {
        unsigned o = __shfl_up_sync(kFull, cc, d);
        if (lane >= d)
        {
          cc += o;
        }
      }
      if (lane < R && cc >= kK && (cc - r) < kK) // first crossing of K
      {
        sh.mstar  = sh.slot[lane];
        sh.nAstar = cc - r; // #items strictly above M*
      }
    }
    __syncthreads();
    const unsigned Mstar = sh.mstar;
    const unsigned nA    = sh.nAstar;
    // single scatter pass: tier A packs 0..nA-1; boundary ties fill nA..15, surplus dropped
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (tw[i] > Mstar)
      {
        unsigned t = atomicAdd(&sh.cntA, 1u);
        out_v[t]   = v[i];
        out_i[t]   = idx[i];
      }
      else if (tw[i] == Mstar)
      {
        unsigned t   = atomicAdd(&sh.cntB, 1u);
        unsigned pos = nA + t;
        if (pos < kK)
        {
          out_v[pos] = v[i];
          out_i[pos] = idx[i];
        }
      }
    }
  }

  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    if (threadIdx.x < kK)
    {
      sh.slot[threadIdx.x] = 0;
      sh.rh[threadIdx.x]   = 0;
    }
    if (threadIdx.x == kBlock - 1)
    {
      sh.cnt  = 0;
      sh.cnt2 = 0;
      sh.cntA = 0;
      sh.cntB = 0;
    }
    unsigned tw[4];
    unsigned s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      tw[i] = twiddle(v[i]);
      s[i]  = tw[i];
    }
    sort4_desc(s);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r)
    {
      atomicMax(&sh.slot[r], s[0]);
      __syncthreads();
      const unsigned M = sh.slot[r];
      const bool w     = (s[0] == M);
      s[0]             = w ? s[1] : s[0];
      s[1]             = w ? s[2] : s[1];
      s[2]             = w ? s[3] : s[2];
      s[3]             = w ? 0u : s[3];
    }
    // early flood check: if >= K items sit on/above slot[1], the boundary is already known
    const unsigned sl1 = sh.slot[1];
    unsigned myc2      = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      myc2 += (tw[i] >= sl1);
    }
    atomicAdd(&sh.cnt2, myc2);
    __syncthreads();
    const unsigned c2 = sh.cnt2;
    if (c2 >= kK)
    {
      resolve<2>(v, idx, tw, sh, sl1, c2, myc2, out_v, out_i);
      __syncthreads();
      return;
    }
#pragma unroll
    for (int r = 2; r < kK; ++r)
    {
      atomicMax(&sh.slot[r], s[0]);
      __syncthreads();
      const unsigned M = sh.slot[r];
      const bool w     = (s[0] == M); // shifting zeros on depleted flood rounds is harmless
      s[0]             = w ? s[1] : s[0];
      s[1]             = w ? s[2] : s[1];
      s[2]             = w ? s[3] : s[2];
      s[3]             = w ? 0u : s[3];
    }
    // count items on/above the 16th extracted value
    const unsigned sl15 = sh.slot[kK - 1];
    unsigned myc        = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      myc += (tw[i] >= sl15);
    }
    atomicAdd(&sh.cnt, myc);
    __syncthreads();
    resolve<kK>(v, idx, tw, sh, sl15, sh.cnt, myc, out_v, out_i);
    __syncthreads();
  }
};

struct ProtoHierExtract
{
  static constexpr const char* name = "hier_extract (warp redux x16 -> merge)";
  static constexpr bool is_half     = false;
  struct Smem
  {
    unsigned cv[128];
    int ci[128];
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned s[4];
    int ix[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      s[i]  = twiddle(v[i]);
      ix[i] = idx[i];
    }
    sort4_desc_kv(s, ix);
    // phase 1: each warp extracts its top-16 of 128, barrier-free
#pragma unroll
    for (int r = 0; r < kK; ++r)
    {
      unsigned M    = __reduce_max_sync(kFull, s[0]);
      unsigned ball = __ballot_sync(kFull, s[0] == M);
      if ((int) (__ffs(ball) - 1) == lane)
      {
        sh.cv[warp * kK + r] = M;
        sh.ci[warp * kK + r] = ix[0];
        shift1_kv(s, ix);
      }
    }
    __syncthreads();
    // phase 2: warp 0 extracts the top-16 of the 128 candidates
    if (warp == 0)
    {
      unsigned c[4];
      int ci2[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        c[i]   = sh.cv[lane + 32 * i];
        ci2[i] = sh.ci[lane + 32 * i];
      }
      sort4_desc_kv(c, ci2);
#pragma unroll
      for (int r = 0; r < kK; ++r)
      {
        unsigned M    = __reduce_max_sync(kFull, c[0]);
        unsigned ball = __ballot_sync(kFull, c[0] == M);
        if ((int) (__ffs(ball) - 1) == lane)
        {
          out_v[r] = untwiddle(M);
          out_i[r] = ci2[0];
          shift1_kv(c, ci2);
        }
      }
    }
    __syncthreads();
  }
};

struct ProtoHistNarrow
{
  static constexpr const char* name = "hist_narrow (radix narrow, 1-warp scan)";
  static constexpr bool is_half     = false;
  // Histogram bins live at stride-9 slots (bin d -> word 9*(d/8) + d%8) so the one-warp scan
  // (lane l reads words 9l..9l+7) is bank-conflict-free. Two buffers: while pass p histograms
  // into buf[p&1], threads reset buf[(p+1)&1] -- consumed a pass ago -- saving a barrier.
  struct Smem
  {
    unsigned hist[2][288];
    unsigned state_bucket, state_above, state_cnt;
    unsigned cntA, cntB;
  };
  __device__ __forceinline__ static unsigned slot_of(unsigned d)
  {
    return d + (d >> 3);
  }
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    unsigned tw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      tw[i] = twiddle(v[i]);
    }
#pragma unroll
    for (int i = threadIdx.x; i < 288; i += kBlock)
    {
      sh.hist[0][i] = 0;
      sh.hist[1][i] = 0;
    }
    if (threadIdx.x == 0)
    {
      sh.cntA = 0;
      sh.cntB = 0;
    }
    __syncthreads();

    unsigned prefix = 0, pmask = 0;
    unsigned krem = kK;
#pragma unroll 1
    for (int pass = 0; pass < 4; ++pass)
    {
      const int shift    = 24 - 8 * pass;
      unsigned (&cur)[288]  = sh.hist[pass & 1];
      unsigned (&next)[288] = sh.hist[(pass + 1) & 1];
      // histogram over surviving candidates only (fire-and-forget adds)
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if ((tw[i] & pmask) == prefix)
        {
          atomicAdd(&cur[slot_of((tw[i] >> shift) & 0xffu)], 1u);
        }
      }
      if (pass > 0)
      {
        // reset the other buffer for pass+1; it was last read before the previous state barrier
#pragma unroll
        for (int i = threadIdx.x; i < 288; i += kBlock)
        {
          next[i] = 0;
        }
      }
      __syncthreads();
      // one warp: suffix counts from the top digit downwards, locate the k-th bucket
      if (threadIdx.x < 32)
      {
        const int lane = threadIdx.x;
        unsigned h[8];
        unsigned s = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
          h[j] = cur[9 * lane + j];
          s += h[j];
        }
        unsigned S = s; // suffix-inclusive across lanes: S_l = sum_{m >= l} s_m
#pragma unroll
        for (int d = 1; d < 32; d <<= 1)
        {
          unsigned o = __shfl_down_sync(kFull, S, d);
          if (lane + d < 32)
          {
            S += o;
          }
        }
        unsigned Snext = __shfl_down_sync(kFull, S, 1);
        if (lane == 31)
        {
          Snext = 0;
        }
        unsigned cum = Snext;
#pragma unroll
        for (int j = 7; j >= 0; --j)
        {
          const unsigned cprev = cum; // count of items in digits > lane*8+j
          cum += h[j];
          if (cum >= krem && cprev < krem)
          {
            sh.state_bucket = lane * 8 + j;
            sh.state_above  = cprev;
            sh.state_cnt    = h[j];
          }
        }
      }
      __syncthreads();
      const unsigned bucket = sh.state_bucket;
      const unsigned above  = sh.state_above;
      const unsigned bcnt   = sh.state_cnt;
      prefix |= bucket << shift;
      pmask |= 0xffu << shift;
      krem -= above;
      if (bcnt == krem)
      {
        break; // every remaining candidate is selected
      }
    }
    // scatter: tier A = strictly above the k-th prefix; tier B = ties on the prefix
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if ((tw[i] & pmask) > prefix)
      {
        unsigned t = atomicAdd(&sh.cntA, 1u);
        out_v[t]   = v[i];
        out_i[t]   = idx[i];
      }
    }
    __syncthreads();
    const unsigned nA = sh.cntA;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if ((tw[i] & pmask) == prefix)
      {
        unsigned t = atomicAdd(&sh.cntB, 1u);
        int pos    = kK - 1 - (int) t;
        if (pos >= (int) nA)
        {
          out_v[pos] = v[i];
          out_i[pos] = idx[i];
        }
      }
    }
    __syncthreads();
  }
};

struct ProtoAtomicIterHalf
{
  static constexpr const char* name = "atomic_iter_h (__half, packed u32)";
  static constexpr bool is_half     = true;
  struct Smem
  {
    unsigned slot[kK];
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    if (threadIdx.x < kK)
    {
      sh.slot[threadIdx.x] = 0;
    }
    unsigned p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      p[i] = (twiddle16(__float2half(v[i])) << 16) | (unsigned) idx[i]; // index fits in 10 bits
    }
    sort4_desc(p);
    __syncthreads();
    // Rounds are fully branchless: the packed word is unique, so the winner just shifts
    // (predicated moves); slot[r] itself IS the r-th output. No in-loop output writes.
#pragma unroll
    for (int r = 0; r < kK; ++r)
    {
      atomicMax(&sh.slot[r], p[0]);
      __syncthreads();
      const unsigned M = sh.slot[r];
      const bool w     = (p[0] == M);
      p[0]             = w ? p[1] : p[0];
      p[1]             = w ? p[2] : p[1];
      p[2]             = w ? p[3] : p[2];
      p[3]             = w ? 0u : p[3];
    }
    // unpack the slots (descending top-16) once at the end
    if (threadIdx.x < kK)
    {
      const unsigned M     = sh.slot[threadIdx.x];
      out_v[threadIdx.x]   = __half2float(untwiddle16(M >> 16));
      out_i[threadIdx.x]   = (int) (M & 0xffffu);
    }
    __syncthreads();
  }
};

// Diagnostics (not top-k implementations; excluded from correctness): isolate the cost of the
// bare round skeletons inside this harness, to attribute the algorithm cost on top of them.
struct ProtoDiagBlockRounds
{
  static constexpr const char* name = "DIAG 16x block atomic rounds (skeleton)";
  static constexpr bool is_half     = false;
  static constexpr bool is_diag     = true;
  struct Smem
  {
    unsigned slot[kK];
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    if (threadIdx.x < kK)
    {
      sh.slot[threadIdx.x] = 0;
    }
    __syncthreads();
    unsigned x = twiddle(v[0]);
#pragma unroll
    for (int r = 0; r < kK; ++r)
    {
      atomicMax(&sh.slot[r], x);
      __syncthreads();
      x = sh.slot[r] + threadIdx.x;
    }
    if (threadIdx.x == 0)
    {
      out_v[0] = (float) (x & 0xffffu);
      out_i[0] = 0;
    }
    __syncthreads();
  }
};

struct ProtoDiagWarpRounds
{
  static constexpr const char* name = "DIAG 2x16 warp redux rounds (skeleton)";
  static constexpr bool is_half     = false;
  static constexpr bool is_diag     = true;
  struct Smem
  {
    int dummy;
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    unsigned x = twiddle(v[0]);
#pragma unroll
    for (int r = 0; r < 2 * kK; ++r)
    {
      unsigned M    = __reduce_max_sync(kFull, x);
      unsigned ball = __ballot_sync(kFull, x == M);
      x += __ffs(ball) + (x == M ? 1u : 2u);
    }
    if (threadIdx.x == 0)
    {
      out_v[0] = (float) (x & 0xffffu);
      out_i[0] = 0;
    }
    __syncthreads();
  }
};

// nb stripped to prologue + 16 branchless rounds (slots dumped as output; no post-hoc, no tier)
struct ProtoDiagNbRounds
{
  static constexpr const char* name = "DIAG nb rounds only";
  static constexpr bool is_half     = false;
  struct Smem
  {
    unsigned slot[kK];
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    if (threadIdx.x < kK)
    {
      sh.slot[threadIdx.x] = 0;
    }
    unsigned s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      s[i] = twiddle(v[i]);
    }
    sort4_desc(s);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kK; ++r)
    {
      atomicMax(&sh.slot[r], s[0]);
      __syncthreads();
      const unsigned M = sh.slot[r];
      const bool w     = (s[0] == M);
      s[0]             = w ? s[1] : s[0];
      s[1]             = w ? s[2] : s[1];
      s[2]             = w ? s[3] : s[2];
      s[3]             = w ? 0u : s[3];
    }
    if (threadIdx.x < kK)
    {
      out_v[threadIdx.x] = untwiddle(sh.slot[threadIdx.x]);
      out_i[threadIdx.x] = 0;
    }
    __syncthreads();
  }
};

// nb without the tier scatter (rounds + post-hoc M*; M* dumped as output)
struct ProtoDiagNbNoTier
{
  static constexpr const char* name = "DIAG nb rounds + post-hoc (no tier)";
  static constexpr bool is_half     = false;
  using Smem                        = ProtoAtomicAdaptive::Smem;
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    if (threadIdx.x < kK)
    {
      sh.slot[threadIdx.x] = 0;
      sh.rh[threadIdx.x]   = 0;
    }
    unsigned tw[4];
    unsigned s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      tw[i] = twiddle(v[i]);
      s[i]  = tw[i];
    }
    sort4_desc(s);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kK; ++r)
    {
      atomicMax(&sh.slot[r], s[0]);
      __syncthreads();
      const unsigned M = sh.slot[r];
      const bool w     = (s[0] == M);
      s[0]             = w ? s[1] : s[0];
      s[1]             = w ? s[2] : s[1];
      s[2]             = w ? s[3] : s[2];
      s[3]             = w ? 0u : s[3];
    }
    unsigned sl[kK];
#pragma unroll
    for (int j = 0; j < kK; ++j)
    {
      sl[j] = sh.slot[j];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      unsigned jj = 0;
      bool eq     = false;
#pragma unroll
      for (int j = 0; j < kK; ++j)
      {
        jj += (sl[j] > tw[i]);
        eq |= (sl[j] == tw[i]) && (sl[j] != 0);
      }
      if (eq)
      {
        atomicAdd(&sh.rh[jj], 1u);
      }
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
      const int lane = threadIdx.x;
      unsigned r     = (lane < kK) ? sh.rh[lane] : 0;
      unsigned c     = r;
#pragma unroll
      for (int d = 1; d < kK; d <<= 1)
      {
        unsigned o = __shfl_up_sync(kFull, c, d);
        if (lane >= d)
        {
          c += o;
        }
      }
      if (lane < kK && c >= kK && (c - r) < kK)
      {
        sh.mstar = sh.slot[lane];
      }
    }
    __syncthreads();
    if (threadIdx.x < kK)
    {
      out_v[threadIdx.x] = untwiddle(sh.mstar);
      out_i[threadIdx.x] = 0;
    }
    __syncthreads();
  }
};

template <class P>
constexpr bool is_diag_v = false;
template <>
constexpr bool is_diag_v<ProtoDiagBlockRounds> = true;
template <>
constexpr bool is_diag_v<ProtoDiagWarpRounds> = true;
template <>
constexpr bool is_diag_v<ProtoDiagNbRounds> = true;
template <>
constexpr bool is_diag_v<ProtoDiagNbNoTier> = true;

// ------------------------------------------------------------------ kernels
template <class P>
struct Box
{
  typename P::Smem s;
  float out_v[kK];
  int out_i[kK];
};

template <class P>
__global__ void __launch_bounds__(kBlock) correct_kernel(const float* in, float* out_v, int* out_i)
{
  __shared__ Box<P> box;
  float v[4];
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

template <class P>
__global__ void __launch_bounds__(kBlock) lat_kernel(const float* in, int chain, long long* out)
{
  __shared__ Box<P> box;
  float v0[4];
  int idx[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    idx[i] = threadIdx.x * 4 + i;
    v0[i]  = in[idx[i]];
  }
  P::run(v0, idx, box.s, box.out_v, box.out_i); // warmup; initializes box.out_v
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
      float v[4];
      v[0] = fmaf(box.out_v[0], 0.0f, v0[0]); // serialize: call n depends on call n-1's output
      v[1] = v0[1];
      v[2] = v0[2];
      v[3] = v0[3];
      P::run(v, idx, box.s, box.out_v, box.out_i);
    }
    float f = box.out_v[0];
    asm volatile("" ::"f"(f)); // consume before reading the end clock
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
__global__ void __launch_bounds__(kBlock) thr_kernel(const float* in, int outer, float* sink)
{
  __shared__ Box<P> box;
  float v0[4];
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
    float v[4];
    v[0] = fmaf(box.out_v[0], 0.0f, v0[0]);
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

// ------------------------------------------------------------------ host: data patterns
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

static std::vector<float> gen_pattern(const std::string& p, unsigned seed)
{
  std::vector<float> v(kN);
  Lcg rng(seed * 2654435761u + 12345u);
  auto normal = [&]() {
    float u1 = std::max(rng.uniform(), 1e-7f);
    float u2 = rng.uniform();
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.28318530718f * u2);
  };
  auto quantize = [](float base) {
    float r  = std::rint(base);
    float fr = std::rint((base - r) * 32.0f);
    return r + fr / 32.0f;
  };
  if (p == "random")
  {
    for (auto& x : v)
    {
      x = normal();
    }
  }
  else if (p == "quantized_random")
  {
    for (auto& x : v)
    {
      x = quantize(normal());
    }
  }
  else if (p == "relu_quantized")
  {
    for (auto& x : v)
    {
      x = quantize(std::max(normal(), 0.f));
    }
  }
  else if (p == "tie_heavy")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = (float) (j % 64) / 64.f;
    }
  }
  else if (p == "pivot_tie4" || p == "pivot_tie40")
  {
    const int gt = (p == "pivot_tie4") ? 4 : 40;
    for (auto& x : v)
    {
      x = 1.f;
    }
    int placed = 0;
    while (placed < gt)
    {
      unsigned pos = rng.next() % kN;
      if (v[pos] != 2.f)
      {
        v[pos] = 2.f;
        ++placed;
      }
    }
  }
  else if (p == "all_equal")
  {
    for (auto& x : v)
    {
      x = 1.f;
    }
  }
  else if (p == "sorted_asc")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = (float) j / kN;
    }
  }
  else
  {
    printf("unknown pattern %s\n", p.c_str());
    exit(1);
  }
  return v;
}

static float half_round(float f)
{
  return __half2float(__float2half(f));
}

// set-semantics validation (§8)
static bool validate(const std::vector<float>& in, bool as_half, const float* ov, const int* oi, std::string& why)
{
  std::vector<float> ref(kN);
  for (int i = 0; i < kN; ++i)
  {
    ref[i] = as_half ? half_round(in[i]) : in[i];
  }
  std::vector<float> sorted = ref;
  std::sort(sorted.begin(), sorted.end(), [](float a, float b) {
    return a > b;
  });
  std::vector<float> want(sorted.begin(), sorted.begin() + kK);
  std::vector<float> got(ov, ov + kK);
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
    if (!(ref[oi[i]] == ov[i]))
    {
      char b[128];
      snprintf(b, sizeof b, "index %d holds %g but output value is %g", oi[i], ref[oi[i]], ov[i]);
      why = b;
      return false;
    }
  }
  return true;
}

// ------------------------------------------------------------------ host: runners
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

#define CHECK(call)                                                              \
  do                                                                             \
  {                                                                              \
    cudaError_t e = (call);                                                      \
    if (e != cudaSuccess)                                                        \
    {                                                                            \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);     \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

template <class P>
void run_correct()
{
  if constexpr (is_diag_v<P>)
  {
    return;
  }
  const char* pats[] = {
    "random", "quantized_random", "relu_quantized", "tie_heavy", "pivot_tie4", "pivot_tie40", "all_equal", "sorted_asc"};
  float *d_in, *d_ov;
  int* d_oi;
  CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CHECK(cudaMalloc(&d_ov, kK * sizeof(float)));
  CHECK(cudaMalloc(&d_oi, kK * sizeof(int)));
  int fails = 0, runs = 0;
  for (const char* pat : pats)
  {
    const int seeds = (std::string(pat) == "random" || std::string(pat) == "quantized_random") ? 8 : 2;
    for (int seed = 0; seed < seeds; ++seed)
    {
      auto in = gen_pattern(pat, seed);
      CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
      correct_kernel<P><<<1, kBlock>>>(d_in, d_ov, d_oi);
      CHECK(cudaDeviceSynchronize());
      float ov[kK];
      int oi[kK];
      CHECK(cudaMemcpy(ov, d_ov, sizeof ov, cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(oi, d_oi, sizeof oi, cudaMemcpyDeviceToHost));
      std::string why;
      ++runs;
      if (!validate(in, P::is_half, ov, oi, why))
      {
        ++fails;
        printf("    FAIL %-16s seed %d: %s\n", pat, seed, why.c_str());
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
  const char* pats[] = {"random", "tie_heavy", "pivot_tie40", "sorted_asc"};
  float* d_in;
  long long* d_out;
  CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CHECK(cudaMalloc(&d_out, sizeof(long long)));
  printf("  %-40s", P::name);
  for (const char* pat : pats)
  {
    auto in = gen_pattern(pat, 0);
    CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
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
  constexpr int grid = 2048, outer = 32;
  float *d_in, *d_sink;
  CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CHECK(cudaMalloc(&d_sink, (size_t) grid * kK * sizeof(float)));
  auto in = gen_pattern("random", 0);
  CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
  thr_kernel<P><<<grid, kBlock>>>(d_in, outer, d_sink); // warmup
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
  const double calls = (double) grid * (outer + 1);
  const double gelems = calls * kN / (best_ms * 1e-3) / 1e9;
  printf("  %-40s %8.1f G elem/s   (%.3f ms, %d blocks x %d calls)\n", P::name, gelems, best_ms, grid, outer + 1);
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
         P::name, a.numRegs, a.sharedSizeBytes, a.localSizeBytes, occ);
}

#define FOREACH_PROTO(X)  \
  X(ProtoAir)             \
  X(ProtoBitonicHier)     \
  X(ProtoReduxIter)       \
  X(ProtoAtomicIter)      \
  X(ProtoAtomicAdaptive)  \
  X(ProtoHybrid)          \
  X(ProtoHierExtract)     \
  X(ProtoHistNarrow)      \
  X(ProtoAtomicIterHalf)  \
  X(ProtoDiagBlockRounds) \
  X(ProtoDiagWarpRounds)  \
  X(ProtoDiagNbRounds)    \
  X(ProtoDiagNbNoTier)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness (set semantics, all patterns) ===\n");
#define RUNC(P) run_correct<P>();
    FOREACH_PROTO(RUNC)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== latency: slope cyc/call, single block, per data pattern ===\n");
#define RUNL(P) run_lat<P>();
    FOREACH_PROTO(RUNL)
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== throughput: fixed workload, %d SMs saturated ===\n", p.multiProcessorCount);
#define RUNT(P) run_thr<P>();
    FOREACH_PROTO(RUNT)
  }
  if (mode == "res" || mode == "all")
  {
    printf("\n=== resources (thr kernel) ===\n");
#define RUNR(P) run_res<P>();
    FOREACH_PROTO(RUNR)
  }
  return 0;
}
