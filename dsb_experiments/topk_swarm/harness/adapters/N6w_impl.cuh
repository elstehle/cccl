// Implementation shared by the N6 warp adapters (equi-width bucket select over a value bracket),
// the W3 adapters (warp radix select with ballot-count histograms) and the N1 adapter (bit-serial
// ballot select). Not an adapter by itself.
//
// Common state: a warp-uniform range [tmin, tmax] of order-preserving key images (twiddled
// 32-bit words), the number of candidates n inside it and the number r of them still needed.
// Every pass refines the range: a bracket pass splits the candidates' value range into 32 equal
// width bins, a digit pass splits the next d key bits into 2^d buckets, a bit pass splits one bit.
// Items strictly above the range are selected, items below are dropped, both without any per-item
// state: the final emission compares every item's image against the threshold.
//
// Keys: fp32 and bf16, widened to fp32 with minus zero folded onto plus zero. The image of a bf16
// key keeps its 16 twiddled bits in the high half and zero in the low half, so one comparison
// order serves both types and the bf16 digit passes cover bits 31 down to 16 only.
//
// Ballot-count histogram: for every item slot the lanes ballot each bit of the bucket index (plus
// one ballot of the candidate flag when not every item is a candidate), lane j combines the masks
// into the mask of lanes whose item is in bucket j and counts it with popc. A suffix scan over the
// lanes locates the bucket holding the r-th largest candidate.
//
// Issue order: the ballots of a chunk of items are issued before any of them is consumed. With one
// ballot group per item followed by its LOP3 and POPC consumers in program order, the in-order
// issue stalls on every item for the ballot latency, and the pass costs about 70 cycles per item
// instead of the 6 cycles per pipelined ballot of T17 (measured in the first N6w_ballot batch).
//
// Emission: the selected items (image above the threshold, plus the first `take` items equal to it
// in index order) are written at their rank among the selected items, computed from one ballot per
// slot in lane-major order, which is index order on the blocked arrangement. prefer_smaller_index
// therefore costs nothing beyond the tie rank pass, which only runs when a tie class is cut.
//
// NaN: unsupported (the image of NaN is not ordered), focus-set inputs contain no NaN. Infinities
// and denormals are ordinary keys.
#pragma once

#include <cuda_bf16.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

#include "../topk_harness.cuh"

namespace n6w
{

inline constexpr unsigned FULL = 0xffffffffu;
inline constexpr uint32_t IMG0 = 0x4b000000u; // bits of 2^23, the image of bin 0
inline constexpr float TWO23   = 8388608.0f;
inline constexpr int ALL_TIES  = 0x7fffffff;

__host__ __device__ constexpr int ceil_log2(int x)
{
  int r = 0;
  while ((1 << r) < x)
  {
    ++r;
  }
  return r;
}
__host__ __device__ constexpr int next_pow2(int x)
{
  return 1 << ceil_log2(x);
}
__host__ __device__ constexpr int cmin(int a, int b)
{
  return a < b ? a : b;
}
__host__ __device__ constexpr int cmax(int a, int b)
{
  return a > b ? a : b;
}
// items whose ballots are issued back to back before any consumer
__host__ __device__ constexpr int chunk_items(int ipt)
{
  const int want = ipt >= 128 ? 4 : 8;
  return (ipt % want == 0) ? want : (ipt % 4 == 0) ? 4 : (ipt % 2 == 0) ? 2 : 1;
}

template <typename KeyT>
inline constexpr int key_bits = std::is_same_v<KeyT, __nv_bfloat16> ? 16 : 32;

// ------------------------------------------------------------------------------------------------
// Key images
// ------------------------------------------------------------------------------------------------
template <typename KeyT>
__device__ __forceinline__ float widen(KeyT k)
{
  if constexpr (std::is_same_v<KeyT, __nv_bfloat16>)
  {
    return __bfloat162float(k);
  }
  else
  {
    return k;
  }
}

// order preserving image of a float whose minus zero was folded onto plus zero
__device__ __forceinline__ uint32_t tw_f32(float f)
{
  const uint32_t u = __float_as_uint(f);
  const uint32_t s = static_cast<uint32_t>(static_cast<int32_t>(u) >> 31);
  return u ^ (s | 0x80000000u);
}

template <typename KeyT>
__device__ __forceinline__ uint32_t tw_of(float f)
{
  if constexpr (std::is_same_v<KeyT, __nv_bfloat16>)
  {
    return tw_f32(f) & 0xffff0000u;
  }
  else
  {
    return tw_f32(f);
  }
}

__device__ __forceinline__ float untw(uint32_t t)
{
  const uint32_t s = static_cast<uint32_t>(static_cast<int32_t>(~t) >> 31);
  return __uint_as_float(t ^ (s | 0x80000000u));
}

__device__ __forceinline__ uint32_t lanemask_lt()
{
  uint32_t m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

template <typename KeyT, int IPT>
__device__ __forceinline__ void load_values(const KeyT (&keys)[IPT], float (&v)[IPT])
{
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    v[i] = __fadd_rn(widen<KeyT>(keys[i]), 0.0f);
  }
}

template <typename KeyT, int IPT>
__device__ __forceinline__ void load_images(const KeyT (&keys)[IPT], uint32_t (&tw)[IPT])
{
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    tw[i] = tw_of<KeyT>(__fadd_rn(widen<KeyT>(keys[i]), 0.0f));
  }
}

template <typename KeyT, int IPT>
__device__ __forceinline__ void to_images(const float (&v)[IPT], uint32_t (&tw)[IPT])
{
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    tw[i] = tw_of<KeyT>(v[i]);
  }
}

// ------------------------------------------------------------------------------------------------
// Per-lane top-M on floats (sorted blocks of M and pairwise top-M merges, min and max only)
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void fswap_desc(float& a, float& b)
{
  const float lo = fminf(a, b);
  a              = fmaxf(a, b);
  b              = lo;
}

template <int LO, int HI, int R, int P>
__device__ __forceinline__ void oe_merge_f(float (&a)[P])
{
  constexpr int STEP = 2 * R;
  if constexpr (STEP < HI - LO)
  {
    oe_merge_f<LO, HI, STEP>(a);
    oe_merge_f<LO + R, HI, STEP>(a);
#pragma unroll
    for (int i = LO + R; i < HI - R; i += STEP)
    {
      fswap_desc(a[i], a[i + R]);
    }
  }
  else
  {
    fswap_desc(a[LO], a[LO + R]);
  }
}

template <int LO, int HI, int P>
__device__ __forceinline__ void oe_sort_desc_f(float (&a)[P])
{
  if constexpr (HI - LO >= 1)
  {
    constexpr int MID = LO + (HI - LO) / 2;
    oe_sort_desc_f<LO, MID>(a);
    oe_sort_desc_f<MID + 1, HI>(a);
    oe_merge_f<LO, HI, 1>(a);
  }
}

// list (descending) becomes the M largest of list and blk (descending)
template <int M>
__device__ __forceinline__ void merge_top_f(float (&list)[M], const float (&blk)[M])
{
#pragma unroll
  for (int i = 0; i < M; ++i)
  {
    list[i] = fmaxf(list[i], blk[M - 1 - i]);
  }
#pragma unroll
  for (int stride = M >> 1; stride > 0; stride >>= 1)
  {
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
      const int p = i ^ stride;
      if (p > i)
      {
        fswap_desc(list[i], list[p]);
      }
    }
  }
}

template <int M, int OFF, int IPT>
__device__ __forceinline__ void take_block(const float (&v)[IPT], float (&blk)[M])
{
#pragma unroll
  for (int j = 0; j < M; ++j)
  {
    blk[j] = v[OFF + j];
  }
  oe_sort_desc_f<0, M - 1>(blk);
}

template <int M, int IPT, int G>
__device__ __forceinline__ void top_blocks(const float (&v)[IPT], float (&list)[M])
{
  if constexpr (G < IPT / M)
  {
    float blk[M];
    take_block<M, G * M>(v, blk);
    merge_top_f<M>(list, blk);
    top_blocks<M, IPT, G + 1>(v, list);
  }
}

// mx = lane maximum, mth = the lane's M-th largest (M 0: the lane minimum)
template <int IPT, int M>
__device__ __forceinline__ void lane_top(const float (&v)[IPT], float& mx, float& mth)
{
  if constexpr (M == 0)
  {
    float a = v[0], b = v[0];
#pragma unroll
    for (int i = 1; i < IPT; ++i)
    {
      a = fmaxf(a, v[i]);
      b = fminf(b, v[i]);
    }
    mx  = a;
    mth = b;
  }
  else if constexpr (M == 1)
  {
    float a = v[0];
#pragma unroll
    for (int i = 1; i < IPT; ++i)
    {
      a = fmaxf(a, v[i]);
    }
    mx  = a;
    mth = a;
  }
  else if constexpr (M == 2)
  {
    float a = fmaxf(v[0], v[1]);
    float b = fminf(v[0], v[1]);
#pragma unroll
    for (int i = 2; i < IPT; ++i)
    {
      b = fmaxf(b, fminf(a, v[i]));
      a = fmaxf(a, v[i]);
    }
    mx  = a;
    mth = b;
  }
  else
  {
    static_assert(IPT % M == 0, "block top-M needs IPT a multiple of M");
    float list[M];
    take_block<M, 0>(v, list);
    top_blocks<M, IPT, 1>(v, list);
    mx  = list[0];
    mth = list[M - 1];
  }
}

// ------------------------------------------------------------------------------------------------
// Range state, scan and pivot location
// ------------------------------------------------------------------------------------------------
struct range
{
  uint32_t tmin; // smallest image a candidate may have
  uint32_t tmax; // largest image a candidate may have
  int r; // candidates still needed
  int n; // candidates inside the range
};

// s = sum of c over lanes >= this lane
__device__ __forceinline__ int suffix_scan(int c, int lane)
{
  int s = c;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1)
  {
    const int t = __shfl_down_sync(FULL, s, d);
    s += (lane + d < 32) ? t : 0;
  }
  return s;
}

// lanes with s >= r form the prefix 0..p, p is the pivot bucket, cnt_p its count, above the
// candidates in buckets beyond it
__device__ __forceinline__ void locate(int s, int c, int r, int& p, int& cnt_p, int& above)
{
  const unsigned b = __ballot_sync(FULL, s >= r);
  p                = __popc(b) - 1;
  const uint32_t w = __shfl_sync(FULL, (static_cast<uint32_t>(s) << 16) | static_cast<uint32_t>(c), p);
  cnt_p            = static_cast<int>(w & 0xffffu);
  above            = static_cast<int>(w >> 16) - cnt_p;
}

// per-lane inversion masks for the bucket id = lane
template <int D>
struct bucket_masks
{
  uint32_t inv[D];
  __device__ __forceinline__ explicit bucket_masks(int lane)
  {
#pragma unroll
    for (int b = 0; b < D; ++b)
    {
      inv[b] = ((lane >> b) & 1) ? 0u : ~0u;
    }
  }
  // mask of lanes whose bucket bits (balloted into bal[b]) equal this lane's bucket id
  __device__ __forceinline__ uint32_t match(const uint32_t (&bal)[D]) const
  {
    uint32_t m = bal[0] ^ inv[0];
#pragma unroll
    for (int b = 1; b < D; ++b)
    {
      m &= bal[b] ^ inv[b];
    }
    return m;
  }
};

// 5-bit bin per item, six items per word
template <int IPT>
struct bins5
{
  static constexpr int W = (IPT + 5) / 6;
  uint32_t w[W];
  __device__ __forceinline__ bins5()
  {
#pragma unroll
    for (int j = 0; j < W; ++j)
    {
      w[j] = 0u;
    }
  }
  __device__ __forceinline__ void set(int i, uint32_t x)
  {
    w[i / 6] |= (x & 31u) << (5 * (i % 6));
  }
  __device__ __forceinline__ uint32_t get(int i) const
  {
    return (w[i / 6] >> (5 * (i % 6))) & 31u;
  }
};

// ------------------------------------------------------------------------------------------------
// Bracket pass (N6): 32 equal width bins over the value range of the current candidates
// ------------------------------------------------------------------------------------------------
// bin image: u = bits(fma(v - lo, scale, 2^23)) clamped into [IMG0, IMG0 + 31], monotone in v
__device__ __forceinline__ uint32_t bin_image(float v, float lo, float scale)
{
  const float d   = v - lo;
  const int32_t x = __float_as_int(fmaf(d, scale, TWO23));
  const int32_t y = ::max(x, static_cast<int32_t>(IMG0));
  return ::min(static_cast<uint32_t>(y), IMG0 + 31u);
}

template <typename KeyT, int IPT, bool ALL>
__device__ __forceinline__ void bracket_pass(const float (&v)[IPT], range& st, int lane)
{
  constexpr int CH  = chunk_items(IPT);
  const float lo    = fmaxf(untw(st.tmin), -FLT_MAX);
  const float hi    = fminf(untw(st.tmax), FLT_MAX);
  const float width = hi - lo;
  // a degenerate bracket (zero or overflowing width) keeps a tiny positive scale: a zero scale
  // would turn infinite items into NaN images (inf times 0) and put -inf into the top bin
  float scale = 32.0f / width;
  scale       = (scale > 0.0f && scale < FLT_MAX) ? scale : FLT_MIN; // 32 / inf is +0, not caught by width > 0
  const bucket_masks<5> bm(lane);

  bins5<IPT> bins;
  int cnt = 0;
#pragma unroll
  for (int c0 = 0; c0 < IPT; c0 += CH)
  {
    uint32_t bal[CH][5];
    uint32_t cm[CH];
#pragma unroll
    for (int j = 0; j < CH; ++j)
    {
      const int i = c0 + j;
      uint32_t x  = bin_image(v[i], lo, scale);
      cm[j]       = FULL;
      if constexpr (!ALL)
      {
        const uint32_t t = tw_of<KeyT>(v[i]);
        const bool c     = (t >= st.tmin) & (t <= st.tmax);
        x                = c ? x : 0u;
        cm[j]            = __ballot_sync(FULL, c);
      }
      bins.set(i, x);
#pragma unroll
      for (int b = 0; b < 5; ++b)
      {
        bal[j][b] = __ballot_sync(FULL, (x >> b) & 1u);
      }
    }
#pragma unroll
    for (int j = 0; j < CH; ++j)
    {
      cnt += __popc(bm.match(bal[j]) & cm[j]);
    }
  }
  const int s = suffix_scan(cnt, lane);
  int p, np, above;
  locate(s, cnt, st.r, p, np, above);

  const uint32_t pb = static_cast<uint32_t>(p);
  uint32_t tmin = FULL, tmax = 0u;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const uint32_t t = tw_of<KeyT>(v[i]);
    bool c           = (bins.get(i) == pb);
    if constexpr (!ALL)
    {
      c = c & (t >= st.tmin) & (t <= st.tmax);
    }
    tmin = c ? ::min(tmin, t) : tmin;
    tmax = c ? ::max(tmax, t) : tmax;
  }
  st.tmin = __reduce_min_sync(FULL, tmin);
  st.tmax = __reduce_max_sync(FULL, tmax);
  st.r -= above;
  st.n = np;
}

// ------------------------------------------------------------------------------------------------
// Digit pass (W3): the next d <= D unresolved key bits, 2^d buckets, candidates by image range
// ------------------------------------------------------------------------------------------------
// VALID: the array is a compacted one with nv valid slots (slot i of lane l is valid when l + 32 i < nv)
template <typename KeyT, int IPT, int D, bool ALL, bool VALID = false>
__device__ __forceinline__ void digit_pass(const uint32_t (&tw)[IPT], range& st, int lane, int nv = 0)
{
  static_assert(D >= 1 && D <= 5, "one bucket per lane");
  constexpr int CH     = chunk_items(IPT);
  constexpr int KB     = key_bits<KeyT>;
  const int nb         = ALL ? 0 : __clz(st.tmin ^ st.tmax); // resolved bits from the top
  const int d          = ::min(D, KB - nb);
  const int shift      = 32 - nb - d;
  const uint32_t dmask = (1u << d) - 1u;
  const bucket_masks<D> bm(lane);

  int cnt = 0;
#pragma unroll
  for (int c0 = 0; c0 < IPT; c0 += CH)
  {
    uint32_t bal[CH][D];
    uint32_t cm[CH];
#pragma unroll
    for (int j = 0; j < CH; ++j)
    {
      const int i      = c0 + j;
      const uint32_t t = tw[i];
      uint32_t dig     = (t >> shift) & dmask;
      cm[j]            = FULL;
      if constexpr (!ALL)
      {
        bool c = (t >= st.tmin) & (t <= st.tmax);
        if constexpr (VALID)
        {
          c = c & (lane + 32 * i < nv);
        }
        dig   = c ? dig : 0u;
        cm[j] = __ballot_sync(FULL, c);
      }
#pragma unroll
      for (int b = 0; b < D; ++b)
      {
        bal[j][b] = __ballot_sync(FULL, (dig >> b) & 1u);
      }
    }
#pragma unroll
    for (int j = 0; j < CH; ++j)
    {
      cnt += __popc(bm.match(bal[j]) & cm[j]);
    }
  }
  // lanes beyond the 2^d buckets of this pass share the masks of lower lanes and must not count
  cnt         = (lane < (1 << d)) ? cnt : 0;
  const int s = suffix_scan(cnt, lane);
  int p, np, above;
  locate(s, cnt, st.r, p, np, above);

  const uint32_t keep = ~((2u << (shift + d - 1)) - 1u); // bits above the digit
  const uint32_t base = (st.tmin & keep) | (static_cast<uint32_t>(p) << shift);
  const uint32_t low  = (1u << shift) - 1u;
  st.tmin             = ::max(st.tmin, base);
  st.tmax             = ::min(st.tmax, base | low);
  st.r -= above;
  st.n = np;
}

// ------------------------------------------------------------------------------------------------
// Bit pass (N1): one key bit, the count of candidates with the bit set decides
// ------------------------------------------------------------------------------------------------
template <typename KeyT, int IPT, bool ALL>
__device__ __forceinline__ void bit_pass(const uint32_t (&tw)[IPT], range& st, int lane)
{
  constexpr int CH = chunk_items(IPT);
  const int nb     = ALL ? 0 : __clz(st.tmin ^ st.tmax);
  const int shift  = 31 - nb;
  int c1           = 0;
#pragma unroll
  for (int c0 = 0; c0 < IPT; c0 += CH)
  {
    uint32_t bal[CH];
#pragma unroll
    for (int j = 0; j < CH; ++j)
    {
      const uint32_t t = tw[c0 + j];
      bool f           = (t >> shift) & 1u;
      if constexpr (!ALL)
      {
        f = f & (t >= st.tmin) & (t <= st.tmax);
      }
      bal[j] = __ballot_sync(FULL, f);
    }
#pragma unroll
    for (int j = 0; j < CH; ++j)
    {
      c1 += __popc(bal[j]);
    }
  }
  const uint32_t keep = ~((2u << shift) - 1u);
  const uint32_t low  = (1u << shift) - 1u;
  const uint32_t one  = 1u << shift;
  const bool up       = c1 >= st.r;
  const uint32_t base = (st.tmin & keep) | (up ? one : 0u);
  st.tmin             = ::max(st.tmin, base);
  st.tmax             = ::min(st.tmax, base | low);
  st.r                = up ? st.r : st.r - c1;
  st.n                = up ? c1 : st.n - c1;
}

// ------------------------------------------------------------------------------------------------
// Emission: write val(image, index) of every flagged item at its rank among the flagged items.
// LM (lane-major) ranks are index order on the blocked arrangement, slot-major ranks are index
// order on a compacted array (slot q = lane + 32 j holds the q-th candidate in index order).
// ------------------------------------------------------------------------------------------------
template <int IPT>
struct flags
{
  static constexpr int W = (IPT + 31) / 32;
  uint32_t w[W];
  __device__ __forceinline__ flags()
  {
#pragma unroll
    for (int j = 0; j < W; ++j)
    {
      w[j] = 0u;
    }
  }
  __device__ __forceinline__ void set(int i, bool f)
  {
    w[i / 32] |= static_cast<uint32_t>(f) << (i % 32);
  }
  __device__ __forceinline__ bool get(int i) const
  {
    return (w[i / 32] >> (i % 32)) & 1u;
  }
};

struct idx_blocked
{
  int base;
  __device__ __forceinline__ int operator()(int i) const
  {
    return base + i;
  }
};

template <int C>
struct idx_array
{
  const uint32_t (&a)[C];
  __device__ __forceinline__ int operator()(int i) const
  {
    return static_cast<int>(a[i]);
  }
};

struct val_index
{
  __device__ __forceinline__ uint32_t operator()(uint32_t, int idx) const
  {
    return static_cast<uint32_t>(idx);
  }
};

// one ballot per slot, chunked, into bal[IPT]
template <int IPT>
__device__ __forceinline__ void ballot_flags(const flags<IPT>& fl, uint32_t (&bal)[IPT])
{
  constexpr int CH = chunk_items(IPT);
#pragma unroll
  for (int c0 = 0; c0 < IPT; c0 += CH)
  {
#pragma unroll
    for (int j = 0; j < CH; ++j)
    {
      bal[c0 + j] = __ballot_sync(FULL, fl.get(c0 + j));
    }
  }
}

// lane-major: flagged items on lower lanes precede this lane's items
template <int IPT>
__device__ __forceinline__ int rank_base(const flags<IPT>& fl, uint32_t lt)
{
  uint32_t bal[IPT];
  ballot_flags<IPT>(fl, bal);
  int R = 0;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    R += __popc(bal[i] & lt);
  }
  return R;
}

template <int IPT, bool LM, class Idx, class Val>
__device__ __forceinline__ void
scatter_flags(const flags<IPT>& fl, const uint32_t (&tw)[IPT], Idx idx, Val val, uint32_t* dst, uint32_t lt)
{
  if constexpr (LM)
  {
    const int R = rank_base<IPT>(fl, lt);
    int own     = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      const bool f = fl.get(i);
      if (f)
      {
        dst[R + own] = val(tw[i], idx(i));
      }
      own += f ? 1 : 0;
    }
  }
  else
  {
    uint32_t bal[IPT];
    ballot_flags<IPT>(fl, bal);
    int base = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      if (fl.get(i))
      {
        dst[base + __popc(bal[i] & lt)] = val(tw[i], idx(i));
      }
      base += __popc(bal[i]);
    }
  }
}

// selected: image > thr, plus the first `take` items with image == thr in index order
template <int IPT, bool LM, class Idx>
__device__ __forceinline__ void
emit_threshold(const uint32_t (&tw)[IPT], uint32_t thr, int take, int* out_idx, Idx idx, uint32_t lt)
{
  flags<IPT> sel;
  if (take == ALL_TIES)
  {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      sel.set(i, tw[i] >= thr);
    }
  }
  else
  {
    flags<IPT> tie;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      tie.set(i, tw[i] == thr);
    }
    if constexpr (LM)
    {
      const int R = rank_base<IPT>(tie, lt);
      int own     = 0;
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        const bool t = tie.get(i);
        sel.set(i, (tw[i] > thr) | (t & (R + own < take)));
        own += t ? 1 : 0;
      }
    }
    else
    {
      uint32_t bal[IPT];
      ballot_flags<IPT>(tie, bal);
      int base = 0;
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        const bool t = tie.get(i);
        sel.set(i, (tw[i] > thr) | (t & (base + __popc(bal[i] & lt) < take)));
        base += __popc(bal[i]);
      }
    }
  }
  scatter_flags<IPT, LM>(sel, tw, idx, val_index{}, reinterpret_cast<uint32_t*>(out_idx), lt);
}

// final emission from a resolved range: all candidates when n == r, else the tie class at tmin.
// Images of real keys are never 0, so a threshold of at least 1 excludes padding slots (image 0).
template <int IPT, bool LM, class Idx>
__device__ __forceinline__ void emit_range(const uint32_t (&tw)[IPT], const range& st, int* out_idx, Idx idx, uint32_t lt)
{
  if (st.n == st.r)
  {
    emit_threshold<IPT, LM>(tw, ::max(st.tmin, 1u), ALL_TIES, out_idx, idx, lt);
  }
  else
  {
    emit_threshold<IPT, LM>(tw, st.tmin, st.r, out_idx, idx, lt);
  }
}

// ------------------------------------------------------------------------------------------------
// Candidate compaction: the items above the range are emitted at out_idx[0 .. above), the items
// inside the range go to shared memory in index order and come back as CIPT slots per lane (slot
// j of lane l holds candidate l + 32 j), padded with image 0. The caller continues on the
// compacted slots and emits their selection at out_idx + above.
// ------------------------------------------------------------------------------------------------
template <int IPT, int CIPT, class Idx>
__device__ __forceinline__ void compact(
  const uint32_t (&tw)[IPT],
  Idx idx,
  const range& st,
  uint32_t* smem_tw,
  uint32_t* smem_idx,
  uint32_t (&ctw)[CIPT],
  uint32_t (&cidx)[CIPT],
  int* out_idx,
  int lane,
  uint32_t lt)
{
  flags<IPT> in, up;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    in.set(i, (tw[i] >= st.tmin) & (tw[i] <= st.tmax));
    up.set(i, tw[i] > st.tmax);
  }
  scatter_flags<IPT, true>(up, tw, idx, val_index{}, reinterpret_cast<uint32_t*>(out_idx), lt);
  const int R = rank_base<IPT>(in, lt);
  int own     = 0;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const bool f = in.get(i);
    if (f)
    {
      smem_tw[R + own]  = tw[i];
      smem_idx[R + own] = static_cast<uint32_t>(idx(i));
    }
    own += f ? 1 : 0;
  }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < CIPT; ++j)
  {
    const int q      = lane + 32 * j;
    const bool valid = q < st.n;
    ctw[j]           = valid ? smem_tw[q] : 0u;
    cidx[j]          = valid ? smem_idx[q] : 0u;
  }
}

// ------------------------------------------------------------------------------------------------
// Packed W2 finish on the candidates of a small range: compact (image, index) words into shared
// memory in index order, then r rounds of redux.max over per-lane lists of FIPT words
// ------------------------------------------------------------------------------------------------
template <typename KeyT, int IdxBits>
struct val_packed
{
  uint32_t tmin;
  __device__ __forceinline__ uint32_t operator()(uint32_t t, int idx) const
  {
    constexpr uint32_t imask = (1u << IdxBits) - 1u;
    const uint32_t inv       = imask ^ static_cast<uint32_t>(idx);
    if constexpr (std::is_same_v<KeyT, __nv_bfloat16>)
    {
      return t | inv; // the bf16 image has a zero low half
    }
    else
    {
      return ((t - tmin + 1u) << IdxBits) | inv;
    }
  }
};

template <int IdxBits>
__device__ __forceinline__ int unpack_index(uint32_t w)
{
  constexpr uint32_t imask = (1u << IdxBits) - 1u;
  return static_cast<int>(imask ^ (w & imask));
}

template <typename KeyT, int IdxBits>
__device__ __forceinline__ bool packing_fits(uint32_t tmin, uint32_t tmax)
{
  if constexpr (std::is_same_v<KeyT, __nv_bfloat16>)
  {
    return true;
  }
  else
  {
    return (tmax - tmin) < ((1u << (32 - IdxBits)) - 1u);
  }
}

__device__ __forceinline__ void cswap_desc_u(uint32_t& x, uint32_t& y)
{
  const uint32_t lo = ::min(x, y);
  x                 = ::max(x, y);
  y                 = lo;
}

template <int L>
__device__ __forceinline__ void sort_words_desc(uint32_t (&a)[L])
{
  if constexpr (L == 2)
  {
    cswap_desc_u(a[0], a[1]);
  }
  else if constexpr (L == 4)
  {
    cswap_desc_u(a[0], a[1]);
    cswap_desc_u(a[2], a[3]);
    cswap_desc_u(a[0], a[2]);
    cswap_desc_u(a[1], a[3]);
    cswap_desc_u(a[1], a[2]);
  }
  else
  {
    static_assert(L == 1, "finish list of 1, 2 or 4 words");
  }
}

// words[0, n) hold the candidates (written by this warp before the call), r of them are emitted
// to out_idx[above + 0 .. above + r - 1] in (key descending, index ascending) order
template <int FIPT, int IdxBits>
__device__ __forceinline__ void w2_rounds(const uint32_t* words, int n, int r, int above, int* out_idx, int lane)
{
  __syncwarp();
  uint32_t list[FIPT];
#pragma unroll
  for (int j = 0; j < FIPT; ++j)
  {
    const int q = lane + 32 * j;
    list[j]     = (q < n) ? words[q] : 0u;
  }
  sort_words_desc<FIPT>(list);
#pragma unroll 1
  for (int j = 0; j < r; ++j)
  {
    const uint32_t m = __reduce_max_sync(FULL, list[0]);
    const bool win   = (list[0] == m);
    if (lane == (j & 31))
    {
      out_idx[above + j] = unpack_index<IdxBits>(m);
    }
#pragma unroll
    for (int q = 0; q < FIPT - 1; ++q)
    {
      list[q] = win ? list[q + 1] : list[q];
    }
    list[FIPT - 1] = win ? 0u : list[FIPT - 1];
  }
}

// items above the range are emitted at ranks 0 .. above - 1, the range's candidates are compacted
// into words (index order) and finished by w2_rounds
template <typename KeyT, int IPT, int FIPT, int IdxBits>
__device__ __forceinline__ void w2_finish(
  const uint32_t (&tw)[IPT], const range& st, int above, uint32_t* words, int* out_idx, int lane, int idx_base, uint32_t lt)
{
  flags<IPT> up, in;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const uint32_t t = tw[i];
    up.set(i, t > st.tmax);
    in.set(i, (t >= st.tmin) & (t <= st.tmax));
  }
  const idx_blocked idx{idx_base};
  scatter_flags<IPT, true>(up, tw, idx, val_index{}, reinterpret_cast<uint32_t*>(out_idx), lt);
  scatter_flags<IPT, true>(in, tw, idx, val_packed<KeyT, IdxBits>{st.tmin}, words, lt);
  w2_rounds<FIPT, IdxBits>(words, st.n, st.r, above, out_idx, lane);
}

// Compile-time support predicate shared by the adapters: warp scope, bf16 or fp32, N a multiple
// of 32 up to 32768 (the scan packs two 16-bit counts into one word), k in [1, N/2]
__host__ __device__ constexpr bool supported_shape(topk::key_type kt, int threads, int n, int k, topk::requirement req)
{
  const bool key_ok   = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
  const bool shape_ok = (threads == 32) && (n >= 32) && (n % 32 == 0) && (n <= 32768) && (k >= 1) && (2 * k <= n);
  const bool req_ok   = (req != topk::requirement::prefer_larger_index);
  return key_ok && shape_ok && req_ok;
}

} // namespace n6w
