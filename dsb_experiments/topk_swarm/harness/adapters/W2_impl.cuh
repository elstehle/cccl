// Implementation of approach W2 (redux based iterative extraction on packed words), shared by the
// W2 warp and sub-warp adapters and by the B5 block composition. Not an adapter by itself.
//
// Representation: every item becomes one unique 32-bit word that orders exactly like the pair
// (key descending, index ascending). The word is `twiddled_key_bits | inverted_index` where the
// index occupies the low ceil(log2 N) bits and the key the bits above. bf16 keeps all 16 key bits.
// fp32 keeps the top 32 - ceil(log2 N) bits of the twiddled key (idea N4): the k rounds are then
// exact on (truncated key, index) and a tie pass restores the exact top-k by full key inside the
// tie class of the k-th truncated key. Minus zero is mapped onto plus zero before packing so the
// two compare equal, as the harness oracle requires.
//
// Round: `m = redux.max(head)` over the group, the lane whose head equals m is the unique winner
// (words are unique), it shifts its sorted per-lane list by one with predicated selects. There is
// no ballot, no ffs and no election in the round (TECHNIQUES.md T1, T3, T19). The winner index is
// decoded from the uniform m and stored by lane r mod T, off the critical path.
//
// Prologue: each lane packs its items and selects its L largest in sorted order with a bitonic
// network built from sort blocks of L and pairwise top-L merges, where L is the smallest power of
// two not below min(k, items per lane). Positions of the list that can never reach the head
// before the last round are dead after unrolling and the compiler drops their shifts.
//
// Both tie-break requirements run the same code: the packed order is prefer_smaller_index by
// construction, and a prefer_smaller_index result is a valid not_guaranteed result.
//
// NaN: a NaN key with the sign bit clear twiddles above +inf and is selected first, a NaN with the
// sign bit set twiddles below -inf and is never selected. The all-ones bit pattern (a negative NaN
// with a full mantissa) packs to a word that can collide with the empty-list sentinel 0 and may be
// dropped. Focus-set inputs contain no NaN.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

#include "../topk_harness.cuh"

namespace w2
{

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

// ------------------------------------------------------------------------------------------------
// Twiddles: order preserving unsigned images of the key bits, minus zero mapped onto plus zero
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t twiddle_f32(float f)
{
  const uint32_t u    = __float_as_uint(f);
  const uint32_t mask = static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u;
  const uint32_t t    = u ^ mask;
  return t + (t == 0x7fffffffu ? 1u : 0u);
}

__device__ __forceinline__ uint32_t twiddle_bf16(__nv_bfloat16 h)
{
  const uint32_t u    = __bfloat16_as_ushort(h);
  const uint32_t mask = (u & 0x8000u) ? 0xffffu : 0x8000u;
  const uint32_t t    = u ^ mask;
  return t + (t == 0x7fffu ? 1u : 0u);
}

template <typename KeyT>
__device__ __forceinline__ uint32_t twiddle(KeyT k)
{
  if constexpr (std::is_same_v<KeyT, __nv_bfloat16>)
  {
    return twiddle_bf16(k);
  }
  else
  {
    return twiddle_f32(k);
  }
}

// Packed word: key field above the index field, inverted index in the low IdxBits bits
template <typename KeyT, int IdxBits>
__device__ __forceinline__ uint32_t pack(KeyT key, uint32_t idx)
{
  constexpr uint32_t idx_mask = (1u << IdxBits) - 1u;
  const uint32_t inv          = idx_mask ^ idx;
  if constexpr (std::is_same_v<KeyT, __nv_bfloat16>)
  {
    static_assert(IdxBits <= 16, "bf16 packing needs the index in 16 bits");
    return (twiddle_bf16(key) << 16) | inv;
  }
  else
  {
    return (twiddle_f32(key) & ~idx_mask) | inv;
  }
}

template <int IdxBits>
__device__ __forceinline__ int unpack_idx(uint32_t w)
{
  constexpr uint32_t idx_mask = (1u << IdxBits) - 1u;
  return static_cast<int>(idx_mask ^ (w & idx_mask));
}

// ------------------------------------------------------------------------------------------------
// Per-lane sorting networks on register arrays with static indices (min and max only, T20)
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void cswap_desc(uint32_t& a, uint32_t& b)
{
  const uint32_t lo = ::min(a, b);
  a                 = ::max(a, b);
  b                 = lo;
}
__device__ __forceinline__ void cswap_asc(uint32_t& a, uint32_t& b)
{
  const uint32_t hi = ::max(a, b);
  a                 = ::min(a, b);
  b                 = hi;
}

// Variant policies. v1 is the baseline: bitonic sort blocks and a counting tie pass. v2 adds the
// next-word tie check (one redux after the rounds instead of a count over every item). v3 adds
// Batcher odd-even merge sort blocks (19 comparators at 8 and 63 at 16 against 24 and 80).
struct policy_v1
{
  static constexpr bool next_word_check = false;
  static constexpr bool oddeven_sort    = false;
};
struct policy_v2
{
  static constexpr bool next_word_check = true;
  static constexpr bool oddeven_sort    = false;
};
struct policy_v3
{
  static constexpr bool next_word_check = true;
  static constexpr bool oddeven_sort    = true;
};

// Batcher odd-even merge sort of a[LO, HI] (inclusive bounds) into descending order
template <int LO, int HI, int R, int P>
__device__ __forceinline__ void oe_merge(uint32_t (&a)[P])
{
  constexpr int STEP = 2 * R;
  if constexpr (STEP < HI - LO)
  {
    oe_merge<LO, HI, STEP>(a);
    oe_merge<LO + R, HI, STEP>(a);
#pragma unroll
    for (int i = LO + R; i < HI - R; i += STEP)
    {
      cswap_desc(a[i], a[i + R]);
    }
  }
  else
  {
    cswap_desc(a[LO], a[LO + R]);
  }
}

template <int LO, int HI, int P>
__device__ __forceinline__ void oe_sort_desc(uint32_t (&a)[P])
{
  if constexpr (HI - LO >= 1)
  {
    constexpr int MID = LO + (HI - LO) / 2;
    oe_sort_desc<LO, MID>(a);
    oe_sort_desc<MID + 1, HI>(a);
    oe_merge<LO, HI, 1>(a);
  }
}

// Bitonic sort of a[OFF, OFF + L) into descending order
template <int L, int OFF, int P>
__device__ __forceinline__ void bitonic_sort_desc(uint32_t (&a)[P])
{
#pragma unroll
  for (int size = 2; size <= L; size <<= 1)
  {
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1)
    {
#pragma unroll
      for (int i = 0; i < L; ++i)
      {
        const int p = i ^ stride;
        if (p > i)
        {
          if ((i & size) == 0)
          {
            cswap_desc(a[OFF + i], a[OFF + p]);
          }
          else
          {
            cswap_asc(a[OFF + i], a[OFF + p]);
          }
        }
      }
    }
  }
}

template <int L, int OFF, int P, bool OE>
__device__ __forceinline__ void sort_block_desc(uint32_t (&a)[P])
{
  if constexpr (OE)
  {
    oe_sort_desc<OFF, OFF + L - 1>(a);
  }
  else
  {
    bitonic_sort_desc<L, OFF>(a);
  }
}

// list (descending) becomes the L largest of list and a[OFF, OFF + L) (descending), sorted.
// With TrackSpare, spare becomes the largest word dropped so far (the (L+1)-th largest of
// everything merged into list), which is the max of the min side of the elementwise max step.
template <int L, int OFF, int P, bool TrackSpare = false>
__device__ __forceinline__ void top_merge_desc(uint32_t (&list)[L], const uint32_t (&a)[P], uint32_t& spare)
{
#pragma unroll
  for (int i = 0; i < L; ++i)
  {
    const uint32_t x = a[OFF + L - 1 - i];
    if constexpr (TrackSpare)
    {
      spare = ::max(spare, ::min(list[i], x));
    }
    list[i] = ::max(list[i], x);
  }
#pragma unroll
  for (int stride = L >> 1; stride > 0; stride >>= 1)
  {
#pragma unroll
    for (int i = 0; i < L; ++i)
    {
      const int p = i ^ stride;
      if (p > i)
      {
        cswap_desc(list[i], list[p]);
      }
    }
  }
}

template <int L, int P, int G, bool OE, bool TrackSpare>
__device__ __forceinline__ void select_top_groups(uint32_t (&a)[P], uint32_t (&list)[L], uint32_t& spare)
{
  if constexpr (G < P / L)
  {
    sort_block_desc<L, G * L, P, OE>(a);
    top_merge_desc<L, G * L, P, TrackSpare>(list, a, spare);
    select_top_groups<L, P, G + 1, OE, TrackSpare>(a, list, spare);
  }
}

// list = the L largest of a[0, P), descending, spare = the (L+1)-th largest (0 when P == L or
// when TrackSpare is off). P is a multiple of L, both powers of two.
template <int L, int P, bool OE = false, bool TrackSpare = false>
__device__ __forceinline__ void select_top(uint32_t (&a)[P], uint32_t (&list)[L], uint32_t& spare)
{
  static_assert(P % L == 0, "P must be a multiple of L");
  sort_block_desc<L, 0, P, OE>(a);
#pragma unroll
  for (int i = 0; i < L; ++i)
  {
    list[i] = a[i];
  }
  spare = 0u;
  select_top_groups<L, P, 1, OE, TrackSpare>(a, list, spare);
}

template <int L, int P>
__device__ __forceinline__ void select_top(uint32_t (&a)[P], uint32_t (&list)[L])
{
  uint32_t spare;
  select_top<L, P, false, false>(a, list, spare);
}

// ------------------------------------------------------------------------------------------------
// Extraction rounds
// ------------------------------------------------------------------------------------------------
// One round: group max of the heads, the unique winner shifts its list, lane r mod T emits.
// Emit is a functor (word, round) so the warp phase of B5 can store words instead of indices.
template <int T, int L, class Emit>
__device__ __forceinline__ uint32_t one_round(uint32_t (&list)[L], int r, int lane, unsigned mask, Emit emit)
{
  const uint32_t m = __reduce_max_sync(mask, list[0]);
  const bool win   = (list[0] == m);
  if (lane == (r & (T - 1)))
  {
    emit(m, r);
  }
#pragma unroll
  for (int i = 0; i < L - 1; ++i)
  {
    list[i] = win ? list[i + 1] : list[i];
  }
  list[L - 1] = win ? 0u : list[L - 1];
  return m;
}

// K rounds, returns the word extracted in the last round (group uniform)
template <int T, int K, int L, class Emit>
__device__ __forceinline__ uint32_t extract_rounds(uint32_t (&list)[L], int lane, unsigned mask, Emit emit)
{
  uint32_t m = 0;
  if constexpr (K <= 64)
  {
#pragma unroll
    for (int r = 0; r < K; ++r)
    {
      m = one_round<T, L>(list, r, lane, mask, emit);
    }
  }
  else
  {
#pragma unroll 4
    for (int r = 0; r < K; ++r)
    {
      m = one_round<T, L>(list, r, lane, mask, emit);
    }
  }
  return m;
}

template <int IdxBits>
struct emit_index
{
  int* out_idx;
  __device__ __forceinline__ void operator()(uint32_t m, int r) const
  {
    out_idx[r] = unpack_idx<IdxBits>(m);
  }
};

struct emit_word
{
  uint32_t* out;
  __device__ __forceinline__ void operator()(uint32_t m, int r) const
  {
    out[r] = m;
  }
};

// ------------------------------------------------------------------------------------------------
// N4 tie pass for fp32 (group scope): the rounds were exact on (truncated key, index). Let T be
// the truncated key of the last extracted word. Class = items with truncated key T. Members with
// word >= m_last were emitted (E of them, in the last E output slots), members with a larger
// index were not. If the class has more members than were emitted, the right E members are the
// top E of the class by (full key descending, index ascending): value rounds over the class with
// a redux.max of the full twiddled key and a redux.min of the index among the holders of that key.
// The fast path (class fully emitted, the common case on random data) is one count and one
// redux.add. The slow path is group uniform.
// ------------------------------------------------------------------------------------------------
// Slow path: e = emitted class members (those with word >= m_last, in the last e output slots),
// replaced by the top e class members under (full key descending, index ascending).
template <int T, int IdxBits, int K, int IPT>
__device__ __forceinline__ void
n4_slow_path(const float (&keys)[IPT], uint32_t m_last, int* out_idx, int lane, unsigned mask, int idx_base)
{
  constexpr uint32_t idx_mask = (1u << IdxBits) - 1u;
  uint32_t tw[IPT];
  bool mem[IPT];
  uint32_t cnt = 0;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    tw[i]            = twiddle_f32(keys[i]);
    const uint32_t w = (tw[i] & ~idx_mask) | (idx_mask ^ static_cast<uint32_t>(idx_base + i));
    mem[i]           = ((w ^ m_last) <= idx_mask);
    cnt += (mem[i] && w >= m_last) ? 1u : 0u;
  }
  const int e = static_cast<int>(__reduce_add_sync(mask, cnt));
#pragma unroll 1
  for (int j = 0; j < e; ++j)
  {
    uint32_t cand = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      cand = mem[i] ? ::max(cand, tw[i]) : cand;
    }
    const uint32_t m = __reduce_max_sync(mask, cand);
    uint32_t myidx   = 0xffffffffu;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      myidx = (mem[i] && tw[i] == m) ? ::min(myidx, static_cast<uint32_t>(idx_base + i)) : myidx;
    }
    const uint32_t widx = __reduce_min_sync(mask, myidx);
    if (lane == (j & (T - 1)))
    {
      out_idx[K - e + j] = static_cast<int>(widx);
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      mem[i] = mem[i] && (static_cast<uint32_t>(idx_base + i) != widx);
    }
  }
}

// v1 check: count class members and emitted members over every item, one redux.add
template <int T, int IdxBits, int K, int IPT>
__device__ __forceinline__ void
n4_tie_pass(const float (&keys)[IPT], uint32_t m_last, int* out_idx, int lane, unsigned mask, int idx_base)
{
  constexpr uint32_t idx_mask = (1u << IdxBits) - 1u;
  uint32_t cnt                = 0; // low half: class members, high half: members already emitted
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const uint32_t w  = pack<float, IdxBits>(keys[i], static_cast<uint32_t>(idx_base + i));
    const bool member = ((w ^ m_last) <= idx_mask);
    cnt += member ? (w >= m_last ? 0x10001u : 1u) : 0u;
  }
  const uint32_t tot = __reduce_add_sync(mask, cnt);
  if ((tot & 0xffffu) == (tot >> 16))
  {
    return;
  }
  n4_slow_path<T, IdxBits, K, IPT>(keys, m_last, out_idx, lane, mask, idx_base);
}

// v2 check: the largest word not emitted is the group max over max(head, spare), one redux. A tie
// member was left out exactly when that word has the truncated key of the k-th word.
template <int T, int IdxBits, int K, int IPT, int L>
__device__ __forceinline__ void n4_tie_pass_next(
  const float (&keys)[IPT],
  const uint32_t (&list)[L],
  uint32_t spare,
  uint32_t m_last,
  int* out_idx,
  int lane,
  unsigned mask,
  int idx_base)
{
  constexpr uint32_t idx_mask = (1u << IdxBits) - 1u;
  const uint32_t m_next       = __reduce_max_sync(mask, ::max(list[0], spare));
  if ((m_next ^ m_last) > idx_mask)
  {
    return;
  }
  n4_slow_path<T, IdxBits, K, IPT>(keys, m_last, out_idx, lane, mask, idx_base);
}

// ------------------------------------------------------------------------------------------------
// The group primitive: T lanes hold N keys blocked (lane t has items t*IPT .. t*IPT+IPT-1), write
// K indices to out_idx.
// ------------------------------------------------------------------------------------------------
template <typename KeyT, int T, int N, int K, class Policy = policy_v1>
struct primitive
{
  static constexpr int IPT     = N / T;
  static constexpr int IdxBits = ceil_log2(N);
  static constexpr int L       = next_pow2(cmin(K, IPT));
  static constexpr int P       = topk::round_up(IPT, L);

  __device__ __forceinline__ static void run(const KeyT (&keys)[IPT], int* out_idx)
  {
    const int lane      = topk::group_lane<T>();
    const unsigned mask = topk::member_mask<T>();
    const int idx_base  = lane * IPT;
    uint32_t w[P];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      w[i] = pack<KeyT, IdxBits>(keys[i], static_cast<uint32_t>(idx_base + i));
    }
#pragma unroll
    for (int i = IPT; i < P; ++i)
    {
      w[i] = 0u;
    }
    constexpr bool track_spare = std::is_same_v<KeyT, float> && Policy::next_word_check;
    uint32_t list[L];
    uint32_t spare;
    select_top<L, P, Policy::oddeven_sort, track_spare>(w, list, spare);
    const uint32_t m_last = extract_rounds<T, K, L>(list, lane, mask, emit_index<IdxBits>{out_idx});
    if constexpr (std::is_same_v<KeyT, float>)
    {
      if constexpr (Policy::next_word_check)
      {
        n4_tie_pass_next<T, IdxBits, K, IPT, L>(keys, list, spare, m_last, out_idx, lane, mask, idx_base);
      }
      else
      {
        n4_tie_pass<T, IdxBits, K, IPT>(keys, m_last, out_idx, lane, mask, idx_base);
      }
    }
  }
};

// Compile-time support predicate shared by the adapters
__host__ __device__ constexpr bool
supported_shape(topk::key_type kt, int threads, int n, int k, topk::requirement req, int min_threads, int max_threads)
{
  const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
  const bool thread_ok = (threads >= min_threads) && (threads <= max_threads) && ((threads & (threads - 1)) == 0);
  const bool shape_ok  = (n >= threads) && (n % threads == 0) && (k >= 1) && (2 * k <= n);
  // bf16 keeps 16 key bits above a 16-bit index, fp32 keeps at least 17 key bits
  const bool width_ok = (kt == topk::key_type::bf16) ? (n <= 65536) : (n <= 32768);
  const bool req_ok   = (req != topk::requirement::prefer_larger_index);
  return key_ok && thread_ok && shape_ok && width_ok && req_ok;
}

} // namespace w2
