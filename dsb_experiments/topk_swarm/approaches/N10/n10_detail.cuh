// N10: fp32 keys-only bitonic top-k network with the indices recovered afterwards by ballot rank.
//
// The fp32 key is twiddled into an order-preserving unsigned 32-bit word (sign flip, -0.0 mapped
// to +0.0) and the words run through the min and max exchange network of approaches/W1
// (warp_level_topk.cuh stage machinery, word_ops<uint32_t>, T42): one shuffle and two predicated
// VIMNMX per shuffle stage, no index moves. After the network the k-th largest word t is known:
//
//   full      : the final merge of tile 0 is complete, t is the word at position k - 1 (one shuffle)
//   truncated : the final merge of tile 0 stops after the half-cleaner stages with stride >= k,
//               positions [0, k) then hold the k largest words as a bitonic sequence and t is their
//               minimum (one redux.min, full warp, the identity in the lanes of other logical warps)
//
// Recovery: every lane compares its original words against t. One ballot per slot for "above t"
// and one for "equal to t" (slot-major index order), popc prefixes give every item above t a unique
// output slot in index order and the items equal to t take the next slots in index order until k
// is reached. That is exact prefer_smaller_index by construction, and the same code serves
// not_guaranteed. Output order: the items above t in index order, then the tied items in index
// order (not sorted by key).
//
// Logical warps of 4, 8, 16 and 32 lanes. All lanes of the hardware warp run the same code in
// lockstep (the harness guarantees it), so shuffles, ballots and redux use the full warp mask with
// the segment handled by masks and identities (T37, T40), never a per-group collective mask.
//
// NaN: a NaN twiddles above +inf (positive sign) or below -inf (negative sign) and is treated as
// an ordinary word, so it is selected or dropped deterministically by its sign bit. Focus-set
// inputs contain no NaN. -0.0 and +0.0 are the same word.
#pragma once

#include <cstdint>
#include <type_traits>

#include "../W1/w1_common.cuh"
#include "../W1/warp_level_topk.cuh"
#include "topk_harness.cuh"

namespace n10
{
__host__ __device__ inline constexpr int next_pow2(int x)
{
  int p = 1;
  while (p < x)
  {
    p *= 2;
  }
  return p;
}
__host__ __device__ inline constexpr int max_int(int a, int b)
{
  return a > b ? a : b;
}
__host__ __device__ inline constexpr int max_k_of(int threads, int k)
{
  return max_int(threads, next_pow2(k));
}

__device__ __forceinline__ uint32_t twiddle(float f)
{
  uint32_t u          = __float_as_uint(f);
  u                   = (u == 0x80000000u) ? 0u : u; // -0.0 equals +0.0
  const uint32_t sign = 0u - (u >> 31); // all ones for a negative key
  return u ^ (sign | 0x80000000u);
}

// Minimum over the caller's logical warp of T lanes: one full-warp redux for T 32, one full-warp
// redux per logical warp with the identity in the other lanes below that (T37, T40).
template <int T>
__device__ __forceinline__ uint32_t group_min(uint32_t v, int group)
{
  if constexpr (T >= 32)
  {
    return __reduce_min_sync(0xffffffffu, v);
  }
  else
  {
    constexpr int G = 32 / T;
    uint32_t m      = 0xffffffffu;
#pragma unroll
    for (int g = 0; g < G; ++g)
    {
      const uint32_t r = __reduce_min_sync(0xffffffffu, (group == g) ? v : 0xffffffffu);
      m                = (group == g) ? r : m;
    }
    return m;
  }
}

// ------------------------------------------------------------------------------------------------
// Network drivers on the stage machinery of w1::level_topk. L = M x LW words per tile, T tiles,
// R = M x T words per lane, striped positions r x LW + lane. The final merge of tile 0 (size L,
// descending) runs the strides L/2 down to MinStride only: MinStride 1 is the full merge,
// MinStride 2^j leaves the 2^j largest words in positions [0, 2^j) as a bitonic sequence.
// ------------------------------------------------------------------------------------------------
template <int LW, int M, int T, int MinStride, typename Shfl = w1::shfl_split32, typename Ops = w1::word_ops<uint32_t>>
struct common_net
{
  using W                = uint32_t;
  using net              = w1::level_topk<W, LW, M, T, Shfl, Ops>;
  static constexpr int L = net::L;
  static constexpr int R = net::R;
  static_assert(MinStride >= 1 && MinStride <= L && w1::is_pow2(MinStride), "MinStride must be a power of two up to L");

  template <int TI, bool Desc, int Size, int Stride>
  __device__ __forceinline__ static void sort_strides(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (Stride >= 1)
    {
      net::template stage<TI * M, Size, Stride, Desc>(k, lane, mask);
      sort_strides<TI, Desc, Size, Stride / 2>(k, lane, mask);
    }
  }
  // bitonic sort of tile TI for the sizes Size .. MaxSize (MaxSize L is the complete sort)
  template <int TI, bool Desc, int Size, int MaxSize>
  __device__ __forceinline__ static void sort_sizes(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (Size <= MaxSize)
    {
      sort_strides<TI, Desc, Size, Size / 2>(k, lane, mask);
      sort_sizes<TI, Desc, Size * 2, MaxSize>(k, lane, mask);
    }
  }
  // final merge of tile 0, descending, strides Stride down to MinS
  template <int Stride, int MinS>
  __device__ __forceinline__ static void merge0(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (Stride >= MinS)
    {
      net::template stage<0, L, Stride, true>(k, lane, mask);
      merge0<Stride / 2, MinS>(k, lane, mask);
    }
  }
  // one tile only: sort the sizes below L, then the truncated size L merge
  __device__ __forceinline__ static void single_tile(W (&k)[R], int lane, unsigned mask)
  {
    sort_sizes<0, true, 2, L / 2>(k, lane, mask);
    merge0<L / 2, MinStride>(k, lane, mask);
  }
};

// Eager tile loop (the cub::detail::WarpBitonicTopK order): sort tile 0 descending, then per tile
// sort it ascending, keep the larger word per position, merge tile 0 descending. The last merge is
// truncated.
template <int LW, int M, int T, int MinStride, typename Shfl = w1::shfl_split32, typename Ops = w1::word_ops<uint32_t>>
struct eager_net : common_net<LW, M, T, MinStride, Shfl, Ops>
{
  using base = common_net<LW, M, T, MinStride, Shfl, Ops>;
  using W    = typename base::W;
  using net  = typename base::net;
  static constexpr int L = base::L;
  static constexpr int R = base::R;

  template <int TI, int Rr>
  __device__ __forceinline__ static void pair_tile(W (&k)[R])
  {
    if constexpr (Rr < M)
    {
      k[Rr] = Ops::max(k[Rr], k[TI * M + Rr]);
      pair_tile<TI, Rr + 1>(k);
    }
  }
  template <int TI>
  __device__ __forceinline__ static void tiles(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (TI < T)
    {
      base::template sort_sizes<TI, false, 2, L>(k, lane, mask);
      pair_tile<TI, 0>(k);
      if constexpr (TI + 1 < T)
      {
        base::template merge0<L / 2, 1>(k, lane, mask);
      }
      else
      {
        base::template merge0<L / 2, MinStride>(k, lane, mask);
      }
      tiles<TI + 1>(k, lane, mask);
    }
  }
  __device__ __forceinline__ static void run(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (T == 1)
    {
      base::single_tile(k, lane, mask);
    }
    else
    {
      base::template sort_sizes<0, true, 2, L>(k, lane, mask);
      tiles<1>(k, lane, mask);
    }
  }
};

// Level order (approaches/W1/warp_level_topk.cuh): sort all tiles at once in alternating
// directions, then halve the tile count per level. The merge of the last level is truncated.
template <int LW, int M, int T, int MinStride, typename Shfl = w1::shfl_split32, typename Ops = w1::word_ops<uint32_t>>
struct level_net : common_net<LW, M, T, MinStride, Shfl, Ops>
{
  using base = common_net<LW, M, T, MinStride, Shfl, Ops>;
  using W    = typename base::W;
  using net  = typename base::net;
  static constexpr int L = base::L;
  static constexpr int R = base::R;

  template <int Tn>
  __device__ __forceinline__ static void levels(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (Tn >= 1)
    {
      net::template pair_tiles<Tn, 0, 0>(k);
      if constexpr (Tn == 1)
      {
        base::template merge0<L / 2, MinStride>(k, lane, mask);
      }
      else
      {
        net::template merge_strides<Tn, L / 2>(k, lane, mask);
        levels<Tn / 2>(k, lane, mask);
      }
    }
  }
  __device__ __forceinline__ static void run(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (T == 1)
    {
      base::single_tile(k, lane, mask);
    }
    else
    {
      net::template sort_sizes<2>(k, lane, mask);
      levels<T / 2>(k, lane, mask);
    }
  }
};

// ------------------------------------------------------------------------------------------------
// Index recovery. w0 are the lane's original words (slot i of lane l is source index i x LW + l),
// t the k-th largest word of the group, seg the group's lane mask, lt the group's lanes below the
// caller. Full-warp ballots, popc prefixes, one predicated store per item.
//
// two_class: one "above t" and one "equal to t" ballot per slot, exact in one pass.
// fast_path: one "at least t" ballot per slot. When the group holds exactly k words at least t
//   (no excess ties at rank k, the random fp32 case) the prefix rank over those words is the output
//   slot. Otherwise a warp-uniform branch runs the two-class pass with the "above" ballots added
//   (the "equal" ballots are then the difference). At sub-warp scope the branch condition is made
//   uniform over the hardware warp with one any-vote, so the collectives inside stay full-warp.
// ------------------------------------------------------------------------------------------------
template <int IPT, int LW, int K>
__device__ __forceinline__ void
emit_two_class(const uint32_t (&w0)[IPT], uint32_t t, const uint32_t (&bgt)[IPT], const uint32_t (&beq)[IPT], unsigned lt, int lane, int* out_idx)
{
  uint32_t above = 0;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    above += static_cast<uint32_t>(__popc(bgt[i]));
  }
  uint32_t pa = 0;
  uint32_t pe = above;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const uint32_t sg = pa + static_cast<uint32_t>(__popc(bgt[i] & lt));
    const uint32_t se = pe + static_cast<uint32_t>(__popc(beq[i] & lt));
    const bool gt     = w0[i] > t;
    const bool eq     = w0[i] == t;
    const bool take   = gt | (eq & (se < static_cast<uint32_t>(K)));
    const int slot    = gt ? static_cast<int>(sg) : static_cast<int>(se);
    if (take)
    {
      out_idx[slot] = i * LW + lane;
    }
    pa += static_cast<uint32_t>(__popc(bgt[i]));
    pe += static_cast<uint32_t>(__popc(beq[i]));
  }
}

template <int IPT, int LW, int K>
__device__ __forceinline__ void
recover_two_class(const uint32_t (&w0)[IPT], uint32_t t, unsigned seg, unsigned lt, int lane, int* out_idx)
{
  uint32_t bgt[IPT];
  uint32_t beq[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    bgt[i] = __ballot_sync(0xffffffffu, w0[i] > t) & seg;
    beq[i] = __ballot_sync(0xffffffffu, w0[i] == t) & seg;
  }
  emit_two_class<IPT, LW, K>(w0, t, bgt, beq, lt, lane, out_idx);
}

template <int IPT, int LW, int K>
__device__ __forceinline__ void
recover_fast_path(const uint32_t (&w0)[IPT], uint32_t t, unsigned seg, unsigned lt, int lane, int* out_idx)
{
  uint32_t bge[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    bge[i] = __ballot_sync(0xffffffffu, w0[i] >= t) & seg;
  }
  uint32_t total = 0;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    total += static_cast<uint32_t>(__popc(bge[i]));
  }
  bool excess = total > static_cast<uint32_t>(K);
  if constexpr (LW < 32)
  {
    excess = __any_sync(0xffffffffu, excess);
  }
  if (!excess)
  {
    uint32_t p = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      const int slot = static_cast<int>(p + static_cast<uint32_t>(__popc(bge[i] & lt)));
      if (w0[i] >= t)
      {
        out_idx[slot] = i * LW + lane;
      }
      p += static_cast<uint32_t>(__popc(bge[i]));
    }
  }
  else
  {
    uint32_t bgt[IPT];
    uint32_t beq[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      bgt[i] = __ballot_sync(0xffffffffu, w0[i] > t) & seg;
      beq[i] = bge[i] & ~bgt[i];
    }
    emit_two_class<IPT, LW, K>(w0, t, bgt, beq, lt, lane, out_idx);
  }
}

// ------------------------------------------------------------------------------------------------
// Adapter body. Trunc selects the truncated final merge plus redux threshold (power of two k only,
// other k fall back to the full merge), Level the level tile order instead of the eager one,
// FastPath the one-ballot recovery with the two-class pass behind a uniform branch.
// ------------------------------------------------------------------------------------------------
__host__ __device__ inline constexpr bool
supported(topk::key_type kt, int threads, int n, int k, topk::requirement req, bool subwarp)
{
  const bool key_ok    = kt == topk::key_type::fp32;
  const bool thread_ok = subwarp ? (threads == 4 || threads == 8 || threads == 16) : (threads == 32);
  const int max_k      = max_k_of(threads, k);
  const bool shape_ok  = (k >= 1) && (n % threads == 0) && (max_k <= n) && (n % max_k == 0) && topk::is_pow2(n / max_k);
  const bool req_ok    = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
  return key_ok && thread_ok && shape_ok && req_ok;
}

template <typename KeyT, int Threads, int N, int K, topk::requirement Req, bool Trunc, bool Level, bool FastPath = false>
struct adapter_impl
{
  static_assert(std::is_same_v<KeyT, float>, "N10 is the fp32 route");
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int LW                              = Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr int max_k                           = max_k_of(Threads, K);
  static constexpr int M                               = max_k / Threads;
  static constexpr int T                               = N / max_k;
  static constexpr int k_top                           = next_pow2(K);
  static constexpr bool trunc                          = Trunc && (k_top == K);
  static constexpr int min_stride                      = trunc ? k_top : 1;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  struct TempStorage
  {};
  using Net = std::conditional_t<Level, level_net<LW, M, T, min_stride>, eager_net<LW, M, T, min_stride>>;
  static_assert(Net::R == IPT, "words per lane must equal items per thread");

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    const int lane     = topk::group_lane<Threads>();
    const int hw       = static_cast<int>(threadIdx.x) & 31;
    const unsigned seg = topk::member_mask<Threads>();
    const unsigned lt  = seg & ((1u << hw) - 1u);
    uint32_t w0[IPT];
    uint32_t w[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      w0[i] = twiddle(keys[i]);
      w[i]  = w0[i];
    }
    Net::run(w, lane, 0xffffffffu);
    uint32_t t;
    if constexpr (trunc)
    {
      uint32_t v;
      if constexpr (K >= LW)
      {
        v = w[0];
#pragma unroll
        for (int r = 1; r < K / LW; ++r)
        {
          v = min(v, w[r]);
        }
      }
      else
      {
        v = (lane < K) ? w[0] : 0xffffffffu;
      }
      t = group_min<Threads>(v, hw / Threads);
    }
    else
    {
      t = __shfl_sync(0xffffffffu, w[(K - 1) / LW], (K - 1) % LW, LW);
    }
    if constexpr (FastPath)
    {
      recover_fast_path<IPT, LW, K>(w0, t, seg, lt, lane, out_idx);
    }
    else
    {
      recover_two_class<IPT, LW, K>(w0, t, seg, lt, lane, out_idx);
    }
  }
};

// The combined form: truncated final merge always, level tile order from four tiles on (at one or
// two tiles the two orders are the same dependency graph), one-ballot recovery from four items per
// lane on (below that the two-class pass is cheaper than the branch).
template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct auto_adapter
    : adapter_impl<KeyT, Threads, N, K, Req, true, (N / max_k_of(Threads, K) >= 4), (N / Threads >= 4)>
{};
} // namespace n10
