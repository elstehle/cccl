// Implementation of idea N7 (two-level per-lane list for the packed W2 rounds), warp scope. Not an
// adapter by itself. Builds on W2_impl.cuh: the same packed words, the same redux round, the same
// fp32 route (idea N4) and slow path.
//
// Per lane the tile of IPT words is split into a sorted head of H words (positions 0 to H - 1 of
// the register array) and an unsorted remainder (positions H to IPT - 1). The prologue sorts block
// 0 with the odd-even network and merges every further block of H into the head with a
// write-back merge: elementwise max with the reversed sorted block, the mins written back into the
// block, then a half-cleaner on the head. The remainder therefore holds exactly the words that are
// not in the head, and the head is the lane's sorted top H.
//
// Rounds are W2's (one redux of the heads, the unique winner shifts its head by one), so the round
// costs the depth-8 price (T19, T41) at any items per lane. The loop over blocks is rolled so the
// refill network is emitted once. The head of a lane can only run dry after H wins. Rounds run in
// blocks of B (H >= 4 B). At the top of block b every lane tests whether it had fewer than 2 B
// valid head words at the end of block b - 1 and still has remainder words, one __any_sync over
// the warp decides, and the refill (every remainder block sorted again and merged into the head
// with write-back) runs under that warp-uniform branch after the rounds of block b. Invariant: at
// the start of every block each lane holds at least B valid head words or an empty remainder, so
// no lane can run dry inside a block, and the vote of block b overlaps the rounds of block b
// instead of sitting on the redux chain. A refill needs H - 2 B + 1 wins of one lane since its
// last refill (five at H 8, B 2), rare on random input.
//
// fp32: after the rounds the largest word not emitted is the warp max over max(head[0], remainder
// max), W2's next-word check decides the slow path, which is W2's (blocked arrangement).
// NaN and minus zero as W2_impl.cuh.
#pragma once

#include <cstdint>
#include <type_traits>

#include "../topk_harness.cuh"
#include "W2_impl.cuh"

namespace n7
{

// Merge the sorted descending block a[LO, LO + H) into the sorted descending head a[0, H): the head
// becomes the top H of both, the block receives the dropped words.
template <int H, int LO, int P>
__device__ __forceinline__ void merge_block_writeback(uint32_t (&a)[P])
{
#pragma unroll
  for (int i = 0; i < H; ++i)
  {
    const uint32_t x  = a[LO + H - 1 - i];
    const uint32_t hi = ::max(a[i], x);
    a[LO + H - 1 - i] = ::min(a[i], x);
    a[i]              = hi;
  }
#pragma unroll
  for (int stride = H / 2; stride > 0; stride >>= 1)
  {
#pragma unroll
    for (int i = 0; i < H; ++i)
    {
      const int p = i ^ stride;
      if (p > i)
      {
        w2::cswap_desc(a[i], a[p]);
      }
    }
  }
}

// Sort every remainder block from G on and merge it into the head with write-back
template <int H, int G, int P>
__device__ __forceinline__ void refill_blocks(uint32_t (&a)[P])
{
  if constexpr (G < P / H)
  {
    w2::oe_sort_desc<G * H, G * H + H - 1>(a);
    merge_block_writeback<H, G * H>(a);
    refill_blocks<H, G + 1>(a);
  }
}

template <int H, int P>
__device__ __forceinline__ int head_count(const uint32_t (&a)[P])
{
  int c = 0;
#pragma unroll
  for (int i = 0; i < H; ++i)
  {
    c += (a[i] != 0u) ? 1 : 0;
  }
  return c;
}

// One block of B rounds with the lookahead vote, the refill after the block when any lane needs it
template <int T, int H, int B, int P, class Red, class Emit>
__device__ __forceinline__ uint32_t
round_block(uint32_t (&a)[P], int& rem_count, int r0, bool last_block, int lane, const Red& red, Emit emit)
{
  // valid head words below 2 B at the end of the previous block: the lane could run dry in the
  // block after this one (head sorted, zeros only at the tail, so a[2 B - 1] == 0 is the test)
  const bool need     = (a[2 * B - 1] == 0u) && (rem_count > 0);
  const bool any_need = __any_sync(0xffffffffu, need);
  uint32_t m          = 0;
  uint32_t head[H];
#pragma unroll
  for (int i = 0; i < H; ++i)
  {
    head[i] = a[i];
  }
#pragma unroll
  for (int r = 0; r < B; ++r)
  {
    m = w2::one_round<T, H>(head, r0 + r, lane, red, emit);
  }
#pragma unroll
  for (int i = 0; i < H; ++i)
  {
    a[i] = head[i];
  }
  if (any_need && !last_block)
  {
    const int before = head_count<H>(a);
    refill_blocks<H, 1>(a);
    const int after = head_count<H>(a);
    rem_count -= (after - before);
  }
  return m;
}

template <int T, int K, int H, int B, int P, class Red, class Emit>
__device__ __forceinline__ uint32_t twolevel_rounds(uint32_t (&a)[P], int lane, const Red& red, Emit emit)
{
  static_assert(K % B == 0, "k must be a multiple of the block length");
  constexpr int NB = K / B;
  int rem_count    = P - H;
  uint32_t m       = 0;
  // The block loop stays rolled at every k: an unrolled loop inlines the refill network once per
  // block (15 copies at k 64, 3700 min max instructions in the kernel) and the instruction cache
  // then costs more per round than the deep list it replaces.
#pragma unroll 1
  for (int b = 0; b < NB; ++b)
  {
    m = round_block<T, H, B>(a, rem_count, b * B, b + 1 == NB, lane, red, emit);
  }
  return m;
}

// The warp primitive: T lanes hold N keys blocked (lane t has items t*IPT .. t*IPT+IPT-1), write K
// indices to out_idx. H head words, blocks of B rounds, H == 2 B.
template <typename KeyT, int T, int N, int K, int H = 8, int B = 2>
struct primitive
{
  static constexpr int IPT     = N / T;
  static constexpr int IdxBits = w2::ceil_log2(N);
  static_assert(IPT % H == 0 && IPT >= 2 * H, "the two-level list needs at least two blocks of H per lane");
  // The refill is flagged when a lane has fewer than 2 B valid head words, that is after H - 2 B + 1
  // wins since its last refill. With H 8 and B 4 one win is enough and the refill ran in nearly
  // every block (measured 70 to 190 cycles per round, batch n7b), with B 2 it takes five wins.
  static_assert(H >= 4 * B, "the head must hold four blocks of rounds for the refill to be rare");

  __device__ __forceinline__ static void run(const KeyT (&keys)[IPT], int* out_idx)
  {
    const int lane     = topk::group_lane<T>();
    const int idx_base = lane * IPT;
    const w2::masked_reducer<T> red{topk::member_mask<T>()};
    uint32_t a[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      a[i] = w2::pack<KeyT, IdxBits>(keys[i], static_cast<uint32_t>(idx_base + i));
    }
    w2::oe_sort_desc<0, H - 1>(a);
    refill_blocks<H, 1>(a);
    const uint32_t m_last = twolevel_rounds<T, K, H, B>(a, lane, red, w2::emit_index<IdxBits>{out_idx});
    if constexpr (std::is_same_v<KeyT, float>)
    {
      constexpr uint32_t idx_mask = (1u << IdxBits) - 1u;
      uint32_t rest               = a[0];
#pragma unroll
      for (int i = H; i < IPT; ++i)
      {
        rest = ::max(rest, a[i]);
      }
      const uint32_t m_next = red.max(rest);
      if ((m_next ^ m_last) <= idx_mask)
      {
        w2::n4_slow_path<T, IdxBits, K, IPT>(keys, m_last, out_idx, lane, red, idx_base);
      }
    }
  }
};

__host__ __device__ constexpr bool supported_shape(topk::key_type kt, int threads, int n, int k, topk::requirement req)
{
  // at least two blocks of 8 per lane, k a multiple of the block length 4, rounds rolled above 64
  return w2::supported_shape(kt, threads, n, k, req, 32, 32) && (n / threads >= 16) && (k % 4 == 0) && (k <= 256);
}

} // namespace n7
