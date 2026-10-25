// Level-parallel bitonic top-k on a logical warp (idea card N2, built here as the W1
// restructuring). Words are unsigned integers whose order is the selection order (larger wins),
// see w1_common.cuh for the packing. The layout is striped: word r of lane l is position
// r * LW + l. MaxK = M * LW words form one tile and the input holds T tiles (N = T * MaxK, T a
// power of two).
//
// Phase 1 sorts all T tiles at once: every stage of the bitonic sorting network is issued for all
// tiles before the next stage, so the T independent dependency chains overlap. Even tiles sort
// descending, odd tiles ascending.
// Phase 2 halves the tile count per level: tile j becomes the elementwise max of tiles 2j and
// 2j+1. The two are sorted in opposite directions, so the max is the larger half of their union as
// a bitonic sequence (it is the first half-cleaner of their concatenation), and one bitonic merge
// of log2(MaxK) stages sorts it, again in alternating directions. After log2(T) levels tile 0
// holds the MaxK largest words in descending order and the first k positions are the result.
//
// Stage shapes: a compare-exchange at stride >= LW is between two words of the same lane (no
// shuffle), at stride < LW it is a shuffle exchange with the partner lane.
//
// Code shape: template recursion only, every array index is a compile-time constant, no lambdas
// and no loops. A lambda that captures the word array by reference and is not inlined, or a loop
// that is not unrolled, makes the array escape to local memory, which happened in the latency
// kernel of the first versions at N 256 and at 32 items per lane.
#pragma once

#include "w1_common.cuh"

namespace w1
{
template <typename W, int LW, int M, int T, typename Shfl = shfl_split32>
struct level_topk
{
  static_assert(is_pow2(LW) && is_pow2(M) && is_pow2(T), "logical warp, tile words and tile count must be powers of two");
  static constexpr int L     = M * LW; // words per tile (MaxK)
  static constexpr int R     = M * T; // words per lane
  static constexpr int log_l = log2_int(L);
  static constexpr int log_t = log2_int(T);

  __host__ __device__ static constexpr bool tile_desc(int t)
  {
    return (t & 1) == 0;
  }

  // One compare-exchange stage of the bitonic network over the tile at words k[Base..Base+M):
  // partners at distance Stride inside blocks of Size positions. The block direction alternates
  // with the Size bit of the position, DescTile flips the whole tile. Word r of the tile is handled
  // by stage_word, recursion over r keeps every array index a compile-time constant.
  template <int Base, int Size, int Stride, bool DescTile, int Rr>
  __device__ __forceinline__ static void stage_word(W (&k)[R], int lane, unsigned mask, bool upper)
  {
    if constexpr (Rr < M)
    {
      if constexpr (Stride >= LW)
      {
        constexpr int rs = Stride / LW;
        if constexpr ((Rr & rs) == 0)
        {
          constexpr bool desc = (((Rr * LW) & Size) != 0) != DescTile;
          word_ops<W>::exchange_regs(k[Base + Rr], k[Base + (Rr | rs)], desc);
        }
      }
      else
      {
        bool desc;
        if constexpr (Size >= L)
        {
          desc = DescTile;
        }
        else if constexpr (Size >= LW)
        {
          desc = (((Rr * LW) & Size) != 0) != DescTile;
        }
        else
        {
          desc = ((lane & Size) != 0) != DescTile;
        }
        const bool keep_max = upper != desc;
        const W other       = Shfl::xor_(k[Base + Rr], Stride, mask);
        word_ops<W>::exchange(k[Base + Rr], other, keep_max);
      }
      stage_word<Base, Size, Stride, DescTile, Rr + 1>(k, lane, mask, upper);
    }
  }
  template <int Base, int Size, int Stride, bool DescTile>
  __device__ __forceinline__ static void stage(W (&k)[R], int lane, unsigned mask)
  {
    const bool upper = (lane & Stride) != 0;
    stage_word<Base, Size, Stride, DescTile, 0>(k, lane, mask, upper);
  }

  // phase 1: one stage for every tile, all strides of one size, all sizes
  template <int TI, int Size, int Stride>
  __device__ __forceinline__ static void stage_all_tiles(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (TI < T)
    {
      stage<TI * M, Size, Stride, tile_desc(TI)>(k, lane, mask);
      stage_all_tiles<TI + 1, Size, Stride>(k, lane, mask);
    }
  }
  template <int Size, int Stride>
  __device__ __forceinline__ static void sort_strides(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (Stride >= 1)
    {
      stage_all_tiles<0, Size, Stride>(k, lane, mask);
      sort_strides<Size, Stride / 2>(k, lane, mask);
    }
  }
  template <int Size>
  __device__ __forceinline__ static void sort_sizes(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (Size <= L)
    {
      sort_strides<Size, Size / 2>(k, lane, mask);
      sort_sizes<Size * 2>(k, lane, mask);
    }
  }

  // phase 2: pair the tiles, merge the survivors with the stages interleaved, repeat
  template <int JI, int Tn, int Stride>
  __device__ __forceinline__ static void merge_all_tiles(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (JI < Tn)
    {
      stage<JI * M, L, Stride, tile_desc(JI)>(k, lane, mask);
      merge_all_tiles<JI + 1, Tn, Stride>(k, lane, mask);
    }
  }
  template <int Tn, int Stride>
  __device__ __forceinline__ static void merge_strides(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (Stride >= 1)
    {
      merge_all_tiles<0, Tn, Stride>(k, lane, mask);
      merge_strides<Tn, Stride / 2>(k, lane, mask);
    }
  }
  template <int Tn, int JI, int Rr>
  __device__ __forceinline__ static void pair_tiles(W (&k)[R])
  {
    if constexpr (JI < Tn)
    {
      if constexpr (Rr < M)
      {
        k[JI * M + Rr] = word_ops<W>::max(k[2 * JI * M + Rr], k[(2 * JI + 1) * M + Rr]);
        pair_tiles<Tn, JI, Rr + 1>(k);
      }
      else
      {
        pair_tiles<Tn, JI + 1, 0>(k);
      }
    }
  }
  template <int Tn>
  __device__ __forceinline__ static void levels(W (&k)[R], int lane, unsigned mask)
  {
    if constexpr (Tn >= 1)
    {
      pair_tiles<Tn, 0, 0>(k);
      merge_strides<Tn, L / 2>(k, lane, mask);
      levels<Tn / 2>(k, lane, mask);
    }
  }

  __device__ __forceinline__ static void run(W (&k)[R], int lane, unsigned mask)
  {
    sort_sizes<2>(k, lane, mask);
    levels<T / 2>(k, lane, mask);
  }
};
} // namespace w1
