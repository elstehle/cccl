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
#pragma once

#include <cuda/__utility/static_for.h>

#include "w1_common.cuh"

namespace w1
{
template <typename W, int LW, int M, int T>
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

  // One compare-exchange stage of the bitonic network over one tile (words k[0..M)): partners at
  // distance Stride inside blocks of Size positions. The block direction alternates with the Size
  // bit of the position, DescTile flips the whole tile.
  template <int Size, int Stride, bool DescTile>
  __device__ __forceinline__ static void stage(W* k, int lane, unsigned mask)
  {
    if constexpr (Stride >= LW)
    {
      constexpr int rs = Stride / LW;
      ::cuda::static_for<M>([&](auto ri) {
        constexpr int r = ri();
        if constexpr ((r & rs) == 0)
        {
          constexpr bool desc = (((r * LW) & Size) != 0) != DescTile;
          word_ops<W>::template exchange_regs<desc>(k[r], k[r | rs]);
        }
      });
    }
    else
    {
      const bool upper = (lane & Stride) != 0;
      ::cuda::static_for<M>([&](auto ri) {
        constexpr int r = ri();
        bool desc;
        if constexpr (Size >= L)
        {
          desc = DescTile;
        }
        else if constexpr (Size >= LW)
        {
          desc = (((r * LW) & Size) != 0) != DescTile;
        }
        else
        {
          desc = ((lane & Size) != 0) != DescTile;
        }
        const bool keep_max = upper != desc;
        const W other       = shfl_xor(k[r], Stride, mask);
        word_ops<W>::exchange(k[r], other, keep_max);
      });
    }
  }

  __device__ __forceinline__ static void run(W (&k)[R], int lane, unsigned mask)
  {
    // phase 1: sort every tile, stages interleaved across tiles
    ::cuda::static_for<log_l>([&](auto si) {
      constexpr int Size = 2 << si();
      ::cuda::static_for<si() + 1>([&](auto sj) {
        constexpr int Stride = Size >> (1 + sj());
        ::cuda::static_for<T>([&](auto ti) {
          stage<Size, Stride, tile_desc(ti())>(k + ti() * M, lane, mask);
        });
      });
    });
    // phase 2: halve the tile count per level
    ::cuda::static_for<log_t>([&](auto li) {
      constexpr int Tn = T >> (li() + 1);
      ::cuda::static_for<Tn>([&](auto ji) {
        constexpr int j = ji();
        ::cuda::static_for<M>([&](auto ri) {
          constexpr int r = ri();
          k[j * M + r]    = word_ops<W>::max(k[2 * j * M + r], k[(2 * j + 1) * M + r]);
        });
      });
      ::cuda::static_for<log_l>([&](auto sj) {
        constexpr int Stride = L >> (1 + sj());
        ::cuda::static_for<Tn>([&](auto ji) {
          stage<L, Stride, tile_desc(ji())>(k + ji() * M, lane, mask);
        });
      });
    });
  }
};
} // namespace w1
