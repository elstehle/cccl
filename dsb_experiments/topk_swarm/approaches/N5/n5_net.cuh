// The eager bitonic top-k network of approaches/W1/warp_level_topk.cuh (opt-W1's template
// recursion, the eager tile loop of cub::detail::WarpBitonicTopK) copied for N5 with one change:
// the stage direction is a single bit test on a precomputed per-lane word instead of a function
// of two lane bits recomputed per stage.
//
// Why: with the predicated bf16x2 exchange (two HMNMX2 under a branch that ptxas if-converts,
// see n5_detail.cuh ops_bf16) the direction predicate of every stage is rebuilt from the lane id
// as two half-rate instructions (xor of two shifted lane copies, then a bit test), where the
// one-word VIMNMX route gets one LOP3 straight into a predicate register (T42 SASS, T47 pipe
// rates). Every direction of the network is the parity of at most two lane bits, bit s and bit
// s + d: the word x[d] = lane ^ (lane >> d) holds all parities of distance d at bit position s,
// and x[0] = lane holds the single-bit directions of the merge stages. Four words at 32 lanes,
// computed once per call (eight instructions, loop invariant for the throughput kernel), and
// every stage direction is one `(x[d] >> s) & 1` test.
//
// Everything else is the W1 network unchanged: striped layout (word r of lane l is position
// r * LW + l), MaxK = M * LW words per tile, T tiles, the exchange forms through the Ops policy.
#pragma once

#include "../W1/w1_common.cuh"

namespace n5
{
template <int LW>
struct dir_words
{
  static constexpr int log_lw = w1::log2_int(LW);
  uint32_t x[log_lw > 0 ? log_lw : 1];

  __device__ __forceinline__ explicit dir_words(int lane)
  {
    x[0] = static_cast<uint32_t>(lane);
#pragma unroll
    for (int d = 1; d < log_lw; ++d)
    {
      x[d] = static_cast<uint32_t>(lane) ^ (static_cast<uint32_t>(lane) >> d);
    }
  }
  // parity of lane bits S and S + D (D 0: bit S alone)
  template <int D, int S>
  __device__ __forceinline__ bool parity() const
  {
    static_assert(D >= 0 && D < log_lw + 1 && S >= 0 && S + D < log_lw, "lane bits out of range");
    return ((x[D] >> S) & 1u) != 0;
  }
};

template <typename W, int LW, int M, int T, typename Shfl, typename Ops>
struct level_topk_dir
{
  static_assert(w1::is_pow2(LW) && w1::is_pow2(M) && w1::is_pow2(T),
                "logical warp, tile words and tile count must be powers of two");
  static constexpr int L     = M * LW; // words per tile (MaxK)
  static constexpr int R     = M * T; // words per lane
  static constexpr int log_l = w1::log2_int(L);
  static constexpr int log_t = w1::log2_int(T);
  using dir_t                = dir_words<LW>;

  __host__ __device__ static constexpr bool tile_desc(int t)
  {
    return (t & 1) == 0;
  }

  // keep_max of the shuffle exchange at (Size, Stride < LW) for word Rr of a tile with direction
  // DescTile: upper = lane bit log2(Stride), desc = DescTile for Size >= L, DescTile xor a
  // compile-time bit of the word index for LW <= Size < L, DescTile xor lane bit log2(Size) for
  // Size < LW. keep_max = upper != desc.
  template <int Size, int Stride, bool DescTile, int Rr>
  __device__ __forceinline__ static bool keep_max(const dir_t& dir)
  {
    constexpr int s = w1::log2_int(Stride);
    if constexpr (Size >= LW)
    {
      constexpr bool desc_c = (Size >= L) ? DescTile : ((((Rr * LW) & Size) != 0) != DescTile);
      return dir.template parity<0, s>() != desc_c;
    }
    else
    {
      constexpr int z = w1::log2_int(Size);
      return dir.template parity<z - s, s>() != DescTile;
    }
  }

  template <int Base, int Size, int Stride, bool DescTile, int Rr>
  __device__ __forceinline__ static void stage_word(W (&k)[R], const dir_t& dir, unsigned mask)
  {
    if constexpr (Rr < M)
    {
      if constexpr (Stride >= LW)
      {
        constexpr int rs = Stride / LW;
        if constexpr ((Rr & rs) == 0)
        {
          constexpr bool desc = (((Rr * LW) & Size) != 0) != DescTile;
          Ops::exchange_regs(k[Base + Rr], k[Base + (Rr | rs)], desc);
        }
      }
      else
      {
        const bool km = keep_max<Size, Stride, DescTile, Rr>(dir);
        const W other = Shfl::xor_(k[Base + Rr], Stride, mask);
        Ops::exchange(k[Base + Rr], other, km);
      }
      stage_word<Base, Size, Stride, DescTile, Rr + 1>(k, dir, mask);
    }
  }
  template <int Base, int Size, int Stride, bool DescTile>
  __device__ __forceinline__ static void stage(W (&k)[R], const dir_t& dir, unsigned mask)
  {
    stage_word<Base, Size, Stride, DescTile, 0>(k, dir, mask);
  }
};

// The eager tile loop (sort tile 0 descending, then per tile: sort ascending, keep the larger word
// per position, merge tile 0 descending), as w1::eager_topk.
template <typename W, int LW, int M, int T, typename Shfl, typename Ops>
struct eager_topk_dir
{
  using net              = level_topk_dir<W, LW, M, T, Shfl, Ops>;
  using dir_t            = typename net::dir_t;
  static constexpr int L = net::L;
  static constexpr int R = net::R;

  template <int TI, bool Desc, int Size, int Stride>
  __device__ __forceinline__ static void sort_tile_strides(W (&k)[R], const dir_t& dir, unsigned mask)
  {
    if constexpr (Stride >= 1)
    {
      net::template stage<TI * M, Size, Stride, Desc>(k, dir, mask);
      sort_tile_strides<TI, Desc, Size, Stride / 2>(k, dir, mask);
    }
  }
  template <int TI, bool Desc, int Size>
  __device__ __forceinline__ static void sort_tile_sizes(W (&k)[R], const dir_t& dir, unsigned mask)
  {
    if constexpr (Size <= L)
    {
      sort_tile_strides<TI, Desc, Size, Size / 2>(k, dir, mask);
      sort_tile_sizes<TI, Desc, Size * 2>(k, dir, mask);
    }
  }
  template <int TI, int Rr>
  __device__ __forceinline__ static void pair_tile(W (&k)[R])
  {
    if constexpr (Rr < M)
    {
      k[Rr] = Ops::max(k[Rr], k[TI * M + Rr]);
      pair_tile<TI, Rr + 1>(k);
    }
  }
  template <int Stride>
  __device__ __forceinline__ static void merge_tile0(W (&k)[R], const dir_t& dir, unsigned mask)
  {
    if constexpr (Stride >= 1)
    {
      net::template stage<0, L, Stride, true>(k, dir, mask);
      merge_tile0<Stride / 2>(k, dir, mask);
    }
  }
  template <int TI>
  __device__ __forceinline__ static void tiles(W (&k)[R], const dir_t& dir, unsigned mask)
  {
    if constexpr (TI < T)
    {
      sort_tile_sizes<TI, false, 2>(k, dir, mask);
      pair_tile<TI, 0>(k);
      merge_tile0<L / 2>(k, dir, mask);
      tiles<TI + 1>(k, dir, mask);
    }
  }

  __device__ __forceinline__ static void run(W (&k)[R], const dir_t& dir, unsigned mask)
  {
    sort_tile_sizes<0, true, 2>(k, dir, mask);
    tiles<1>(k, dir, mask);
  }
};
} // namespace n5
