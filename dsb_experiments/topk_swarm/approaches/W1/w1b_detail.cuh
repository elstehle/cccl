// W1b shared pieces: a register-resident bitonic network over an "element policy", the eager and the
// level-parallel (N2) top-k loops on top of it, and the key packings (bf16 key with inverted index in
// one 32-bit word, fp32 key with inverted index in one 64-bit word). Not an adapter: the adapters
// W1_<variant>.cuh include this file and pick a policy family and a loop.
//
// Conventions (same as cub::detail::WarpBitonicSortNetwork): a tile of M registers over LW lanes holds
// position p = i * LW + lane in register i. "Forward" order puts the element that ranks first (the
// largest key, and for equal keys the smaller index when the policy encodes it) at position 0.
// Reverse order is the mirror image. A compare-exchange is written so that the lane keeping the
// larger element and the lane keeping the smaller one both keep their own element on a tie, which
// keeps pairs consistent without a second comparison.
//
// NaN: fp32 keys compare with fmaxf and fminf (a NaN loses against any number) or through the
// order-preserving integer twiddle (a NaN ranks above +inf). bf16 keys go through the twiddle only.
// Focus-set inputs contain no NaN. -0.0 is canonicalized to +0.0 before packing, so the two zeros
// are equal under every requirement.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

#include "topk_harness.cuh"

namespace w1b
{

__host__ __device__ inline constexpr int log2_int(int x)
{
  int r = 0;
  while ((1 << r) < x)
  {
    ++r;
  }
  return r;
}
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

// ------------------------------------------------------------------------------------------------
// Shuffles on the logical warp
// ------------------------------------------------------------------------------------------------
template <int LW>
__device__ __forceinline__ uint32_t shfl_xor_u32(uint32_t v, int stride, unsigned mask)
{
  return __shfl_xor_sync(mask, v, stride, LW);
}

// ------------------------------------------------------------------------------------------------
// Element policies
//   elem_t          the network element
//   shfl_xor(e, s, mask)  partner's element
//   cx(mine, other, keep_min)  mine becomes the smaller (keep_min) or the larger of the two, own on tie
//   cx2(a, b)       in-register exchange, a becomes the larger, b the smaller
//   index(e)        the source index of the element
// ------------------------------------------------------------------------------------------------

// One 32-bit word per element: bits 31..16 the twiddled bf16 key, bits 15..0 the inverted index.
// Words are unique, so the larger word ranks first and equal keys resolve to the smaller index.
template <int LW>
struct policy_u32
{
  using elem_t = uint32_t;

  __device__ __forceinline__ static elem_t shfl_xor(elem_t e, int stride, unsigned mask)
  {
    return shfl_xor_u32<LW>(e, stride, mask);
  }
  __device__ __forceinline__ static void cx(elem_t& mine, elem_t other, bool keep_min)
  {
    const bool take = (other > mine) != keep_min;
    mine            = take ? other : mine;
  }
  __device__ __forceinline__ static void cx2(elem_t& a, elem_t& b)
  {
    const elem_t hi = max(a, b);
    const elem_t lo = min(a, b);
    a               = hi;
    b               = lo;
  }
  __device__ __forceinline__ static int index(elem_t e)
  {
    return 0xffff - static_cast<int>(e & 0xffffu);
  }
};

// One 64-bit word per element: bits 63..32 the twiddled fp32 key, bits 31..0 the inverted index.
template <int LW>
struct policy_u64
{
  using elem_t = unsigned long long;

  __device__ __forceinline__ static elem_t shfl_xor(elem_t e, int stride, unsigned mask)
  {
    const uint32_t lo = shfl_xor_u32<LW>(static_cast<uint32_t>(e), stride, mask);
    const uint32_t hi = shfl_xor_u32<LW>(static_cast<uint32_t>(e >> 32), stride, mask);
    return (static_cast<elem_t>(hi) << 32) | lo;
  }
  __device__ __forceinline__ static void cx(elem_t& mine, elem_t other, bool keep_min)
  {
    const bool take = (other > mine) != keep_min;
    mine            = take ? other : mine;
  }
  __device__ __forceinline__ static void cx2(elem_t& a, elem_t& b)
  {
    const elem_t hi = max(a, b);
    const elem_t lo = min(a, b);
    a               = hi;
    b               = lo;
  }
  __device__ __forceinline__ static int index(elem_t e)
  {
    return static_cast<int>(0xffffffffu - static_cast<uint32_t>(e));
  }
};

// fp32 key and int32 index as a pair, keys compared natively with fmaxf and fminf, the index follows
// the key (the lane whose key changed takes the partner's index). Ties keep both elements in place,
// not_guaranteed only.
struct pair_f32
{
  float k;
  int i;
};

template <int LW>
struct policy_pair_f32
{
  using elem_t = pair_f32;

  __device__ __forceinline__ static elem_t shfl_xor(elem_t e, int stride, unsigned mask)
  {
    elem_t r;
    r.k = __uint_as_float(shfl_xor_u32<LW>(__float_as_uint(e.k), stride, mask));
    r.i = static_cast<int>(shfl_xor_u32<LW>(static_cast<uint32_t>(e.i), stride, mask));
    return r;
  }
  __device__ __forceinline__ static void cx(elem_t& mine, elem_t other, bool keep_min)
  {
    const float nk  = keep_min ? fminf(mine.k, other.k) : fmaxf(mine.k, other.k);
    const bool take = nk != mine.k;
    mine.i          = take ? other.i : mine.i;
    mine.k          = nk;
  }
  __device__ __forceinline__ static void cx2(elem_t& a, elem_t& b)
  {
    const bool swap = b.k > a.k;
    const float hi  = fmaxf(a.k, b.k);
    const float lo  = fminf(a.k, b.k);
    const int ai    = swap ? b.i : a.i;
    const int bi    = swap ? a.i : b.i;
    a.k             = hi;
    b.k             = lo;
    a.i             = ai;
    b.i             = bi;
  }
  __device__ __forceinline__ static int index(elem_t e)
  {
    return e.i;
  }
};

// bf16 key and int32 index as a pair, keys compared natively with __hmax and __hmin, not_guaranteed
// only, the same tie handling as the fp32 pairs.
struct pair_bf16
{
  __nv_bfloat16 k;
  int i;
};

template <int LW>
struct policy_pair_bf16
{
  using elem_t = pair_bf16;

  __device__ __forceinline__ static elem_t shfl_xor(elem_t e, int stride, unsigned mask)
  {
    elem_t r;
    r.k = __ushort_as_bfloat16(
      static_cast<unsigned short>(shfl_xor_u32<LW>(__bfloat16_as_ushort(e.k), stride, mask)));
    r.i = static_cast<int>(shfl_xor_u32<LW>(static_cast<uint32_t>(e.i), stride, mask));
    return r;
  }
  __device__ __forceinline__ static void cx(elem_t& mine, elem_t other, bool keep_min)
  {
    const __nv_bfloat16 nk = keep_min ? __hmin(mine.k, other.k) : __hmax(mine.k, other.k);
    const bool take        = __hne(nk, mine.k);
    mine.i                 = take ? other.i : mine.i;
    mine.k                 = nk;
  }
  __device__ __forceinline__ static void cx2(elem_t& a, elem_t& b)
  {
    const bool swap        = __hgt(b.k, a.k);
    const __nv_bfloat16 hi = __hmax(a.k, b.k);
    const __nv_bfloat16 lo = __hmin(a.k, b.k);
    const int ai           = swap ? b.i : a.i;
    const int bi           = swap ? a.i : b.i;
    a.k                    = hi;
    b.k                    = lo;
    a.i                    = ai;
    b.i                    = bi;
  }
  __device__ __forceinline__ static int index(elem_t e)
  {
    return e.i;
  }
};

// ------------------------------------------------------------------------------------------------
// Packing
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t twiddle_bf16(__nv_bfloat16 h)
{
  uint16_t b;
  memcpy(&b, &h, sizeof(b));
  b                   = (b == 0x8000u) ? uint16_t{0} : b; // -0.0 becomes +0.0
  const uint16_t flip = (b & 0x8000u) ? uint16_t{0xffffu} : uint16_t{0x8000u};
  return static_cast<uint32_t>(static_cast<uint16_t>(b ^ flip));
}

__device__ __forceinline__ uint32_t pack_bf16_index(__nv_bfloat16 h, int idx)
{
  return (twiddle_bf16(h) << 16) | (0xffffu - static_cast<uint32_t>(idx));
}

__device__ __forceinline__ uint32_t twiddle_f32(float f)
{
  uint32_t b          = __float_as_uint(f);
  b                   = (b == 0x80000000u) ? 0u : b; // -0.0 becomes +0.0
  const uint32_t flip = (b & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return b ^ flip;
}

__device__ __forceinline__ unsigned long long pack_f32_index(float f, int idx)
{
  return (static_cast<unsigned long long>(twiddle_f32(f)) << 32) | (0xffffffffu - static_cast<uint32_t>(idx));
}

// ------------------------------------------------------------------------------------------------
// The network. LW lanes, M registers per tile (M a power of two), Policy P.
// ------------------------------------------------------------------------------------------------
template <class P, int LW>
struct net
{
  using E                         = typename P::elem_t;
  static constexpr int num_stages = log2_int(LW);

  // One stage of the shuffle network on a single register: strides 2^Stage down to 1. The sort
  // direction of a lane's group alternates with the group id so that the next stage sees a bitonic
  // sequence, exactly as in cub::detail::WarpBitonicSortNetwork<KeyT, 1, LW>::merge_stage.
  template <bool Reverse, int Stage>
  __device__ __forceinline__ static void stage(E& e, int lane, unsigned mask)
  {
    bool group_reverse = Reverse;
    if constexpr (Stage != num_stages - 1)
    {
      group_reverse ^= ((lane >> (Stage + 1)) & 1) != 0;
    }
#pragma unroll
    for (int stride = (1 << Stage); stride > 0; stride /= 2)
    {
      const bool upper    = (lane & stride) != 0;
      const bool keep_min = group_reverse != upper;
      const E other       = P::shfl_xor(e, stride, mask);
      P::cx(e, other, keep_min);
    }
  }

  template <bool Reverse, int S>
  __device__ __forceinline__ static void stages_from(E& e, int lane, unsigned mask)
  {
    if constexpr (S < num_stages)
    {
      stage<Reverse, S>(e, lane, mask);
      stages_from<Reverse, S + 1>(e, lane, mask);
    }
  }

  // Sort M registers x LW lanes (positions i * LW + lane) in the given direction.
  template <int M, bool Reverse>
  __device__ __forceinline__ static void sort(E* e, int lane, unsigned mask)
  {
    if constexpr (M == 1)
    {
      stages_from<Reverse, 0>(*e, lane, mask);
    }
    else
    {
      sort<M / 2, !Reverse>(e, lane, mask);
      sort<M / 2, Reverse>(e + M / 2, lane, mask);
      merge<M, Reverse>(e, lane, mask);
    }
  }

  // Merge a bitonic sequence of M registers x LW lanes into the given direction.
  template <int M, bool Reverse>
  __device__ __forceinline__ static void merge(E* e, int lane, unsigned mask)
  {
    if constexpr (M == 1)
    {
      if constexpr (num_stages > 0)
      {
        stage<Reverse, num_stages - 1>(*e, lane, mask);
      }
    }
    else
    {
#pragma unroll
      for (int i = 0; i < M / 2; ++i)
      {
        if constexpr (Reverse)
        {
          P::cx2(e[i + M / 2], e[i]);
        }
        else
        {
          P::cx2(e[i], e[i + M / 2]);
        }
      }
      merge<M / 2, Reverse>(e, lane, mask);
      merge<M / 2, Reverse>(e + M / 2, lane, mask);
    }
  }

  // Keep the larger element per position: a (forward sorted) against b (reverse sorted) leaves a
  // bitonic sequence in a.
  template <int M>
  __device__ __forceinline__ static void keep_larger(E* a, const E* b)
  {
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
      E x = a[i];
      E y = b[i];
      P::cx2(x, y);
      a[i] = x;
    }
  }

  // Eager top-k (the PR #12222 loop): sort tile 0 forward, then for every further tile sort it in
  // reverse, keep the larger per position, merge. T tiles of M registers.
  template <int M, int T>
  __device__ __forceinline__ static void topk_eager(E* e, int lane, unsigned mask)
  {
    sort<M, false>(e, lane, mask);
#pragma unroll
    for (int t = 1; t < T; ++t)
    {
      sort<M, true>(e + t * M, lane, mask);
      keep_larger<M>(e, e + t * M);
      merge<M, false>(e, lane, mask);
    }
  }

  // Level-parallel top-k (N2): sort all T tiles at once (tiles 0 to T/2 - 1 forward, the rest in
  // reverse), then per level pair tile i with tile i + S, keep the larger per position and merge
  // the S survivors with the directions the next level needs. T a power of two.
  template <int M, int T>
  __device__ __forceinline__ static void sort_all(E* e, int lane, unsigned mask)
  {
#pragma unroll
    for (int t = 0; t < T; ++t)
    {
      if (t < T / 2)
      {
        sort<M, false>(e + t * M, lane, mask);
      }
      else
      {
        sort<M, true>(e + t * M, lane, mask);
      }
    }
  }

  template <int M, int S>
  __device__ __forceinline__ static void level(E* e, int lane, unsigned mask)
  {
    // S survivors out of 2S tiles
#pragma unroll
    for (int i = 0; i < S; ++i)
    {
      keep_larger<M>(e + i * M, e + (i + S) * M);
    }
    if constexpr (S == 1)
    {
      merge<M, false>(e, lane, mask);
    }
    else
    {
#pragma unroll
      for (int i = 0; i < S; ++i)
      {
        if (i < S / 2)
        {
          merge<M, false>(e + i * M, lane, mask);
        }
        else
        {
          merge<M, true>(e + i * M, lane, mask);
        }
      }
      level<M, S / 2>(e, lane, mask);
    }
  }

  template <int M, int T>
  __device__ __forceinline__ static void topk_level_parallel(E* e, int lane, unsigned mask)
  {
    static_assert(topk::is_pow2(T), "level-parallel needs a power of two tile count");
    if constexpr (T == 1)
    {
      sort<M, false>(e, lane, mask);
    }
    else
    {
      sort_all<M, T>(e, lane, mask);
      level<M, T / 2>(e, lane, mask);
    }
  }
};

// Write the first K striped positions of tile 0 as indices.
template <class P, int LW, int M>
__device__ __forceinline__ void emit_indices(const typename P::elem_t* e, int lane, int* out_idx, int k)
{
#pragma unroll
  for (int i = 0; i < M; ++i)
  {
    const int pos = i * LW + lane;
    if (pos < k)
    {
      out_idx[pos] = P::index(e[i]);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Adapter body shared by W1_eager_same, W1_level_same, W1_minmax_pairs and W1_level_pairs (and
// their sub-warp wrappers). Pairs false: bf16 goes through the 32-bit index-in-key word and fp32
// through the 64-bit index-in-key word, both requirements. Pairs true: native (key, index) pairs
// compared with min and max, not_guaranteed only.
// ------------------------------------------------------------------------------------------------
__host__ __device__ inline constexpr int max_k_of(int threads, int k)
{
  return max_int(threads, next_pow2(k));
}

__host__ __device__ inline constexpr bool
supported(topk::key_type kt, int threads, int n, int k, topk::requirement req, bool level_parallel, bool pairs)
{
  const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
  const bool thread_ok = (threads == 4) || (threads == 8) || (threads == 16) || (threads == 32);
  const int max_k      = max_k_of(threads, k);
  const bool shape_ok  = (k >= 1) && (n % threads == 0) && (max_k <= n) && (n % max_k == 0);
  const bool tiles_ok  = !level_parallel || topk::is_pow2(n / max_k);
  const bool index_ok  = pairs || (kt != topk::key_type::bf16) || (n <= 65536);
  const bool req_ok    = (req == topk::requirement::not_guaranteed)
                   || (!pairs && req == topk::requirement::prefer_smaller_index);
  return key_ok && thread_ok && shape_ok && tiles_ok && index_ok && req_ok;
}

// Pairs false: packed words (bf16 u32, fp32 u64), both requirements. Pairs true: native pairs,
// not_guaranteed only.
template <typename KeyT, int LW, bool Pairs>
struct policy_for;

template <int LW>
struct policy_for<__nv_bfloat16, LW, false>
{
  using type = policy_u32<LW>;
  __device__ __forceinline__ static uint32_t pack(__nv_bfloat16 h, int idx)
  {
    return pack_bf16_index(h, idx);
  }
};

template <int LW>
struct policy_for<float, LW, false>
{
  using type = policy_u64<LW>;
  __device__ __forceinline__ static unsigned long long pack(float f, int idx)
  {
    return pack_f32_index(f, idx);
  }
};

template <int LW>
struct policy_for<float, LW, true>
{
  using type = policy_pair_f32<LW>;
  __device__ __forceinline__ static pair_f32 pack(float f, int idx)
  {
    return pair_f32{f, idx};
  }
};

template <int LW>
struct policy_for<__nv_bfloat16, LW, true>
{
  using type = policy_pair_bf16<LW>;
  __device__ __forceinline__ static pair_bf16 pack(__nv_bfloat16 h, int idx)
  {
    return pair_bf16{h, idx};
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req, bool LevelParallel, bool Pairs>
struct adapter_impl
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int lw                              = Threads;
  static constexpr int max_k                           = max_k_of(Threads, K);
  static constexpr int m                               = max_k / Threads;
  static constexpr int tiles                           = items_per_thread / m;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  struct TempStorage
  {
    int unused;
  };
  using PF  = policy_for<KeyT, lw, Pairs>;
  using P   = typename PF::type;
  using E   = typename P::elem_t;
  using Net = net<P, lw>;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    const int lane      = topk::group_lane<Threads>();
    const unsigned mask = topk::member_mask<Threads>();
    E e[items_per_thread];
#pragma unroll
    for (int i = 0; i < items_per_thread; ++i)
    {
      e[i] = PF::pack(keys[i], idx[i]);
    }
    if constexpr (LevelParallel)
    {
      Net::template topk_level_parallel<m, tiles>(e, lane, mask);
    }
    else
    {
      Net::template topk_eager<m, tiles>(e, lane, mask);
    }
    emit_indices<P, lw, m>(e, lane, out_idx, K);
  }
};

} // namespace w1b
