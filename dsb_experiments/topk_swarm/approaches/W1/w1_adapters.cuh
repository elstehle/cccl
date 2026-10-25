// Adapter templates shared by the W1 adapter headers at warp and sub-warp scope. Threads is the
// logical warp size, the input is striped (item i of lane l is input item i * Threads + l), the
// output is the k selected source indices in the harness's shared output array, keys are never
// materialized. Contract details are in the adapter headers that instantiate these.
#pragma once

#include <cub/warp/warp_bitonic_topk.cuh>

#include "topk_harness.cuh"
#include "w1_common.cuh"
#include "warp_level_topk.cuh"

namespace w1
{
__host__ __device__ constexpr bool key_ok(topk::key_type kt)
{
  return kt == topk::key_type::fp32 || kt == topk::key_type::bf16;
}
__host__ __device__ constexpr bool threads_ok(int threads, bool subwarp)
{
  return subwarp ? (threads == 4 || threads == 8 || threads == 16) : (threads == 32);
}
__host__ __device__ constexpr bool index_fits(topk::key_type kt, int n)
{
  return kt == topk::key_type::bf16 ? (n <= 65536) : true;
}
__host__ __device__ constexpr bool shape_ok(int threads, int n, int k)
{
  return n % threads == 0 && k >= 1 && topk::round_up(k, threads) <= n;
}

// 1. The eager CUB network on key-index pairs (the structure of the reference) at any logical
//    warp size. not_guaranteed only: ties are resolved by network position, not by index.
__host__ __device__ constexpr bool
eager_pairs_supported(topk::key_type kt, int threads, int n, int k, topk::requirement req, bool subwarp)
{
  return key_ok(kt) && threads_ok(threads, subwarp) && shape_ok(threads, n, k)
      && req == topk::requirement::not_guaranteed;
}

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct eager_pairs_adapter
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int max_k                           = topk::round_up(K, Threads);
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t      = cub::detail::WarpBitonicTopK<KeyT, max_k, Threads, int>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t{ts}.TopK(keys, idx, topk::greater_t<KeyT>{}, K);
    const int lane = topk::group_lane<Threads>();
#pragma unroll
    for (int i = 0; i < max_k / Threads; ++i)
    {
      const int pos = i * Threads + lane;
      if (pos < K)
      {
        out_idx[pos] = idx[i];
      }
    }
  }
};

// 2. The eager CUB network, keys only, on packed words (key and inverted index in one word, see
//    w1_common.cuh). Both unsorted requirements, the tie-break is a property of the word order.
__host__ __device__ constexpr bool
packed_supported(topk::key_type kt, int threads, int n, int k, topk::requirement req, bool subwarp)
{
  return key_ok(kt) && threads_ok(threads, subwarp) && shape_ok(threads, n, k) && index_fits(kt, n)
      && (req == topk::requirement::not_guaranteed || req == topk::requirement::prefer_smaller_index);
}

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct eager_packed_adapter
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int max_k                           = topk::round_up(K, Threads);
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using P                                              = packer<KeyT>;
  using W                                              = typename P::word_t;
  using topk_t                                         = cub::detail::WarpBitonicTopK<W, max_k, Threads>;
  using TempStorage                                    = typename topk_t::TempStorage;

  struct greater_word
  {
    __device__ __forceinline__ bool operator()(W a, W b) const
    {
      return a > b;
    }
  };

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    W words[items_per_thread];
#pragma unroll
    for (int i = 0; i < items_per_thread; ++i)
    {
      words[i] = P::pack(keys[i], idx[i]);
    }
    topk_t{ts}.TopK(words, greater_word{}, K);
    const int lane = topk::group_lane<Threads>();
#pragma unroll
    for (int i = 0; i < max_k / Threads; ++i)
    {
      const int pos = i * Threads + lane;
      if (pos < K)
      {
        out_idx[pos] = P::index(words[i]);
      }
    }
  }
};

// 3. The level-parallel network (warp_level_topk.cuh) on packed words. Needs N / MaxK to be a
//    power of two.
__host__ __device__ constexpr bool
level_supported(topk::key_type kt, int threads, int n, int k, topk::requirement req, bool subwarp)
{
  return packed_supported(kt, threads, n, k, req, subwarp) && n % topk::round_up(k, threads) == 0
      && is_pow2(n / topk::round_up(k, threads));
}

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct level_packed_adapter
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int max_k                           = topk::round_up(K, Threads);
  static constexpr int M                               = max_k / Threads;
  static constexpr int T                               = N / max_k;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using P                                              = packer<KeyT>;
  using W                                              = typename P::word_t;
  using net                                            = level_topk<W, Threads, M, T>;
  struct TempStorage
  {};

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    W words[items_per_thread];
#pragma unroll
    for (int i = 0; i < items_per_thread; ++i)
    {
      words[i] = P::pack(keys[i], idx[i]);
    }
    const int lane = topk::group_lane<Threads>();
    net::run(words, lane, topk::member_mask<Threads>());
#pragma unroll
    for (int r = 0; r < M; ++r)
    {
      const int pos = r * Threads + lane;
      if (pos < K)
      {
        out_idx[pos] = P::index(words[r]);
      }
    }
  }
};
} // namespace w1
