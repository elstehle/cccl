// Adapter W1_kvcmp: W1_baseline with the composite (key, index) comparator written branch-free.
// W1_baseline's comparator uses the short-circuit forms || and &&, which the compiler turns into a
// branch region per compare-exchange of the network. Here the same predicate is built with the
// bitwise | and & on bools, so every compare stays predicated. One change against W1_baseline,
// prefer_smaller_index only (the not_guaranteed path of W1_baseline is the unchanged reference).
//
// Contract:
//   scope warp or sub-warp (4, 8, 16, 32 lanes), striped input in registers, N = LW x IPT, k in
//   [1, N] with round_up(k, LW) <= N, keys fp32 or bf16 compared natively (bf16 through __hgt and
//   __heq), both unsorted requirements. Output: k indices in the harness output array, sorted by
//   key descending (and index ascending among equal keys under prefer_smaller_index). Temp storage
//   none. Sync: shuffles only. NaN: comparisons with NaN are false, a NaN key lands at an
//   unspecified position, never produced by focus-set inputs. -0.0 and +0.0 compare equal under
//   both requirements.
#pragma once

#include <cub/warp/warp_bitonic_topk.cuh>

#include "../topk_harness.cuh"

namespace w1_kvcmp_detail
{
template <typename KeyT>
struct kv
{
  KeyT k;
  int i;
};

template <typename KeyT>
struct kv_greater
{
  __device__ __forceinline__ bool operator()(const kv<KeyT>& a, const kv<KeyT>& b) const
  {
    if constexpr (std::is_same_v<KeyT, __nv_bfloat16>)
    {
      return __hgt(a.k, b.k) | (__heq(a.k, b.k) & (a.i < b.i));
    }
    else
    {
      return (a.k > b.k) | ((a.k == b.k) & (a.i < b.i));
    }
  }
};
} // namespace w1_kvcmp_detail

struct W1_kvcmp_info
{
  static constexpr const char* id       = "W1_kvcmp";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "WarpBitonicTopK keys-only on a (key, index) struct with a branch-free composite comparator, "
    "prefer_smaller_index only, logical warps 4 to 32, MaxK = round_up(k, LW)";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 4) || (threads == 8) || (threads == 16) || (threads == 32);
    const bool shape_ok  = (n % threads == 0) && (k >= 1) && (topk::round_up(k, threads) <= n);
    const bool req_ok = (req == topk::requirement::prefer_smaller_index);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_kvcmp
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int max_k                           = topk::round_up(K, Threads);
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  struct TempStorage
  {
    int unused;
  };

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    const int lane = topk::group_lane<Threads>();
    if constexpr (Req == topk::requirement::not_guaranteed)
    {
      using topk_t = cub::detail::WarpBitonicTopK<KeyT, max_k, Threads, int>;
      typename topk_t::TempStorage ts;
      topk_t{ts}.TopK(keys, idx, topk::greater_t<KeyT>{}, K);
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
    else
    {
      using kv_t   = w1_kvcmp_detail::kv<KeyT>;
      using topk_t = cub::detail::WarpBitonicTopK<kv_t, max_k, Threads>;
      kv_t e[items_per_thread];
#pragma unroll
      for (int i = 0; i < items_per_thread; ++i)
      {
        e[i] = kv_t{keys[i], idx[i]};
      }
      typename topk_t::TempStorage ts;
      topk_t{ts}.TopK(e, w1_kvcmp_detail::kv_greater<KeyT>{}, K);
#pragma unroll
      for (int i = 0; i < max_k / Threads; ++i)
      {
        const int pos = i * Threads + lane;
        if (pos < K)
        {
          out_idx[pos] = e[i].i;
        }
      }
    }
  }
};
