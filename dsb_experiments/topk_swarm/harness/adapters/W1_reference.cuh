// Adapter W1_reference: cub::detail::WarpBitonicTopK (PR #12222) at 32 lanes with key-index pairs,
// index-only output. The warp sorts and merges the striped register tile with MaxK rounded up to
// the next multiple of 32, the first K striped positions hold the selected pairs (sorted), the
// indices are written to the harness's output array and the keys are dropped.
//
// Contract:
//   scope warp, 32 lanes, striped input in registers, N = 32 x IPT with IPT >= 1, k in [1, N] with
//   round_up(k, 32) <= N, keys fp32 or bf16 compared natively (bf16 through __hgt), requirement
//   not_guaranteed only (the network is deterministic but ties are not resolved by index).
//   Output: k indices in the harness's shared output array, sorted by key descending. Temp storage
//   none (an empty TempStorage object). Sync: shuffles only, no barriers assumed or performed.
//   NaN: comparisons with NaN are false, a NaN key is therefore ordered unpredictably, never
//   produced by focus-set inputs. -0.0 and +0.0 compare equal.
#pragma once

#include <cub/warp/warp_bitonic_topk.cuh>

#include "../topk_harness.cuh"

struct W1_reference_info
{
  static constexpr const char* id            = "W1_reference";
  static constexpr topk::scope scope         = topk::scope::warp;
  static constexpr bool data_dependent       = false;
  static constexpr const char* contract      =
    "WarpBitonicTopK pairs, striped registers in, indices to smem out, MaxK = round_up(k, 32), not_guaranteed only";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok   = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool shape_ok = (threads == 32) && (n % 32 == 0) && (k >= 1) && (topk::round_up(k, 32) <= n);
    const bool req_ok   = (req == topk::requirement::not_guaranteed);
    return key_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_reference
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
    // The selected pairs occupy the first K striped positions: item i of lane l is position i*32+l
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
