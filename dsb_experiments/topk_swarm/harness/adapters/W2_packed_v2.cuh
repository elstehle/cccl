// Adapter W2_packed_v2: W2_packed with the next-word tie check (policy_v2 in W2_impl.cuh): the fp32
// tie pass is one redux over the remaining heads and spares instead of a count over every item.
// (bf16 exact, fp32 through the N4 truncated-key route with an exact tie pass). See W2_impl.cuh.
//
// Contract:
//   scope warp, 32 lanes, blocked input in registers (lane t holds items t*IPT .. t*IPT+IPT-1),
//   N = 32 x IPT with IPT >= 1 and N <= 65536 (bf16) or 32768 (fp32), k in [1, N/2], keys bf16 or
//   fp32 (twiddled internally, minus zero normalized to plus zero), int32 indices derived from the
//   position. Requirements not_guaranteed and prefer_smaller_index run the same code, the packed
//   order is prefer_smaller_index by construction. Output: k indices in the harness's shared output
//   array, in descending key order (slot r holds the r-th largest, ties by smaller index), except
//   that the fp32 tie pass may rewrite the last slots of the tie class in (key, index) order.
//   Temp storage none. Sync: redux.sync and one redux.add (fp32) over the full warp mask, no
//   barriers. NaN: see W2_impl.cuh, focus-set inputs contain no NaN.
#pragma once

#include "W2_impl.cuh"

struct W2_packed_v2_info
{
  static constexpr const char* id       = "W2_packed_v2";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "packed redux extraction, blocked registers in, indices to smem out, bf16 exact and fp32 via N4, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w2::supported_shape(kt, threads, n, k, req, 32, 32);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W2_packed_v2
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  struct TempStorage
  {
    int unused;
  };

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    w2::primitive<KeyT, Threads, N, K, w2::policy_v2>::run(keys, out_idx);
  }
};
