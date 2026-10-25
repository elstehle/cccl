// Adapter W2_packed_sw_v2: W2_packed_sw with one full-warp redux per logical warp (the other
// groups contribute the identity) instead of the masked redux (policy_sw_v2 in W2_impl.cuh). The
// same primitive as W2_packed with the redux and the tie pass masked to the group's lanes. Several
// groups share one hardware warp and run their rounds in lockstep. See W2_impl.cuh.
//
// Contract:
//   scope subwarp, T in {4, 8, 16} lanes, blocked input in registers, N = T x IPT with IPT >= 1,
//   k in [1, N/2], keys bf16 or fp32, int32 indices derived from the position, both tie-break
//   requirements by construction. Output: k indices in the group's shared output array in
//   descending key order (see W2_packed for the fp32 tie pass exception). Temp storage none.
//   Sync: masked redux.sync on the member mask only, no barriers. NaN: see W2_impl.cuh.
#pragma once

#include "W2_impl.cuh"

struct W2_packed_sw_v2_info
{
  static constexpr const char* id       = "W2_packed_sw_v2";
  static constexpr topk::scope scope    = topk::scope::subwarp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "packed redux extraction on logical warps of 4 to 16 lanes, masked redux, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w2::supported_shape(kt, threads, n, k, req, 4, 16);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W2_packed_sw_v2
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
    w2::primitive<KeyT, Threads, N, K, w2::policy_sw_v2>::run(keys, out_idx);
  }
};
