// Adapter W5_v3: W5_v2 with a predicate-free filter (flags from the sign bit of the subtraction,
// integer count tree, unconditional stores with a per-lane dummy slot). See W5_impl_v3.cuh.
// Control: W5_v2. Contract as W5_v2 (temp storage 32 more words for the dummy slots).
#pragma once

#include "W5_impl_v3.cuh"

struct W5_v3_info
{
  static constexpr const char* id       = "W5_v3";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "W5 with a predicate-free filter (sign-bit flags, unconditional scatter with dummy slots), striped registers in, "
    "indices to smem out, bf16 exact and fp32 via N4, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w5::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W5_v3
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using prim                                           = w5v3::primitive<KeyT, N, K>;
  using TempStorage                                    = typename prim::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    prim::run(keys, ts, out_idx);
  }
};
