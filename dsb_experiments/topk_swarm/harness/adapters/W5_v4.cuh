// Adapter W5_v4: W5_v3 with a predicate-free gather (minus zero normalized at load by one FADD,
// compare-free twiddle, unconditional loads masked by the sign of p - tot). See W5_impl_v4.cuh.
// Control: W5_v3. Contract as W5_v3.
#pragma once

#include "W5_impl_v4.cuh"

struct W5_v4_info
{
  static constexpr const char* id       = "W5_v4";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "W5 with predicate-free filter and gather, striped registers in, indices to smem out, bf16 exact and fp32 via N4, "
    "both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w5::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W5_v4
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using prim                                           = w5v4::primitive<KeyT, N, K>;
  using TempStorage                                    = typename prim::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    prim::run(keys, ts, out_idx);
  }
};
