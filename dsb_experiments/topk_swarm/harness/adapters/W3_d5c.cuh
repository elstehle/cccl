// Adapter W3_d5c: W3_d4c with 5-bit digits (bf16 5+5+5+1, fp32 5x6+2). See W3_impl.cuh.
// Contract as W3_d4c.
#pragma once

#include "W3_impl.cuh"

struct W3_d5c_info
{
  static constexpr const char* id       = "W3_d5c";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "radix select by 5-bit digits and ballot counts with candidate compaction at N/4, blocked registers in, "
    "indices to smem out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n6w::supported_shape(kt, threads, n, k, req) && (n >= 128);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W3_d5c
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using impl                                           = w3::primitive<KeyT, N, K, 5, true>;
  using TempStorage                                    = typename impl::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    impl::run(keys, ts, out_idx);
  }
};
