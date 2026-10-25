// Adapter W3_d4: warp radix select, 4-bit digits, ballot-count histograms, no compaction (every
// pass sweeps the whole register tile). The control for the compaction of W3_d4c. See W3_impl.cuh.
// Contract as W3_d4c without the N >= 128 limit and with no temp storage.
#pragma once

#include "W3_impl.cuh"

struct W3_d4_info
{
  static constexpr const char* id       = "W3_d4";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "radix select by 4-bit digits and ballot counts over the whole tile every pass, blocked registers in, "
    "indices to smem out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n6w::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W3_d4
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using impl                                           = w3::primitive<KeyT, N, K, 4, false>;
  using TempStorage                                    = typename impl::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    impl::run(keys, ts, out_idx);
  }
};
