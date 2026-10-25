// Adapter W4_chunk4: k-truncated merge of 4 sorted runs at block scope (approach W4). The tile
// of N keys in blocked arrangement is 4 consecutive runs of N / 4 items, each sorted by (key
// descending, index ascending) as a per-chunk top-k emits them (the harness provides such input
// through its sorted-runs option). Output: the k largest of the union, sorted, as indices.
// Contract and NaN behavior in W4_impl.cuh. Control: B3_tuned (unsorted selection over the same
// candidates) and B6_radix (full sort).
#pragma once

#include "W4_impl.cuh"

struct W4_chunk4_info
{
  static constexpr const char* id       = "W4_chunk4";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "k-truncated merge-path merge of 4 sorted runs (blocked), block 256, sorted indices to smem, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w4::chunk_supported(kt, threads, n, k, req, 4);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W4_chunk4
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using impl                                           = w4::chunk_adapter<KeyT, Threads, N, K, 4>;
  using TempStorage                                    = typename impl::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    impl::run(keys, ts, out_idx);
  }
};
