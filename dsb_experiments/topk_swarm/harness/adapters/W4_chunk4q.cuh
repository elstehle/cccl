// Adapter W4_chunk4q: W4_chunk4 with a 4-ary merge-path search (three probes per step, six
// independent loads in flight, half the dependent steps). Control: W4_chunk4 and B3_tuned.
// Contract and NaN behavior in W4_impl.cuh.
#pragma once

#include "W4_impl.cuh"

struct W4_chunk4q_info
{
  static constexpr const char* id       = "W4_chunk4q";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "k-truncated merge of 4 sorted runs with a 4-ary merge path, block 256, sorted indices to smem, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w4::chunk_supported(kt, threads, n, k, req, 4);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W4_chunk4q
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using impl                                           = w4::chunk_adapter<KeyT, Threads, N, K, 4, true>;
  using TempStorage                                    = typename impl::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    impl::run(keys, ts, out_idx);
  }
};
