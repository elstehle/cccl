// Adapter W4_wmerge: warp truncated merge sort (approach W4). Per-lane Batcher sort of the
// lane's IPT items (blocked), then five merge-path levels through shared memory that keep only
// the k largest of every merged pair, sorted indices out. Contract and NaN behavior in
// W4_impl.cuh. Controls: W2_packed (v1 form) at k 1024, W5_v4 at k 64.
#pragma once

#include "W4_impl.cuh"

struct W4_wmerge_info
{
  static constexpr const char* id       = "W4_wmerge";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "per-lane sort plus k-truncated merge-path levels, warp 32, blocked registers in, sorted indices to smem, both "
    "requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w4::wmerge_supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W4_wmerge
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using impl                                           = w4::wmerge_adapter<KeyT, N, K>;
  using TempStorage                                    = typename impl::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    impl::run(keys, ts, out_idx);
  }
};
