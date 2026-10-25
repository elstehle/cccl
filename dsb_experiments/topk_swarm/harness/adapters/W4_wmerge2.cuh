// Adapter W4_wmerge2: W4_wmerge with the per-lane sort in two halves (two sorted runs of IPT / 2
// per lane, half the words live during the network, one more merge level whose split is
// trivial). Control: W4_wmerge. Contract and NaN behavior in W4_impl.cuh.
#pragma once

#include "W4_impl.cuh"

struct W4_wmerge2_info
{
  static constexpr const char* id       = "W4_wmerge2";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "per-lane sort plus k-truncated merge-path levels, warp 32, blocked registers in, sorted indices to smem, both "
    "requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w4::wmerge_supported(kt, threads, n, k, req, 2);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W4_wmerge2
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using impl                                           = w4::wmerge_adapter<KeyT, N, K, 2>;
  using TempStorage                                    = typename impl::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    impl::run(keys, ts, out_idx);
  }
};
