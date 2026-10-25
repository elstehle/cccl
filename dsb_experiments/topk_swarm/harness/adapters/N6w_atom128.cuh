// Adapter N6w_atom128: N6w_atom32 with 128 bins (four per lane in the scan). The finer bins
// shrink the pivot bin four times, so more rows exit without a finish and the W2 finish runs
// fewer rounds. See N6w_atomic_impl.cuh. Contract as N6w_ballot, temp storage 640 bytes more.
#pragma once

#include "N6w_atomic_impl.cuh"

struct N6w_atom128_info
{
  static constexpr const char* id       = "N6w_atom128";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "N6w_ballot with a 128-bin shared atomic histogram, blocked registers in, indices to smem out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n6w::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N6w_atom128
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using impl                                           = n6w::atomic_primitive<KeyT, N, K, 128>;
  using TempStorage                                    = typename impl::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    impl::run(keys, ts, out_idx);
  }
};
