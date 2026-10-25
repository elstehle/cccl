// Adapter N6w_atom32f: N6w_atom32 with the trimmed finish (w2_finish_fused: flags recomputed, both
// ranks from one loop, rounds unrolled by four) and unpacked bins up to 32 items per lane.
// Adapter N6w_atom32: N6w_ballot with the 32-bin histogram counted by one shared-memory atomic per
// item instead of five ballots. Same bins, same finish, isolates the counting mechanism. See
// N6w_atomic_impl.cuh. Contract as N6w_ballot, temp storage 256 bytes more (histogram and dummy
// words), __syncwarp around the histogram.
#pragma once

#include "N6w_atomic_impl.cuh"

struct N6w_atom32f_info
{
  static constexpr const char* id       = "N6w_atom32f";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "N6w_ballot with a 32-bin shared atomic histogram, blocked registers in, indices to smem out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n6w::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N6w_atom32f
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using impl                                           = n6w::atomic_primitive<KeyT, N, K, 32, true>;
  using TempStorage                                    = typename impl::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    impl::run(keys, ts, out_idx);
  }
};
