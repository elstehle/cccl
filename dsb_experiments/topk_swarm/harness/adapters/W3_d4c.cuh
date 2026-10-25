// Adapter W3_d4c: warp radix select, 4-bit digits, ballot-count histograms, candidate compaction
// into shared memory once the candidates fit into N/4 slots. See W3_impl.cuh.
//
// Contract:
//   scope warp, 32 lanes, blocked input in registers, N a multiple of 32 up to 32768 (N >= 128 for
//   the compaction slots), k in [1, N/2], keys bf16 or fp32 (minus zero folded onto plus zero),
//   int32 indices from the position. Both requirements run the same code (the tie class at the
//   k-th key is taken in index order). Output: k indices in the harness's shared output array in
//   index order. Temp storage N/2 words per warp (two compaction arrays of N/4). Sync: ballots,
//   shuffles, __syncwarp, no barriers. NaN: unsupported, focus-set inputs contain no NaN.
#pragma once

#include "W3_impl.cuh"

struct W3_d4c_info
{
  static constexpr const char* id       = "W3_d4c";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "radix select by 4-bit digits and ballot counts with candidate compaction at N/4, blocked registers in, "
    "indices to smem out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n6w::supported_shape(kt, threads, n, k, req) && (n >= 128);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W3_d4c
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using impl                                           = w3::primitive<KeyT, N, K, 4, true>;
  using TempStorage                                    = typename impl::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    impl::run(keys, ts, out_idx);
  }
};
