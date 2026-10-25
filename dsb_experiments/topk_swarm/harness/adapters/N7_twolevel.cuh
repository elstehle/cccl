// Adapter N7_twolevel: W2's packed redux extraction with a two-level per-lane list (sorted head of
// 8, unsorted remainder, write-back merges, lookahead vote every 4 rounds, warp-uniform refill).
// See N7_impl.cuh. Control: W2_packed_v3 (same words, same rounds, full-depth sorted list).
//
// Contract:
//   scope warp, 32 lanes, blocked input in registers (lane t holds items t*IPT .. t*IPT+IPT-1),
//   N = 32 x IPT with IPT >= 16 a power of two, N <= 65536 (bf16) or 32768 (fp32), k a multiple of
//   4 in [4, 256] with 2k <= N, keys bf16 or fp32 (twiddled internally, minus zero normalized),
//   int32 indices from the position. Both unsorted requirements run the same code (the packed
//   order is prefer_smaller_index). Output: k indices in descending key order (ties by smaller
//   index), the fp32 tie pass may rewrite the last slots of the tie class in (key, index) order.
//   Temp storage none. Sync: redux.sync and one vote per 4 rounds over the full warp, no barriers.
//   NaN: as W2_impl.cuh, focus-set inputs contain no NaN.
#pragma once

#include "N7_impl.cuh"

struct N7_twolevel_info
{
  static constexpr const char* id       = "N7_twolevel";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "packed redux extraction on a two-level per-lane list (head 8, lookahead vote per 4 rounds), blocked registers in, "
    "indices to smem out, bf16 exact and fp32 via N4, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n7::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N7_twolevel
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  struct TempStorage
  {
    int unused;
  };

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    n7::primitive<KeyT, Threads, N, K>::run(keys, out_idx);
  }
};
