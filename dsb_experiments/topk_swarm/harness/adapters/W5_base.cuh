// Adapter W5_base: threshold filter with a certified lower bound from grouped lane maxima, ballot
// prefix compaction through shared memory, packed W2 finish on the candidates, exact radix select
// fallback on overflow. See W5_impl.cuh.
//
// Contract:
//   scope warp, 32 lanes, striped input in registers (lane t holds items t, t + 32, ...), N a power
//   of two from 256 to 65536 (bf16) or 32768 (fp32), k in [4, 128] with 2k <= N, keys bf16 or fp32
//   (twiddled internally, minus zero normalized to plus zero), int32 indices derived from the
//   position. Requirements not_guaranteed and prefer_smaller_index run the same code, the packed
//   order is prefer_smaller_index by construction. Output: k indices in the harness's shared output
//   array, in descending key order (ties by smaller index) on the filter path, bf16 exact and fp32
//   by truncated key with the tie class of the k-th key rewritten in (key, index) order, and in
//   ticket order (unsorted) on the overflow fallback path. Temp storage: 32 x D words (bf16) or
//   pairs (fp32) with D 4, 8 or 16 by k (512 bytes to 4 KB). Sync: __syncwarp between the scatter
//   and the gather, redux.sync, ballot and shuffles over the full warp mask, no barriers.
//   NaN: see W5_impl.cuh, focus-set inputs contain no NaN.
#pragma once

#include "W5_impl.cuh"

struct W5_base_info
{
  static constexpr const char* id       = "W5_base";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "certified grouped lane bound, compare filter, smem compaction, packed W2 finish, radix select fallback, striped "
    "registers in, indices to smem out, bf16 exact and fp32 via N4, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w5::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W5_base
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using prim                                           = w5::primitive<KeyT, N, K>;
  using TempStorage                                    = typename prim::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    prim::run(keys, ts, out_idx);
  }
};
