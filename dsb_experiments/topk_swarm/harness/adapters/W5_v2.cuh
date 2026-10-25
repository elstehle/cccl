// Adapter W5_v2: W5_base with the filter on raw float keys (bf16 widened by a shift), candidates
// twiddled and packed only after the gather, tree count, IPT / 8 scatter position chains, pair-tree
// top 2. See W5_impl_v2.cuh and W5_impl.cuh. Control: W5_base.
//
// Contract: as W5_base (striped registers in, k indices to smem out, both unsorted requirements,
// bf16 exact and fp32 via N4 with the exact tie pass, radix select fallback on overflow, temp
// storage 32 x D words or pairs, __syncwarp, redux, ballot and shuffles over the full warp).
#pragma once

#include "W5_impl_v2.cuh"

struct W5_v2_info
{
  static constexpr const char* id       = "W5_v2";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "W5 with the filter on raw float keys, pack after the gather, tree count and chunked scatter chains, striped "
    "registers in, indices to smem out, bf16 exact and fp32 via N4, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w5::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W5_v2
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using prim                                           = w5v2::primitive<KeyT, N, K>;
  using TempStorage                                    = typename prim::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    prim::run(keys, ts, out_idx);
  }
};
