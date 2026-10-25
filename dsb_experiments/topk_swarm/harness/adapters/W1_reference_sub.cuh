// Adapter W1_reference_sub: cub::detail::WarpBitonicTopK (PR #12222) on a logical warp of 4, 8 or
// 16 lanes with key-index pairs, index-only output. The sub-warp control: the same structure as
// W1_reference with MaxK rounded up to a multiple of the logical warp size.
//
// Contract:
//   scope subwarp, logical warp of 4, 8 or 16 lanes, striped input in registers, N a multiple of
//   the logical warp size, k in [1, N] with round_up(k, LW) <= N, keys fp32 or bf16 compared
//   natively (bf16 through __hgt), requirement not_guaranteed only (ties are not index resolved).
//   Output: k indices in the harness's shared output array, sorted by key descending. Temp storage
//   none. Sync: shuffles inside the logical warp only. NaN: comparisons with NaN are false, a NaN
//   key lands at an unspecified position, never produced by focus-set inputs.
#pragma once

#include "../../approaches/W1/w1_adapters.cuh"

struct W1_reference_sub_info
{
  static constexpr const char* id       = "W1_reference_sub";
  static constexpr topk::scope scope    = topk::scope::subwarp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "WarpBitonicTopK pairs on a logical warp of 4, 8 or 16 lanes, striped registers in, indices to smem out, MaxK = "
    "round_up(k, LW), not_guaranteed only";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w1::eager_pairs_supported(kt, threads, n, k, req, true);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_reference_sub : w1::eager_pairs_adapter<KeyT, Threads, N, K, Req>
{};
