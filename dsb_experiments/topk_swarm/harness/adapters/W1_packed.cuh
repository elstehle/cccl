// Adapter W1_packed: the eager cub::detail::WarpBitonicTopK network (PR #12222) at 32 lanes, keys
// only, on packed words: bf16 keys become one 32-bit word (order preserving key bits high, inverted
// 16-bit index low), fp32 keys one 64-bit word (key bits high, inverted 32-bit index low). The
// network moves one word per item and per stage, and the integer word order gives the
// prefer_smaller_index tie-break for free. See approaches/W1/w1_common.cuh.
//
// Contract:
//   scope warp, 32 lanes, striped input in registers, N = 32 x IPT with IPT >= 1 (bf16: N at most
//   65536), k in [1, N] with round_up(k, 32) <= N, keys bf16 or fp32 twiddled to unsigned words,
//   requirements not_guaranteed and prefer_smaller_index (same code, the word order is exact).
//   Output: k indices in the harness's shared output array, sorted by (key descending, index
//   ascending). Temp storage none. Sync: shuffles only. NaN: ordered by its sign bit (positive NaN
//   above every finite key), never produced by focus-set inputs. -0.0 is mapped to +0.0.
#pragma once

#include "../../approaches/W1/w1_adapters.cuh"

struct W1_packed_info
{
  static constexpr const char* id       = "W1_packed";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "WarpBitonicTopK keys only on packed key-index words (bf16 u32, fp32 u64), striped registers in, indices to smem "
    "out, MaxK = round_up(k, 32), both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w1::packed_supported(kt, threads, n, k, req, false);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_packed : w1::eager_packed_adapter<KeyT, Threads, N, K, Req>
{};
