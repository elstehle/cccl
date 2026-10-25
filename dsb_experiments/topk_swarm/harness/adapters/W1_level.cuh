// Adapter W1_level: the level-parallel bitonic top-k of idea card N2 (approaches/W1/
// warp_level_topk.cuh) at 32 lanes on the same packed words as W1_packed. All N / MaxK tiles are
// sorted at once with the stages interleaved, then the tile count is halved per level with an
// elementwise max and one bitonic merge of log2(MaxK) stages.
//
// Contract:
//   scope warp, 32 lanes, striped input in registers, N = 32 x IPT with N / round_up(k, 32) a
//   power of two (bf16: N at most 65536), k in [1, N] with round_up(k, 32) <= N, keys bf16 or fp32
//   twiddled to unsigned words, requirements not_guaranteed and prefer_smaller_index (same code).
//   Output: k indices in the harness's shared output array, sorted by (key descending, index
//   ascending). Temp storage none. Sync: shuffles only. NaN: ordered by its sign bit, never
//   produced by focus-set inputs. -0.0 is mapped to +0.0.
#pragma once

#include "../../approaches/W1/w1_adapters.cuh"

struct W1_level_info
{
  static constexpr const char* id       = "W1_level";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "level-parallel bitonic top-k on packed key-index words (bf16 u32, fp32 u64), striped registers in, indices to "
    "smem out, MaxK = round_up(k, 32), N / MaxK a power of two, both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w1::level_supported(kt, threads, n, k, req, false);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_level : w1::level_packed_adapter<KeyT, Threads, N, K, Req>
{};
