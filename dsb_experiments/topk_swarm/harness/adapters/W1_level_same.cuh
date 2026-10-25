// Adapter W1_level_same: the level-parallel top-k of idea card N2 on the same packed words and the same network code as
// W1_eager_same: all tiles are sorted at once (first half forward, second half reverse), then per
// level tile i is paired with tile i + S, the larger per position kept and the survivors merged.
// Same shuffle count as the eager loop, only the dependency structure differs.
// Element policies and the network are in approaches/W1/w1b_detail.cuh.
//
// Contract:
//   scope warp or sub-warp (4, 8, 16, 32 lanes), striped input in registers, N = LW x IPT, k in
//   [1, N] with MaxK = max(LW, next_pow2(k)) dividing N and N over MaxK a power of two, keys fp32 or bf16,
//   both unsorted requirements (the word order gives prefer_smaller_index), bf16 needs N <= 65536. Output: k indices in
//   the harness output array, sorted by key descending. Temp storage none. Sync: shuffles only.
//   NaN: see w1b_detail.cuh, never produced by focus-set inputs. -0.0 and +0.0 are equal.
#pragma once

#include "../../approaches/W1/w1b_detail.cuh"
#include "../topk_harness.cuh"

struct W1_level_same_info
{
  static constexpr const char* id       = "W1_level_same";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = "level-parallel tile loop (N2) on packed key-index words, same network and packing as W1_eager_same, N over MaxK a power of two, both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w1b::supported(kt, threads, n, k, req, true, false);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_level_same : w1b::adapter_impl<KeyT, Threads, N, K, Req, true, false>
{};
