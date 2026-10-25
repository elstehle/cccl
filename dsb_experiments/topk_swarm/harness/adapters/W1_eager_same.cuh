// Adapter W1_eager_same: the eager tile loop of WarpBitonicTopK (sort tile 0, then for every further tile sort it in
// reverse, keep the larger per position, merge) on packed words: bf16 keys travel as one 32-bit word
// (twiddled key high, inverted index low), fp32 keys as one 64-bit word, so the network is keys-only
// and prefer_smaller_index is a property of the word order. Control of W1_level_same.
// Element policies and the network are in approaches/W1/w1b_detail.cuh.
//
// Contract:
//   scope warp or sub-warp (4, 8, 16, 32 lanes), striped input in registers, N = LW x IPT, k in
//   [1, N] with MaxK = max(LW, next_pow2(k)) dividing N, keys fp32 or bf16,
//   both unsorted requirements (the word order gives prefer_smaller_index), bf16 needs N <= 65536. Output: k indices in
//   the harness output array, sorted by key descending. Temp storage none. Sync: shuffles only.
//   NaN: see w1b_detail.cuh, never produced by focus-set inputs. -0.0 and +0.0 are equal.
#pragma once

#include "../../approaches/W1/w1b_detail.cuh"
#include "../topk_harness.cuh"

struct W1_eager_same_info
{
  static constexpr const char* id       = "W1_eager_same";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = "eager tile loop on packed key-index words (bf16 u32, fp32 u64), logical warps 4 to 32, MaxK = max(LW, next_pow2(k)), both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w1b::supported(kt, threads, n, k, req, false, false);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_eager_same : w1b::adapter_impl<KeyT, Threads, N, K, Req, false, false>
{};
