// Adapter N10_ctrl_u64rec: control for N10, the fp32 index-in-key u64 word (twiddled key high, inverted index
// low) through the same template recursion and eager tile order as the N10 variants
// (approaches/W1/warp_level_topk.cuh eager_topk, word_ops<uint64_t>: compare and select, there is
// no 64-bit VIMNMX). It isolates the keys-only network plus recovery from the code structure,
// W1_eager_same is the same word in the opt-W1b code structure.
//
// Contract:
//   scope warp (32 lanes) and sub-warp through the _sub adapter, striped input in registers,
//   N a multiple of MaxK = round_up(k, LW) with N / MaxK a power of two, keys fp32, both unsorted
//   requirements (the word order gives prefer_smaller_index). Output: k indices sorted by
//   (key descending, index ascending). Temp storage none. Sync: shuffles on the member mask.
//   NaN: ordered by its sign bit, never produced by focus-set inputs. -0.0 is mapped to +0.0.
#pragma once

#include "../../approaches/W1/w1_adapters.cuh"
#include "../topk_harness.cuh"

struct N10_ctrl_u64rec_info
{
  static constexpr const char* id       = "N10_ctrl_u64rec";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "control: fp32 index-in-key u64 word in the eager template recursion of W1x (compare and select exchange), both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return kt == topk::key_type::fp32 && w1::eager_rec_supported(kt, threads, n, k, req, false)
        && w1::is_pow2(n / topk::round_up(k, threads));
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N10_ctrl_u64rec
    : w1::level_packed_adapter<KeyT, Threads, N, K, Req, w1::shfl_split32, w1::word_ops<uint64_t>, true>
{};
