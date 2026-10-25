// Adapter W1_level_sub: W1_level on a logical warp of 4, 8 or 16 lanes (the level-parallel
// network of approaches/W1/warp_level_topk.cuh on packed key-index words, see W1_level.cuh).
//
// Contract: as W1_level with the logical warp size LW in {4, 8, 16} in place of 32: N a multiple
// of LW with N / round_up(k, LW) a power of two, k in [1, N] with round_up(k, LW) <= N, both
// unsorted requirements, shuffles inside the logical warp only.
#pragma once

#include "../../approaches/W1/w1_adapters.cuh"

struct W1_level_sub_info
{
  static constexpr const char* id       = "W1_level_sub";
  static constexpr topk::scope scope    = topk::scope::subwarp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "level-parallel bitonic top-k on packed key-index words (bf16 u32, fp32 u64) on a logical warp of 4, 8 or 16 "
    "lanes, MaxK = round_up(k, LW), N / MaxK a power of two, both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w1::level_supported(kt, threads, n, k, req, true);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_level_sub : w1::level_packed_adapter<KeyT, Threads, N, K, Req>
{};
