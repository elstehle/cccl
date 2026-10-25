// Adapter W1_level_same_sub: W1_level_same declared at sub-warp scope so run.py expands the sub-warp cell grid
// (logical warps of 4, 8 and 16 lanes). Same code, same contract as W1_level_same.cuh.
#pragma once

#include "W1_level_same.cuh"

struct W1_level_same_sub_info
{
  static constexpr const char* id       = "W1_level_same_sub";
  static constexpr topk::scope scope    = topk::scope::subwarp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = W1_level_same_info::contract;

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return threads < 32 && W1_level_same_info::supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_level_same_sub : W1_level_same<KeyT, Threads, N, K, Req>
{};
