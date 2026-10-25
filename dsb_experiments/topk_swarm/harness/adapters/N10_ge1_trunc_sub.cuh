// Adapter N10_ge1_trunc_sub: N10_ge1_trunc declared at sub-warp scope so run.py expands the sub-warp cell grid
// (logical warps of 4, 8 and 16 lanes). Same code, same contract as N10_ge1_trunc.cuh.
#pragma once

#include "N10_ge1_trunc.cuh"

struct N10_ge1_trunc_sub_info
{
  static constexpr const char* id       = "N10_ge1_trunc_sub";
  static constexpr topk::scope scope    = topk::scope::subwarp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = N10_ge1_trunc_info::contract;

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n10::supported(kt, threads, n, k, req, true);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N10_ge1_trunc_sub : N10_ge1_trunc<KeyT, Threads, N, K, Req>
{};
