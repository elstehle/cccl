// Adapter N10_u32_ltrunc_sub: N10_u32_ltrunc declared at sub-warp scope so run.py expands the sub-warp cell grid
// (logical warps of 4, 8 and 16 lanes). Same code, same contract as N10_u32_ltrunc.cuh.
#pragma once

#include "N10_u32_ltrunc.cuh"

struct N10_u32_ltrunc_sub_info
{
  static constexpr const char* id       = "N10_u32_ltrunc_sub";
  static constexpr topk::scope scope    = topk::scope::subwarp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = N10_u32_ltrunc_info::contract;

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n10::supported(kt, threads, n, k, req, true);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N10_u32_ltrunc_sub : N10_u32_ltrunc<KeyT, Threads, N, K, Req>
{};
