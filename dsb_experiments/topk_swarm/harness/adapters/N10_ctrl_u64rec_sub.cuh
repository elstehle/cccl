// Adapter N10_ctrl_u64rec_sub: N10_ctrl_u64rec declared at sub-warp scope (logical warps of 4, 8 and 16
// lanes). Same code, same contract as N10_ctrl_u64rec.cuh.
#pragma once

#include "N10_ctrl_u64rec.cuh"

struct N10_ctrl_u64rec_sub_info
{
  static constexpr const char* id       = "N10_ctrl_u64rec_sub";
  static constexpr topk::scope scope    = topk::scope::subwarp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = N10_ctrl_u64rec_info::contract;

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return kt == topk::key_type::fp32 && w1::eager_rec_supported(kt, threads, n, k, req, true)
        && w1::is_pow2(n / topk::round_up(k, threads));
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N10_ctrl_u64rec_sub : N10_ctrl_u64rec<KeyT, Threads, N, K, Req>
{};
