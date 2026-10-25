// Adapter N5_split_f_sub: N5_split_f declared at sub-warp scope so run.py expands the sub-warp cell grid
// (logical warps of 8 and 16 lanes, 4 where the shape allows). Same code, same contract as N5_split_f.cuh.
#pragma once

#include "N5_split_f.cuh"

struct N5_split_f_sub_info
{
  static constexpr const char* id       = "N5_split_f_sub";
  static constexpr topk::scope scope    = topk::scope::subwarp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = N5_split_f_info::contract;

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return threads < 32 && N5_split_f_info::supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N5_split_f_sub : N5_split_f<KeyT, Threads, N, K, Req>
{};
