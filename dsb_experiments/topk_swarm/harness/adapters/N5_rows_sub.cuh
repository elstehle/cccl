// Adapter N5_rows_sub: N5_rows declared at sub-warp scope so run.py expands the sub-warp cell grid
// (logical warps of 8 and 16 lanes, 4 where the shape allows). Same code, same contract as N5_rows.cuh.
#pragma once

#include "N5_rows.cuh"

struct N5_rows_sub_info
{
  static constexpr const char* id       = "N5_rows_sub";
  static constexpr topk::scope scope    = topk::scope::subwarp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = N5_rows_info::contract;

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return threads < 32 && N5_rows_info::supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N5_rows_sub : N5_rows<KeyT, Threads, N, K, Req>
{};
