// Adapter W1_level_n64: W1_level with the 64-bit words shuffled through the __shfl_xor_sync
// intrinsic on the whole word (the CUDA headers split it with asm volatile moves) instead of two
// 32-bit shuffles. Identical to W1_level for bf16, the ablation for fp32.
//
// Contract: as W1_level.
#pragma once

#include "../../approaches/W1/w1_adapters.cuh"

struct W1_level_n64_info
{
  static constexpr const char* id       = "W1_level_n64";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "W1_level with the 64-bit word shuffled by the whole-word intrinsic, striped registers in, indices to smem out, "
    "MaxK = round_up(k, 32), N / MaxK a power of two, both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w1::level_supported(kt, threads, n, k, req, false);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_level_n64 : w1::level_packed_adapter<KeyT, Threads, N, K, Req, w1::shfl_intrinsic>
{};
