// Adapter W1_level_pairs: the level-parallel top-k (N2) on the native min and max pairs of W1_minmax_pairs, same network
// code, same policies, control W1_minmax_pairs.
// Element policies and the network are in approaches/W1/w1b_detail.cuh.
//
// Contract:
//   scope warp or sub-warp (4, 8, 16, 32 lanes), striped input in registers, N = LW x IPT, k in
//   [1, N] with MaxK = max(LW, next_pow2(k)) dividing N and N over MaxK a power of two, keys fp32 or bf16,
//   requirement not_guaranteed (ties keep their network position). Output: k indices in
//   the harness output array, sorted by key descending. Temp storage none. Sync: shuffles only.
//   NaN: see w1b_detail.cuh, never produced by focus-set inputs. -0.0 and +0.0 are equal.
#pragma once

#include "../../approaches/W1/w1b_detail.cuh"
#include "../topk_harness.cuh"

struct W1_level_pairs_info
{
  static constexpr const char* id       = "W1_level_pairs";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = "level-parallel tile loop (N2) on native min max pairs, same network and policies as W1_minmax_pairs, N over MaxK a power of two, not_guaranteed only";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w1b::supported(kt, threads, n, k, req, true, true);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_level_pairs : w1b::adapter_impl<KeyT, Threads, N, K, Req, true, true>
{};
