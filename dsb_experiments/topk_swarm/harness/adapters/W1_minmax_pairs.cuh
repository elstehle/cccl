// Adapter W1_minmax_pairs: the eager tile loop on native (key, index) pairs where the compare-exchange is a min and a max of
// the keys (issue #12160) and the index follows the key that moved: fp32 with fmaxf and fminf, bf16
// with __hmax and __hmin. Two shuffles per stage as the reference, one FMNMX on the key path.
// Element policies and the network are in approaches/W1/w1b_detail.cuh.
//
// Contract:
//   scope warp or sub-warp (4, 8, 16, 32 lanes), striped input in registers, N = LW x IPT, k in
//   [1, N] with MaxK = max(LW, next_pow2(k)) dividing N, keys fp32 or bf16,
//   requirement not_guaranteed (ties keep their network position). Output: k indices in
//   the harness output array, sorted by key descending. Temp storage none. Sync: shuffles only.
//   NaN: see w1b_detail.cuh, never produced by focus-set inputs. -0.0 and +0.0 are equal.
#pragma once

#include "../../approaches/W1/w1b_detail.cuh"
#include "../topk_harness.cuh"

struct W1_minmax_pairs_info
{
  static constexpr const char* id       = "W1_minmax_pairs";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = "eager tile loop on native pairs, key path through min and max (fp32 fmaxf fminf, bf16 __hmax __hmin), index follows the key, not_guaranteed only";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w1b::supported(kt, threads, n, k, req, false, true);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1_minmax_pairs : w1b::adapter_impl<KeyT, Threads, N, K, Req, false, true>
{};
