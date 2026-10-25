// Adapter N10_ge1_trunc: fp32 keys-only network with index recovery by ballot rank (idea N10).
// eager tile order, twiddled fp32 u32 words through the min and max exchange, final merge truncated at stride k, k-th word by redux.min, index recovery by one ballot per slot with the two-class pass behind a uniform branch for excess ties
// Network, threshold and recovery are in approaches/N10/n10_detail.cuh.
//
// Contract:
//   scope warp (32 lanes) and sub-warp (4, 8, 16 lanes through the _sub adapter), striped input in
//   registers, N = LW x IPT with MaxK = max(LW, next_pow2(k)) dividing N and N / MaxK a power of two,
//   keys fp32 only, both unsorted requirements (prefer_smaller_index exact by construction, same
//   code for not_guaranteed). Output: k source indices in the harness output array, unsorted (the
//   items above the k-th key in index order, then the tied items in index order). Temp storage
//   none. Sync: shuffles, ballots and redux on the full warp, all lanes of the hardware warp in
//   lockstep. NaN: ordered by its sign bit, never produced by focus-set inputs. -0.0 equals +0.0.
#pragma once

#include "../../approaches/N10/n10_detail.cuh"
#include "../topk_harness.cuh"

struct N10_ge1_trunc_info
{
  static constexpr const char* id       = "N10_ge1_trunc";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = "eager tile order, twiddled fp32 u32 words through the min and max exchange, final merge truncated at stride k, k-th word by redux.min, index recovery by one ballot per slot with the two-class pass behind a uniform branch for excess ties";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n10::supported(kt, threads, n, k, req, false);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N10_ge1_trunc : n10::adapter_impl<KeyT, Threads, N, K, Req, true, false, true>
{};
