// Adapter N5_split: one bf16 row of N keys per call, split into its even and odd items as the two
// halves of 32-bit words (the harness's packed_striped arrangement), the W1 eager bitonic top-k
// network run once on both sets with max.bf16x2 and min.bf16x2, the two top-MaxK results merged
// by one more bitonic merge on one-key words, indices recovered from the k-th key (greater keys by
// lane scan, equal keys in index order by ballot rank). approaches/N5/n5_detail.cuh. The single-row
// form of N5_rows, a drop-in for the one-word W1 route.
//
// Contract:
//   scope warp (32 lanes here, the _sub adapter declares 8 and 16), packed_striped input (word j
//   of lane l holds items 2 (j LW + l) and 2 (j LW + l) + 1), N = 2 x LW x words with
//   MaxK = round_up(k, LW) dividing N / 2 and (N / 2) / MaxK a power of two, k in [1, N / 2],
//   bf16 keys only, both unsorted requirements with the same code (prefer_smaller_index is exact
//   by construction). Output: k indices in out_idx[0, K), unsorted (greater keys in lane order,
//   then the tie class in index order). Temp storage none. Sync: shuffles and ballots over the
//   full warp. -0.0 equals +0.0. NaN keys are never selected and leave output slots unwritten,
//   never produced by focus-set inputs. N 32 at warp scope is unsupported (each set would hold
//   fewer keys than lanes). fp32 keys are unsupported: two fp32 keys do not fit one register.
#pragma once

#include "../../approaches/N5/n5_detail.cuh"

struct N5_split_info
{
  static constexpr const char* id       = "N5_split";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "one bf16 row split into even and odd items as bf16x2 words, eager bitonic top-k with max and min bf16x2 on both "
    "sets, one merge, index recovery from the k-th key, MaxK = round_up(k, LW), both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n5::split_supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N5_split : n5::split_adapter<KeyT, Threads, N, K, Req, n5::ops_bf16>
{};
