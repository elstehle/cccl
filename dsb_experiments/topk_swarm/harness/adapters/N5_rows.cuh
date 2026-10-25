// Adapter N5_rows: two independent bf16 rows of N keys per call in the halves of one 32-bit word,
// the W1 eager bitonic top-k network (template recursion, T42 exchange) run once on both rows with
// max.bf16x2 and min.bf16x2, indices recovered after the network from each row's k-th key (greater
// keys by lane scan, equal keys in index order by ballot rank). approaches/N5/n5_detail.cuh.
//
// Contract:
//   scope warp (32 lanes here, the _sub adapter declares 8 and 16), rows 2, the harness's
//   packed_rows arrangement (word j of lane l is item j * LW + l of both rows, row 0 low, row 1
//   high), N = LW x IPT with MaxK = round_up(k, LW) dividing N and N / MaxK a power of two,
//   k in [1, N / 2], bf16 keys only (two per word), both unsorted requirements with the same code
//   (prefer_smaller_index is exact by construction). Output: row r's k indices at
//   out_idx[r * K, (r + 1) * K), unsorted (greater keys in lane order, then the tie class in index
//   order). Temp storage none. Sync: shuffles and ballots over the full warp. -0.0 equals +0.0.
//   NaN keys are never selected and leave output slots unwritten, never produced by focus-set
//   inputs. fp32 keys are unsupported: two fp32 keys do not fit one register.
#pragma once

#include "../../approaches/N5/n5_detail.cuh"

struct N5_rows_info
{
  static constexpr const char* id       = "N5_rows";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "two bf16 rows per call in bf16x2 words, eager bitonic top-k with max and min bf16x2, index recovery from the k-th "
    "key, MaxK = round_up(k, LW), both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n5::rows_supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N5_rows : n5::rows_adapter<KeyT, Threads, N, K, Req, n5::ops_bf16>
{};
