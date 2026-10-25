// Adapter N9_w2fin2: N9_w2fin with the tickets of the epilogue drawn from one class-indexed
// counter (T23 and T39: a constant-address ticket is warp-aggregated by ptxas). B1_n6b (N6 fixed-point bucket select with per-bucket tie-class tracking) with
// one change, idea N9: when the pivot bin after the first pass holds at most 32 items and the pass
// did not end the selection, the items above the pivot bin are scattered by ticket and the pivot
// bin finishes on warp 0 by packed redux rounds (approaches/B4/block_topk_n6_w2.cuh), instead of
// the second pass over the whole tile. Measured against B1_n6b, the same code without the branch.
//
// Contract: as B1_n6b (block, striped registers in, index-only output, both requirements, N6's
// NaN and signed-zero behavior), plus 264 bytes of temp storage for the pivot buffer.
#pragma once

#include "../../approaches/B4/block_topk_n6_w2.cuh"
#include "../topk_harness.cuh"

struct N9_w2fin2_info
{
  static constexpr const char* id       = "N9_w2fin2";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "N6 (B1_n6b) with the second pass replaced by a packed redux finish on one warp when the pivot "
    "bin holds at most 32 items, class-indexed tickets, striped registers in, index-only out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (n <= 32768) && (k >= 1) && (k < n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    const bool ipt_ok    = (req == topk::requirement::not_guaranteed) || (n / threads) <= 64;
    return key_ok && thread_ok && shape_ok && req_ok && ipt_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N9_w2fin2
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  static constexpr bool det                            = (Req == topk::requirement::prefer_smaller_index);
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t = topk_b1::block_topk_n6w2<KeyT, Threads, items_per_thread, det,
                                          det ? topk_b1::tie_break::smaller_index : topk_b1::tie_break::none, true, true>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).select_indices_striped(keys, K, out_idx);
  }
};
