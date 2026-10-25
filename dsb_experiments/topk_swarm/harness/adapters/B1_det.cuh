// Adapter B1_det: B1_idx with the requirement selecting the tie-break. not_guaranteed: identical to B1_idx
// (tickets, no -0.0 normalization). prefer_smaller_index: -0.0 is normalized to +0.0 once in the prologue so
// that both zeros form one tie class, and the items tied with the k-th key are taken in index order by the
// warp-redundant rank (knowledge base 4.5 approach C): per warp and tile slot a ballot of the candidates, a
// small table in shared memory, one barrier, every warp derives the bases itself by a shuffle scan, and the
// taken ties are written at total_selected + rank without an atomic. Striped tile order is index order, so
// the rank needs no index compare. Everything else as B1_idx (see that header).
#pragma once

#include "../../approaches/B1/block_topk_b1.cuh"
#include "../topk_harness.cuh"

struct B1_det_info
{
  static constexpr const char* id       = "B1_det";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "AIR radix select, striped registers in, index-only scatter, prefer_smaller_index by warp-redundant rank, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (k >= 1) && (k < n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    const bool ipt_ok    = (n / threads) <= 64; // the tie decision is kept in a 64-bit mask per thread
    return key_ok && thread_ok && shape_ok && req_ok && ipt_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B1_det
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  static constexpr bool det                            = (Req == topk::requirement::prefer_smaller_index);
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t = topk_b1::block_topk_b1<KeyT, Threads, items_per_thread, 8, det,
                                        det ? topk_b1::tie_break::smaller_index : topk_b1::tie_break::none,
                                        topk_b1::ticket_mode::per_item>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).select_indices_striped(keys, K, out_idx);
  }
};
