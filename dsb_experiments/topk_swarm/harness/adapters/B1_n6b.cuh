// Adapter B1_n6b: B1_n6 with per-bucket tracking of the candidates' key min and max from the second pass on,
// so the tie-class exit fires as soon as the pivot bucket is a tie class (and not one pass later when the
// whole candidate set is one). Everything else as B1_n6 (see that header and approaches/B1/block_topk_n6.cuh).
#pragma once

#include "../../approaches/B1/block_topk_n6.cuh"
#include "../topk_harness.cuh"

struct B1_n6b_info
{
  static constexpr const char* id       = "B1_n6b";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "N6 fixed-point bucket select, per-bucket tie-class tracking, striped registers in, index-only scatter, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (k >= 1) && (k < n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    const bool ipt_ok    = (req == topk::requirement::not_guaranteed) || (n / threads) <= 64;
    return key_ok && thread_ok && shape_ok && req_ok && ipt_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B1_n6b
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  static constexpr bool det                            = (Req == topk::requirement::prefer_smaller_index);
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t = topk_b1::block_topk_n6<KeyT, Threads, items_per_thread, det,
                                        det ? topk_b1::tie_break::smaller_index : topk_b1::tie_break::none, true>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).select_indices_striped(keys, K, out_idx);
  }
};
