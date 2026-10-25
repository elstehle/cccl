// Adapter B1_nz: B1_idx plus the prologue -0.0 to +0.0 normalization (NormalizeSignedZero), nothing else.
// Isolates the cost of the normalization from the cost of the deterministic tie-break code of B1_det.
// Same contract as B1_idx (not_guaranteed only).
#pragma once

#include "../../approaches/B1/block_topk_b1.cuh"
#include "../topk_harness.cuh"

struct B1_nz_info
{
  static constexpr const char* id       = "B1_nz";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "AIR radix select, striped registers in, index-only ticket scatter, -0.0 normalized to +0.0 once, not_guaranteed only";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (k >= 1) && (k < n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B1_nz
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t = topk_b1::block_topk_b1<KeyT, Threads, items_per_thread, 8, true, topk_b1::tie_break::none,
                                        topk_b1::ticket_mode::per_item>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).select_indices_striped(keys, K, out_idx);
  }
};
