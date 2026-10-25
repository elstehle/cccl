// Adapter B3_lean: B3_digit9 with the guarded per-item memory operations written as plain C++
// in the shape ptxas predicates without a reconvergence region: exactly one store inside each
// if, addresses and values computed before it, positions advanced with selects. No inline PTX
// (ptxas turns a predicated red.shared back into a branch anyway, see the B3 report). Otherwise
// as B3_digit9.
#pragma once

#include "../../approaches/B3/block_topk_compact.cuh"
#include "../topk_harness.cuh"

struct B3_lean_info
{
  static constexpr const char* id       = "B3_lean";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "compacting radix select, 9-bit first digit, padded layout, lean store loops, indices to smem out";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 256);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (n <= 32768) && (k >= 1) && (k < n);
    return key_ok && thread_ok && shape_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B3_lean
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t = b3::block_topk_compact<KeyT, Threads, items_per_thread, 4, 9, b3::cand_layout::padded, b3::store_mode::lean>;
  using TempStorage = typename topk_t::TempStorage;

  static constexpr b3::tie_break tb =
    (Req == topk::requirement::prefer_smaller_index) ? b3::tie_break::smaller_index
    : (Req == topk::requirement::prefer_larger_index) ? b3::tie_break::larger_index
                                                      : b3::tie_break::any;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).template select_max_indices<tb>(keys, K, out_idx);
  }
};
