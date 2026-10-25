// Adapter B3_digit9: B3_straight with a 9-bit first digit (sign plus the whole exponent for
// float keys), so that one first-pass bucket is one binade and the pivot bucket on normal data
// holds about 14 percent of N at most instead of 29 percent with the 8-bit digit that pairs two
// binades. The histogram has 512 bins (two per thread in the scan), later digits are 8 bits
// (fp32: 9, 8, 8, 7 bits over four passes, bf16: 9 and 7 bits over two). Otherwise as B3_straight.
#pragma once

#include "../../approaches/B3/block_topk_compact.cuh"
#include "../topk_harness.cuh"

struct B3_digit9_info
{
  static constexpr const char* id       = "B3_digit9";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "compacting radix select, 9-bit first digit, padded layout, predicated stores, indices to smem out";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 256);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (n <= 32768) && (k >= 1) && (k < n);
    return key_ok && thread_ok && shape_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B3_digit9
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t = b3::block_topk_compact<KeyT, Threads, items_per_thread, 4, 9, b3::cand_layout::padded, b3::store_mode::predicated>;
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
