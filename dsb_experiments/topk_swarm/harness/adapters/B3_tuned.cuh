// Adapter B3_tuned: the B3 configuration chosen by tile size, on top of B3_digit9's code with
// the scans and zeroing limited to the live bins of each pass (256 bins after the 9-bit first
// digit instead of 512). From 16 items per thread for 32-bit keys and from 32 for 16-bit keys:
// 9-bit first digit, compaction into a buffer of N / 4 candidates, padded layout, predicated
// stores. Below that: 8-bit digits and no compaction (tile sweeps plus the scan-based index
// emission), where compaction measured as a loss (fp32 has two more passes to save than bf16,
// so it pays one tile size earlier). Contract as B3_baseline.
#pragma once

#include "../../approaches/B3/block_topk_compact.cuh"
#include "../topk_harness.cuh"

struct B3_tuned_info
{
  static constexpr const char* id       = "B3_tuned";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "compacting radix select, compaction and 9-bit first digit from 16 (fp32) or 32 (bf16) items per thread";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 256);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (n <= 32768) && (k >= 1) && (k < n);
    return key_ok && thread_ok && shape_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B3_tuned
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];

  static constexpr int compact_from  = (sizeof(KeyT) >= 4) ? 16 : 32;
  static constexpr bool big          = items_per_thread >= compact_from;
  static constexpr int cap_divisor   = big ? 4 : 0;
  static constexpr int first_digit   = big ? 9 : 8;
  using topk_t = b3::block_topk_compact<KeyT,
                                        Threads,
                                        items_per_thread,
                                        cap_divisor,
                                        first_digit,
                                        b3::cand_layout::padded,
                                        b3::store_mode::predicated>;
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
