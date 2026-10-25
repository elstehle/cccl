// Adapter B3_baseline: compacting radix select (approaches/B3/block_topk_compact.cuh), the
// simplest correct form. AIR pass over the tile, then the items above the pivot bucket are
// emitted and the pivot bucket is compacted into shared memory (capacity N / 4) in index order,
// later passes sweep the compacted candidates in registers. Positions from block scans, no
// atomic tickets, so both tie-break requirements run the same code.
//
// Contract:
//   scope block, 256 threads, blocked input in registers, N = 256 x IPT with IPT >= 1 and
//   N <= 32768, k in [1, N), keys fp32 or bf16, requirements not_guaranteed, prefer_smaller_index
//   and prefer_larger_index (inherent, index-ordered ranks among the ties at the k-th key).
//   Output: k indices in the harness's shared output array, unsorted (above-pivot items first
//   in index order, then the candidates in index order). Temp storage: two 256-bin histograms,
//   scan storage, pass state, N / 4 candidate words (4 bytes bf16, 8 bytes fp32).
//   Sync: __syncthreads inside, assumes the caller separates consecutive calls that reuse the
//   temp storage by a barrier (the harness does). NaN: positive NaN ranks above +inf and is
//   selected first, negative NaN ranks below -inf. -0.0 is normalized to +0.0 for ranking.
#pragma once

#include "../../approaches/B3/block_topk_compact.cuh"
#include "../topk_harness.cuh"

struct B3_baseline_info
{
  static constexpr const char* id       = "B3_baseline";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "compacting radix select, blocked registers in, indices to smem out, all tie-break requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 256);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (n <= 32768) && (k >= 1) && (k < n);
    return key_ok && thread_ok && shape_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B3_baseline
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t      = b3::block_topk_compact<KeyT, Threads, items_per_thread>;
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
