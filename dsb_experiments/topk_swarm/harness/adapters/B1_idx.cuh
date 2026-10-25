// Adapter B1_idx: block_topk_air's radix selection (striped branch) with an index-only epilogue and no key
// memoization (approaches/B1/block_topk_b1.cuh). The selected items' striped source indices are scattered
// straight into the harness's output array by per-item shared atomic tickets. Raw shift-and-mask digit
// extractor, no -0.0 normalization (any valid top-k set is accepted under not_guaranteed and -0.0 == +0.0
// for the oracle).
//
// Contract:
//   scope block, 256 threads (128 and 512 accepted), striped input in registers (thread t holds items
//   t + i * Threads), N = Threads x IPT with IPT >= 1, k in [1, N), keys fp32 or bf16, requirement
//   not_guaranteed only in this variant (ties at the k-th key by atomic ticket order). Output: k indices in
//   the harness's shared output array, unsorted. Temp storage: histogram double buffer, scan storage, pass
//   state and two counters, independent of N. Sync: __syncthreads inside, assumes the caller separates
//   consecutive calls by a barrier (the harness does). NaN: twiddled as the largest key, never produced by
//   focus-set inputs. The input keys are clobbered (twiddled in place).
#pragma once

#include "../../approaches/B1/block_topk_b1.cuh"
#include "../topk_harness.cuh"

struct B1_idx_info
{
  static constexpr const char* id       = "B1_idx";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "AIR radix select, striped registers in, index-only ticket scatter to smem out, no key memoization, not_guaranteed only";

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
struct B1_idx
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t = topk_b1::block_topk_b1<KeyT, Threads, items_per_thread, 8, false, topk_b1::tie_break::none,
                                        topk_b1::ticket_mode::per_item>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).select_indices_striped(keys, K, out_idx);
  }
};
