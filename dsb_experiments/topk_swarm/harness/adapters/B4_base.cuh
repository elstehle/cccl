// Adapter B4_base: exact-bound prefilter, compaction and a small finish
// (approaches/B4/block_topk_bound.cuh). The bound is the minimum over threads of the per-thread
// maximum (Groups = 1 for k up to 2 x Threads, the candidate count is checked against k at run
// time), or of the per-thread group maxima for larger k. Candidates go to a shared buffer of
// Cap items (4096 for k up to 256, 8192 up to 1024, 16384 above, never more than N), the finish
// runs one image pass over the candidates and resolves the pivot bin on one warp. Both
// requirements, one code path (prefer_smaller_index folds -0.0 onto +0.0 and ranks tie classes
// larger than the warp finish by index).
//
// Contract:
//   scope block, 256 threads (128 and 512 accepted), striped input in registers (thread t holds
//   items t + i * Threads), N = Threads x IPT with N <= 32768, k in [1, N), keys fp32 or bf16,
//   requirements not_guaranteed and prefer_smaller_index. Output: k indices in the harness's
//   shared output array, unsorted. Temp storage: histograms and trackers (6 KB), scan storage,
//   the candidate buffer (Cap x 4 bytes for bf16, x 8 for fp32). Sync: __syncthreads inside,
//   assumes the caller separates consecutive calls by a barrier (the harness does). NaN: see the
//   primitive header, focus-set inputs contain no NaN.
#pragma once

#include "../../approaches/B4/block_topk_bound.cuh"
#include "../topk_harness.cuh"

struct B4_base_info
{
  static constexpr const char* id       = "B4_base";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "thread-max bound, compare sweep, striped compaction, one image pass and a warp finish over "
    "the candidates, scan emission, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (n <= 32768) && (k >= 1) && (k < n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B4_base
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];

  static constexpr int cap    = (K <= 256) ? 4096 : (K <= 1024) ? 8192 : 16384;
  static constexpr int groups = (K + 2 * Threads - 1) / (2 * Threads) < 1 ? 1 : (K + 2 * Threads - 1) / (2 * Threads);
  static constexpr b4::tie_break tb =
    (Req == topk::requirement::prefer_smaller_index) ? b4::tie_break::smaller_index : b4::tie_break::any;
  using topk_t      = b4::block_topk_bound<KeyT, Threads, items_per_thread, cap, groups, tb>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).select(keys, K, out_idx);
  }
};
