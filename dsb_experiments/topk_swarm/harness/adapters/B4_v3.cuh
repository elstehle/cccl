// Adapter B4_v3: B4_v2 with the register sweeps in the value domain (raw candidate words, twiddle at
// candidate load), unconditional clamped candidate loads and the single-pass ticket emission
// (approaches/B4/block_topk_bound_v3.cuh). v2 was B4_base restructured after the phase profile:
// the candidate image computed once at load time, one histogram atomic per pivot item and a lazy
// block min and max for the tie-class test, slot loops in uniform groups of four, the emission
// classified once into bitmasks, four accumulators in the count sweep. Same algorithm, bound,
// buffer policy and contract as B4_base.
#pragma once

#include "../../approaches/B4/block_topk_bound_v3.cuh"
#include "../topk_harness.cuh"

struct B4_v3_info
{
  static constexpr const char* id       = "B4_v3";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "thread-max bound, compare sweep, striped compaction, one image pass and a warp finish over "
    "the candidates, scan emission, both requirements (v3 code shape)";

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
struct B4_v3
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];

  static constexpr int cap    = (K <= 256) ? 4096 : (K <= 1024) ? 8192 : 16384;
  static constexpr int groups = (K + 2 * Threads - 1) / (2 * Threads) < 1 ? 1 : (K + 2 * Threads - 1) / (2 * Threads);
  static constexpr b4v3::tie_break tb =
    (Req == topk::requirement::prefer_smaller_index) ? b4v3::tie_break::smaller_index : b4v3::tie_break::any;
  using topk_t      = b4v3::block_topk_bound<KeyT, Threads, items_per_thread, cap, groups, tb>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).select(keys, K, out_idx);
  }
};
