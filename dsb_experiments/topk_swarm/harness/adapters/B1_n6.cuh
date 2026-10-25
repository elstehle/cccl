// Adapter B1_n6: N6 equi-width bucket select (approaches/B1/block_topk_n6.cuh). The keys are mapped onto a
// 23-bit fixed-point image over the tile's value bracket (hi the tile max, lo the min over threads of the
// per-thread max for k <= Threads, else the tile min) and selected by block_topk_air's pass machinery on 8,
// 8 and 7 bit digits of that image, with exact exits (pivot bucket count equals the remaining k, or the
// candidates are one tie class by tracked key min and max) and radix passes on the key bits over the
// candidates as the exact fallback. Index-only epilogue as B1_idx. The requirement selects the tie-break:
// not_guaranteed tickets, prefer_smaller_index the warp-redundant rank with -0.0 normalized to +0.0.
//
// Contract:
//   scope block, 256 threads (128 and 512 accepted), striped input in registers, N = Threads x IPT with
//   IPT >= 1 (IPT <= 64 under prefer_smaller_index), k in [1, N), keys fp32 or bf16, both unsorted
//   requirements. Output: k indices in the harness's shared output array, unsorted. Temp storage:
//   histogram double buffer, scan storage, pass state, bracket partials, key min and max trackers, two
//   counters, independent of N. Sync: __syncthreads inside, assumes a barrier between consecutive calls.
//   NaN not supported (never produced by focus-set inputs), infinities clamp into the end bins and are
//   ordered exactly by the key passes if the pivot falls on them. The input keys are not modified.
#pragma once

#include "../../approaches/B1/block_topk_n6.cuh"
#include "../topk_harness.cuh"

struct B1_n6_info
{
  static constexpr const char* id       = "B1_n6";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "N6 fixed-point bucket select over the value bracket, striped registers in, index-only scatter, both requirements";

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
struct B1_n6
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  static constexpr bool det                            = (Req == topk::requirement::prefer_smaller_index);
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t = topk_b1::block_topk_n6<KeyT, Threads, items_per_thread, det,
                                        det ? topk_b1::tie_break::smaller_index : topk_b1::tie_break::none>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).select_indices_striped(keys, K, out_idx);
  }
};
