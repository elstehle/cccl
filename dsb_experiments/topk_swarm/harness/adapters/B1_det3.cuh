// Adapter B1_det3: B1_det2 with the tie phase of the deterministic epilogue done per candidate instead of per
// item slot. not_guaranteed: identical to B1_idx (tickets, no -0.0 normalization). prefer_smaller_index:
// -0.0 is normalized to +0.0 once in the prologue so that both zeros form one tie class, and the items tied
// with the k-th key are taken in index order by rank: the ballot word of every slot that holds a candidate
// somewhere in the warp goes to a (warp, slot) table (above 8 items per thread the warps OR their candidate
// masks first and ballot only those slots, up to 8 every slot is balloted), one barrier, every warp with a
// candidate derives its bases by a shuffle scan into its own row of a base table, and only the threads that
// hold a candidate look their rank up (two shared loads and a popcount per candidate, unrolled per slot up
// to 8 items per thread, a loop over the set mask bits above) and write out[total_selected + rank] when the
// rank is below the remaining k.
// Nothing per slot stays in registers but one candidate bit. The emission ticket of the strictly selected
// items is the class-indexed counter of B1_det2 (T57). Everything else as B1_idx (see that header).
#pragma once

#include "../../approaches/B1/block_topk_b1_det3.cuh"
#include "../topk_harness.cuh"

struct B1_det3_info
{
  static constexpr const char* id       = "B1_det3";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "AIR radix select, striped registers in, index-only scatter, prefer_smaller_index by per-candidate rank lookup, class-indexed ticket, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (k >= 1) && (k < n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    const bool ipt_ok    = (n / threads) <= 64; // the candidate mask is 64 bits per thread
    return key_ok && thread_ok && shape_ok && req_ok && ipt_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B1_det3
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  static constexpr bool det                            = (Req == topk::requirement::prefer_smaller_index);
  using key_storage                                    = KeyT[items_per_thread];
  // Per-slot ballots up to 8 items per thread (the sparse form's redux, zeroing and loop cost more than 8
  // ballots), warp-sparse ballots above.
  static constexpr topk_b1_det3::tie_ballots ballots =
    (items_per_thread <= 8) ? topk_b1_det3::tie_ballots::per_slot : topk_b1_det3::tie_ballots::warp_sparse;
  using topk_t = topk_b1_det3::block_topk_b1<KeyT, Threads, items_per_thread, 8, det,
                                        det ? topk_b1_det3::tie_break::smaller_index : topk_b1_det3::tie_break::none,
                                        topk_b1_det3::ticket_mode::per_item, ballots>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).select_indices_striped(keys, K, out_idx);
  }
};
