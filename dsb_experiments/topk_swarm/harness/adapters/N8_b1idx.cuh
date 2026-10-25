// Adapter N8_b1idx: B1_idx (AIR radix select on striped registers with the index-only ticket
// epilogue, approaches/B1/block_topk_b1.cuh) with the block size opened as a tuning dimension.
// The primitive is B1_idx's, unchanged and inherited, so the 128, 256 and 512 thread
// instantiations compile to the same code as B1_idx at those sizes. The only change is the
// support predicate: 64-thread blocks are accepted, so that the block incumbent is available as
// the same-shape control of the 64-thread B2 cells (the N8 study measures B2 at 64, 128 and 256
// threads in one lock section against this adapter at the same block size).
//
// What the block size changes in this primitive: the histogram sweep per pass is IPT = N / Threads
// items per thread (the dependent part of the pass), the scan over 256 bins is 256 / Threads bins
// per thread (4 at 64 threads, striped bin slots), the barrier costs 16, 20, 28 or 44 cycles at
// 2, 4, 8 or 16 warps (T11), and __launch_bounds__(512) caps the registers at 128 per thread.
//
// Contract: as B1_idx (block scope, striped input in registers, N = Threads x IPT with IPT >= 1,
// k in [1, N), keys fp32 or bf16, not_guaranteed only, k indices unsorted to smem, keys
// clobbered, NaN twiddled as the largest key), with Threads in {64, 128, 256, 512}.
#pragma once

#include "../topk_harness.cuh"
#include "B1_idx.cuh"

struct N8_b1idx_info
{
  static constexpr const char* id       = "N8_b1idx";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "B1_idx with the block size as a tuning dimension (64 to 512 threads), AIR radix select, striped registers in, "
    "index-only ticket scatter to smem out, not_guaranteed only";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 64) || (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (k >= 1) && (k < n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N8_b1idx : B1_idx<KeyT, Threads, N, K, Req>
{};
