// Adapter N8_b3tuned: B3_tuned (compacting radix select with the scan-based index emission,
// approaches/B3/block_topk_compact.cuh, configuration chosen by tile size) with the block size
// opened as a tuning dimension. The primitive is B3_tuned's, unchanged and inherited, so the
// 256-thread instantiation compiles to the same code as B3_tuned. The only change is the support
// predicate: 128 and 512 threads are accepted (B3_tuned declares 256 only, the primitive itself
// is generic in the thread count, B3 report section 6 item 7).
//
// What the block size changes in this primitive: the tile sweeps are IPT = N / Threads items per
// thread, the compacted candidate set is N / 4 / Threads words per thread (16 instead of 32 at
// N 32768 with 512 threads), the first-digit scan is 512 / Threads bins per thread, the barrier
// costs 20, 28 or 44 cycles at 4, 8 or 16 warps (T11), and __launch_bounds__(512) caps the
// registers at 128 per thread. The compaction threshold stays per items per thread (16 for 32-bit
// keys, 32 for 16-bit keys), so at 512 threads and N 16384 the bf16 tile (IPT 32) still compacts
// and at N 8192 it would not.
//
// Contract: as B3_tuned (block scope, blocked input in registers, N = Threads x IPT with IPT >= 1
// and N <= 32768, k in [1, N), keys fp32 or bf16, all three requirements at the same cost, k
// indices unsorted to smem), with Threads in {128, 256, 512}.
#pragma once

#include "../topk_harness.cuh"
#include "B3_tuned.cuh"

struct N8_b3tuned_info
{
  static constexpr const char* id       = "N8_b3tuned";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "B3_tuned with the block size as a tuning dimension (128 to 512 threads), compacting radix select, compaction "
    "and 9-bit first digit from 16 (fp32) or 32 (bf16) items per thread";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (n <= 32768) && (k >= 1) && (k < n);
    return key_ok && thread_ok && shape_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N8_b3tuned : B3_tuned<KeyT, Threads, N, K, Req>
{};
