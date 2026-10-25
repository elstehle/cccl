// Adapter N8_b2packed: B2_packed (atomics based iterative extraction on packed key-index words,
// full per-thread sorted list) with the block size opened as a tuning dimension. The primitive is
// B2_packed's, unchanged and inherited, so the 128, 256 and 512 thread instantiations compile to
// the same code as B2_packed at those sizes. The only change is the support predicate: 64-thread
// blocks (2 warps) are accepted, and the per-thread list is capped at 64 words for both key types
// (B2_packed's bf16 list of 128 words goes to local memory, see T34).
//
// What the block size changes in this primitive: the round is one barrier plus one aggregated
// post per warp plus one read back (T11, T26), so fewer warps per block make the round cheaper,
// while the per-thread list and its sorting network grow with N / Threads (the prologue and the
// round's predicated shift, pruned to min(k, IPT) positions by the compiler, T38 and T41).
//
// Contract: as B2_packed (block scope, blocked input in registers, N = Threads x IPT with IPT a
// power of two, k in [1, min(N/2, 256)], both requirements with one binary, output sorted by key
// descending then index ascending, NaN above +inf or below -inf by twiddle, -0.0 equals +0.0),
// with Threads in {64, 128, 256, 512} and IPT <= 64 for both key types.
#pragma once

#include "../topk_harness.cuh"
#include "B2_packed.cuh"

struct N8_b2packed_info
{
  static constexpr const char* id       = "N8_b2packed";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "B2_packed with the block size as a tuning dimension (64 to 512 threads), packed key-index words, k "
    "fire-and-forget atomicMax rounds, blocked registers in, sorted indices to smem out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 64) || (threads == 128) || (threads == 256) || (threads == 512);
    const int ipt        = (threads > 0) ? n / threads : 0;
    // IPT 128 puts the sorted list into local memory for either key type (T34), the index must fit
    // in 16 bits next to a 16-bit key, k rounds above 256 cost more than any radix path
    const bool shape_ok = (n % threads == 0) && topk::is_pow2(ipt) && (ipt <= 64) && (n <= 65536) && (k >= 1)
                       && (k <= 256) && (2 * k <= n);
    const bool req_ok = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N8_b2packed : B2_packed<KeyT, Threads, N, K, Req>
{};
