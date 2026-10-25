// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// B1_det3 (opt-B1det3): a copy of block_topk_b1_det2.cuh in its own namespace whose functional change is the
// tie phase of the deterministic (tie_break::smaller_index) path, see resolve_ties. The copy exists so that
// the control B1_det2 (namespace topk_b1_det2) and this variant can share one translation unit.
//
// The tie phase of B1_det2 is per item slot: a ballot per slot and warp with its popcount stored in a table,
// one barrier, then per slot a base shuffle and a rank popcount, with one ballot word and one rank word per
// slot held in registers across the barrier and into the epilogue. B1_det3 is per candidate:
//
//   * The ballot word itself goes to the (warp, slot) table, its popcount is taken when the bases are
//     summed. Nothing per slot stays in registers but one bit per slot (the candidate mask).
//   * With tie_ballots::warp_sparse a warp first ORs its candidate masks (one redux per mask word), zeros its
//     table row, and ballots only the slots that hold a candidate somewhere in the warp (a loop over the set
//     bits of the union, warp uniform, per 32-bit half of the mask). With tie_ballots::per_slot every slot is
//     balloted as in B1_det2 (cheaper up to 8 slots, the adapter chooses by items per thread). In both forms
//     the warps without a candidate (one any-vote) skip the base scan and the lookup.
//   * Every warp derives its own (warp, slot) bases by the shuffle scan as before, with the rounds of 32 slots
//     loaded, summed and scanned side by side.
//   * Only the threads that hold a candidate look their ranks up: a shared load and a popcount give
//     rank = base + popc(ballot[warp][slot] & lt_mask), and the item is written to out[total_selected + rank]
//     when the rank is below the remaining k. Up to 8 slots this is one predicated lookup per slot with the
//     base shuffled from its owning lane and compile-time table offsets, above it the bases go to the warp's
//     row of a base table (warp sync, no block barrier) and a loop over the set bits of the candidate mask
//     (per 32-bit half) reads base and ballot. The epilogue's ticket loop keeps the class-indexed ticket (T57)
//     and no longer branches on a taken mask.
//
// Everything else is B1_det2 (prologue -0.0 normalization under prefer_smaller_index, two radix passes for
// bf16 and up to four for fp32, index-only epilogue by class-indexed tickets), see that header.
//
// Full tiles only (N = Threads x ItemsPerThread), k in [1, N), select the largest.

#include <cub/config.cuh>

#include <cub/block/block_scan.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/util_ptx.cuh>

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/bit>
#include <cuda/std/cstdint>

// Optional phase profile (approaches/B1/profile_det3_phases.cu only): with B1_PROFILE defined, thread 0
// writes clock64 stamps to topk_b1_det3::g_stamps (16 slots per call) at the phase boundaries of
// select_indices_striped, and the end stamp sits behind an extra block barrier. Compiled out otherwise.
// The macro names differ from the B1_det2 header's so that both headers can share a translation unit.
#ifdef B1_PROFILE
#  define B1D3_STAMP(i)                                 \
    do                                                  \
    {                                                   \
      if (threadIdx.x == 0)                             \
      {                                                 \
        ::topk_b1_det3::g_stamps[(i)] = clock64();      \
      }                                                 \
    } while (0)
#  define B1D3_STAMP_SYNC(i) \
    do                       \
    {                        \
      __syncthreads();       \
      B1D3_STAMP(i);         \
    } while (0)
#  define B1D3_STAMP_FLAG(i, v)                           \
    do                                                    \
    {                                                     \
      if (threadIdx.x == 0)                               \
      {                                                   \
        ::topk_b1_det3::g_stamps[(i)] = (long long) (v);  \
      }                                                   \
    } while (0)
#else
#  define B1D3_STAMP(i) \
    do                  \
    {                   \
    } while (0)
#  define B1D3_STAMP_SYNC(i) \
    do                       \
    {                        \
    } while (0)
#  define B1D3_STAMP_FLAG(i, v) \
    do                          \
    {                           \
    } while (0)
#endif

namespace topk_b1_det3
{
#ifdef B1_PROFILE
__device__ long long g_stamps[16];
#endif
using namespace ::cub;
using namespace ::cub::detail;
namespace detail = ::cub::detail;

enum class tie_break
{
  none, // any subset of the items tied with the k-th key, in atomic ticket order
  smaller_index // the tied items with the smallest source indices
};

enum class ticket_mode
{
  per_item, // every selected item draws its own ticket with a shared atomic
  warp_aggregated // ballot per tile slot, one atomic per warp and slot, popc rank inside the warp
};

enum class tie_ballots
{
  per_slot, // one ballot per item slot and warp, as B1_det2
  warp_sparse // one ballot per slot that holds a candidate somewhere in the warp, warps without ties skip
};

// Shared atomic add with return on a 32-bit shared-space address (block scope, relaxed), the ticket of the
// deterministic epilogue. Spelled in PTX so that the address is a plain register value taken once: through
// the generic temp storage reference the compiler recomputes the CTA's shared window base (S2UR SR_CgaCtaId,
// UMOV, ULEA) in front of every ticket atomic once its register target is tight, and the special register
// read lands on every item's ticket chain.
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::uint32_t
shared_atomic_add(::cuda::std::uint32_t shared_addr, ::cuda::std::uint32_t value)
{
  ::cuda::std::uint32_t old;
  asm volatile("atom.relaxed.cta.shared.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(shared_addr), "r"(value));
  return old;
}

template <typename KeyT,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int RadixBits                = 8,
          bool NormalizeSignedZero     = false,
          tie_break TieBreak           = tie_break::none,
          ticket_mode Tickets          = ticket_mode::per_item,
          tie_ballots Ballots          = tie_ballots::warp_sparse>
class block_topk_b1
{
private:
  static constexpr int threads_per_block = ThreadsPerBlock;
  static constexpr int items_per_thread  = ItemsPerThread;
  static constexpr int tile_items        = threads_per_block * items_per_thread;
  static constexpr int num_buckets       = int{1u << RadixBits};
  static constexpr int buckets_per_thread = ::cuda::ceil_div(num_buckets, threads_per_block);
  static constexpr int num_warps          = threads_per_block / 32;
  static constexpr bool striped_histogram = buckets_per_thread > 1 && (num_buckets % threads_per_block == 0);
  static constexpr bool deterministic_ties = TieBreak == tie_break::smaller_index;

  static_assert(threads_per_block % 32 == 0, "whole warps");

  using histo_counter_t = ::cuda::std::uint32_t;
  using block_scan_t    = BlockScan<histo_counter_t, threads_per_block, BLOCK_SCAN_WARP_SCANS>;

  using traits                 = detail::radix::traits_t<KeyT>;
  using bit_ordered_type       = typename traits::bit_ordered_type;
  using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;
  using tie_mask_t = ::cuda::std::conditional_t<(items_per_thread > 32), ::cuda::std::uint64_t, ::cuda::std::uint32_t>;
  static_assert(items_per_thread <= 64 || !deterministic_ties, "the candidate mask is 64 bits per thread");

  struct digit_extractor_t
  {
    ::cuda::std::uint32_t bit_start;
    ::cuda::std::uint32_t mask;
    [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::uint32_t Digit(bit_ordered_type key) const
    {
      return static_cast<::cuda::std::uint32_t>(key >> bit_start) & mask;
    }
  };

  struct TempStorage_
  {
    struct
    {
      // Double-buffered: pass p histograms into buffer p%2 while re-zeroing buffer (p+1)%2
      histo_counter_t histogram[2][num_buckets];
      typename block_scan_t::TempStorage scan_temp_storage;
    } passes;

    struct
    {
      histo_counter_t selected;
      histo_counter_t candidates;
      int bucket;
    } pass_state;

    // Preset before the radix passes (ordered by their barriers), so the scatter needs no setup barrier
    histo_counter_t selected_offset[2];

    // Per (warp, tile slot) candidate ballot words and rank bases of the deterministic tie-break. A warp
    // reads every row of tie_table (after the barrier) and only its own row of base_table (after a warp sync).
    histo_counter_t tie_table[deterministic_ties ? num_warps : 1][deterministic_ties ? items_per_thread : 1];
    int base_table[deterministic_ties ? num_warps : 1][deterministic_ties ? items_per_thread : 1];
  };

  TempStorage_& storage;
  int linear_tid;

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE int item_slot(int i) const
  {
    return i * threads_per_block + linear_tid;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static constexpr int bin_slot(int bin)
  {
    if constexpr (striped_histogram)
    {
      return (bin % buckets_per_thread) * threads_per_block + bin / buckets_per_thread;
    }
    else
    {
      return bin;
    }
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void zero_histogram(histo_counter_t (&histogram)[num_buckets])
  {
    int histo_offset = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + threads_per_block <= num_buckets; histo_offset += threads_per_block)
    {
      histogram[histo_offset + linear_tid] = 0;
    }
    if ((num_buckets % threads_per_block != 0) && (histo_offset + linear_tid < num_buckets))
    {
      histogram[histo_offset + linear_tid] = 0;
    }
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void compute_histograms(
    const bit_ordered_type (&unsigned_keys)[items_per_thread],
    digit_extractor_t digit_extractor,
    bit_ordered_type prefix_mask,
    bit_ordered_type kth_key_prefix,
    histo_counter_t (&histogram)[num_buckets])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const bit_ordered_type key = unsigned_keys[i];
      if ((key & prefix_mask) == kth_key_prefix)
      {
        const auto digit  = static_cast<int>(digit_extractor.Digit(key));
        const auto bucket = num_buckets - 1 - digit;
        atomicAdd_block(&histogram[bin_slot(bucket)], histo_counter_t{1});
      }
    }
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  scan_and_choose_bucket(const histo_counter_t (&histogram)[num_buckets], histo_counter_t k)
  {
    histo_counter_t counts[buckets_per_thread]{};
    const int base = linear_tid * buckets_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        counts[i] = histogram[bin_slot(bin_idx)];
      }
    }

    histo_counter_t exclusive_sums[buckets_per_thread];
    block_scan_t(storage.passes.scan_temp_storage).ExclusiveSum(counts, exclusive_sums);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        const histo_counter_t exclusive = exclusive_sums[i];
        const histo_counter_t inclusive = exclusive + counts[i];
        if (exclusive < k && inclusive >= k)
        {
          storage.pass_state.bucket     = bin_idx;
          storage.pass_state.candidates = counts[i];
          storage.pass_state.selected   = exclusive;
        }
      }
    }
  }

  // One radix pass. Returns true when all remaining candidates are amongst the top-k (early exit).
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool run_radix_pass(
    const bit_ordered_type (&unsigned_keys)[items_per_thread],
    int& k,
    int& num_candidates,
    bit_ordered_type& kth_key_prefix,
    bit_ordered_type& prefix_mask,
    int pass,
    int pass_begin_bit,
    int pass_bits,
    bool zero_next_histogram)
  {
    const bit_ordered_type pass_mask = ::cuda::bitmask<bit_ordered_type>(pass_begin_bit, pass_bits);
    histo_counter_t(&histogram)[num_buckets] = storage.passes.histogram[pass % 2];

    const digit_extractor_t digit_extractor{static_cast<::cuda::std::uint32_t>(pass_begin_bit), (1u << pass_bits) - 1u};
    compute_histograms(unsigned_keys, digit_extractor, prefix_mask, kth_key_prefix, histogram);
    if (zero_next_histogram)
    {
      zero_histogram(storage.passes.histogram[(pass + 1) % 2]);
    }
    __syncthreads();

    scan_and_choose_bucket(histogram, static_cast<histo_counter_t>(k));
    __syncthreads();

    k -= storage.pass_state.selected;
    num_candidates = storage.pass_state.candidates;

    const auto kth_key_digit = num_buckets - 1 - storage.pass_state.bucket;
    kth_key_prefix |= bit_ordered_type(kth_key_digit) << pass_begin_bit;
    prefix_mask |= pass_mask;

    return num_candidates == k;
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void get_kth_key_prefix(
    bit_ordered_type (&unsigned_keys)[items_per_thread],
    int k,
    int& total_selected,
    int& num_candidates,
    bit_ordered_type& kth_key_prefix,
    bit_ordered_type& prefix_mask)
  {
    constexpr int max_bit = int(sizeof(KeyT) * 8);
    kth_key_prefix        = 0;
    prefix_mask           = 0;

    zero_histogram(storage.passes.histogram[0]);
    __syncthreads();

    constexpr int num_passes = ::cuda::ceil_div(max_bit, RadixBits);
    int k_remaining          = k;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int pass = 0; pass < num_passes; ++pass)
    {
      const int pass_end_bit   = max_bit - pass * RadixBits;
      const int pass_begin_bit = (::cuda::std::max) (pass_end_bit - RadixBits, 0);
      const bool done = run_radix_pass(
        unsigned_keys,
        k_remaining,
        num_candidates,
        kth_key_prefix,
        prefix_mask,
        pass,
        pass_begin_bit,
        pass_end_bit - pass_begin_bit,
        pass + 1 < num_passes);
      B1D3_STAMP(2 + pass);
      if (done)
      {
        break;
      }
    }
    total_selected = k - k_remaining;
  }

  // Ballot the slots whose bit is set in the warp-uniform word `warp_half` (slots base .. base + 31), one
  // ballot per set slot, lane 0 stores the ballot word into the warp's row of the tie table.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  ballot_slots(::cuda::std::uint32_t warp_half, ::cuda::std::uint32_t lane_half, int base, int warp, int lane)
  {
    while (warp_half != 0)
    {
      const int b = __ffs(static_cast<int>(warp_half)) - 1;
      warp_half &= warp_half - 1u;
      const unsigned ballot = __ballot_sync(0xffffffffu, ((lane_half >> b) & 1u) != 0u);
      if (lane == 0)
      {
        storage.tie_table[warp][base + b] = ballot;
      }
    }
  }

  // Rank lookup for the candidates whose bit is set in this lane's word `lane_half` (slots base .. base + 31):
  // rank = base of the (warp, slot) plus the candidates of the slot in lower lanes, written out when below take.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void lookup_slots(
    ::cuda::std::uint32_t lane_half, int base, int warp, unsigned lt_mask, int take, int total_selected, int* out_idx)
  {
    while (lane_half != 0)
    {
      const int b = __ffs(static_cast<int>(lane_half)) - 1;
      lane_half &= lane_half - 1u;
      const int i    = base + b;
      const int rank = storage.base_table[warp][i] + __popc(storage.tie_table[warp][i] & lt_mask);
      if (rank < take)
      {
        out_idx[total_selected + rank] = item_slot(i);
      }
    }
  }

  // Deterministic tie-break, per candidate. The candidates are the items equal to the k-th key (every bit is
  // in the prefix when this runs) and `take` of them, the ones with the smallest source indices, are written
  // to out_idx[total_selected, total_selected + take). Striped tile order (slot major, then warp, then lane)
  // is index order, so a candidate's rank is the number of candidates in lower slots plus the number of
  // candidates in its own slot in lower threads, and no index compare is needed.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void resolve_ties(
    const bit_ordered_type (&unsigned_keys)[items_per_thread],
    bit_ordered_type prefix_mask,
    bit_ordered_type kth_prefix,
    int take,
    int total_selected,
    int* out_idx)
  {
    const int warp         = linear_tid / 32;
    const int lane         = linear_tid % 32;
    const unsigned lt_mask = (1u << lane) - 1u;
    constexpr int rounds   = ::cuda::ceil_div(items_per_thread, 32);
    constexpr bool wide    = sizeof(tie_mask_t) == 8;

    // One candidate bit per slot (the compiler builds the word as a LOP3 tree, not a chain). The tie phase
    // runs only when every radix pass ran without an early exit, so the prefix covers every key bit and the
    // candidates are the keys equal to the k-th key, no mask needed.
    _CCCL_ASSERT(prefix_mask == static_cast<bit_ordered_type>(~bit_ordered_type(0)), "ties are exact");
    (void) prefix_mask;
    tie_mask_t cand_mask = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const bool cand = unsigned_keys[i] == kth_prefix;
      cand_mask |= static_cast<tie_mask_t>(cand) << i;
    }
    const ::cuda::std::uint32_t cand_lo = static_cast<::cuda::std::uint32_t>(cand_mask);
    const ::cuda::std::uint32_t cand_hi = wide ? static_cast<::cuda::std::uint32_t>(cand_mask >> (wide ? 32 : 0)) : 0u;

    bool warp_has_candidate;
    if constexpr (Ballots == tie_ballots::warp_sparse)
    {
      // Zero the warp's table row, then ballot only the slots that hold a candidate somewhere in the warp.
      // The union is warp uniform, so the loops and their ballots are too.
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int rnd = 0; rnd < rounds; ++rnd)
      {
        const int slot = rnd * 32 + lane;
        if (slot < items_per_thread)
        {
          storage.tie_table[warp][slot] = 0;
        }
      }
      const ::cuda::std::uint32_t union_lo = __reduce_or_sync(0xffffffffu, cand_lo);
      const ::cuda::std::uint32_t union_hi = wide ? __reduce_or_sync(0xffffffffu, cand_hi) : 0u;
      warp_has_candidate                   = (union_lo | union_hi) != 0u;
      __syncwarp();
      ballot_slots(union_lo, cand_lo, 0, warp, lane);
      if constexpr (wide)
      {
        ballot_slots(union_hi, cand_hi, 32, warp, lane);
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const unsigned ballot = __ballot_sync(0xffffffffu, ((cand_mask >> i) & 1) != 0);
        if (lane == (i % 32))
        {
          storage.tie_table[warp][i] = ballot;
        }
      }
      warp_has_candidate = __any_sync(0xffffffffu, cand_mask != 0);
    }
    __syncthreads();
    B1D3_STAMP(6);

    if (warp_has_candidate)
    {
      // Every warp derives its own (slot, warp) bases: lane l owns slot l per round of 32 slots, sums the
      // ballot populations of its slot over the warps, keeps the prefix of the warps below its own, a shuffle
      // scan orders the slots, and the base goes to the warp's row of the base table. The rounds load, sum
      // and scan side by side and are chained only by the running total at the end.
      histo_counter_t words[rounds][num_warps];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int rnd = 0; rnd < rounds; ++rnd)
      {
        const int slot      = rnd * 32 + lane;
        const bool has_slot = slot < items_per_thread;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int w = 0; w < num_warps; ++w)
        {
          words[rnd][w] = has_slot ? storage.tie_table[w][has_slot ? slot : 0] : 0u;
        }
      }
      int total[rounds];
      int wpre[rounds];
      int incl[rounds];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int rnd = 0; rnd < rounds; ++rnd)
      {
        total[rnd] = 0;
        wpre[rnd]  = 0;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int w = 0; w < num_warps; ++w)
        {
          const int v = __popc(words[rnd][w]);
          wpre[rnd] += (w < warp) ? v : 0;
          total[rnd] += v;
        }
        incl[rnd] = total[rnd];
      }
      constexpr int slots_in_round = (items_per_thread < 32) ? items_per_thread : 32;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int d = 1; d < slots_in_round; d <<= 1)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int rnd = 0; rnd < rounds; ++rnd)
        {
          const int n = __shfl_up_sync(0xffffffffu, incl[rnd], d);
          if (lane >= d)
          {
            incl[rnd] += n;
          }
        }
      }
      // Few slots: the bases stay in the owning lanes and reach the slots by one shuffle each, no base table.
      // Many slots: the bases go to the warp's row of the base table for the loop over the set mask bits.
      constexpr bool unrolled_lookup = items_per_thread <= 8;
      int base_lane[rounds];
      int running = 0;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int rnd = 0; rnd < rounds; ++rnd)
      {
        const int slot = rnd * 32 + lane;
        base_lane[rnd] = running + (incl[rnd] - total[rnd]) + wpre[rnd];
        if constexpr (!unrolled_lookup)
        {
          if (slot < items_per_thread)
          {
            storage.base_table[warp][slot] = base_lane[rnd];
          }
        }
        if (rnd + 1 < rounds)
        {
          running += __shfl_sync(0xffffffffu, incl[rnd], slots_in_round - 1);
        }
      }
      if constexpr (!unrolled_lookup)
      {
        __syncwarp();
      }
      B1D3_STAMP(10);

      // Only the threads that hold a candidate look up their ranks, one lookup per candidate.
      if constexpr (unrolled_lookup)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < items_per_thread; ++i)
        {
          const int base = __shfl_sync(0xffffffffu, base_lane[i / 32], i % 32);
          if (((cand_mask >> i) & 1) != 0)
          {
            const int rank = base + __popc(storage.tie_table[warp][i] & lt_mask);
            if (rank < take)
            {
              out_idx[total_selected + rank] = item_slot(i);
            }
          }
        }
      }
      else
      {
        lookup_slots(cand_lo, 0, warp, lt_mask, take, total_selected, out_idx);
        if constexpr (wide)
        {
          lookup_slots(cand_hi, 32, warp, lt_mask, take, total_selected, out_idx);
        }
      }
    }
    B1D3_STAMP(7);
  }

public:
  using TempStorage = Uninitialized<TempStorage_>;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_b1(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(ThreadsPerBlock, 1, 1))
  {}

  //! Selects the k largest keys of a full striped tile and writes their source indices (item i of thread t
  //! is item i * Threads + t) to out_idx[0, k) in unspecified order. The keys are clobbered.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void select_indices_striped(KeyT (&keys)[items_per_thread], int k, int* out_idx)
  {
    _CCCL_ASSERT(k > 0 && k < tile_items, "k must be in (0, tile_items)");
    B1D3_STAMP(0);

    if (linear_tid == 0)
    {
      storage.selected_offset[0] = 0;
      storage.selected_offset[1] = 0;
    }

    bit_ordered_type(&unsigned_keys)[ItemsPerThread] = reinterpret_cast<bit_ordered_type(&)[ItemsPerThread]>(keys);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      unsigned_keys[i] = bit_ordered_conversion::to_bit_ordered(identity_decomposer_t{}, unsigned_keys[i]);
    }
    if constexpr (NormalizeSignedZero && ::cuda::is_floating_point_v<KeyT>)
    {
      const bit_ordered_type twiddled_minus_zero =
        ::cub::Traits<KeyT>::TwiddleIn(bit_ordered_type(1) << (8 * sizeof(bit_ordered_type) - 1));
      const bit_ordered_type twiddled_zero = ::cub::Traits<KeyT>::TwiddleIn(0);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        unsigned_keys[i] = (unsigned_keys[i] == twiddled_minus_zero) ? twiddled_zero : unsigned_keys[i];
      }
    }

    B1D3_STAMP(1);

    bit_ordered_type kth_prefix{};
    bit_ordered_type prefix_mask{};
    int total_selected{};
    int num_candidates = tile_items;
    get_kth_key_prefix(unsigned_keys, k, total_selected, num_candidates, kth_prefix, prefix_mask);

    // All candidates (items equal to the k-th key's prefix) are selected when they exactly fill the
    // remaining slots. Otherwise the candidates are exact ties (every bit is in the prefix) and a subset of
    // them is taken: by ticket order, or by index order with the deterministic tie-break, which writes the
    // taken ties itself.
    const bool select_all_candidates = num_candidates + total_selected == k;

    if constexpr (deterministic_ties)
    {
      if (!select_all_candidates)
      {
        resolve_ties(unsigned_keys, prefix_mask, kth_prefix, k - total_selected, total_selected, out_idx);
      }
    }

    if constexpr (Tickets == ticket_mode::per_item)
    {
      // The counters' shared address, taken once for the deterministic ticket (see shared_atomic_add)
      [[maybe_unused]] const ::cuda::std::uint32_t counters_shared =
        static_cast<::cuda::std::uint32_t>(__cvta_generic_to_shared(&storage.selected_offset[0]));
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const bit_ordered_type key_prefix = unsigned_keys[i] & prefix_mask;
        const bool is_selected            = key_prefix > kth_prefix;
        const bool is_candidate           = key_prefix == kth_prefix;
        if constexpr (deterministic_ties)
        {
          // The ticket is spelled as in the not_guaranteed path: one atomic per item under the union of
          // the two classes, on a counter indexed by the item's class (T57). With a constant counter
          // address and a constant increment the compiler warp-aggregates the atomic and the special
          // register reads land on the critical path of every item. The ties' tickets (class 1) are drawn
          // and dropped, their positions were written by resolve_ties. The atomic takes the class-indexed
          // 32-bit shared address so that the counters' base is a register value and not recomputed per item.
          const int item_class = (!select_all_candidates) && is_candidate ? 1 : 0;
          if (is_selected || is_candidate)
          {
            const auto ticket =
              shared_atomic_add(counters_shared + 4u * static_cast<::cuda::std::uint32_t>(item_class), 1u);
            if (item_class == 0)
            {
              out_idx[ticket] = item_slot(i);
            }
          }
        }
        else
        {
          const int item_class = (!select_all_candidates) && is_candidate ? 1 : 0;
          if (is_selected || is_candidate)
          {
            // Ties draw tickets beyond the remaining k as well, only the first k_remaining of them are written
            const auto ticket = atomicAdd_block(&storage.selected_offset[item_class], histo_counter_t{1});
            const auto pos    = (item_class == 1) ? static_cast<histo_counter_t>(total_selected) + ticket : ticket;
            if (item_class == 0 || static_cast<int>(ticket) < k - total_selected)
            {
              out_idx[pos] = item_slot(i);
            }
          }
        }
      }
    }
    else
    {
      const int lane         = linear_tid % 32;
      const unsigned lt_mask = (1u << lane) - 1u;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const bit_ordered_type key_prefix = unsigned_keys[i] & prefix_mask;
        const bool is_candidate           = key_prefix == kth_prefix;
        const bool is_selected            = key_prefix > kth_prefix || (is_candidate && select_all_candidates);
        const unsigned sel_ballot         = __ballot_sync(0xffffffffu, is_selected);
        if (sel_ballot != 0)
        {
          histo_counter_t base = 0;
          if (lane == 0)
          {
            base = atomicAdd_block(&storage.selected_offset[0], static_cast<histo_counter_t>(__popc(sel_ballot)));
          }
          base = __shfl_sync(0xffffffffu, base, 0);
          if (is_selected)
          {
            out_idx[base + __popc(sel_ballot & lt_mask)] = item_slot(i);
          }
        }
        if constexpr (!deterministic_ties)
        {
          const unsigned cand_ballot = __ballot_sync(0xffffffffu, is_candidate && !select_all_candidates);
          if (cand_ballot != 0)
          {
            histo_counter_t base = 0;
            if (lane == 0)
            {
              base = atomicAdd_block(&storage.selected_offset[1], static_cast<histo_counter_t>(__popc(cand_ballot)));
            }
            base = __shfl_sync(0xffffffffu, base, 0);
            const int tie_pos = static_cast<int>(base) + __popc(cand_ballot & lt_mask);
            if (is_candidate && !select_all_candidates && tie_pos < k - total_selected)
            {
              out_idx[total_selected + tie_pos] = item_slot(i);
            }
          }
        }
      }
    }
    B1D3_STAMP_SYNC(8);
    B1D3_STAMP_FLAG(9, select_all_candidates ? 0 : 1);
  }
};
} // namespace topk_b1_det3
