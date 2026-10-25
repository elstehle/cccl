// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// Prototype of the B1 optimizer (swarm study): block_topk_air's radix selection (striped branch, commit
// c7cf73b09d) with an index-only epilogue. The radix passes (atomic histogram over the surviving candidates,
// fused scan and bucket choice, pass state broadcast) are the striped header's. What changes:
//
//   * Index-only output. The selected items' source indices are scattered straight into the caller's output
//     array (k ints in shared memory) by ticket. There is no exchange buffer, no key gather, no second trip for
//     the values, and no register copy of the original keys, so the temp storage is independent of the tile
//     size and the keys may be clobbered.
//   * Source indices are derived from the striped tile position (item i of thread t is item i * Threads + t)
//     instead of being carried in registers.
//   * Raw shift-and-mask digit extractor (knowledge base 4.4): the -0.0 to +0.0 mapping that
//     ShiftDigitExtractor repeats on every key in every pass is dropped. NormalizeSignedZero selects whether
//     the prologue maps twiddled -0.0 onto +0.0 once (needed when -0.0 and +0.0 must form one tie class, as
//     under prefer_smaller_index) or leaves the keys as they are (any valid top-k set under not_guaranteed,
//     where the oracle compares keys with ==).
//   * Ticket scheme for the selected items: one shared atomic with return per selected item (per_item), or
//     one ballot per tile slot per warp, one atomic per warp and slot, and a popc rank within the warp
//     (warp_aggregated).
//   * Tie-break among the items equal to the k-th key (tie_break::smaller_index): the warp-redundant rank of
//     knowledge base 4.5 approach C. Per warp and slot a ballot of the candidates goes to a small table, one
//     barrier, every warp derives the (slot, warp) bases itself by a shuffle scan, and a candidate is taken
//     when its rank in striped tile order (which is index order) is below the remaining k. Taken ties are
//     written at total_selected + rank, without an atomic.
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

namespace topk_b1
{
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

template <typename KeyT,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int RadixBits                = 8,
          bool NormalizeSignedZero     = false,
          tie_break TieBreak           = tie_break::none,
          ticket_mode Tickets          = ticket_mode::per_item>
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
  static_assert(items_per_thread <= 64 || !deterministic_ties, "the tie decision is kept in a 64-bit mask");

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

    // Per (warp, tile slot) candidate counts of the deterministic tie-break
    histo_counter_t tie_table[deterministic_ties ? num_warps : 1][deterministic_ties ? items_per_thread : 1];
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
      if (run_radix_pass(
            unsigned_keys,
            k_remaining,
            num_candidates,
            kth_key_prefix,
            prefix_mask,
            pass,
            pass_begin_bit,
            pass_end_bit - pass_begin_bit,
            pass + 1 < num_passes))
      {
        break;
      }
    }
    total_selected = k - k_remaining;
  }

  // Deterministic tie-break: a bit per item, set when the item is a candidate whose rank in striped tile
  // order (index order) is below `take`. Also returns the rank base per slot through the lambda's captures.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void resolve_ties(
    const bit_ordered_type (&unsigned_keys)[items_per_thread],
    bit_ordered_type prefix_mask,
    bit_ordered_type kth_prefix,
    int take,
    tie_mask_t& taken,
    int (&tie_rank)[items_per_thread])
  {
    const int warp         = linear_tid / 32;
    const int lane         = linear_tid % 32;
    const unsigned lt_mask = (1u << lane) - 1u;
    taken                  = 0;

    unsigned ballots[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const bool cand = (unsigned_keys[i] & prefix_mask) == kth_prefix;
      ballots[i]      = __ballot_sync(0xffffffffu, cand);
      if (lane == (i % 32))
      {
        storage.tie_table[warp][i] = __popc(ballots[i]);
      }
    }
    __syncthreads();

    // Every warp derives the (slot, warp) bases itself: lane l owns slot l per round of 32 slots, reads the
    // counts of its slot, keeps the prefix of the warps below its own, and a shuffle scan orders the slots.
    constexpr int rounds = ::cuda::ceil_div(items_per_thread, 32);
    int base_lane[rounds];
    int running = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int rnd = 0; rnd < rounds; ++rnd)
    {
      const int slot      = rnd * 32 + lane;
      const bool has_slot = slot < items_per_thread;
      int total           = 0;
      int wpre            = 0;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int w = 0; w < num_warps; ++w)
      {
        const int v = has_slot ? static_cast<int>(storage.tie_table[w][has_slot ? slot : 0]) : 0;
        wpre += (w < warp) ? v : 0;
        total += v;
      }
      constexpr int slots_in_round = (items_per_thread < 32) ? items_per_thread : 32;
      int incl                     = total;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int d = 1; d < slots_in_round; d <<= 1)
      {
        const int n = __shfl_up_sync(0xffffffffu, incl, d);
        if (lane >= d)
        {
          incl += n;
        }
      }
      base_lane[rnd] = running + (incl - total) + wpre;
      running += __shfl_sync(0xffffffffu, incl, slots_in_round - 1);
    }
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const int base = __shfl_sync(0xffffffffu, base_lane[i / 32], i % 32);
      tie_rank[i]    = base + __popc(ballots[i] & lt_mask);
      taken |= static_cast<tie_mask_t>(tie_rank[i] < take) << i;
    }
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

    bit_ordered_type kth_prefix{};
    bit_ordered_type prefix_mask{};
    int total_selected{};
    int num_candidates = tile_items;
    get_kth_key_prefix(unsigned_keys, k, total_selected, num_candidates, kth_prefix, prefix_mask);

    // All candidates (items equal to the k-th key's prefix) are selected when they exactly fill the
    // remaining slots. Otherwise the candidates are exact ties (every bit is in the prefix) and a subset of
    // them is taken: by ticket order, or by index order with the deterministic tie-break.
    const bool select_all_candidates = num_candidates + total_selected == k;

    [[maybe_unused]] tie_mask_t taken = 0;
    [[maybe_unused]] int tie_rank[items_per_thread];
    if constexpr (deterministic_ties)
    {
      if (!select_all_candidates)
      {
        resolve_ties(unsigned_keys, prefix_mask, kth_prefix, k - total_selected, taken, tie_rank);
      }
    }

    if constexpr (Tickets == ticket_mode::per_item)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const bit_ordered_type key_prefix = unsigned_keys[i] & prefix_mask;
        const bool is_selected            = key_prefix > kth_prefix;
        const bool is_candidate           = key_prefix == kth_prefix;
        if constexpr (deterministic_ties)
        {
          if (is_selected || (is_candidate && select_all_candidates))
          {
            // The counter index goes through an empty asm statement: with a provably uniform address and a
            // constant increment the compiler aggregates the atomic across the warp (VOTE, POPC, FLO and
            // S2R SR_LTMASK per site), and the special register reads land on the critical path of every
            // item. An index the compiler cannot prove uniform keeps the plain per-item atomic.
            int sel_class = 0;
            asm volatile("" : "+r"(sel_class));
            const auto ticket = atomicAdd_block(&storage.selected_offset[sel_class], histo_counter_t{1});
            out_idx[ticket]   = item_slot(i);
          }
          else if (is_candidate && ((taken >> i) & 1))
          {
            out_idx[total_selected + tie_rank[i]] = item_slot(i);
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
        if constexpr (deterministic_ties)
        {
          if (is_candidate && !select_all_candidates && ((taken >> i) & 1))
          {
            out_idx[total_selected + tie_rank[i]] = item_slot(i);
          }
        }
        else
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
  }
};
} // namespace topk_b1
