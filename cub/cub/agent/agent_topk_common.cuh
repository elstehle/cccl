// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Shared device-side primitives consumed by both the single-problem top-k agents in
//! `cub/agent/agent_topk.cuh` and the segmented multi-CTA top-k agents in
//! `cub/agent/agent_batched_topk.cuh`.
//!
//! Contents:
//!   - `AgentTopKPolicy`                  — pure NTTP policy knobs.
//!   - `key_prefix_storage_t` and helpers — kth-key bit accumulator and start-bit math.
//!   - `counter`                          — device-side coordination state for the top-k passes.
//!   - `init_histogram` / `merge_histogram` / `finalize_pass` — block/grid histogram primitives.
//!   - `block_identify_kth_bucket`        — last-block prefix-sum + bucket-finder.
//!   - `sink_mode` enum                   — filter-partition agent's runtime sink selector.
//!   - Filter / callback adapter ops      — predicates and callbacks consumed by the block primitives.
//!
//! Nothing in this header is single-problem-specific: every type and free function takes either
//! per-launch arguments at call site or works on data the caller passes in. The single-problem
//! agents bind these to a single `counter` / `global_histogram` / `retirement_count`; the segmented
//! agents bind them per segment via `[queue_idx]`-indexed slabs.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
//! @brief Parameterizable tuning policy type for AgentTopK
//!
//! @tparam ThreadsPerBlock
//!   Threads per thread block
//!
//! @tparam ItemsPerThread
//!   Items per thread (per tile of input)
//!
//! @tparam BitsPerPass
//!   Number of bits processed per pass
//!
//! @tparam ScanAlgorithm
//!   The BlockScan algorithm to use
//!
//! @tparam KeysTileLoadKind
//!   The `tile_load_kind` used by the keys-stream `TileDataSource`. Architecture
//!   §2.4: unifies sync `BlockLoadAlgorithm` choices and adds async TMA, so this
//!   subsumes what was historically expressed as a `BlockLoadAlgorithm`.
//!
//! @tparam AccumulatingBufferCapacity
//!   Number of smem slots in the per-stream buffer for the
//!   `AccumulatingCandidates` partition strategy / `AccumulatingFilter` filter
//!   strategy (irrelevant for the non-accumulating strategies). Also reused
//!   as the candidate-stream buffer capacity for the `SpeculativeBoth` /
//!   `SpeculativeFilter` strategies.
//!
//! @tparam SpeculativeSelectedBufferCapacity
//!   Number of smem slots in the selected-stream buffer for the
//!   `SpeculativeBoth` partition strategy (the agent's selected stream goes
//!   to a typically-dense output, so a smaller capacity than the candidate
//!   buffer often pays best). `0` short-circuits the selected smem buffer
//!   to pure-Atomics, useful when the selected stream is dense enough that
//!   buffering does not pay. Ignored by every other strategy.
//!
template <int ThreadsPerBlock,
          int ItemsPerThread,
          int BitsPerPass,
          BlockScanAlgorithm ScanAlgorithm,
          tile_load_kind KeysTileLoadKind         = tile_load_kind::block_load_vectorize,
          int AccumulatingBufferCapacity          = 256,
          int SpeculativeSelectedBufferCapacity   = 128>
struct AgentTopKPolicy
{
  static constexpr int block_threads                  = ThreadsPerBlock;
  static constexpr int items_per_thread               = ItemsPerThread;
  static constexpr int bits_per_pass                  = BitsPerPass;
  static constexpr BlockScanAlgorithm scan_algorithm  = ScanAlgorithm;
  // Architecture §2.4: unifies sync `BlockLoadAlgorithm` choices and adds async TMA.
  // Used by the new agents to pick a TileDataSource specialization for the keys
  // stream. Defaults to the legacy `BLOCK_LOAD_VECTORIZE` mapping so existing call
  // sites that don't set it preserve current behavior.
  static constexpr tile_load_kind keys_tile_load_kind         = KeysTileLoadKind;
  static constexpr int accumulating_buffer_capacity           = AccumulatingBufferCapacity;
  static constexpr int speculative_selected_buffer_capacity   = SpeculativeSelectedBufferCapacity;
};

template <typename KeyT, bool CanTwiddle = detail::radix::can_twiddle<KeyT>>
struct key_prefix_storage_t;

template <typename KeyT>
struct key_prefix_storage_t<KeyT, true>
{
  using bits_t = typename Traits<KeyT>::UnsignedBits;
  bits_t bits;
};

// Calculates the number of passes needed for a type T with BitsPerPass bits processed per pass.
template <typename T>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int calc_num_passes(int bits_per_pass)
{
  return ::cuda::ceil_div<int>(sizeof(T) * 8, bits_per_pass);
}

template <int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_num_passes(const int total_bits)
{
  return ::cuda::ceil_div<int>(total_bits, BitsPerPass);
}

// Calculates the starting bit for a given pass (bit 0 is the least significant (rightmost) bit).
// We process the input from the most to the least significant bit. This way, we can skip some passes in the end.
template <typename T, int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int calc_start_bit(const int pass)
{
  int start_bit = int{sizeof(T)} * 8 - (pass + 1) * BitsPerPass;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}

template <int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_start_bit(const int total_bits, const int pass)
{
  int start_bit = total_bits - (pass + 1) * BitsPerPass;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}

// Bit-vector for accumulating prefix digits via funnel shift. Each pass shifts the existing
// contents left by BitsPerPass and ORs the new bucket at the bottom. Sized to hold all
// decomposed bits of KeyT plus headroom for the shift padding of the last pass.
template <typename KeyT>
struct key_prefix_storage_t<KeyT, false>
{
  static constexpr int num_words = ::cuda::ceil_div<int>(sizeof(KeyT) * 8 + 31, 32);
  unsigned int words[num_words];

  // Funnel-shifts the entire bit-vector left by `shift` positions and inserts `value` into the
  // vacated low bits. Each word receives carry bits from its lower neighbor (high-to-low order
  // so each word reads its neighbor's original value). The final word is filled from `value`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void shift_or(int shift, unsigned int value)
  {
    _CCCL_ASSERT(shift > 0 && shift < 32, "shift_or requires 0 < shift < 32");
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = num_words - 1; i > 0; --i)
    {
      words[i] = __funnelshift_l(words[i - 1], words[i], shift);
    }
    words[0] = (words[0] << shift) | value;
  }
};

// Template parameters are ordered with `BitsPerPass` first so callers only have
// to spell the non-deducible compile-time constant; `KeyT` is deduced from the
// `key_prefix_storage_t<KeyT>&` argument.
template <int BitsPerPass, typename KeyT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
set_kth_key_bits(key_prefix_storage_t<KeyT>& prefix, const int pass, const int bin_index)
{
  if constexpr (detail::radix::can_twiddle<KeyT>)
  {
    using bits_t        = typename Traits<KeyT>::UnsignedBits;
    const int start_bit = calc_start_bit<KeyT, BitsPerPass>(pass);
    bits_t bucket       = bin_index;
    prefix.bits |= static_cast<bits_t>(bucket) << start_bit;
  }
  else
  {
    prefix.shift_or(BitsPerPass, bin_index);
  }
}

// Device-side coordination state for the top-k passes. Fields are organized into three groups
// by access pattern:
//
//   (1) Cross-pass scalar state -- single-thread-written at the last-block epilogue, broadcast-read
//       at the next pass's entry. These fields share a single cache line (no internal alignof) so
//       the entry-side reads hit one coherent line and the epilogue stores coalesce.
//
//   (2) Per-pass scratch atomics -- atomicAdd / atomicInc from many blocks within a pass; reset at
//       the epilogue or wrap each pass. Isolated on their own cache lines so per-block atomic
//       traffic does not invalidate the cross-pass scalar state line that other blocks may be
//       reading at the start of the next pass.
//
//   (3) Cross-pass cumulative atomics -- monotonic atomicAdd from many blocks across all passes.
//       Isolated on their own cache lines for the same reason as (2).
//
// Pass-over-pass life cycle of the cross-pass scalar fields (group 1):
//
//   Entry of pass P reads:
//     `num_candidates_in`  -- length of the input stream pass P loads. Drives the tile-loop
//                             sizing (`ceil_div(num_candidates_in, tile_items)`) and the
//                             universal early-exit (`num_candidates_in == 0` -> return).
//     `num_candidates_out` -- count of candidates the previous pass passed through (the bin
//                             count of the bucket containing the kth element). Drives the
//                             pass's *branch selection* via the two comparisons
//                                 `early_stop  = (num_candidates_out == k)`
//                                 `will_buffer = (num_candidates_out <= buffer_capacity)`.
//                             NOT used to size the tile loop -- that is `num_candidates_in`.
//     `k`                  -- top-k items still to find; broadcast to `find_kth_bucket`.
//     `load_from_candidates_buffer` -- source-of-truth bit for the multi-source key/value
//                             loads (false -> `d_keys_in`/`d_values_in`, true -> back buffer).
//     `kth_key_bits`       -- radix prefix accumulated across passes; fed to the
//                             `identify_candidates_op` for candidate classification.
//
//   The last-block epilogue of pass P writes (via `on_kth_bucket` and `counter_update_fn`):
//     `k                  = current_k - num_selected`     (decrement by what this pass wrote out)
//     `num_candidates_out = bin_count`                    (size of the new kth bucket)
//     `kth_key_bits[pass] = bin_index`                    (accumulate the pass's radix digit)
//
//   and, depending on the pass's branch:
//     - histogram         : `num_candidates_in = num_items`; `num_candidates_written = 0`.
//     - filter early_stop : `num_candidates_in = 0`         (universal early-exit signal;
//                                                           remaining fields intentionally
//                                                           left stale -- never read again).
//     - filter buffered   : `num_candidates_in = num_candidates_out`
//                                                           (the back buffer now holds exactly
//                                                            `num_candidates_out` items);
//                          `load_from_candidates_buffer = true`  (sticks for all subsequent
//                                                                 passes);
//                          `num_candidates_written = 0`.
//     - filter unbuffered : no writes -- `num_candidates_in` stays at `num_items`,
//                          `load_from_candidates_buffer` stays false (until/if a later pass
//                          transitions into the buffered chain).
//
// Cross-pass invariants:
//   - `num_candidates_in == num_candidates_out` only along the buffered chain.
//   - Along the unbuffered chain `num_candidates_in == num_items` for every pass.
//   - `load_from_candidates_buffer` is monotonic: false -> true exactly once on the first
//     buffered pass, and sticks (the candidate set is monotonically non-increasing).
//   - `num_candidates_in` is monotonically non-increasing across the buffered chain (the
//     candidate set can only shrink); the early-stop epilogue terminates the chain by writing
//     0 here.
// Cross-pass scalar state. The fields are grouped (and the storage type of
// `load_from_candidates_buffer` widened from `bool` to `::cuda::std::uint32_t`) so that the
// entry-of-pass `LDG.E.128` against this struct lands four clean 32-bit values in consecutive
// scalar registers. With `bool` storage, ptxas inserted a `PRMT` to extract byte 12 out of the
// 16-byte vector load; that `PRMT` is a per-thread op that blocks the `R2UR` heuristic from
// moving any of the loaded fields into uniform registers (see the register-pressure
// investigation notes for the batched filter kernel). The widened uint32 lets the load
// produce four `R2UR`-eligible scalars.
//
// `OffsetT` and `OutOffsetT` are independent at the type level but must each be 32-bit for the
// `LDG.E.128` to cover the whole struct in one transaction (16 bytes = 4*4). The dispatch
// already fixes the batched OffsetT/OutOffsetT to `::cuda::std::uint32_t`; the single-problem
// path also lands here when the user picks 32-bit offsets, which is the throughput-sensitive
// case for the per-pass counter read. Wider offset types still work, they just spill the load
// into multiple `LDG.E.64` transactions and forfeit the `R2UR` win.
template <typename OffsetT, typename OutOffsetT>
struct alignas(16) counter_cross_pass_state
{
  // Top-k items still to be identified after this pass commits its writes. Updated by
  // `on_kth_bucket` to `current_k - num_selected` at each pass's epilogue.
  OutOffsetT k;

  // Count of candidates this pass produced for the next pass to consume (= bin count from this
  // pass's prefix-sum). Read by the next pass to drive the early_stop / buffered / unbuffered
  // mode dispatch.
  OffsetT num_candidates_out;

  // Actual length of the input stream the next pass will load (NOT a candidate count). Equals
  // `num_items` whenever `load_from_candidates_buffer == false` (the unbuffered chain) and the
  // producing pass's `num_candidates_out` once we transition to the buffered chain. The
  // `early_stop` epilogue writes 0 here to act as the universal early-exit signal for all
  // subsequent passes.
  OffsetT num_candidates_in;

  // Whether the next pass should load its input keys/values from `in_key_buf`/`in_val_buf` (the
  // previous pass's candidate buffer) instead of from the original `d_keys_in`/`d_values_in`.
  // Initialized to `false` by the dispatch-side `cudaMemsetAsync` over the counter blob and
  // flipped to `true` exactly once when the first filter pass writes to the candidate buffer; it
  // then sticks because the candidate set is monotonically non-increasing across passes (once we
  // fit in the back buffer we keep fitting).
  //
  // Stored as `uint32_t` rather than `bool` purely to keep the four-field-in-one-cache-line load
  // free of byte-extracting PRMT (see the struct-level note above). Read/write sites keep their
  // `bool` semantics via implicit conversions.
  ::cuda::std::uint32_t load_from_candidates_buffer;
};

template <typename KeyInT, typename OffsetT, typename OutOffsetT>
struct alignas(128) counter : counter_cross_pass_state<OffsetT, OutOffsetT>
{
  // ----- (1) Cross-pass scalar state -----
  //
  // Inherited from `counter_cross_pass_state` so they land at offsets [0..16) of `counter`,
  // sized to one `LDG.E.128`. See the base struct's comment block for the grouping rationale.
  //
  // Inherited members (so existing `counter->k` / `counter->num_candidates_*` /
  // `counter->load_from_candidates_buffer` access patterns keep compiling):
  //   - `OutOffsetT k`
  //   - `OffsetT    num_candidates_out`
  //   - `OffsetT    num_candidates_in`
  //   - `uint32_t   load_from_candidates_buffer`

  // We determine the bits of the k_th key inside the mask processed by the pass. The already
  // known bits are stored in `kth_key_bits`. It's used to discriminate whether an element is a
  // result (written to `out`), a candidate for next pass (written to `out_buf`), or not useful
  // (discarded). The bits that are not yet processed do not matter for this purpose.
  key_prefix_storage_t<KeyInT> kth_key_bits;

  // ----- (2) Per-pass scratch atomics -----

  // Used to determine the write-offset into `out_buf`. Reset by the buffered-pass epilogue.
  alignas(128) OffsetT num_candidates_written;

  // Used to count the number of retired thread blocks. The counter is used to determine the last
  // block to retire and execute the `block_identify_kth_bucket` epilogue (prefix sum + bucket
  // selection). Wraps each pass via `atomicInc`.
  alignas(128) unsigned int finished_block_cnt;

  // ----- (3) Cross-pass cumulative atomics -----

  // Used to determine the write-offset for selected items into the user-provided output
  // iterators across all passes.
  alignas(128) OutOffsetT num_selected_written;

  // Records the number of tied items (crossing the k-th boundary during the last pass) that have
  // been written to the back of the user-provided output iterators. This counter is used to
  // coordinate writes that fill up the gap between definitely selected items at the front and
  // the candidates at the back, making sure we do not overflow beyond the k items the user asked
  // us for.
  alignas(128) OutOffsetT num_ties_written_to_back;
};

//---------------------------------------------------------------------
// Free functions for common histogram operations
//---------------------------------------------------------------------

// Zero-initialize a histogram in shared or global memory
template <int BlockThreads, int NumBuckets, typename CounterT>
_CCCL_DEVICE _CCCL_FORCEINLINE void init_histogram(CounterT* histogram)
{
  int histo_offset = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (; histo_offset + BlockThreads <= NumBuckets; histo_offset += BlockThreads)
  {
    histogram[histo_offset + threadIdx.x] = 0;
  }
  if ((NumBuckets % BlockThreads != 0) && (histo_offset + static_cast<int>(threadIdx.x) < NumBuckets))
  {
    histogram[histo_offset + threadIdx.x] = 0;
  }
}

//---------------------------------------------------------------------
// Atomically merge a block-local histogram into a global histogram. The local and global counters
// may have different types (e.g. a 32-bit local histogram merged into a 32- or 64-bit global one);
// each non-zero local value is widened to the global counter type before the atomic add.
//---------------------------------------------------------------------
template <int BlockThreads, int NumBuckets, typename LocalCounterT, typename GlobalCounterT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
merge_histogram(const LocalCounterT* local_histogram, GlobalCounterT* global_histogram)
{
  int histo_offset = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (; histo_offset + BlockThreads <= NumBuckets; histo_offset += BlockThreads)
  {
    const LocalCounterT local_value = local_histogram[histo_offset + threadIdx.x];
    if (local_value != 0)
    {
      atomicAdd(global_histogram + (histo_offset + threadIdx.x), static_cast<GlobalCounterT>(local_value));
    }
  }
  if ((NumBuckets % BlockThreads != 0) && (histo_offset + static_cast<int>(threadIdx.x) < NumBuckets))
  {
    const LocalCounterT local_value = local_histogram[histo_offset + threadIdx.x];
    if (local_value != 0)
    {
      atomicAdd(global_histogram + (histo_offset + threadIdx.x), static_cast<GlobalCounterT>(local_value));
    }
  }
}

//---------------------------------------------------------------------
// Last-block coordination primitive.
// Fences pending writes, atomically detects the last-finishing block via `retired_block_counter`, and invokes `epilogue_op` exactly once on that block. 
// The epilogue owns whatever smem it needs and decides what "finalization" means (top-k uses it to run `block_identify_kth_bucket::find_kth_bucket` plus
// any per-mode counter bookkeeping). 
// `expected_block_count` is the number of blocks expected to retire (e.g., `gridDim.x`.
//---------------------------------------------------------------------
template <typename BlockCountT, typename EpilogueOpT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
finalize_pass(BlockCountT* retired_block_counter, unsigned int expected_block_count, EpilogueOpT epilogue_op)
{
  __threadfence();

  bool is_last_block = false;
  if (threadIdx.x == 0)
  {
    const unsigned int wrap_at = expected_block_count - 1u;
    const unsigned int retired = atomicInc(retired_block_counter, wrap_at);
    is_last_block              = (retired == wrap_at);
  }

  if (__syncthreads_or(is_last_block))
  {
    epilogue_op();
  }
}

//---------------------------------------------------------------------
// Computes the prefix-sum over bins and finds the k-th item bucket.
// Exposes a the `find_kth_bucket()` entry point that invokes a callback exactly once with the kth-bucket index and the 
// The primitive performs:
//   1) load of the per-bin counts from `input_histogram` into a blocked arrangement;
//   2) `BlockScan::InclusiveSum` over that chunk, keeping prefix sums in registers;
//   3) one boundary write/read per thread so each thread's right neighbour can use its right-edge prefix sum as the
//   `prev` for its bin 0; 4) per-thread search for the bucket whose prefix-sum range straddles `current_k`, with
//   `on_kth_bucket` invoked on the single thread that owns that bucket.
//
// Note, the chosen `LoadAlgorithm` must produce a blocked arrangement.
//---------------------------------------------------------------------
template <int BlockThreads,
          int BitsPerPass,
          BlockScanAlgorithm ScanAlgorithm,
          typename OffsetT,
          typename OutOffsetT,
          BlockLoadAlgorithm LoadAlgorithm = BLOCK_LOAD_TRANSPOSE>
struct block_identify_kth_bucket
{
  // Prefix-sum and boundary-buffer assumes thread `tid` owns the contiguous bin chunk in blocked arrangement
  static_assert(
    LoadAlgorithm == BLOCK_LOAD_DIRECT //
      || LoadAlgorithm == BLOCK_LOAD_VECTORIZE //
      || LoadAlgorithm == BLOCK_LOAD_TRANSPOSE //
      || LoadAlgorithm == BLOCK_LOAD_WARP_TRANSPOSE //
      || LoadAlgorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
    "block_identify_kth_bucket requires a blocked-layout BlockLoadAlgorithm: "
    "BLOCK_LOAD_DIRECT, BLOCK_LOAD_VECTORIZE, BLOCK_LOAD_TRANSPOSE, "
    "BLOCK_LOAD_WARP_TRANSPOSE, or BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED. "
    "Striped layouts (e.g. BLOCK_LOAD_STRIPED) are not supported.");

  static constexpr int num_buckets     = 1 << BitsPerPass;
  static constexpr int bins_per_thread = ::cuda::ceil_div(num_buckets, BlockThreads);

  // Only threads owning at least one in-range bin contribute a boundary, so the buffer is capped at `num_buckets` slots
  // when `num_buckets < BlockThreads`.
  static constexpr int boundaries_size = (BlockThreads < num_buckets) ? BlockThreads : num_buckets;

  using block_load_t = BlockLoad<OffsetT, BlockThreads, bins_per_thread, LoadAlgorithm>;
  using block_scan_t = BlockScan<OffsetT, BlockThreads, ScanAlgorithm>;

  // The boundary buffer shares storage with the load/scan scratch since the boundary exchange runs strictly after both
  // have completed.
  struct TempStorage
  {
    union
    {
      typename block_load_t::TempStorage load;
      typename block_scan_t::TempStorage scan;
      OffsetT boundaries[boundaries_size];
    };
  };

  TempStorage& storage;

  _CCCL_DEVICE _CCCL_FORCEINLINE explicit block_identify_kth_bucket(TempStorage& s)
      : storage(s)
  {}

  // Inclusive-prefix-sums `input_histogram`, identifies the bucket containing the `current_k`-th element, and invokes
  // `on_kth_bucket` exactly once on the unique thread that owns that bucket. `on_kth_bucket` callback signature and
  // arguments: `on_kth_bucket(OutOffsetT current_k, int bin_index, OffsetT num_selected, OffsetT num_candidates)`
  // - current_k: echoed back from the `current_k` argument so the callback can compute `current_k - num_selected`
  // (number of candidates in the k-th item's bucket)
  //    without having to capture it.
  // - bin_index: the index of the bucket containing the kth element.
  // - num_selected: count of items in higher-priority buckets, i.e. items already known to be in the top-k. The new k
  //    for the next pass is `current_k - num_selected`.
  // - num_candidates: count of items inside `bin_index` itself, i.e. the number of candidates the next filter pass will
  //    write.
  template <typename KthBucketFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  find_kth_bucket(const OffsetT* input_histogram, OutOffsetT current_k, KthBucketFn on_kth_bucket)
  {
    // Full tiles (of bins) will skip the out-of-bounds checks
    static constexpr bool is_full_tile = (num_buckets == BlockThreads * bins_per_thread);

    // Each thread loads its contiguous chunk of bins into registers
    OffsetT thread_data[bins_per_thread];
    if constexpr (is_full_tile)
    {
      block_load_t(storage.load).Load(input_histogram, thread_data);
    }
    else
    {
      block_load_t(storage.load).Load(input_histogram, thread_data, num_buckets);
    }

    // Ensure we can reuse temporary storage for the prefix-sum
    __syncthreads();

    // Compute the prefix-sum over the bins
    block_scan_t(storage.scan).InclusiveSum(thread_data, thread_data);

    // Ensure we can reuse temporary storage for the boundary exchange
    __syncthreads();

    // Publish each thread's right-edge prefix sum so its right neighbour can
    // read it as the `prev` for its leftmost bin.
    if constexpr (is_full_tile)
    {
      storage.boundaries[threadIdx.x] = thread_data[bins_per_thread - 1];
    }
    else
    {
      if (static_cast<int>(threadIdx.x) < boundaries_size)
      {
        storage.boundaries[threadIdx.x] = thread_data[bins_per_thread - 1];
      }
    }
    // Ensure all threads finished writing their boundary value
    __syncthreads();

    // Get the previous thread's right-edge prefix sum
    const OffsetT prev_boundary =
      (threadIdx.x > 0 && (is_full_tile || static_cast<int>(threadIdx.x) <= boundaries_size))
        ? storage.boundaries[threadIdx.x - 1]
        : OffsetT{0};

    // The first bin this thread is assigned to
    const int base_bin = static_cast<int>(threadIdx.x) * bins_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < bins_per_thread; ++i)
    {
      const int bin_index = base_bin + i;
      if (is_full_tile || bin_index < num_buckets)
      {
        const OffsetT prev = (i == 0) ? prev_boundary : thread_data[i - 1];
        const OffsetT cur  = thread_data[i];
        if (prev < current_k && cur >= current_k)
        {
          on_kth_bucket(current_k, bin_index, prev, cur - prev);
        }
      }
    }
  }
};

//---------------------------------------------------------------------
// AgentTopKHistogram filter helpers
//---------------------------------------------------------------------
//
// The histogram agent accepts an arbitrary unary predicate `FilterOpT(key) ->
// bool` that decides whether a given key contributes to the histogram. The two
// canonical specializations live here:
//
//  - `topk_pass_through_filter_op` -- pass-0 default. Always returns `true`.
//  - `topk_candidate_filter_op<IdentifyCandidatesOpT>` -- thin wrapper used by the unbuffered filter pass. It wraps the kernel's `identify_candidates_op` and returns `true` only for keys classified as `candidate`
struct topk_pass_through_filter_op
{
  template <typename T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr bool operator()(const T&) const
  {
    return true;
  }
};

template <typename IdentifyCandidatesOpT>
struct topk_candidate_filter_op
{
  IdentifyCandidatesOpT identify_candidates_op;

  template <typename T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(const T& key) const
  {
    return identify_candidates_op(key) == candidate_class::candidate;
  }
};

//---------------------------------------------------------------------
// agent_topk_filter_partition & agent_topk_last_filter helpers
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// sink_mode: compile-time selector for agent_topk_filter_partition.
//
// Each mode corresponds to one behavior of the legacy sink_* family and drives a
// single-kernel runtime switch in DeviceTopKFilterKernel. `early_stop` is the final
// collapsed pass; `buffered` writes output and stages remaining candidates into a
// back buffer. The "scout" mode that only builds a histogram (formerly
// `unbuffered`) is now handled by `AgentTopKHistogram` with a candidate filter,
// and is intentionally no longer part of this enum.
//---------------------------------------------------------------------

enum class sink_mode
{
  early_stop, // selected + candidate -> d_keys_out front; no histogram
  buffered, // selected -> d_keys_out; candidate -> out_buf; histogram over candidates
};

// No-op candidate callback. Used by `agent_topk_last_filter` (no histogram is
// accumulated on the last filter pass; the partition primitive still requires
// a callable).
struct topk_noop_candidate_callback_op
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(const T&) const
  {}
};

// Wraps an `IdentifyCandidatesOp` (3-state classifier returning `candidate_class`)
// into a unary `bool` predicate suitable for the single-stream `BlockFilter`
// primitive. Folds the architecture's "early_stop collapses candidate -> selected"
// rule into the wrapper: any non-rejected item is kept.
template <typename IdentifyCandidatesOpT>
struct topk_identify_selected_op
{
  IdentifyCandidatesOpT identify_candidates_op;

  template <typename KeyT>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(const KeyT& key) const
  {
    return identify_candidates_op(key) != candidate_class::rejected;
  }
};

// Histogram callback: increments the agent's smem histogram for every
// `candidate`-classified key. Mirrors the legacy `process_tile`'s inline atomicAdd
// per item; architecture §10.2 describes the same pattern as a callback.
template <typename ExtractBinOpT, typename CounterT>
struct topk_histogram_callback_op
{
  ExtractBinOpT extract_bin_op;
  CounterT* smem_histogram;

  _CCCL_DEVICE _CCCL_FORCEINLINE topk_histogram_callback_op(ExtractBinOpT eb, CounterT* hist)
      : extract_bin_op(eb)
      , smem_histogram(hist)
  {}

  template <typename KeyT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(const KeyT& key) const
  {
    const int bucket = extract_bin_op(key);
    atomicAdd(smem_histogram + bucket, CounterT{1});
  }
};

} // namespace detail::topk

CUB_NAMESPACE_END
