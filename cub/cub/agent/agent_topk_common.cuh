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
//!   - `init_histogram` / `merge_histogram` — block/grid histogram primitives.
//!   - `block_identify_kth_bucket`        — last-block prefix-sum + bucket-finder.
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
#include <cub/detail/topk/empty_storage.cuh>
#include <cub/detail/topk/key_prefix_storage.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// The generic primitives stay in detail::topk; bring them into scope so the batched
// shared-layer symbols below can refer to them unqualified.
using namespace detail::topk;
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
//!   The `tile_load_kind` used by the keys-stream `TileDataSource`.
//!
//! @tparam AccumulatingBufferCapacity
//!   Number of smem slots in the per-stream candidate buffer.
//!
//! @tparam SpeculativeSelectedBufferCapacity
//!   Number of smem slots in the selected-stream buffer. `0` short-circuits it
//!   to pure atomics.
//!
template <int ThreadsPerBlock,
          int ItemsPerThread,
          int BitsPerPass,
          BlockScanAlgorithm ScanAlgorithm,
          tile_load_kind KeysTileLoadKind = tile_load_kind::block_load_vectorize>
struct AgentTopKPolicy
{
  static constexpr int block_threads                 = ThreadsPerBlock;
  static constexpr int items_per_thread              = ItemsPerThread;
  static constexpr int bits_per_pass                 = BitsPerPass;
  static constexpr BlockScanAlgorithm scan_algorithm = ScanAlgorithm;
  // Picks a TileDataSource specialization for the keys stream. Defaults to the
  // `BLOCK_LOAD_VECTORIZE` mapping.
  static constexpr tile_load_kind keys_tile_load_kind = KeysTileLoadKind;
};

template <int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_num_passes(const int total_bits)
{
  return ::cuda::ceil_div<int>(total_bits, BitsPerPass);
}

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
// by access pattern, each isolated on its own cache line(s) to avoid false sharing:
//   (1) Cross-pass scalar state -- written once at the last-block epilogue, broadcast-read at
//       the next pass's entry.
//   (2) Per-pass scratch atomics -- atomicAdd / atomicInc from many blocks within a pass; reset
//       each pass.
//   (3) Cross-pass cumulative atomics -- monotonic atomicAdd from many blocks across all passes.
//
// Cross-pass invariants:
//   - `num_candidates_in == num_candidates_out` only along the buffered chain; along the
//     unbuffered chain `num_candidates_in == num_items` for every pass.
//   - `load_from_candidates_buffer` is monotonic: false -> true exactly once on the first
//     buffered pass, then sticks (the candidate set is monotonically non-increasing).
//   - `num_candidates_in` is monotonically non-increasing across the buffered chain; the
//     early-stop epilogue terminates the chain by writing 0 here.
// Cross-pass scalar state. The four fields are grouped and `load_from_candidates_buffer` is
// stored as `uint32_t` (not `bool`) so that with 32-bit `OffsetT`/`OutOffsetT` the struct is
// four consecutive 32-bit values and the per-pass entry read is a single 16-byte vector load.
// Wider offset types still work; they just split the load into multiple transactions.
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
  // previous pass's candidate buffer) instead of the original `d_keys_in`/`d_values_in`. Starts
  // `false`, flips to `true` exactly once on the first buffered pass, then sticks. Stored as
  // `uint32_t` rather than `bool` for the layout reason noted on the struct above; read/write
  // sites keep `bool` semantics via implicit conversions.
  ::cuda::std::uint32_t load_from_candidates_buffer;
};

template <typename KeyT, typename OffsetT, typename OutOffsetT>
struct alignas(128) counter : counter_cross_pass_state<OffsetT, OutOffsetT>
{
  // ----- (1) Cross-pass scalar state -----
  //
  // Inherited from `counter_cross_pass_state` (`k`, `num_candidates_out`, `num_candidates_in`,
  // `load_from_candidates_buffer`) so existing `counter->...` access patterns keep compiling.
  // See the base struct for the grouping rationale.

  // We determine the bits of the k_th key inside the mask processed by the pass. The already
  // known bits are stored in `kth_key_bits`. It's used to discriminate whether an element is a
  // result (written to `out`), a candidate for next pass (written to `out_buf`), or not useful
  // (discarded). The bits that are not yet processed do not matter for this purpose.
  key_prefix_storage_t<KeyT> kth_key_bits;

  // ----- (2) Per-pass scratch atomics -----

  // Used to determine the write-offset into `out_buf`. Reset by the buffered-pass epilogue.
  alignas(128) OffsetT num_candidates_written;

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
// Copy a histogram from `src` into `dst`. The read-side complement of `merge_histogram`: used to
// prime a block-local (smem) histogram from a previously-accumulated (global) one instead of
// zero-initializing it. The two counter types may differ; each source value is widened/narrowed
// to the destination counter type.
//---------------------------------------------------------------------
template <int BlockThreads, int NumBuckets, typename DstCounterT, typename SrcCounterT>
_CCCL_DEVICE _CCCL_FORCEINLINE void load_histogram(DstCounterT* dst, const SrcCounterT* src)
{
  int histo_offset = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (; histo_offset + BlockThreads <= NumBuckets; histo_offset += BlockThreads)
  {
    dst[histo_offset + threadIdx.x] = static_cast<DstCounterT>(src[histo_offset + threadIdx.x]);
  }
  if ((NumBuckets % BlockThreads != 0) && (histo_offset + static_cast<int>(threadIdx.x) < NumBuckets))
  {
    dst[histo_offset + threadIdx.x] = static_cast<DstCounterT>(src[histo_offset + threadIdx.x]);
  }
}

//---------------------------------------------------------------------
// Computes the prefix-sum over bins and finds the k-th item bucket. The `find_kth_bucket()`
// entry point invokes `on_kth_bucket` exactly once, on the single thread owning that bucket.
// Steps: load per-bin counts into a blocked arrangement; `BlockScan::InclusiveSum`; exchange
// each thread's right-edge prefix sum so its neighbour has the `prev` for its bin 0; per-thread
// search for the bucket whose prefix-sum range straddles `current_k`.
//
// Note: the chosen `LoadAlgorithm` must produce a blocked arrangement.
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
  // True when the bins exactly fill `BlockThreads * bins_per_thread` (no ragged last chunk); lets
  // the full-tile paths skip the out-of-bounds checks.
  static constexpr bool is_full_tile = (num_buckets == BlockThreads * bins_per_thread);

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

  // Read a histogram laid out by bin index (e.g. an smem slab) into this thread's blocked register
  // chunk: thread `t` owns bins `[t*bins_per_thread, (t+1)*bins_per_thread)`. Direct (non-
  // transposed) read: an smem source has no coalescing benefit, and skipping the transpose frees
  // the load scratch so a caller can alias the histogram storage with this primitive's
  // `TempStorage`. The ragged last chunk is zero-filled.
  _CCCL_DEVICE _CCCL_FORCEINLINE static void
  load_blocked(const OffsetT* histogram, OffsetT (&thread_data)[bins_per_thread])
  {
    const int base_bin = static_cast<int>(threadIdx.x) * bins_per_thread;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < bins_per_thread; ++i)
    {
      const int bin  = base_bin + i;
      thread_data[i] = (is_full_tile || bin < num_buckets) ? histogram[bin] : OffsetT{0};
    }
  }

  // Bucket-finder over a histogram already resident in this thread's blocked registers
  // (`thread_data[i]` holds the count for bin `threadIdx.x * bins_per_thread + i`). Like the
  // `const OffsetT*` overload but skips the load, so the caller can feed a histogram it already
  // has in registers (e.g. via `load_blocked`).
  //
  // Invokes `on_kth_bucket` exactly once, on the unique thread owning the bucket that contains
  // the `current_k`-th element:
  //   `on_kth_bucket(OutOffsetT current_k, int bin_index, OffsetT num_selected, OffsetT num_candidates)`
  // - current_k:      echoed back so the callback can compute `current_k - num_selected`.
  // - bin_index:      index of the bucket containing the kth element.
  // - num_selected:   count of items in higher-priority buckets (already in the top-k); the next
  //                   pass's k is `current_k - num_selected`.
  // - num_candidates: count of items inside `bin_index` (candidates the next filter pass writes).
  //
  // Storage contract: the first op is an in-place block scan into `storage.scan`. If `thread_data`
  // was sourced from memory aliasing this primitive's `TempStorage` (e.g. a unioned smem
  // histogram), the caller must `__syncthreads()` first so all reads complete before the scan
  // overwrites the shared storage.
  template <typename KthBucketFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  find_kth_bucket(OffsetT (&thread_data)[bins_per_thread], OutOffsetT current_k, KthBucketFn on_kth_bucket)
  {
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

  // Loads `input_histogram` (typically a global slab) into blocked registers via a coalesced
  // `BlockLoad`, then delegates to the in-registers overload above. See that overload for the
  // `on_kth_bucket` contract.
  template <typename KthBucketFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  find_kth_bucket(const OffsetT* input_histogram, OutOffsetT current_k, KthBucketFn on_kth_bucket)
  {
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

    find_kth_bucket(thread_data, current_k, on_kth_bucket);
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
//  - `topk_candidate_filter_op<IdentifyCandidatesOpT>` -- thin wrapper used by the unbuffered filter pass. It wraps the
//  kernel's `identify_candidates_op` and returns `true` only for keys classified as `candidate`
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

// No-op candidate callback. Used by `agent_topk_last_filter` (no histogram is
// accumulated on the last filter pass; the partition primitive still requires
// a callable).
struct topk_noop_candidate_callback_op
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(const T&) const
  {}
};

// Wraps an `IdentifyCandidatesOp` (3-state classifier returning `candidate_class`) into a unary
// `bool` predicate for the single-stream `BlockFilter` primitive. Implements the early_stop rule
// (candidate collapses to selected): any non-rejected item is kept.
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

// Histogram callback: increments the agent's smem histogram for every `candidate`-classified key.
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

//---------------------------------------------------------------------
// tile_histogram: block-wide primitive that owns a shared-memory histogram and provides the
// populate / drain / read operations shared by the top-k histogram and filter agents (and the
// finalize-histogram kernel).
//
// Owns the smem histogram plus the per-tile binning and the histogram lifecycle (`reset` /
// `add_*` / `load_from` / `flush`). Tile iteration, segment handling, and data loading stay in the
// owning agent/kernel. Deliberately *sync-free*: no method issues an internal `__syncthreads()`,
// so the owner controls all barrier placement (performance-critical at segment boundaries).
//
// The bins are exposed via `data()` so collaborating primitives can read them
// (`block_identify_kth_bucket`) or write them (`topk_histogram_callback_op`, via
// `make_callback()`); the owner remains the single manager of the buffer's lifecycle.
//
// `extract_bin_op` is fixed for the histogram's lifetime, so it is stored. The per-item accept
// predicate `filter_op` is taken per `add_*` call (defaulting to pass-through).
//---------------------------------------------------------------------
template <int BlockThreads, int NumBuckets, typename CounterT, typename ExtractBinOpT>
class tile_histogram
{
public:
  // Shared-memory state: one counter per bucket.
  struct TempStorage
  {
    CounterT bins[NumBuckets];
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE tile_histogram(TempStorage& temp_storage, ExtractBinOpT extract_bin_op)
      : temp_storage(temp_storage)
      , extract_bin_op(extract_bin_op)
  {}

  // Zero the local histogram. No internal sync; the caller brackets.
  _CCCL_DEVICE _CCCL_FORCEINLINE void reset()
  {
    init_histogram<BlockThreads, NumBuckets>(temp_storage.bins);
  }

  // Initialize the local histogram from a histogram in (typically global) memory -- the read-side
  // complement of `flush`. No internal sync; the caller brackets.
  template <typename SrcCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_from(const SrcCounterT* src)
  {
    load_histogram<BlockThreads, NumBuckets>(temp_storage.bins, src);
  }

  // Bin a full tile of per-thread `items` into the local histogram. Every item that passes
  // `filter_op` contributes one count to its `extract_bin_op` bucket.
  template <typename KeyT, int ItemsPerThread, typename FilterOpT = topk_pass_through_filter_op>
  _CCCL_DEVICE _CCCL_FORCEINLINE void add_full(const KeyT (&items)[ItemsPerThread], FilterOpT filter_op = {})
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (filter_op(items[j]))
      {
        const int bucket = extract_bin_op(items[j]);
        atomicAdd(temp_storage.bins + bucket, CounterT{1});
      }
    }
  }

  // Bin the trailing partial tile's `num_thread_items` items per thread.
  template <typename KeyT, int ItemsPerThread, typename FilterOpT = topk_pass_through_filter_op>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  add_partial(const KeyT (&items)[ItemsPerThread], int num_thread_items, FilterOpT filter_op = {})
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (j < num_thread_items && filter_op(items[j]))
      {
        const int bucket = extract_bin_op(items[j]);
        atomicAdd(temp_storage.bins + bucket, CounterT{1});
      }
    }
  }

  // Atomically merge the local histogram into a destination histogram (typically the segment's
  // global slab). No internal sync; the caller brackets.
  template <typename GlobalCounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void flush(GlobalCounterT* dst)
  {
    merge_histogram<BlockThreads, NumBuckets>(temp_storage.bins, dst);
  }

  // Raw access to the local bins, for collaborating block primitives that read
  // (`block_identify_kth_bucket`) or write (`topk_histogram_callback_op`) the histogram.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE CounterT* data() const
  {
    return temp_storage.bins;
  }

  // Build a per-item bin callback targeting this histogram, for the buffered partition primitive
  // (which bins each surviving candidate as a side effect of classification).
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE topk_histogram_callback_op<ExtractBinOpT, CounterT> make_callback() const
  {
    return topk_histogram_callback_op<ExtractBinOpT, CounterT>{extract_bin_op, temp_storage.bins};
  }

private:
  TempStorage& temp_storage;
  ExtractBinOpT extract_bin_op;
};
} // namespace detail::batched_topk

CUB_NAMESPACE_END
