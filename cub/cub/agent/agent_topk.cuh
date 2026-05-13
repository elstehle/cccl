// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in device-wide topK.
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
#include <cub/detail/topk/block_filter.cuh>
#include <cub/detail/topk/block_filter_accumulating.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/block_partition_accumulating.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

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
//!   strategy (irrelevant for the non-accumulating strategies).
//!
template <int ThreadsPerBlock,
          int ItemsPerThread,
          int BitsPerPass,
          BlockScanAlgorithm ScanAlgorithm,
          tile_load_kind KeysTileLoadKind = tile_load_kind::block_load_vectorize,
          int AccumulatingBufferCapacity  = 256>
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
  static constexpr tile_load_kind keys_tile_load_kind = KeysTileLoadKind;
  static constexpr int accumulating_buffer_capacity   = AccumulatingBufferCapacity;
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
template <typename KeyInT, typename OffsetT, typename OutOffsetT>
struct alignas(128) counter
{
  // ----- (1) Cross-pass scalar state -----

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
  bool load_from_candidates_buffer;

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
// AgentTopKHistogram: histogram agent shared by pass 0 and the unbuffered
// filter passes. Owns _TempStorage, instantiates a TileDataSource for the
// keys stream (per `AgentTopKPolicyT::keys_tile_load_kind`), and runs the
// §4.3 tile loop with full/partial dispatch into `process_tile`.
//
// Each input key is gated by the `FilterOpT` predicate before contributing
// to the histogram. The pass-0 caller leaves the default
// `topk_pass_through_filter_op` in place; the unbuffered filter caller
// supplies a `topk_candidate_filter_op` that wraps its
// `identify_candidates_op`.
//
// Single-source invariant for the unbuffered usage: the unbuffered filter
// branch is reached only when the surviving candidate count exceeds the
// back-buffer capacity (`current_len > buffer_length`). The candidate set is
// monotonically non-increasing across passes, so once an unbuffered pass is
// taken, every prior pass was also unbuffered, which means no candidate has
// ever been written to `in_key_buf`. Therefore the unbuffered path always
// loads from the original `d_keys_in`, and a single `tile_data_source_t<
// KeyInputIteratorT, ...>` is sufficient -- we don't need the
// `multi_source_data_source` machinery the partition agent uses. If we ever
// allow unbuffered passes to follow buffered ones (e.g. an adaptive
// strategy that pauses output writes mid-decode), this agent will need to
// be re-templated on the keys data source so the caller can hand in a
// multi_source over `(d_keys_in, in_key_buf)`.
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT,
          typename FilterOpT = topk_pass_through_filter_op>
struct AgentTopKHistogram
{
  using key_in_t = it_value_t<KeyInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;

  using keys_source_t =
    tile_data_source_t<KeyInputIteratorT, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using identify_kth_bucket_t =
    block_identify_kth_bucket<block_threads, bits_per_pass, AgentTopKPolicyT::scan_algorithm, OffsetT, OutOffsetT>;

  struct _TempStorage
  {
    // Persistent region (architecture §2.2).
    OffsetT histogram[num_buckets];
    typename keys_source_t::TempStorage keys_source_state;

    // Method-call scratch (mutually exclusive in time).
    union
    {
      typename keys_source_t::ScratchStorage keys_source_scratch;
      typename identify_kth_bucket_t::TempStorage prefix_sum;
    } scratch;
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& temp_storage;
  KeyInputIteratorT d_keys_in;
  OffsetT num_items;
  OutOffsetT k;
  ExtractBinOpT extract_bin_op;
  FilterOpT filter_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentTopKHistogram(
    TempStorage& ts,
    const KeyInputIteratorT d_keys_in,
    OffsetT num_items,
    OutOffsetT k,
    ExtractBinOpT extract_bin_op,
    FilterOpT filter_op = {})
      : temp_storage(ts.Alias())
      , d_keys_in(d_keys_in)
      , num_items(num_items)
      , k(k)
      , extract_bin_op(extract_bin_op)
      , filter_op(filter_op)
  {}

  // process_tile: filter + classify + per-bucket atomicAdd into smem histogram.
  // The full overload omits the per-item bound check; the partial overload
  // bound-checks against `num_thread_items`. The default `FilterOpT` returns a
  // constexpr `true`, so the predicate call folds away and the pass-0 inner
  // loop reduces to the original `extract_bin_op + atomicAdd` sequence.
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile_full(const key_in_t (&items)[items_per_thread])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      if (filter_op(items[j]))
      {
        const int bucket = extract_bin_op(items[j]);
        atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  process_tile_partial(const key_in_t (&items)[items_per_thread], int num_thread_items)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      if (j < num_thread_items && filter_op(items[j]))
      {
        const int bucket = extract_bin_op(items[j]);
        atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
      }
    }
  }

  // Both callbacks run only on the last finishing block:
  //   `counter_update_fn` runs on thread 0 (before the prefix-sum / bucket
  //     selection). The pass-0 caller passes a lambda that resets
  //     `num_candidates_in`/`num_candidates_written`; the unbuffered caller passes its
  //     mode-dependent counter update (early-stop vs non-early-stop logic).
  //   `on_kth_bucket`     runs on the single thread that owns the bucket
  //     containing the kth element. See `block_identify_kth_bucket::find_kth_bucket` for the
  //     callback signature.
  template <typename CounterUpdateFn, typename KthBucketFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  invoke(unsigned int* retired_block_counter,
         OffsetT* global_histogram,
         bool reset_histogram,
         CounterUpdateFn counter_update_fn,
         KthBucketFn on_kth_bucket)
  {
    init_histogram<block_threads, num_buckets>(temp_storage.histogram);
    __syncthreads();

    keys_source_t keys_source{d_keys_in, temp_storage.keys_source_state};

    // Split the tile space into full tiles and a possible single trailing
    // partial tile. The hot-path loop below processes only full tiles. The
    // partial tile, if any, is handled exactly once after the loop by the
    // unique block that would have owned it under the original grid-strided
    // schedule.
    const OffsetT num_full_tiles = num_items / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items  = num_items - num_full_tiles * static_cast<OffsetT>(tile_items);

    // --- full-tile loop --------------------------------------------------
    for (OffsetT tile_id = static_cast<OffsetT>(blockIdx.x); tile_id < num_full_tiles;
         tile_id += static_cast<OffsetT>(gridDim.x))
    {
      const OffsetT tile_base = tile_id * static_cast<OffsetT>(tile_items);

      keys_source.set_tile_base(tile_base);

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(temp_storage.scratch.keys_source_scratch);
      h.complete_load(items);
      process_tile_full(items);
    }

    // --- trailing partial tile (handled by exactly one block) ------------
    //
    // Under the original grid-strided schedule the partial tile (tile_id =
    // num_full_tiles) was claimed by the block whose blockIdx.x equals
    // num_full_tiles % gridDim.x. We preserve that ownership.
    if (partial_items > 0)
    {
      const unsigned partial_owner = static_cast<unsigned>(num_full_tiles % static_cast<OffsetT>(gridDim.x));
      if (blockIdx.x == partial_owner)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);

        keys_source.set_tile_base(tile_base);

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(temp_storage.scratch.keys_source_scratch, partial_items);
        h.complete_load(items);
        const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
        const int num_thread_items =
          (thread_offset >= num_items)
            ? 0
            : static_cast<int>((::cuda::std::min) (static_cast<OffsetT>(items_per_thread), num_items - thread_offset));
        process_tile_partial(items, num_thread_items);
      }
    }

    __syncthreads();
    merge_histogram<block_threads, num_buckets>(temp_storage.histogram, global_histogram);

    // Last-block epilogue: per-mode counter update on thread 0, then the
    // prefix-sum + bucket selection whose `on_kth_bucket` callback writes the
    // discovered next-pass inputs into `counter`, then optional histogram
    // reset for the next pass.
    auto epilogue_op = [&] {
      if (threadIdx.x == 0)
      {
        counter_update_fn();
      }
      identify_kth_bucket_t{temp_storage.scratch.prefix_sum}.find_kth_bucket(global_histogram, k, on_kth_bucket);
      if (reset_histogram)
      {
        // TODO (elstehle): We could skip this reset when we detect an early-stop condition. However, it would require a block-wide broadcast. This short-circuit needs to be evaluated experimentally. 
        init_histogram<block_threads, num_buckets>(global_histogram);
      }
    };

    finalize_pass(retired_block_counter, gridDim.x, epilogue_op);
  }
};

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

//---------------------------------------------------------------------
// agent_topk_filter_partition: agent for passes 1..num_passes-1 (the
// "filter passes" between the histogram pass and the dedicated last-filter
// pass).
//
// Handles BOTH `sink_mode::early_stop` and `sink_mode::buffered` in one
// agent type. The agent's `run()` takes a runtime `sink_mode mode` arg and
// dispatches over it with a two-way `if`:
//   - early_stop  -> constructs `BlockFilter[Accumulating]` (single-stream
//                    "filter" primitive) with sinks + identify-selected op
//                    captured at ctor.
//   - buffered    -> constructs `BlockPartition[Accumulating]` (two-stream
//                    "partition" primitive) with sinks + identify op +
//                    histogram callback captured at ctor.
// The mode-agnostic `drive_tile_loop<Primitive, StorageLayout>` helper then
// iterates the tile space and finishes with `primitive.epilogue()`. The
// "scout" mode that only builds a histogram is handled separately by
// `AgentTopKHistogram` driven by a candidate-filter predicate.
//
// Members hold the inputs/outputs that both modes consume; the buffered-only
// candidate outputs (`out_key_buf`, `out_val_buf`, `p_num_candidates_written`)
// are passed to `run()` rather than being stored on the agent.
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

template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          typename OffsetT,
          typename OutOffsetT,
          BlockPartitionStrategy BufferedPartStrat = BlockPartitionStrategy::AtomicsPreClassify,
          BlockFilterStrategy EarlyStopFilterStrat = BlockFilterStrategy::AtomicsPreClassify,
          // Experimental: when `true`, the per-tile `values[ItemsPerThread]`
          // register array is NOT pre-loaded; the partition's scatter loop
          // instead pulls each surviving value via the value channel's
          // `data_source.gather_one(j)`. Mimics `main`'s legacy "only fetch
          // values that survive the filter" behavior. Defaults to `false`
          // (current branch's eager-load behavior). Forced off for keys-only.
          bool LazyValueLoad = false>
struct agent_topk_filter_partition
{
  using key_in_t   = it_value_t<KeyInputIteratorT>;
  using value_in_t = it_value_t<ValueInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  // The epilogue primitive used to find the k-th bucket to conclude the histogram stage
  using identify_kth_bucket_t =
    block_identify_kth_bucket<block_threads, bits_per_pass, AgentTopKPolicyT::scan_algorithm, OffsetT, OutOffsetT>;


  // Lazy value-load only matters when there's a value channel to load lazily.
  static constexpr bool effective_lazy_value_load = LazyValueLoad && !keys_only;

  // Keys data source: multi_source over (d_keys_in source, in_key_buf source).
  using key_source_input_t =
    tile_data_source_t<KeyInputIteratorT, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using key_source_buffer_t =
    tile_data_source_t<key_in_t*, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using keys_source_t = multi_source_data_source<key_source_input_t, key_source_buffer_t, OffsetT>;

  // Value channel: multi_source over (d_values_in, in_val_buf)
  using value_source_input_t = direct_data_source<ValueInputIteratorT, block_threads, items_per_thread, OffsetT>;
  using value_source_buffer_t = direct_data_source<value_in_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t = multi_source_data_source<value_source_input_t, value_source_buffer_t, OffsetT>;

  
  using val_out_t = ValueOutputIteratorT;
  // Buffered mode: the candidate iterators are `key_in_t*` / `value_in_t*` (the
  // back buffers). Early-stop mode has only the selected stream.
  using buffered_cand_val_out_t = value_in_t*;
  using buffered_cand_key_out_t = key_in_t*;

  // Buffered-mode sinks bundle: 2-stream `value_channel_sinks_t` (selected +
  // candidate). Empty tuple for keys-only.
  using buffered_value_channel_sinks_for_agent_t =
    value_channel_sinks_t<val_out_t, buffered_cand_val_out_t, ::cuda::std::identity, ::cuda::std::identity>;
  using buffered_value_channel_sinks_tuple_t = ::cuda::std::conditional_t<
    keys_only,
    ::cuda::std::tuple<>,
    ::cuda::std::tuple<buffered_value_channel_sinks_for_agent_t>>;

  // Early-stop sinks bundle: 1-stream `value_channel_sinks_filter_t` (selected only).
  using early_stop_value_channel_sinks_for_agent_t =
    value_channel_sinks_filter_t<val_out_t, ::cuda::std::identity>;
  using early_stop_value_channel_sinks_tuple_t = ::cuda::std::conditional_t<
    keys_only,
    ::cuda::std::tuple<>,
    ::cuda::std::tuple<early_stop_value_channel_sinks_for_agent_t>>;

  // Per-channel `value_t` (sized to one element per channel), supplied to the
  // partition class so it can size its smem `value_t values[N]` arrays.
  using value_types_tuple_t =
    ::cuda::std::conditional_t<keys_only, ::cuda::std::tuple<>, ::cuda::std::tuple<value_in_t>>;
  // Per-channel data-source `ScratchStorage`, supplied to the partition class so
  // it can size its per-channel `load` slots in the Staged / SharedMem scratch.
  using value_data_source_scratch_types_tuple_t =
    ::cuda::std::conditional_t<keys_only,
                               ::cuda::std::tuple<>,
                               ::cuda::std::tuple<typename value_source_t::ScratchStorage>>;
  using value_sources_tuple_t =
    ::cuda::std::conditional_t<keys_only, ::cuda::std::tuple<>, ::cuda::std::tuple<value_source_t>>;

    // Offset types used to index into the selected and candidate iterators (and counter updates)
  using selected_offset_t           = OutOffsetT;
  using buffered_candidate_offset_t = OffsetT;

  // Reserve op types.
  using sel_reserve_op_t           = atomic_reserve_range_op<selected_offset_t>;
  using buffered_cand_reserve_op_t = atomic_reserve_range_op<buffered_candidate_offset_t>;

  // Key-output transforms are always identity for top-k.
  using key_xform_t = ::cuda::std::identity;

  // Callback / classify-hook types for the two modes.
  using histogram_callback_op_t = topk_histogram_callback_op<ExtractBinOpT, OffsetT>;
  using identify_selected_op_t  = topk_identify_selected_op<IdentifyCandidatesOpT>;

  // The buffered-mode primitive (`BlockPartition` or
  // `BlockPartitionAccumulatingCandidates`).
  using buffered_partition_t = strategy_to_partition_class_t<
    BufferedPartStrat,
    block_threads,
    items_per_thread,
    AgentTopKPolicyT::accumulating_buffer_capacity,
    key_in_t,
    selected_offset_t,
    buffered_candidate_offset_t,
    sel_reserve_op_t,
    buffered_cand_reserve_op_t,
    key_xform_t,
    key_xform_t,
    KeyOutputIteratorT,
    buffered_cand_key_out_t,
    IdentifyCandidatesOpT,
    histogram_callback_op_t,
    buffered_value_channel_sinks_tuple_t,
    value_types_tuple_t,
    value_data_source_scratch_types_tuple_t,
    effective_lazy_value_load>;

  // The early-stop-mode primitive (`BlockFilter` or `BlockFilterAccumulating`).
  using early_stop_filter_t = strategy_to_filter_class_t<
    EarlyStopFilterStrat,
    block_threads,
    items_per_thread,
    AgentTopKPolicyT::accumulating_buffer_capacity,
    key_in_t,
    selected_offset_t,
    sel_reserve_op_t,
    key_xform_t,
    KeyOutputIteratorT,
    identify_selected_op_t,
    early_stop_value_channel_sinks_tuple_t,
    value_types_tuple_t,
    value_data_source_scratch_types_tuple_t,
    effective_lazy_value_load>;

  // Per-mode storage layouts. Both expose the same `get_*()` accessors so the
  // mode-agnostic `drive_tile_loop` helper is layout-agnostic. The buffered
  // layout's `prefix_sum` slot holds the kth-bucket scan state; the early-stop
  // layout doesn't run a prefix sum so its `prefix_sum` slot is an empty
  // placeholder.
  struct empty_prefix_sum_t
  {};

  using buffered_storage_layout_t = bp_detail::partition_storage_layout_for_t<
    buffered_partition_t,
    typename keys_source_t::ScratchStorage,
    typename identify_kth_bucket_t::TempStorage>;

  using early_stop_storage_layout_t = bp_detail::partition_storage_layout_for_t<
    early_stop_filter_t,
    typename keys_source_t::ScratchStorage,
    empty_prefix_sum_t>;

  struct _TempStorage
  {
    // Histogram is only written by the buffered branch. Early-stop leaves it
    // untouched; the storage cost is unavoidable because the kernel TempStorage
    // is sized at compile time.
    OffsetT histogram[num_buckets];
    typename keys_source_t::TempStorage keys_source_state;

    // The two mode-specific arenas alias each other -- `run()` activates exactly
    // one of them based on the runtime `mode` arg. Both inner layouts have
    // non-trivial special members (their inner phase unions carry user-defined
    // ctor/dtor), so the wrapping union itself needs explicit ctor/dtor.
    union arena_t
    {
      buffered_storage_layout_t buffered;
      early_stop_storage_layout_t early_stop;

      _CCCL_HOST_DEVICE arena_t() {}
      _CCCL_HOST_DEVICE ~arena_t() {}
    } partition_arena;
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& storage;

  // Inputs/outputs shared across both modes.
  KeyInputIteratorT d_keys_in;
  KeyOutputIteratorT d_keys_out;
  ValueInputIteratorT d_values_in;
  ValueOutputIteratorT d_values_out;
  key_in_t* in_key_buf;
  value_in_t* in_val_buf;

  OutOffsetT* p_num_selected_written;

  OffsetT input_length;
  bool load_from_candidates_buffer;

  ExtractBinOpT extract_bin_op;
  IdentifyCandidatesOpT identify_candidates_op;
  OffsetT* global_histogram;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_topk_filter_partition(
    TempStorage& temp_storage,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    key_in_t* in_key_buf,
    value_in_t* in_val_buf,
    OutOffsetT* p_num_selected_written,
    OffsetT input_length,
    bool load_from_candidates_buffer,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op,
    OffsetT* global_histogram)
      : storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , in_key_buf(in_key_buf)
      , in_val_buf(in_val_buf)
      , p_num_selected_written(p_num_selected_written)
      , input_length(input_length)
      , load_from_candidates_buffer(load_from_candidates_buffer)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
      , global_histogram(global_histogram)
  {}

private:
  // Per-mode sinks-tuple factories. Each returns a tuple of length 0 (keys-only)
  // or 1 (with values).
  _CCCL_DEVICE _CCCL_FORCEINLINE auto
  make_buffered_value_channel_sinks([[maybe_unused]] buffered_cand_val_out_t cand_val_out)
  {
    if constexpr (keys_only)
    {
      return ::cuda::std::tuple<>{};
    }
    else
    {
      return ::cuda::std::tuple<buffered_value_channel_sinks_for_agent_t>{
        buffered_value_channel_sinks_for_agent_t{
          d_values_out, cand_val_out, ::cuda::std::identity{}, ::cuda::std::identity{}}};
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_early_stop_value_channel_sinks()
  {
    if constexpr (keys_only)
    {
      return ::cuda::std::tuple<>{};
    }
    else
    {
      return ::cuda::std::tuple<early_stop_value_channel_sinks_for_agent_t>{
        early_stop_value_channel_sinks_for_agent_t{d_values_out, ::cuda::std::identity{}}};
    }
  }

  // Per-tile value sources tuple. Both modes consume the same value source type.
  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_value_channel_sources(OffsetT tile_base)
  {
    (void) tile_base;
    if constexpr (keys_only)
    {
      return ::cuda::std::tuple<>{};
    }
    else
    {
      typename value_source_input_t::TempStorage val_state_input{};
      typename value_source_buffer_t::TempStorage val_state_buffer{};
      value_source_input_t val_input{d_values_in, val_state_input};
      value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};
      value_source_t val_src{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
      val_src.set_tile_base(tile_base);
      return ::cuda::std::tuple<value_source_t>{val_src};
    }
  }

  // Single mode-agnostic helper that iterates the tile space: full-tile loop +
  // possible trailing partial tile, both calling `primitive.Partition(...)` with
  // just `(scratch, keys, [num_items,] value_sources)`. Finishes with
  // `primitive.epilogue()`. The `arena` parameter is one of the two mode-specific
  // storage layouts; both expose the same `get_*()` accessors.
  template <typename Primitive, typename StorageLayout>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  drive_tile_loop(Primitive& primitive, StorageLayout& arena, keys_source_t& keys_source)
  {
    const OffsetT num_full_tiles = input_length / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items  = input_length - num_full_tiles * static_cast<OffsetT>(tile_items);

    // --- full-tile loop --------------------------------------------------
    for (OffsetT tile_id = static_cast<OffsetT>(blockIdx.x); tile_id < num_full_tiles;
         tile_id += static_cast<OffsetT>(gridDim.x))
    {
      const OffsetT tile_base = tile_id * static_cast<OffsetT>(tile_items);

      keys_source.set_tile_base(tile_base);
      value_sources_tuple_t value_sources = make_value_channel_sources(tile_base);

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(arena.get_keys_source_scratch());
      h.complete_load(items);
      __syncthreads();

      primitive.Partition(arena.get_partition_scratch(), items, value_sources);
    }

    // --- trailing partial tile (handled by exactly one block) ------------
    if (partial_items > 0)
    {
      const unsigned partial_owner = static_cast<unsigned>(num_full_tiles % static_cast<OffsetT>(gridDim.x));
      if (blockIdx.x == partial_owner)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);

        keys_source.set_tile_base(tile_base);
        value_sources_tuple_t value_sources = make_value_channel_sources(tile_base);

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(arena.get_keys_source_scratch(), partial_items);
        h.complete_load(items);
        __syncthreads();

        primitive.Partition(arena.get_partition_scratch(), items, partial_items, value_sources);
      }
    }

    // Terminal flush. No-op on non-accumulating; the accumulating variants drain
    // their leftover smem buffer here.
    primitive.epilogue();
  }

public:
  // --- entry point ----------------------------------------------------
  //
  // The buffered-mode-only outputs (`out_key_buf`, `out_val_buf`,
  // `p_num_candidates_written`) flow through the run signature; the early_stop
  // branch ignores them.
  //
  // `counter_update_fn` and `on_kth_bucket` are the two last-block-only
  // callbacks for `counter` writes; see the same-named docs on
  // `AgentTopKHistogram::invoke` and `block_identify_kth_bucket::find_kth_bucket`.
  template <typename CounterUpdateFn, typename KthBucketFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  run(unsigned int* retired_block_counter,
      OutOffsetT current_k,
      bool reset_histogram,
      CounterUpdateFn counter_update_fn,
      KthBucketFn on_kth_bucket,
      sink_mode mode,
      key_in_t* out_key_buf             = nullptr,
      value_in_t* out_val_buf           = nullptr,
      OffsetT* p_num_candidates_written = nullptr)
  {
    // Construct keys data source (multi_source over d_keys_in / in_key_buf).
    // Stable across the entire run() regardless of mode.
    key_source_input_t key_src_input{d_keys_in, storage.keys_source_state.a};
    key_source_buffer_t key_src_buffer{in_key_buf, storage.keys_source_state.b};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/load_from_candidates_buffer};

    // Mode-shared stack-locals.
    sel_reserve_op_t reserve_sel{p_num_selected_written};
    key_xform_t sel_key_xform{};
    key_xform_t cand_key_xform{};

    // Two-way mode dispatch: build the right primitive (with all sinks +
    // classify hooks captured at ctor), hand it to `drive_tile_loop`. The
    // helper iterates the tile space and finishes with `primitive.epilogue()`.
    if (mode == sink_mode::early_stop)
    {
      // Early-stop branch: no histogram is accumulated, no kth-bucket scan runs.
      // The filter primitive's `Partition()` writes selected items direct to
      // `d_keys_out` via `reserve_sel`.
      identify_selected_op_t identify_selected{identify_candidates_op};
      auto value_channel_sinks = make_early_stop_value_channel_sinks();

      early_stop_filter_t filter{
        storage.partition_arena.early_stop.get_partition_state(),
        reserve_sel,
        sel_key_xform,
        d_keys_out,
        value_channel_sinks,
        identify_selected};

      drive_tile_loop(filter, storage.partition_arena.early_stop, keys_source);
    }
    else
    {
      // Buffered branch: accumulate a histogram over candidates, scatter
      // selected to `d_keys_out` and candidates to `out_key_buf`.
      init_histogram<block_threads, num_buckets>(storage.histogram);
      __syncthreads();

      buffered_cand_key_out_t cand_key_out = out_key_buf;
      buffered_cand_val_out_t cand_val_out = out_val_buf;
      buffered_cand_reserve_op_t reserve_cand{p_num_candidates_written};
      histogram_callback_op_t histogram_cb{extract_bin_op, storage.histogram};
      auto value_channel_sinks = make_buffered_value_channel_sinks(cand_val_out);

      buffered_partition_t partition{
        storage.partition_arena.buffered.get_partition_state(),
        reserve_sel,
        reserve_cand,
        sel_key_xform,
        cand_key_xform,
        d_keys_out,
        cand_key_out,
        value_channel_sinks,
        identify_candidates_op,
        histogram_cb};

      drive_tile_loop(partition, storage.partition_arena.buffered, keys_source);

      __syncthreads();
      merge_histogram<block_threads, num_buckets>(storage.histogram, global_histogram);
    }

    // Last-block epilogue: per-mode counter update on thread 0, then the kth-bucket
    // scan + optional histogram reset (buffered branch only -- the early_stop
    // pass has nothing to finalize beyond the counter write).
    auto epilogue_op = [&] {
      if (threadIdx.x == 0)
      {
        counter_update_fn();
      }
      if (mode == sink_mode::buffered)
      {
        identify_kth_bucket_t{storage.partition_arena.buffered.get_prefix_sum()}.find_kth_bucket(
          global_histogram, current_k, on_kth_bucket);
        if (reset_histogram)
        {
          init_histogram<block_threads, num_buckets>(global_histogram);
        }
      }
    };

    finalize_pass(retired_block_counter, gridDim.x, epilogue_op);
  }
};

//---------------------------------------------------------------------
// agent_topk_last_filter: dedicated agent for the final filter pass.
//
// Unlike `agent_topk_filter_partition` (which covers passes
// 1..num_passes-1), this agent never accumulates a histogram and never runs
// finalize_pass. Its smem footprint is correspondingly smaller: just the
// keys-source persistent state plus the partition's scratch buffer.
//
// The pass-specific outputs (`p_num_ties_written_to_back`, `k_total`, `num_of_kth_needed`)
// flow through `run()` rather than the constructor; they only become known
// after the previous pass has updated the device-side counter.
//
// Selected candidates land at the front of `d_keys_out` via `p_num_selected_written`;
// surplus "kth"-class candidates land at the back of `d_keys_out` via a
// `back_grow_capped_reserve_op` (cap = num_of_kth_needed, anchor = k_total).
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename IdentifyCandidatesOpT,
          typename OffsetT,
          typename OutOffsetT,
          BlockPartitionStrategy PartStrat = BlockPartitionStrategy::AtomicsPreClassify,
          bool LazyValueLoad               = false>
struct agent_topk_last_filter
{
  using key_in_t   = it_value_t<KeyInputIteratorT>;
  using value_in_t = it_value_t<ValueInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  // last_filter operates as a true 2-way partition (it both writes selected items
  // to the front of d_keys_out AND back-grow-cap writes "kth"-class candidates to
  // the back). `BlockPartition` (any non-accumulating strategy) and
  // `BlockPartitionAccumulatingCandidates` are both supported -- the cooperative
  // flush in the accumulating variant honors the `back_grow_capped_reserve_op`'s
  // `may_grant_less` semantics by dropping items beyond the granted count,
  // equivalent (modulo flush-chunk granularity) to the per-item drop the Atomics
  // strategy performs.

  static constexpr bool effective_lazy_value_load = LazyValueLoad && !keys_only;

  using selected_offset_t  = OutOffsetT;
  using candidate_offset_t = OutOffsetT;

  using key_source_input_t =
    tile_data_source_t<KeyInputIteratorT, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using key_source_buffer_t =
    tile_data_source_t<key_in_t*, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using keys_source_t = multi_source_data_source<key_source_input_t, key_source_buffer_t, OffsetT>;

  using value_source_input_t = direct_data_source<ValueInputIteratorT, block_threads, items_per_thread, OffsetT>;
  using value_source_buffer_t = direct_data_source<value_in_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t = multi_source_data_source<value_source_input_t, value_source_buffer_t, OffsetT>;

  using val_out_t      = ValueOutputIteratorT;
  using cand_val_out_t = ValueOutputIteratorT; // selected and candidate share d_values_out

  using value_channel_sinks_for_agent_t =
    value_channel_sinks_t<val_out_t, cand_val_out_t, ::cuda::std::identity, ::cuda::std::identity>;
  using value_channel_sinks_tuple_t =
    ::cuda::std::conditional_t<keys_only, ::cuda::std::tuple<>, ::cuda::std::tuple<value_channel_sinks_for_agent_t>>;
  using value_types_tuple_t =
    ::cuda::std::conditional_t<keys_only, ::cuda::std::tuple<>, ::cuda::std::tuple<value_in_t>>;
  using value_data_source_scratch_types_tuple_t =
    ::cuda::std::conditional_t<keys_only,
                               ::cuda::std::tuple<>,
                               ::cuda::std::tuple<typename value_source_t::ScratchStorage>>;
  using value_sources_tuple_t =
    ::cuda::std::conditional_t<keys_only, ::cuda::std::tuple<>, ::cuda::std::tuple<value_source_t>>;

  // Reserve-op types: selected stream uses an atomic reserve, candidate stream uses
  // back-grow-capped (writes ties at the back of the output).
  using sel_reserve_op_t  = atomic_reserve_range_op<selected_offset_t>;
  using cand_reserve_op_t = back_grow_capped_reserve_op<candidate_offset_t>;

  using key_xform_t = ::cuda::std::identity;

  using partition_t = strategy_to_partition_class_t<
    PartStrat,
    block_threads,
    items_per_thread,
    AgentTopKPolicyT::accumulating_buffer_capacity,
    key_in_t,
    selected_offset_t,
    candidate_offset_t,
    sel_reserve_op_t,
    cand_reserve_op_t,
    key_xform_t,
    key_xform_t,
    KeyOutputIteratorT,
    KeyOutputIteratorT,
    IdentifyCandidatesOpT,
    topk_noop_candidate_callback_op,
    value_channel_sinks_tuple_t,
    value_types_tuple_t,
    value_data_source_scratch_types_tuple_t,
    effective_lazy_value_load>;

  // last_filter's smem: keys-source persistent state plus the same partition layout
  // helper the filter agent uses. For `BlockPartition` the partition state is empty
  // (so the layout collapses to a 2-arm union of `keys_source_scratch` and
  // `partition_scratch`); for `BlockPartitionAccumulatingCandidates` the persistent
  // slot buffers live in `partition_state`. There's no `prefix_sum` here, so we
  // hand the layout helper an empty `prefix_sum` placeholder type.
  struct empty_prefix_sum_t
  {};

  using storage_layout_t = bp_detail::partition_storage_layout_for_t<
    partition_t,
    typename keys_source_t::ScratchStorage,
    empty_prefix_sum_t>;

  struct _TempStorage
  {
    typename keys_source_t::TempStorage keys_source_state;
    storage_layout_t partition_arena;
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& storage;

  KeyInputIteratorT d_keys_in;
  KeyOutputIteratorT d_keys_out;
  ValueInputIteratorT d_values_in;
  ValueOutputIteratorT d_values_out;
  key_in_t* in_key_buf;
  value_in_t* in_val_buf;

  OutOffsetT* p_num_selected_written;
  OffsetT input_length;
  bool load_from_candidates_buffer;
  IdentifyCandidatesOpT identify_candidates_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_topk_last_filter(
    TempStorage& temp_storage,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    key_in_t* in_key_buf,
    value_in_t* in_val_buf,
    OutOffsetT* p_num_selected_written,
    OffsetT input_length,
    bool load_from_candidates_buffer,
    IdentifyCandidatesOpT identify_candidates_op)
      : storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , in_key_buf(in_key_buf)
      , in_val_buf(in_val_buf)
      , p_num_selected_written(p_num_selected_written)
      , input_length(input_length)
      , load_from_candidates_buffer(load_from_candidates_buffer)
      , identify_candidates_op(identify_candidates_op)
  {}

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_value_channel_sinks()
  {
    if constexpr (keys_only)
    {
      return ::cuda::std::tuple<>{};
    }
    else
    {
      // last_filter sends both selected and candidate values to `d_values_out`.
      return ::cuda::std::tuple<value_channel_sinks_for_agent_t>{
        value_channel_sinks_for_agent_t{d_values_out, d_values_out, ::cuda::std::identity{}, ::cuda::std::identity{}}};
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_value_channel_sources(OffsetT tile_base)
  {
    (void) tile_base;
    if constexpr (keys_only)
    {
      return ::cuda::std::tuple<>{};
    }
    else
    {
      typename value_source_input_t::TempStorage val_state_input{};
      typename value_source_buffer_t::TempStorage val_state_buffer{};
      value_source_input_t val_input{d_values_in, val_state_input};
      value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};
      value_source_t val_src{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
      val_src.set_tile_base(tile_base);
      return ::cuda::std::tuple<value_source_t>{val_src};
    }
  }

  template <bool IsFull>
  _CCCL_DEVICE _CCCL_FORCEINLINE void do_partition(
    partition_t& partition,
    const key_in_t (&keys)[items_per_thread],
    OffsetT num_items_in_tile,
    value_sources_tuple_t& value_sources)
  {
    if constexpr (IsFull)
    {
      partition.Partition(storage.partition_arena.get_partition_scratch(), keys, value_sources);
    }
    else
    {
      partition.Partition(storage.partition_arena.get_partition_scratch(), keys, num_items_in_tile, value_sources);
    }
  }

public:
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  run(OutOffsetT* p_num_ties_written_to_back, OutOffsetT k_total, OutOffsetT num_of_kth_needed)
  {
    key_source_input_t key_src_input{d_keys_in, storage.keys_source_state.a};
    key_source_buffer_t key_src_buffer{in_key_buf, storage.keys_source_state.b};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/load_from_candidates_buffer};

    // Build the reserve ops + sinks once, before the tile loop. The partition object
    // captures these references and consults them at every flush.
    sel_reserve_op_t reserve_sel{p_num_selected_written};
    cand_reserve_op_t reserve_cand{
      p_num_ties_written_to_back,
      static_cast<candidate_offset_t>(k_total),
      static_cast<candidate_offset_t>(num_of_kth_needed)};
    key_xform_t sel_key_xform{};
    key_xform_t cand_key_xform{};
    auto value_channel_sinks = make_value_channel_sinks();
    topk_noop_candidate_callback_op callback_op{};

    partition_t partition{
      storage.partition_arena.get_partition_state(),
      reserve_sel,
      reserve_cand,
      sel_key_xform,
      cand_key_xform,
      d_keys_out,
      d_keys_out,
      value_channel_sinks,
      identify_candidates_op,
      callback_op};

    // Split the tile space into full tiles and a possible single trailing
    // partial tile. The hot-path loop below processes only full tiles. The
    // partial tile, if any, is handled exactly once after the loop by the
    // unique block that would have owned it under the original grid-strided
    // schedule.
    const OffsetT num_full_tiles = input_length / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items  = input_length - num_full_tiles * static_cast<OffsetT>(tile_items);

    // --- full-tile loop --------------------------------------------------
    for (OffsetT tile_id = static_cast<OffsetT>(blockIdx.x); tile_id < num_full_tiles;
         tile_id += static_cast<OffsetT>(gridDim.x))
    {
      const OffsetT tile_base = tile_id * static_cast<OffsetT>(tile_items);

      keys_source.set_tile_base(tile_base);
      value_sources_tuple_t value_sources = make_value_channel_sources(tile_base);

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch());
      h.complete_load(items);
      __syncthreads();
      do_partition</*IsFull=*/true>(partition, items, /*num_items_in_tile=*/tile_items, value_sources);
    }

    // --- trailing partial tile (handled by exactly one block) ------------
    if (partial_items > 0)
    {
      const unsigned partial_owner = static_cast<unsigned>(num_full_tiles % static_cast<OffsetT>(gridDim.x));
      if (blockIdx.x == partial_owner)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);

        keys_source.set_tile_base(tile_base);
        value_sources_tuple_t value_sources = make_value_channel_sources(tile_base);

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch(), partial_items);
        h.complete_load(items);
        __syncthreads();
        do_partition</*IsFull=*/false>(partition, items, partial_items, value_sources);
      }
    }

    // Terminal partition flush. No-op on the non-accumulating strategies.
    partition.epilogue();
  }
};
} // namespace detail::topk
CUB_NAMESPACE_END
