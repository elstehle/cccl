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
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
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
//! @tparam LoadAlgorithm
//!   The BlockLoad algorithm to use
//!
//! @tparam ScanAlgorithm
//!   The BlockScan algorithm to use
//!
template <int ThreadsPerBlock,
          int ItemsPerThread,
          int BitsPerPass,
          BlockLoadAlgorithm LoadAlgorithm,
          BlockScanAlgorithm ScanAlgorithm,
          tile_load_kind KeysTileLoadKind = tile_load_kind::block_load_vectorize>
struct AgentTopKPolicy
{
  static constexpr int threads_per_block             = ThreadsPerBlock;
  static constexpr int items_per_thread              = ItemsPerThread;
  static constexpr int bits_per_pass                 = BitsPerPass;
  static constexpr BlockLoadAlgorithm load_algorithm = LoadAlgorithm;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = ScanAlgorithm;
  // Architecture §2.4: unifies sync `BlockLoadAlgorithm` choices and adds async TMA.
  // Used by the new agents to pick a TileDataSource specialization for the keys
  // stream. Defaults to the legacy `BLOCK_LOAD_VECTORIZE` mapping so existing call
  // sites that don't set it preserve current behavior.
  static constexpr tile_load_kind keys_tile_load_kind = KeysTileLoadKind;
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

template <typename KeyT, int BitsPerPass>
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

template <typename KeyInT, typename OffsetT, typename OutOffsetT>
struct alignas(128) Counter
{
  // We are processing the items in multiple passes, from most-significant to least-significant bits. In each pass, we
  // keep the length of input (`len`) and the `k` of current pass, and update them at the end of the pass.
  OutOffsetT k;
  OffsetT len;

  // `previous_len` is the length of the input in the previous pass. Note that `previous_len` rather than `len` is used
  // for the filtering step because filtering is indeed for previous pass.
  OffsetT previous_len;

  // We determine the bits of the k_th key inside the mask processed by the pass. The
  // already known bits are stored in `kth_key_bits`. It's used to discriminate a
  // element is a result (written to `out`), a candidate for next pass (written to
  // `out_buf`), or not useful (discarded). The bits that are not yet processed do not
  // matter for this purpose.
  key_prefix_storage_t<KeyInT> kth_key_bits;

  // Record how many elements have passed filtering. It's used to determine the position
  // in the `out_buf` where an element should be written.
  alignas(128) OffsetT filter_cnt;

  // For a row inside a batch, we may launch multiple thread blocks. This counter is
  // used to determine if the current block is the last running block. If so, this block
  // will execute compute_bin_offsets() and choose_bucket().
  alignas(128) unsigned int finished_block_cnt;

  // Record how many elements have been written to the front of `out`. Elements less (if
  // SelectMin==true) than the k-th key are written from front to back.
  alignas(128) OutOffsetT out_cnt;

  // Record how many elements have been written to the back of `out`. Elements equal to
  // the k-th key are written from back to front. We need to keep count of them
  // separately because the number of elements that <= the k-th key might exceed k.
  alignas(128) OutOffsetT out_back_cnt;
  // The 'alignas' is necessary to improve the performance of global memory accessing by isolating the request,
  // especially for the segment version.
};

// candidate_class is defined in <cub/detail/topk/block_partition.cuh>.

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

// Atomically merge a block-local histogram into a global histogram. The local and global counters
// may have different types (e.g. a 32-bit local histogram merged into a 32- or 64-bit global one);
// each non-zero local value is widened to the global counter type before the atomic add.
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
// Prefix sum over histogram bins: temp storage and compute function
//---------------------------------------------------------------------

template <int BlockThreads, int BitsPerPass, BlockScanAlgorithm ScanAlgorithm, typename OffsetT>
struct BinPrefixSumTempStorage
{
  static constexpr int num_buckets     = 1 << BitsPerPass;
  static constexpr int bins_per_thread = ::cuda::ceil_div(num_buckets, BlockThreads);

  using block_load_t  = BlockLoad<OffsetT, BlockThreads, bins_per_thread, BLOCK_LOAD_TRANSPOSE>;
  using block_scan_t  = BlockScan<OffsetT, BlockThreads, ScanAlgorithm>;
  using block_store_t = BlockStore<OffsetT, BlockThreads, bins_per_thread, BLOCK_STORE_TRANSPOSE>;

  union
  {
    typename block_load_t::TempStorage load;
    typename block_scan_t::TempStorage scan;
    typename block_store_t::TempStorage store;
  };
};

// Compute inclusive prefix sum over histogram bin counts.
// Reads from input_histogram (global), writes prefix sums to output_histogram (shared).
template <int BlockThreads, int BitsPerPass, BlockScanAlgorithm ScanAlgorithm, typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE void compute_bin_offsets(
  BinPrefixSumTempStorage<BlockThreads, BitsPerPass, ScanAlgorithm, OffsetT>& temp,
  volatile OffsetT* input_histogram,
  OffsetT* output_histogram)
{
  using storage_t                        = BinPrefixSumTempStorage<BlockThreads, BitsPerPass, ScanAlgorithm, OffsetT>;
  static constexpr int num_buckets_local = storage_t::num_buckets;
  static constexpr int bins_per_thread   = storage_t::bins_per_thread;

  OffsetT thread_data[bins_per_thread]{};
  typename storage_t::block_load_t(temp.load).Load(input_histogram, thread_data, num_buckets_local);
  __syncthreads();
  typename storage_t::block_scan_t(temp.scan).InclusiveSum(thread_data, thread_data);
  __syncthreads();
  typename storage_t::block_store_t(temp.store).Store(output_histogram, thread_data, num_buckets_local);
}

//---------------------------------------------------------------------
// choose_bucket: identify the bucket containing the k-th element
//---------------------------------------------------------------------

template <int BlockThreads, int BitsPerPass, typename KeyInT, typename OffsetT, typename OutOffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE void choose_bucket(
  const OffsetT* prefix_sum_histogram,
  Counter<KeyInT, OffsetT, OutOffsetT>* counter,
  const OutOffsetT k,
  const int pass)
{
  static constexpr int num_buckets = 1 << BitsPerPass;
  int histo_offset                 = 0;

  auto body = [&] {
    const int bin_idx  = histo_offset + threadIdx.x;
    const OffsetT prev = (bin_idx == 0) ? 0 : prefix_sum_histogram[bin_idx - 1];
    const OffsetT cur  = prefix_sum_histogram[bin_idx];

    if (prev < k && cur >= k)
    {
      counter->k   = k - prev;
      counter->len = cur - prev;
      set_kth_key_bits<KeyInT, BitsPerPass>(
        counter->kth_key_bits, pass, static_cast<unsigned int>(bin_idx));
    }
  };

  _CCCL_PRAGMA_UNROLL_FULL()
  for (; histo_offset + BlockThreads <= num_buckets; histo_offset += BlockThreads)
  {
    body();
  }
  if ((num_buckets % BlockThreads != 0) && (histo_offset + static_cast<int>(threadIdx.x) < num_buckets))
  {
    body();
  }
}

//---------------------------------------------------------------------
// finalize_pass: last-block coordination after histogram accumulation
//---------------------------------------------------------------------

// Ensures global visibility of histogram writes, detects the last finishing block,
// runs the prefix sum, identifies the k-th bucket, and resets the histogram for
// the next pass. counter_update_fn runs on thread 0 of the last block.
template <int BlockThreads,
          int BitsPerPass,
          BlockScanAlgorithm ScanAlgorithm,
          typename KeyInT,
          typename OffsetT,
          typename OutOffsetT,
          typename CounterUpdateFn>
_CCCL_DEVICE _CCCL_FORCEINLINE void finalize_pass(
  BinPrefixSumTempStorage<BlockThreads, BitsPerPass, ScanAlgorithm, OffsetT>& prefix_sum_temp,
  OffsetT* histogram_smem,
  Counter<KeyInT, OffsetT, OutOffsetT>* counter,
  OffsetT* global_histogram,
  OutOffsetT current_k,
  int pass,
  bool is_last_pass,
  CounterUpdateFn counter_update_fn)
{
  static constexpr int num_buckets = 1 << BitsPerPass;

  __threadfence();

  bool is_last_block = false;
  if (threadIdx.x == 0)
  {
    unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
    is_last_block         = (finished == (gridDim.x - 1));
  }

  if (__syncthreads_or(is_last_block))
  {
    if (threadIdx.x == 0)
    {
      counter_update_fn();
    }

    compute_bin_offsets(prefix_sum_temp, global_histogram, histogram_smem);
    __syncthreads();

    choose_bucket<BlockThreads, BitsPerPass>(histogram_smem, counter, current_k, pass);

    if (!is_last_pass)
    {
      init_histogram<BlockThreads, num_buckets>(global_histogram);
    }
  }
}

//---------------------------------------------------------------------
// AgentTopKHistogram filter helpers
//---------------------------------------------------------------------
//
// The histogram agent accepts an arbitrary unary predicate `FilterOpT(key) ->
// bool` that decides whether a given key contributes to the histogram. The two
// canonical specializations live here:
//
//   * `topk_pass_through_filter_op` -- pass-0 default. Always returns `true`,
//     so the optimizer can fold the predicate call away and the inner tile
//     loop reduces to "extract bucket + atomicAdd" exactly as it did before
//     this generalization. SASS for the pass-0 hot loop must remain identical
//     and is verified externally.
//
//   * `topk_candidate_filter_op<IdentifyCandidatesOpT>` -- thin wrapper used
//     by the unbuffered filter pass. It wraps the kernel's
//     `identify_candidates_op` and returns `true` only for keys classified as
//     `candidate_class::candidate`, replicating what `do_histogram_only` did
//     in the previous `agent_topk_filter_partition` unbuffered specialization.

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
  using prefix_sum_temp_t =
    BinPrefixSumTempStorage<block_threads, bits_per_pass, AgentTopKPolicyT::SCAN_ALGORITHM, OffsetT>;

  struct _TempStorage
  {
    // Persistent region (architecture §2.2).
    OffsetT histogram[num_buckets];
    typename keys_source_t::TempStorage keys_source_state;

    // Method-call scratch (mutually exclusive in time).
    union
    {
      typename keys_source_t::ScratchStorage keys_source_scratch;
      prefix_sum_temp_t prefix_sum;
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

  // `counter_update_fn` runs on thread 0 of the last finishing block, inside
  // `finalize_pass`. The pass-0 caller passes a lambda that resets
  // `previous_len`/`filter_cnt`; the unbuffered caller passes its
  // mode-dependent counter update (early-stop vs non-early-stop logic).
  template <typename CounterUpdateFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke(
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OffsetT* global_histogram,
    int pass,
    bool is_last_pass,
    CounterUpdateFn counter_update_fn)
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
      const unsigned partial_owner =
        static_cast<unsigned>(num_full_tiles % static_cast<OffsetT>(gridDim.x));
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

    finalize_pass(temp_storage.scratch.prefix_sum,
                  temp_storage.histogram,
                  counter,
                  global_histogram,
                  k,
                  pass,
                  is_last_pass,
                  counter_update_fn);
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
//
// The dedicated last-filter pass lives in its own agent (`agent_topk_last_filter`)
// and is intentionally not part of this enum either.
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
// Compile-time specialized on `sink_mode` (early_stop / buffered /
// unbuffered); the kernel switches to the matching instantiation based on
// the device-side Counter state. Owns the tile loop with full/partial
// dispatch; keys come in via a `multi_source_data_source` over
// `(d_keys_in, in_key_buf)` selected at runtime by `load_from_original_input`.
// The per-tile partition is delegated to `BlockPartition` with strategy
// `PartStrat`, per-mode reserve ops, and a candidate callback that performs
// the histogram update inline.
//
// Members hold the inputs/outputs that all three modes consume; the
// buffered-only candidate outputs (`out_key_buf`, `out_val_buf`,
// `p_filter_cnt`) are passed to `run()` and flow through `do_partition`
// rather than being stored on the agent.
//---------------------------------------------------------------------

// Identity transform helper used as the per-stream key/value transform op.
struct topk_identity_transform_op
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(const T& x) const
  {
    return x;
  }
};

// No-op candidate callback used by `early_stop` (HasCandidates=false collapses the
// candidate stream onto selected; the callback is statically guaranteed never to
// fire, but the partition primitive still requires a callable).
struct topk_noop_candidate_callback_op
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(const T&) const
  {}
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
          sink_mode Mode,
          BlockPartitionStrategy PartStrat        = BlockPartitionStrategy::Atomics,
          BlockPartitionClassifyMode ClassifyMode = BlockPartitionClassifyMode::precomputed,
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

  using prefix_sum_temp_t =
    BinPrefixSumTempStorage<block_threads, bits_per_pass, AgentTopKPolicyT::SCAN_ALGORITHM, OffsetT>;

  // Compile-time mode plumbing.
  //   selected_offset_t / candidate_offset_t : pointer types of the global counters.
  //   accumulate_histogram : `buffered` accumulates a histogram over candidates;
  //                          `early_stop` does not.
  static constexpr bool accumulate_histogram = (Mode == sink_mode::buffered);

  using selected_offset_t  = OutOffsetT;

  using candidate_offset_t = ::cuda::std::conditional_t<Mode == sink_mode::buffered, OffsetT, OutOffsetT>;
  // The `inlined` classify mode is only valid with the Atomics scatter strategy. When
  // the agent is configured with a different strategy (Staged / SharedMem), silently
  // fall back to `precomputed` so the agent type still instantiates cleanly.
  static constexpr BlockPartitionClassifyMode effective_classify =
    (PartStrat == BlockPartitionStrategy::Atomics)
      ? ClassifyMode
      : BlockPartitionClassifyMode::precomputed;
  // Lazy value-load only makes sense on the Atomics path (the smem-coordinating
  // strategies precompute everything cooperatively) and only when there is
  // actually a value channel to load lazily. Forced off otherwise so the agent
  // template still instantiates cleanly across all configurations.
  static constexpr bool effective_lazy_value_load =
    LazyValueLoad && !keys_only && (PartStrat == BlockPartitionStrategy::Atomics);

  // Keys data source: multi_source over (d_keys_in source, in_key_buf source).
  using key_source_a_t =
    tile_data_source_t<KeyInputIteratorT, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using key_source_b_t =
    tile_data_source_t<key_in_t*, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using keys_source_t = multi_source_data_source<key_source_a_t, key_source_b_t, OffsetT>;

  // Value channel: multi_source over (d_values_in, in_val_buf), each wrapped in
  // `direct_data_source` per the plan. For keys-only the value channel tuple stays
  // empty.
  using val_source_a_t = direct_data_source<ValueInputIteratorT, block_threads, items_per_thread, OffsetT>;
  using val_source_b_t = direct_data_source<value_in_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t = multi_source_data_source<val_source_a_t, val_source_b_t, OffsetT>;

  using val_out_t = ValueOutputIteratorT;
  // For `buffered` the candidate iterator is `value_in_t*` (the back-buffer); for
  // `early_stop` it's `ValueOutputIteratorT`.
  using cand_val_out_t =
    ::cuda::std::conditional_t<Mode == sink_mode::buffered, value_in_t*, ValueOutputIteratorT>;

  struct value_channel_t
  {
    using data_source_t = value_source_t;
    using value_t       = value_in_t;

    data_source_t data_source;
    val_out_t selected_values_out;
    cand_val_out_t candidate_values_out;
    topk_identity_transform_op selected_value_transform;
    topk_identity_transform_op candidate_value_transform;
  };

  using value_channels_tuple_t =
    ::cuda::std::conditional_t<keys_only, ::cuda::std::tuple<>, ::cuda::std::tuple<value_channel_t>>;

  using partition_t = BlockPartition<block_threads,
                                     items_per_thread,
                                     PartStrat,
                                     key_in_t,
                                     selected_offset_t,
                                     candidate_offset_t,
                                     value_channels_tuple_t,
                                     effective_classify>;

  // Smem layout: histogram + keys-source persistent state in the persistent
  // region; method-call scratch is a union of the keys-source scratch, the
  // prefix-sum scratch, and the partition's scratch. Both modes need the
  // histogram array (either to accumulate it directly or to consume it during
  // finalize_pass).
  struct _TempStorage
  {
    OffsetT histogram[num_buckets];
    typename keys_source_t::TempStorage keys_source_state;

    union
    {
      typename keys_source_t::ScratchStorage keys_source_scratch;
      prefix_sum_temp_t prefix_sum;
      typename partition_t::ScratchStorage partition_buf;
    } scratch;
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

  OutOffsetT* p_out_cnt;

  OffsetT input_length;
  bool load_from_original_input;

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
    OutOffsetT* p_out_cnt,
    OffsetT input_length,
    bool load_from_original_input,
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
      , p_out_cnt(p_out_cnt)
      , input_length(input_length)
      , load_from_original_input(load_from_original_input)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
      , global_histogram(global_histogram)
  {}

private:
  // Build the value channel tuple (empty for keys-only). Takes `out_val_buf`
  // because for the buffered mode the candidate-stream output is the back
  // buffer rather than `d_values_out`.
  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_value_channels([[maybe_unused]] value_in_t* out_val_buf)
  {
    if constexpr (keys_only)
    {
      return ::cuda::std::tuple<>{};
    }
    else
    {
      typename val_source_a_t::TempStorage val_state_a{};
      typename val_source_b_t::TempStorage val_state_b{};
      val_source_a_t val_a{d_values_in, val_state_a};
      val_source_b_t val_b{in_val_buf, val_state_b};
      value_source_t val_src{val_a, val_b, /*pick_b=*/!load_from_original_input};

      [[maybe_unused]] cand_val_out_t cand_out{};
      if constexpr (Mode == sink_mode::buffered)
      {
        cand_out = out_val_buf;
      }
      else
      {
        cand_out = d_values_out;
      }
      return ::cuda::std::tuple<value_channel_t>{value_channel_t{
        val_src,
        d_values_out,
        cand_out,
        topk_identity_transform_op{},
        topk_identity_transform_op{}}};
    }
  }

  // Mode-specific partition call. Builds the per-mode reserve ops and dispatches to
  // the matching `BlockPartition::Partition()` overload. The `out_key_buf` and
  // `p_filter_cnt` arguments are only consumed by the buffered branch.
  template <bool IsFull>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  do_partition(const key_in_t (&keys)[items_per_thread],
               OffsetT num_items_in_tile,
               value_channels_tuple_t& channels,
               [[maybe_unused]] key_in_t* out_key_buf,
               [[maybe_unused]] OffsetT* p_filter_cnt)
  {
    partition_t partition{};
    topk_identity_transform_op key_transform{};

    if constexpr (Mode == sink_mode::early_stop)
    {
      // HasCandidates=false: selected and candidate fold into d_keys_out via
      // p_out_cnt; the candidate-side machinery is statically elided.
      atomic_reserve_range_op<selected_offset_t> reserve_sel{p_out_cnt};
      atomic_reserve_range_op<candidate_offset_t> reserve_cand{
        reinterpret_cast<candidate_offset_t*>(p_out_cnt)}; // unused by partition when HasCandidates=false
      topk_noop_candidate_callback_op cb{};
      if constexpr (IsFull)
      {
        partition.template Partition<false, effective_lazy_value_load>(
          storage.scratch.partition_buf,
          keys,
          ::cuda::std::integral_constant<bool, false>{},
          identify_candidates_op,
          cb,
          reserve_sel,
          reserve_cand,
          key_transform,
          key_transform,
          d_keys_out,
          d_keys_out,
          channels);
      }
      else
      {
        partition.template Partition<false, effective_lazy_value_load>(
          storage.scratch.partition_buf,
          keys,
          num_items_in_tile,
          ::cuda::std::integral_constant<bool, false>{},
          identify_candidates_op,
          cb,
          reserve_sel,
          reserve_cand,
          key_transform,
          key_transform,
          d_keys_out,
          d_keys_out,
          channels);
      }
    }
    else
    {
      static_assert(Mode == sink_mode::buffered, "do_partition is only called for output-writing modes");
      // selected -> d_keys_out via p_out_cnt; candidate -> out_key_buf via
      // p_filter_cnt; histogram callback fires per candidate.
      atomic_reserve_range_op<selected_offset_t> reserve_sel{p_out_cnt};
      atomic_reserve_range_op<candidate_offset_t> reserve_cand{p_filter_cnt};
      topk_histogram_callback_op<ExtractBinOpT, OffsetT> cb{extract_bin_op, storage.histogram};
      if constexpr (IsFull)
      {
        partition.template Partition<true, effective_lazy_value_load>(
          storage.scratch.partition_buf,
          keys,
          ::cuda::std::integral_constant<bool, true>{},
          identify_candidates_op,
          cb,
          reserve_sel,
          reserve_cand,
          key_transform,
          key_transform,
          d_keys_out,
          out_key_buf,
          channels);
      }
      else
      {
        partition.template Partition<true, effective_lazy_value_load>(
          storage.scratch.partition_buf,
          keys,
          num_items_in_tile,
          ::cuda::std::integral_constant<bool, true>{},
          identify_candidates_op,
          cb,
          reserve_sel,
          reserve_cand,
          key_transform,
          key_transform,
          d_keys_out,
          out_key_buf,
          channels);
      }
    }
  }

public:
  // --- entry point ----------------------------------------------------
  //
  // The buffered-mode-only outputs (`out_key_buf`, `out_val_buf`,
  // `p_filter_cnt`) flow through the run signature. They default to nullptr
  // for the early_stop caller, which never consumes them (gated by
  // `if constexpr (Mode == sink_mode::buffered)` inside the agent).
  template <typename CounterUpdateFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OutOffsetT current_k,
    int pass,
    bool is_last_pass,
    CounterUpdateFn counter_update_fn,
    key_in_t*  out_key_buf  = nullptr,
    value_in_t* out_val_buf = nullptr,
    OffsetT*   p_filter_cnt = nullptr)
  {
    if constexpr (accumulate_histogram)
    {
      init_histogram<block_threads, num_buckets>(storage.histogram);
      __syncthreads();
    }

    // Construct keys data source (multi_source over d_keys_in / in_key_buf).
    key_source_a_t key_src_a{d_keys_in, storage.keys_source_state.a};
    key_source_b_t key_src_b{in_key_buf, storage.keys_source_state.b};
    keys_source_t keys_source{key_src_a, key_src_b, /*pick_b=*/!load_from_original_input};

    // Split the tile space into full tiles and a possible single trailing
    // partial tile. The hot-path loop below processes only full tiles (no
    // per-iteration full/partial branch). The partial tile, if any, is
    // handled exactly once after the loop by the unique block that would
    // have owned it under the original grid-strided schedule.
    const OffsetT num_full_tiles = input_length / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items  = input_length - num_full_tiles * static_cast<OffsetT>(tile_items);

    // --- full-tile loop --------------------------------------------------
    for (OffsetT tile_id = static_cast<OffsetT>(blockIdx.x); tile_id < num_full_tiles;
         tile_id += static_cast<OffsetT>(gridDim.x))
    {
      const OffsetT tile_base = tile_id * static_cast<OffsetT>(tile_items);

      keys_source.set_tile_base(tile_base);
      value_channels_tuple_t channels = make_value_channels(out_val_buf);
      if constexpr (!keys_only)
      {
        ::cuda::std::get<0>(channels).data_source.set_tile_base(tile_base);
      }

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.scratch.keys_source_scratch);
      h.complete_load(items);
      __syncthreads();
      do_partition</*IsFull=*/true>(items, static_cast<OffsetT>(tile_items), channels, out_key_buf, p_filter_cnt);
    }

    // --- trailing partial tile (handled by exactly one block) ------------
    //
    // Under the original grid-strided schedule the partial tile (tile_id =
    // num_full_tiles) was claimed by the block whose blockIdx.x equals
    // num_full_tiles % gridDim.x. We preserve that ownership so the work
    // distribution across blocks is unchanged.
    if (partial_items > 0)
    {
      const unsigned partial_owner =
        static_cast<unsigned>(num_full_tiles % static_cast<OffsetT>(gridDim.x));
      if (blockIdx.x == partial_owner)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);

        keys_source.set_tile_base(tile_base);
        value_channels_tuple_t channels = make_value_channels(out_val_buf);
        if constexpr (!keys_only)
        {
          ::cuda::std::get<0>(channels).data_source.set_tile_base(tile_base);
        }

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.scratch.keys_source_scratch, partial_items);
        h.complete_load(items);
        __syncthreads();
        do_partition</*IsFull=*/false>(items, partial_items, channels, out_key_buf, p_filter_cnt);
      }
    }

    if constexpr (accumulate_histogram)
    {
      __syncthreads();
      merge_histogram<block_threads, num_buckets>(storage.histogram, global_histogram);
    }

    finalize_pass(storage.scratch.prefix_sum,
                  storage.histogram,
                  counter,
                  global_histogram,
                  current_k,
                  pass,
                  is_last_pass,
                  counter_update_fn);
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
// The pass-specific outputs (`p_out_back_cnt`, `k_total`, `num_of_kth_needed`)
// flow through `run()` rather than the constructor; they only become known
// after the previous pass has updated the device-side Counter.
//
// Selected candidates land at the front of `d_keys_out` via `p_out_cnt`;
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
          BlockPartitionStrategy PartStrat        = BlockPartitionStrategy::Atomics,
          BlockPartitionClassifyMode ClassifyMode = BlockPartitionClassifyMode::precomputed,
          bool LazyValueLoad                       = false>
struct agent_topk_last_filter
{
  using key_in_t   = it_value_t<KeyInputIteratorT>;
  using value_in_t = it_value_t<ValueInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  // last_filter writes output, so unlike `agent_topk_filter_partition`'s
  // `unbuffered` mode we honor the policy's `PartStrat` and don't force
  // Atomics. Same fall-back rules for classify and lazy_value_load apply.
  static constexpr BlockPartitionStrategy effective_strat = PartStrat;
  static constexpr BlockPartitionClassifyMode effective_classify =
    (effective_strat == BlockPartitionStrategy::Atomics)
      ? ClassifyMode
      : BlockPartitionClassifyMode::precomputed;
  static constexpr bool effective_lazy_value_load =
    LazyValueLoad && !keys_only && (effective_strat == BlockPartitionStrategy::Atomics);

  using selected_offset_t  = OutOffsetT;
  using candidate_offset_t = OutOffsetT;

  using key_source_a_t =
    tile_data_source_t<KeyInputIteratorT, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using key_source_b_t =
    tile_data_source_t<key_in_t*, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using keys_source_t = multi_source_data_source<key_source_a_t, key_source_b_t, OffsetT>;

  using val_source_a_t = direct_data_source<ValueInputIteratorT, block_threads, items_per_thread, OffsetT>;
  using val_source_b_t = direct_data_source<value_in_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t = multi_source_data_source<val_source_a_t, val_source_b_t, OffsetT>;

  using val_out_t      = ValueOutputIteratorT;
  using cand_val_out_t = ValueOutputIteratorT; // selected and candidate share d_values_out

  struct value_channel_t
  {
    using data_source_t = value_source_t;
    using value_t       = value_in_t;

    data_source_t data_source;
    val_out_t selected_values_out;
    cand_val_out_t candidate_values_out;
    topk_identity_transform_op selected_value_transform;
    topk_identity_transform_op candidate_value_transform;
  };

  using value_channels_tuple_t =
    ::cuda::std::conditional_t<keys_only, ::cuda::std::tuple<>, ::cuda::std::tuple<value_channel_t>>;

  using partition_t = BlockPartition<block_threads,
                                     items_per_thread,
                                     effective_strat,
                                     key_in_t,
                                     selected_offset_t,
                                     candidate_offset_t,
                                     value_channels_tuple_t,
                                     effective_classify>;

  // last_filter's smem: keys-source persistent state plus a 2-arm scratch
  // union (the keys-source's load scratch and the partition's scratch).
  // No histogram, no prefix-sum scratch.
  struct _TempStorage
  {
    typename keys_source_t::TempStorage keys_source_state;

    union
    {
      typename keys_source_t::ScratchStorage keys_source_scratch;
      typename partition_t::ScratchStorage partition_buf;
    } scratch;
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

  OutOffsetT* p_out_cnt;
  OffsetT input_length;
  bool load_from_original_input;
  IdentifyCandidatesOpT identify_candidates_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_topk_last_filter(
    TempStorage& temp_storage,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    key_in_t* in_key_buf,
    value_in_t* in_val_buf,
    OutOffsetT* p_out_cnt,
    OffsetT input_length,
    bool load_from_original_input,
    IdentifyCandidatesOpT identify_candidates_op)
      : storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , in_key_buf(in_key_buf)
      , in_val_buf(in_val_buf)
      , p_out_cnt(p_out_cnt)
      , input_length(input_length)
      , load_from_original_input(load_from_original_input)
      , identify_candidates_op(identify_candidates_op)
  {}

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_value_channels()
  {
    if constexpr (keys_only)
    {
      return ::cuda::std::tuple<>{};
    }
    else
    {
      typename val_source_a_t::TempStorage val_state_a{};
      typename val_source_b_t::TempStorage val_state_b{};
      val_source_a_t val_a{d_values_in, val_state_a};
      val_source_b_t val_b{in_val_buf, val_state_b};
      value_source_t val_src{val_a, val_b, /*pick_b=*/!load_from_original_input};

      // For last_filter both selected and candidate values feed `d_values_out`.
      return ::cuda::std::tuple<value_channel_t>{value_channel_t{
        val_src,
        d_values_out,
        d_values_out,
        topk_identity_transform_op{},
        topk_identity_transform_op{}}};
    }
  }

  template <bool IsFull>
  _CCCL_DEVICE _CCCL_FORCEINLINE void do_partition(
    const key_in_t (&keys)[items_per_thread],
    OffsetT num_items_in_tile,
    value_channels_tuple_t& channels,
    OutOffsetT* p_out_back_cnt,
    OutOffsetT k_total,
    OutOffsetT num_of_kth_needed)
  {
    partition_t partition{};
    topk_identity_transform_op key_transform{};

    atomic_reserve_range_op<selected_offset_t> reserve_sel{p_out_cnt};
    back_grow_capped_reserve_op<candidate_offset_t> reserve_cand{
      p_out_back_cnt,
      static_cast<candidate_offset_t>(k_total),
      static_cast<candidate_offset_t>(num_of_kth_needed)};
    topk_noop_candidate_callback_op cb{}; // last_filter doesn't accumulate histogram
    if constexpr (IsFull)
    {
      partition.template Partition<true, effective_lazy_value_load>(
        storage.scratch.partition_buf,
        keys,
        ::cuda::std::integral_constant<bool, true>{},
        identify_candidates_op,
        cb,
        reserve_sel,
        reserve_cand,
        key_transform,
        key_transform,
        d_keys_out,
        d_keys_out,
        channels);
    }
    else
    {
      partition.template Partition<true, effective_lazy_value_load>(
        storage.scratch.partition_buf,
        keys,
        num_items_in_tile,
        ::cuda::std::integral_constant<bool, true>{},
        identify_candidates_op,
        cb,
        reserve_sel,
        reserve_cand,
        key_transform,
        key_transform,
        d_keys_out,
        d_keys_out,
        channels);
    }
  }

public:
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(
    OutOffsetT* p_out_back_cnt,
    OutOffsetT k_total,
    OutOffsetT num_of_kth_needed)
  {
    key_source_a_t key_src_a{d_keys_in, storage.keys_source_state.a};
    key_source_b_t key_src_b{in_key_buf, storage.keys_source_state.b};
    keys_source_t keys_source{key_src_a, key_src_b, /*pick_b=*/!load_from_original_input};

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
      value_channels_tuple_t channels = make_value_channels();
      if constexpr (!keys_only)
      {
        ::cuda::std::get<0>(channels).data_source.set_tile_base(tile_base);
      }

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.scratch.keys_source_scratch);
      h.complete_load(items);
      __syncthreads();
      do_partition</*IsFull=*/true>(items,
                                    static_cast<OffsetT>(tile_items),
                                    channels,
                                    p_out_back_cnt,
                                    k_total,
                                    num_of_kth_needed);
    }

    // --- trailing partial tile (handled by exactly one block) ------------
    if (partial_items > 0)
    {
      const unsigned partial_owner =
        static_cast<unsigned>(num_full_tiles % static_cast<OffsetT>(gridDim.x));
      if (blockIdx.x == partial_owner)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);

        keys_source.set_tile_base(tile_base);
        value_channels_tuple_t channels = make_value_channels();
        if constexpr (!keys_only)
        {
          ::cuda::std::get<0>(channels).data_source.set_tile_base(tile_base);
        }

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.scratch.keys_source_scratch, partial_items);
        h.complete_load(items);
        __syncthreads();
        do_partition</*IsFull=*/false>(items,
                                       partial_items,
                                       channels,
                                       p_out_back_cnt,
                                       k_total,
                                       num_of_kth_needed);
      }
    }
  }
};

} // namespace detail::topk
CUB_NAMESPACE_END
