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

#include <cub/agent/topk/block_partition.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
//! @brief Parameterizable tuning policy type for AgentTopK
//!
//! @tparam BlockThreads
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
template <int BlockThreads,
          int ItemsPerThread,
          int BitsPerPass,
          BlockLoadAlgorithm LoadAlgorithm,
          BlockScanAlgorithm ScanAlgorithm>
struct AgentTopKPolicy
{
  static constexpr int block_threads                 = BlockThreads;
  static constexpr int items_per_thread              = ItemsPerThread;
  static constexpr int bits_per_pass                 = BitsPerPass;
  static constexpr BlockLoadAlgorithm load_algorithm = LoadAlgorithm;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = ScanAlgorithm;
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

// candidate_class is defined in <cub/agent/topk/block_partition.cuh>.

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

// Atomically merge a block-local histogram into a global histogram
template <int BlockThreads, int NumBuckets, typename CounterT>
_CCCL_DEVICE _CCCL_FORCEINLINE void merge_histogram(const CounterT* local_histogram, CounterT* global_histogram)
{
  int histo_offset = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (; histo_offset + BlockThreads <= NumBuckets; histo_offset += BlockThreads)
  {
    if (local_histogram[histo_offset + threadIdx.x] != 0)
    {
      atomicAdd(global_histogram + (histo_offset + threadIdx.x), local_histogram[histo_offset + threadIdx.x]);
    }
  }
  if ((NumBuckets % BlockThreads != 0) && (histo_offset + static_cast<int>(threadIdx.x) < NumBuckets))
  {
    if (local_histogram[histo_offset + threadIdx.x] != 0)
    {
      atomicAdd(global_histogram + (histo_offset + threadIdx.x), local_histogram[histo_offset + threadIdx.x]);
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
// process_histogram_only: tile processor that builds a radix histogram
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT>
struct process_histogram_only
{
  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;

  using prefix_sum_temp_t =
    BinPrefixSumTempStorage<block_threads, bits_per_pass, AgentTopKPolicyT::SCAN_ALGORITHM, OffsetT>;

  struct _TempStorage
  {
    prefix_sum_temp_t prefix_sum;
    OffsetT histogram[num_buckets];
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  prefix_sum_temp_t& prefix_sum;
  OffsetT* histogram;
  ExtractBinOpT extract_bin_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE
  process_histogram_only(TempStorage& ts, ExtractBinOpT extract_bin_op)
      : prefix_sum(ts.Alias().prefix_sum)
      , histogram(ts.Alias().histogram)
      , extract_bin_op(extract_bin_op)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE
  process_histogram_only(prefix_sum_temp_t& prefix_sum, OffsetT* histogram, ExtractBinOpT extract_bin_op)
      : prefix_sum(prefix_sum)
      , histogram(histogram)
      , extract_bin_op(extract_bin_op)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void segment_prologue()
  {
    init_histogram<block_threads, num_buckets>(histogram);
    __syncthreads();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile(
    const KeyInT (&items)[items_per_thread], OffsetT /*thread_offset*/, int num_thread_items)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      if (j < num_thread_items)
      {
        const int bucket = extract_bin_op(items[j]);
        atomicAdd(histogram + bucket, OffsetT{1});
      }
    }
  }

  template <typename CounterUpdateFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void segment_epilogue(
    Counter<KeyInT, OffsetT, OutOffsetT>* counter,
    OffsetT* global_histogram,
    OutOffsetT current_k,
    int pass,
    bool is_last_pass,
    CounterUpdateFn counter_update_fn)
  {
    __syncthreads();

    merge_histogram<block_threads, num_buckets>(histogram, global_histogram);

    finalize_pass(
      prefix_sum,
      histogram,
      counter,
      global_histogram,
      current_k,
      pass,
      is_last_pass,
      counter_update_fn);
  }
};

//---------------------------------------------------------------------
// input_channel: wraps multiple iterators of the same value type with
// a runtime selector. Allows BlockTileLoader::consume to accept a
// single argument instead of branching at the call site.
//---------------------------------------------------------------------

template <typename... IteratorTs>
struct input_channel
{
  ::cuda::std::tuple<IteratorTs...> iterators;
  int active;
};

template <typename... IteratorTs>
_CCCL_DEVICE _CCCL_FORCEINLINE auto make_input_channel(int active, IteratorTs... its)
  -> input_channel<IteratorTs...>
{
  return {::cuda::std::tuple<IteratorTs...>{its...}, active};
}

//---------------------------------------------------------------------
// TileQueueT strategies for BlockTileLoader.
//
// grid_stride_queue reproduces the hardcoded grid-stride loop of the legacy loader.
// It is the default and the only strategy shipped today; atomic-counter /
// UGETNEXTWORKID / segmented variants (see plan Q10) plug into the same interface
// without touching the loader.
//
// Required interface:
//   OffsetT next_tile_id();                 // returns sentinel (OffsetT max) when done
//   bool    crosses_segment_boundary();     // true iff the next tile changes segments
//   Any     current_segment();              // identifier of the current segment
//---------------------------------------------------------------------

template <typename OffsetT>
struct grid_stride_queue
{
  static constexpr OffsetT sentinel = ::cuda::std::numeric_limits<OffsetT>::max();

  OffsetT next_block_;
  OffsetT grid_size_;
  OffsetT total_blocks_;

  _CCCL_DEVICE _CCCL_FORCEINLINE grid_stride_queue(OffsetT total_blocks)
      : next_block_(static_cast<OffsetT>(blockIdx.x))
      , grid_size_(static_cast<OffsetT>(gridDim.x))
      , total_blocks_(total_blocks)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT next_tile_id()
  {
    const OffsetT ret = next_block_;
    if (ret >= total_blocks_)
    {
      return sentinel;
    }
    next_block_ += grid_size_;
    return ret;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE bool crosses_segment_boundary() const
  {
    return false;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE int current_segment() const
  {
    return 0;
  }
};

//---------------------------------------------------------------------
// LoadStrategyT tags for BlockTileLoader.
//
// sync_load_strategy is the only strategy shipped today: one synchronous BlockLoad per
// tile, items arrive in the processor's registers via process_tile. The abstraction
// point exists so an async_load_strategy (double-buffered BlockLoadToShared for TMA
// overlap; see plan Q9) can drop in later without changing the loader interface.
//---------------------------------------------------------------------

struct sync_load_strategy
{};

// Detects whether the processor has an on_segment_change(int) hook.
template <typename P, typename = void>
struct processor_has_on_segment_change : ::cuda::std::false_type
{};
template <typename P>
struct processor_has_on_segment_change<
  P,
  ::cuda::std::void_t<decltype(::cuda::std::declval<P&>().on_segment_change(0))>>
    : ::cuda::std::true_type
{};

template <typename P, typename SegmentIdT>
_CCCL_DEVICE _CCCL_FORCEINLINE void maybe_on_segment_change(P& proc, SegmentIdT segment_id)
{
  if constexpr (processor_has_on_segment_change<P>::value)
  {
    proc.on_segment_change(static_cast<int>(segment_id));
  }
  else
  {
    (void) proc;
    (void) segment_id;
  }
}

//---------------------------------------------------------------------
// BlockTileLoader: standalone building block that iterates over a range in tiles
// driven by a TileQueueT, loads each tile via the chosen LoadStrategyT, and
// delegates per-tile processing to a caller-supplied processor.
//
// ProcessorT must provide:
//   void process_tile(const KeyT (&items)[items_per_thread],
//                     OffsetT thread_offset, int num_thread_items);
//
// ProcessorT MAY provide:
//   void on_segment_change(int segment_id);
// (called by segment-aware queues when crosses_segment_boundary() returns true)
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyT,
          typename OffsetT,
          typename TileQueueT    = grid_stride_queue<OffsetT>,
          typename LoadStrategyT = sync_load_strategy>
struct BlockTileLoader
{
  static_assert(::cuda::std::is_same_v<LoadStrategyT, sync_load_strategy>,
                "Only sync_load_strategy is implemented today; async is a deferred plan Q9 follow-up.");

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int tile_items       = block_threads * items_per_thread;

  using block_load_t = BlockLoad<KeyT, block_threads, items_per_thread, AgentTopKPolicyT::load_algorithm>;

  struct _TempStorage
  {
    typename block_load_t::TempStorage load;
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& temp_storage;
  TileQueueT queue_;

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockTileLoader(_TempStorage& temp_storage, TileQueueT queue)
      : temp_storage(temp_storage)
      , queue_(queue)
  {}

  // Convenience constructor for the default grid_stride_queue: derive total_blocks from num_items.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockTileLoader(_TempStorage& temp_storage, OffsetT num_items)
      : temp_storage(temp_storage)
      , queue_(::cuda::ceil_div(num_items, static_cast<OffsetT>(tile_items)))
  {}

  // Plain iterator overload
  template <typename ProcessorT, typename InputItT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume(ProcessorT& proc, InputItT in, OffsetT num_items)
  {
    consume_impl(proc, in, num_items);
  }

  // input_channel overload -- same tiling loop, branch at the Load site
  template <typename ProcessorT, typename... IteratorTs>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume(
    ProcessorT& proc, input_channel<IteratorTs...> channel, OffsetT num_items)
  {
    consume_impl(proc, channel, num_items);
  }

private:
  template <typename InputItT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(InputItT in, OffsetT tile_base, KeyT (&thread_data)[items_per_thread])
  {
    block_load_t(temp_storage.load).Load(in + tile_base, thread_data);
  }

  template <typename InputItT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(
    InputItT in, OffsetT tile_base, KeyT (&thread_data)[items_per_thread], OffsetT valid_items)
  {
    block_load_t(temp_storage.load).Load(in + tile_base, thread_data, valid_items);
  }

  template <typename... IteratorTs>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(
    input_channel<IteratorTs...>& channel, OffsetT tile_base, KeyT (&thread_data)[items_per_thread])
  {
    channel_dispatch(channel, [&](auto& it) {
      block_load_t(temp_storage.load).Load(it + tile_base, thread_data);
    });
  }

  template <typename... IteratorTs>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(
    input_channel<IteratorTs...>& channel,
    OffsetT tile_base,
    KeyT (&thread_data)[items_per_thread],
    OffsetT valid_items)
  {
    channel_dispatch(channel, [&](auto& it) {
      block_load_t(temp_storage.load).Load(it + tile_base, thread_data, valid_items);
    });
  }

  template <typename... IteratorTs, typename Fn>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void channel_dispatch(input_channel<IteratorTs...>& channel, Fn fn)
  {
    channel_dispatch_impl(channel, fn, ::cuda::std::index_sequence_for<IteratorTs...>{});
  }

  template <typename... IteratorTs, typename Fn, ::cuda::std::size_t... Is>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void channel_dispatch_impl(
    input_channel<IteratorTs...>& channel, Fn fn, ::cuda::std::index_sequence<Is...>)
  {
    ((channel.active == static_cast<int>(Is)
        ? (fn(::cuda::std::get<Is>(channel.iterators)), false)
        : true) && ...);
  }

  // Single tiling loop driven by the pluggable TileQueueT. Calls the optional
  // on_segment_change hook when the queue reports a boundary crossing.
  template <typename ProcessorT, typename InputSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_impl(ProcessorT& proc, InputSourceT& in, OffsetT num_items)
  {
    KeyT thread_data[items_per_thread];
    constexpr OffsetT sentinel = ::cuda::std::numeric_limits<OffsetT>::max();

    while (true)
    {
      const OffsetT block_id = queue_.next_tile_id();
      if (block_id == sentinel)
      {
        break;
      }

      if (queue_.crosses_segment_boundary())
      {
        maybe_on_segment_change(proc, queue_.current_segment());
      }

      const OffsetT tile_base  = block_id * static_cast<OffsetT>(tile_items);
      const OffsetT remaining  = num_items - tile_base;
      const bool is_full_tile  = remaining >= static_cast<OffsetT>(tile_items);

      __syncthreads();

      if (is_full_tile)
      {
        load_tile(in, tile_base, thread_data);
      }
      else
      {
        load_tile(in, tile_base, thread_data, remaining);
      }

      const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
      const int num_thread_items =
        is_full_tile
          ? items_per_thread
          : static_cast<int>(
              (thread_offset >= num_items)
                ? OffsetT{0}
                : ::cuda::std::min(static_cast<OffsetT>(items_per_thread), num_items - thread_offset));

      proc.process_tile(thread_data, thread_offset, num_thread_items);
    }
  }
};

//---------------------------------------------------------------------
// AgentTopKHistogram: dedicated agent for the histogram-only pass
// (pass 0). Only needs the input iterator, extract_bin_op, and
// histogram infrastructure — no output iterators, no sinks.
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT>
struct AgentTopKHistogram
{
  using key_in_t = it_value_t<KeyInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;

  using tile_loader_t   = BlockTileLoader<AgentTopKPolicyT, key_in_t, OffsetT>;
  using prefix_sum_temp_t =
    BinPrefixSumTempStorage<block_threads, bits_per_pass, AgentTopKPolicyT::SCAN_ALGORITHM, OffsetT>;

  struct _TempStorage
  {
    union
    {
      typename tile_loader_t::_TempStorage loader;
      prefix_sum_temp_t prefix_sum;
    };
    OffsetT histogram[num_buckets];
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& temp_storage;
  KeyInputIteratorT d_keys_in;
  OffsetT num_items;
  OutOffsetT k;
  ExtractBinOpT extract_bin_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentTopKHistogram(
    TempStorage& temp_storage,
    const KeyInputIteratorT d_keys_in,
    OffsetT num_items,
    OutOffsetT k,
    ExtractBinOpT extract_bin_op)
      : temp_storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , num_items(num_items)
      , k(k)
      , extract_bin_op(extract_bin_op)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke(
    Counter<key_in_t, OffsetT, OutOffsetT>* counter, OffsetT* global_histogram, int pass, bool is_last_pass)
  {
    process_histogram_only<AgentTopKPolicyT, key_in_t, ExtractBinOpT, OffsetT, OutOffsetT> proc(
      temp_storage.prefix_sum, temp_storage.histogram, extract_bin_op);

    tile_loader_t loader(temp_storage.loader, num_items);

    proc.segment_prologue();
    loader.consume(proc, d_keys_in, num_items);
    proc.segment_epilogue(counter, global_histogram, k, pass, is_last_pass, [counter, this] {
      counter->previous_len = num_items;
      counter->filter_cnt   = 0;
    });
  }
};

//---------------------------------------------------------------------
// sink_mode: compile-time selector for agent_topk_filter_partition.
//
// Each mode corresponds to one behavior of the legacy sink_* family and drives a
// single-kernel runtime switch in DeviceTopKFilterKernel. `unbuffered` is the "scout"
// mode that only builds the histogram; `early_stop` is the final collapsed pass; the
// other two write output.
//---------------------------------------------------------------------

enum class sink_mode
{
  early_stop, // selected + candidate -> d_keys_out front; no histogram
  buffered, // selected -> d_keys_out; candidate -> out_buf; histogram over candidates
  unbuffered, // no writes; histogram over candidates only
  last_filter // selected -> d_keys_out front; candidate -> back of d_keys_out with cap
};

// Compile-time flag bundle derived from sink_mode. Keeps the agent's `if constexpr`
// branches readable.
template <sink_mode Mode>
struct sink_flags
{
  static constexpr bool writes_output          = (Mode != sink_mode::unbuffered);
  static constexpr bool accumulate_histogram   = (Mode == sink_mode::buffered) || (Mode == sink_mode::unbuffered);
  static constexpr bool needs_finalize         = (Mode != sink_mode::last_filter);
  static constexpr bool writes_to_cand_buffer  = (Mode == sink_mode::buffered);
  static constexpr bool writes_back_of_output  = (Mode == sink_mode::last_filter);
  static constexpr bool writes_selected_front  = (Mode != sink_mode::unbuffered);
};

//---------------------------------------------------------------------
// agent_topk_filter_partition: unified agent for passes 1..num_passes-1 plus the
// last-filter pass. Replaces the legacy sink-based AgentTopKRefactored.
//
// Compile-time specialized on sink_mode; the kernel switches to the matching
// instantiation based on Counter state. All classify / histogram work is inlined
// into process_tile, scatter is delegated to BlockPartition (parameterized by
// PartStrat) so Atomics / Staged / SharedMem flow the same way.
//---------------------------------------------------------------------

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
          BlockPartitionStrategy PartStrat = BlockPartitionStrategy::Atomics>
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

  using flags             = sink_flags<Mode>;
  using tile_loader_t     = BlockTileLoader<AgentTopKPolicyT, key_in_t, OffsetT>;
  using value_loader_t    = BlockLoad<value_in_t, block_threads, items_per_thread, AgentTopKPolicyT::load_algorithm>;
  using prefix_sum_temp_t =
    BinPrefixSumTempStorage<block_threads, bits_per_pass, AgentTopKPolicyT::SCAN_ALGORITHM, OffsetT>;

  // Compile-time mode/strategy plumbing for the per-tile partition primitive.
  // - selected_offset_t  : OutOffsetT for all modes (out_cnt / out_back_cnt are OutOffsetT*).
  // - candidate_offset_t : OffsetT in `buffered` (filter_cnt is OffsetT*); OutOffsetT otherwise.
  // - has_candidates     : true for modes that actually scatter a separate candidate stream.
  // - has_cap            : true only for `last_filter` (back-of-output candidate cap).
  // - effective_strat    : `unbuffered` skips scatter entirely; force Atomics so the buffer_t
  //                        in the temp-storage union stays empty.
  static constexpr bool has_candidates = (Mode == sink_mode::buffered) || (Mode == sink_mode::last_filter);
  static constexpr bool has_cap        = (Mode == sink_mode::last_filter);
  using selected_offset_t              = OutOffsetT;
  using candidate_offset_t             = ::cuda::std::conditional_t<Mode == sink_mode::buffered, OffsetT, OutOffsetT>;
  static constexpr BlockPartitionStrategy effective_strat =
    (Mode == sink_mode::unbuffered) ? BlockPartitionStrategy::Atomics : PartStrat;
  // For `last_filter`, the candidate stream is written to the back of `d_keys_out` (and
  // `d_values_out`); BlockPartition handles that via `WritesCandidatesToBack` so the
  // candidate iterator stays the original raw pointer / iterator. This preserves the
  // opportunity for vectorized cooperative stores (Q13) in the Staged/SharedMem flush.
  static constexpr bool writes_candidates_to_back = (Mode == sink_mode::last_filter);
  using partition_t                               = BlockPartition<key_in_t,
                                     value_in_t,
                                     selected_offset_t,
                                     candidate_offset_t,
                                     block_threads,
                                     items_per_thread,
                                     effective_strat,
                                     has_candidates,
                                     has_cap,
                                     writes_candidates_to_back>;
  using partition_buffer_t = typename partition_t::buffer_t;

  // Smem layout: tile loader, value loader, partition buffer, and prefix-sum scratch are
  // mutually exclusive in time (one __syncthreads between phases) and share a single
  // union. Histogram is independent and lives outside the union.
  struct _TempStorage
  {
    union
    {
      typename tile_loader_t::_TempStorage loader;
      typename value_loader_t::TempStorage value_loader;
      partition_buffer_t partition_buf;
      prefix_sum_temp_t prefix_sum;
    };
    OffsetT histogram[num_buckets];
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& storage;

  // Input/output (const pointers + iterators, resolved by caller based on Mode)
  KeyInputIteratorT d_keys_in;
  KeyOutputIteratorT d_keys_out;
  ValueInputIteratorT d_values_in;
  ValueOutputIteratorT d_values_out;
  key_in_t* in_key_buf;
  value_in_t* in_val_buf;
  key_in_t* out_key_buf; // only used in buffered mode
  value_in_t* out_val_buf; // only used in buffered mode

  OutOffsetT* p_out_cnt;
  OutOffsetT* p_out_back_cnt; // only used in last_filter
  OffsetT* p_filter_cnt; // only used in buffered / unbuffered

  OutOffsetT k_total; // only used in last_filter; passed as BlockPartition's `back_anchor` so candidates land in [k_total - cap, k_total)
  OutOffsetT num_of_kth_needed; // only used in last_filter (candidate cap)

  OffsetT num_items;
  OffsetT input_length;
  bool load_from_original_input;

  ExtractBinOpT extract_bin_op;
  IdentifyCandidatesOpT identify_candidates_op;
  OffsetT* global_histogram; // used by modes that accumulate_histogram

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_topk_filter_partition(
    TempStorage& temp_storage,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    key_in_t* in_key_buf,
    value_in_t* in_val_buf,
    key_in_t* out_key_buf,
    value_in_t* out_val_buf,
    OutOffsetT* p_out_cnt,
    OutOffsetT* p_out_back_cnt,
    OffsetT* p_filter_cnt,
    OutOffsetT k_total,
    OutOffsetT num_of_kth_needed,
    OffsetT num_items,
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
      , out_key_buf(out_key_buf)
      , out_val_buf(out_val_buf)
      , p_out_cnt(p_out_cnt)
      , p_out_back_cnt(p_out_back_cnt)
      , p_filter_cnt(p_filter_cnt)
      , k_total(k_total)
      , num_of_kth_needed(num_of_kth_needed)
      , num_items(num_items)
      , input_length(input_length)
      , load_from_original_input(load_from_original_input)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
      , global_histogram(global_histogram)
  {}

private:
  // Sequentially load the tile's values into registers from the active value channel
  // (d_values_in for pass 1 / load_from_original_input, in_val_buf otherwise).
  // Reuses the loader smem region (next phase will __syncthreads before partition).
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_values_for_tile(
    OffsetT tile_base, OffsetT remaining, bool is_full_tile, value_in_t (&values)[items_per_thread])
  {
    if (load_from_original_input)
    {
      if (is_full_tile)
      {
        value_loader_t(storage.value_loader).Load(d_values_in + tile_base, values);
      }
      else
      {
        value_loader_t(storage.value_loader).Load(d_values_in + tile_base, values, remaining);
      }
    }
    else
    {
      if (is_full_tile)
      {
        value_loader_t(storage.value_loader).Load(in_val_buf + tile_base, values);
      }
      else
      {
        value_loader_t(storage.value_loader).Load(in_val_buf + tile_base, values, remaining);
      }
    }
  }

  // Mode-specific scatter via BlockPartition. Mode determines:
  //   - which output iterator pair (selected / candidate) and
  //   - which counter pair to atomically advance.
  _CCCL_DEVICE _CCCL_FORCEINLINE void do_partition(
    const key_in_t (&keys)[items_per_thread],
    const value_in_t (&values)[items_per_thread],
    const candidate_class (&classes)[items_per_thread])
  {
    partition_t partition;
    if constexpr (Mode == sink_mode::early_stop)
    {
      // HasCandidates=false: selected and candidate fold into d_keys_out / d_values_out
      // through p_out_cnt; candidate counter is unused.
      if constexpr (effective_strat == BlockPartitionStrategy::Staged)
      {
        partition.PartitionKeys(
          storage.partition_buf,
          keys,
          classes,
          d_keys_out,
          d_keys_out,
          p_out_cnt,
          static_cast<OutOffsetT*>(nullptr));
        if constexpr (!keys_only)
        {
          partition.ScatterValues(storage.partition_buf, values, d_values_out, d_values_out);
        }
      }
      else
      {
        partition.PartitionPairs(
          storage.partition_buf,
          keys,
          values,
          classes,
          d_keys_out,
          d_keys_out,
          d_values_out,
          d_values_out,
          p_out_cnt,
          static_cast<OutOffsetT*>(nullptr));
      }
    }
    else if constexpr (Mode == sink_mode::buffered)
    {
      // selected -> d_keys_out / d_values_out via p_out_cnt;
      // candidate -> out_key_buf / out_val_buf via p_filter_cnt; no cap.
      if constexpr (effective_strat == BlockPartitionStrategy::Staged)
      {
        partition.PartitionKeys(
          storage.partition_buf, keys, classes, d_keys_out, out_key_buf, p_out_cnt, p_filter_cnt);
        if constexpr (!keys_only)
        {
          partition.ScatterValues(storage.partition_buf, values, d_values_out, out_val_buf);
        }
      }
      else
      {
        partition.PartitionPairs(
          storage.partition_buf,
          keys,
          values,
          classes,
          d_keys_out,
          out_key_buf,
          d_values_out,
          out_val_buf,
          p_out_cnt,
          p_filter_cnt);
      }
    }
    else // sink_mode::last_filter
    {
      // selected -> d_keys_out front via p_out_cnt;
      // candidate -> back of d_keys_out (and d_values_out) via BlockPartition's
      // `WritesCandidatesToBack` mode, capped at num_of_kth_needed. The candidate
      // iterators are the original `d_keys_out`/`d_values_out` so raw-pointer cases
      // remain raw pointers all the way down to the BlockPartition flush loops.
      const auto back_anchor = static_cast<candidate_offset_t>(k_total);
      if constexpr (effective_strat == BlockPartitionStrategy::Staged)
      {
        partition.PartitionKeys(
          storage.partition_buf,
          keys,
          classes,
          d_keys_out,
          d_keys_out,
          p_out_cnt,
          p_out_back_cnt,
          num_of_kth_needed,
          back_anchor);
        if constexpr (!keys_only)
        {
          partition.ScatterValues(storage.partition_buf, values, d_values_out, d_values_out);
        }
      }
      else
      {
        partition.PartitionPairs(
          storage.partition_buf,
          keys,
          values,
          classes,
          d_keys_out,
          d_keys_out,
          d_values_out,
          d_values_out,
          p_out_cnt,
          p_out_back_cnt,
          num_of_kth_needed,
          back_anchor);
      }
    }
  }

public:
  // --- processor hook: one call per tile -----------------------------
  // Steps (some elided per Mode via if constexpr):
  //   1. Classify items -> classes[]; out-of-bounds entries forced to `rejected`.
  //   2. Accumulate per-bucket histogram for `candidate` items.
  //   3. (Only if writes_output) BlockLoad values from the active value channel.
  //   4. (Only if writes_output) BlockPartition::PartitionPairs / ScatterValues.
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile(
    const key_in_t (&items)[items_per_thread], OffsetT thread_offset, int num_thread_items)
  {
    candidate_class classes[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      classes[j] =
        (j < num_thread_items) ? identify_candidates_op(items[j]) : candidate_class::rejected;
    }

    if constexpr (flags::accumulate_histogram)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < items_per_thread; ++j)
      {
        if (classes[j] == candidate_class::candidate)
        {
          const int bucket = extract_bin_op(items[j]);
          atomicAdd(storage.histogram + bucket, OffsetT{1});
        }
      }
    }

    if constexpr (!flags::writes_output)
    {
      // unbuffered: histogram only, nothing to scatter.
      (void) thread_offset;
      return;
    }
    else
    {
      [[maybe_unused]] value_in_t values[items_per_thread];
      if constexpr (!keys_only)
      {
        const OffsetT tile_base = thread_offset - static_cast<OffsetT>(threadIdx.x) * items_per_thread;
        const OffsetT remaining = input_length - tile_base;
        const bool is_full_tile = remaining >= static_cast<OffsetT>(tile_items);

        // Wait for all threads to be done with the keys load (loader smem) before
        // overwriting it via the value loader.
        __syncthreads();
        load_values_for_tile(tile_base, remaining, is_full_tile, values);
      }

      // Wait for all threads to finish the value load before reusing the smem region
      // for the partition buffer.
      __syncthreads();
      do_partition(items, values, classes);
    }
  }

  // --- entry point ----------------------------------------------------
  template <typename CounterUpdateFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OutOffsetT current_k,
    int pass,
    bool is_last_pass,
    CounterUpdateFn counter_update_fn)
  {
    if constexpr (flags::accumulate_histogram)
    {
      init_histogram<block_threads, num_buckets>(storage.histogram);
      __syncthreads();
    }

    tile_loader_t loader(storage.loader, input_length);
    const int channel_active = load_from_original_input ? 0 : 1;
    loader.consume(*this, make_input_channel(channel_active, d_keys_in, in_key_buf), input_length);

    if constexpr (flags::accumulate_histogram)
    {
      __syncthreads();
      merge_histogram<block_threads, num_buckets>(storage.histogram, global_histogram);
    }

    if constexpr (flags::needs_finalize)
    {
      finalize_pass(storage.prefix_sum,
                    storage.histogram,
                    counter,
                    global_histogram,
                    current_k,
                    pass,
                    is_last_pass,
                    counter_update_fn);
    }
    else
    {
      (void) counter;
      (void) current_k;
      (void) pass;
      (void) is_last_pass;
      (void) counter_update_fn;
    }
  }
};

} // namespace detail::topk
CUB_NAMESPACE_END
