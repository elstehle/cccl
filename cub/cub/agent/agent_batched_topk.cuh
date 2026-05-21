// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_topk_common.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_topk.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/detail/topk/block_filter.cuh>
#include <cub/detail/topk/block_filter_accumulating.cuh>
#include <cub/detail/topk/block_filter_speculative.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/block_partition_accumulating.cuh>
#include <cub/detail/topk/block_partition_speculative.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_topk_common.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// Atomic counters used *only* by the mixed-path small-segment kernel to (a) enqueue large segments into the
// large-segment work queue and (b) elect the last block to run the epilogue scan over the queued tile counts.
// `alignas(128)` isolates each counter on its own cache line for performance.
//
// The total number of large tiles lives in the trailing slot of `d_large_segments_tile_offsets` (i.e. `d_large_segments_tile_offsets[large_segments_count]`)
// The multi-CTA kernels read it as `d_large_segments_tile_offsets[*large_segments_count_it]`.
template <class NumSegmentsT>
struct batched_topk_counters
{
  // Force unsigned integer type for segment count.
  using segment_count_t = detail::choose_offset_t<NumSegmentsT>;
  // Number of segments enqueued in the large-segment work queue. Atomically incremented (by 1) by the first thread
  // of each block that decides its segment is large.
  alignas(128) segment_count_t large_segments_count;

  // Block retirement counter. Each block atomically increments by 1 when it has finished processing its segment, and
  // the block that observes `gridDim.x - 1` runs the epilogue on the queued large segments tile counts.
  // Assumption: Future support for more than 2^31 - 1 segments will use multiple launches of a slightly modified
  // small-segment kernel instead of additional grid dimensions. Therefore each grid will handle a maximum of 2^31 - 1
  // segments per launch. The counter would not even have to be reset to 0 after each launch if we cleverly make use of
  // its modulo arithmetic.
  alignas(128) unsigned retirement_count;
};

template <typename PolicyGetter, // TODO(bgruber): pass worker_policy as NTTP in C++20
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename LargeSegmentTileOffsetT>
struct agent_batched_topk_worker_per_segment
{
  // -------------------------------------------------------------------------
  // Types and Constants
  // -------------------------------------------------------------------------
  // Derive inner types from Iterator of Iterators
  using key_it_t   = it_value_t<KeyInputItItT>;
  using value_it_t = it_value_t<ValueInputItItT>;

  using key_t   = it_value_t<key_it_t>;
  using value_t = it_value_t<value_it_t>;

  using segment_size_val_t = typename SegmentSizeParameterT::value_type;
  using num_segments_val_t = typename NumSegmentsParameterT::value_type;
  using counters_t         = batched_topk_counters<num_segments_val_t>;

  static constexpr auto policy                 = PolicyGetter{}();
  static constexpr worker_policy active_policy = policy.worker_per_segment_policy;

  // For block-topk (and keys/values load/store):
  static constexpr int threads_per_block = active_policy.threads_per_block;
  static constexpr int items_per_thread  = active_policy.items_per_thread;
  static constexpr int tile_size         = threads_per_block * items_per_thread;

  // For block-scan (and offsets load/store):
  static constexpr int epilogue_items_per_thread = active_policy.epilogue.items_per_thread;
  static constexpr int epilogue_tile_size        = threads_per_block * epilogue_items_per_thread;

  // Number used for preprocessing segment-size data, not for tuning => should not affect performance of this agent.
  static constexpr multi_worker_policy multi_worker_per_segment_policy = policy.multi_worker_per_segment_policy;
  static constexpr int multi_worker_per_segment_tile_size =
    multi_worker_per_segment_policy.threads_per_block * multi_worker_per_segment_policy.items_per_thread;

  // Check if there could be large segments present
  static constexpr bool only_small_segments = params::static_max_value_v<SegmentSizeParameterT> <= tile_size;

  // Check if we are dealing with keys-only or key-value pairs
  static constexpr bool is_keys_only = ::cuda::std::is_same_v<value_t, cub::NullType>;

  // -------------------------------------------------------------------------
  // Primitive Types
  // -------------------------------------------------------------------------
  using block_load_keys_t = BlockLoad<key_t, threads_per_block, items_per_thread, active_policy.load_algorithm>;
  using block_load_vals_t = BlockLoad<value_t, threads_per_block, items_per_thread, active_policy.load_algorithm>;

  using block_topk_t = block_topk<key_t, threads_per_block, items_per_thread, value_t>;

  // TODO (elstehle): Specialize for the case that we statically know k and we can skip passing num_valid_items to
  // Store()
  using block_store_keys_t = BlockStore<key_t, threads_per_block, items_per_thread, active_policy.store_algorithm>;
  using block_store_vals_t = BlockStore<value_t, threads_per_block, items_per_thread, active_policy.store_algorithm>;

  using block_load_epilogue_t =
    BlockLoad<segment_size_val_t, threads_per_block, epilogue_items_per_thread, active_policy.epilogue.load_algorithm>;
  // Type must match the data being scanned (`segment_tile_offsets` -- a tile-count array indexed by
  // large-segment slot). With `segment_size_val_t = int64_t` (the default), scanning into a 32-bit
  // `int` accumulator is a no-overflow but type-mismatched call to `BlockScan::ExclusiveSum` --
  // the function takes `T(&)[]` references and the template `T` is fixed by the `BlockScan`
  // instantiation. Use the same type as the loaded data so the call signature matches and the
  // accumulator can hold the running total across all tiles (a large-segment-rich workload can
  // accumulate well beyond 2^31 tiles).
  using block_scan_epilogue_t =
    BlockScan<segment_size_val_t, threads_per_block, active_policy.epilogue.scan_algorithm>;
  using block_store_epilogue_t =
    BlockStore<segment_size_val_t, threads_per_block, epilogue_items_per_thread, active_policy.epilogue.store_algorithm>;

  // -------------------------------------------------------------------------
  // Shared Memory Storage
  // -------------------------------------------------------------------------
  struct TempStorage_
  {
    union
    {
      typename block_load_keys_t::TempStorage load_keys;
      typename block_load_vals_t::TempStorage load_vals;
      typename block_topk_t::TempStorage topk;
      typename block_store_keys_t::TempStorage store_keys;
      typename block_store_vals_t::TempStorage store_vals;
      typename block_load_epilogue_t::TempStorage load_epilogue;
      typename block_scan_epilogue_t::TempStorage scan_epilogue;
      typename block_store_epilogue_t::TempStorage store_epilogue;
    };
  };

  using TempStorage = Uninitialized<TempStorage_>;

  // -------------------------------------------------------------------------
  // Members
  // -------------------------------------------------------------------------
  TempStorage_& temp_storage;
  KeyInputItItT d_key_segments_it;
  KeyOutputItItT d_key_segments_out_it;
  ValueInputItItT d_value_segments_it;
  ValueOutputItItT d_value_segments_out_it;
  SegmentSizeParameterT segment_sizes;
  KParameterT k_param;
  SelectDirectionParameterT select_directions;
  NumSegmentsParameterT num_segments;
  counters_t* d_counters;
  num_segments_val_t* d_large_segments_ids;
  LargeSegmentTileOffsetT* d_large_segments_tile_offsets;
  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------
  _CCCL_DEVICE_API _CCCL_FORCEINLINE agent_batched_topk_worker_per_segment(
    TempStorage& temp_storage,
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments,
    counters_t* d_counters,
    num_segments_val_t* d_large_segments_ids,
    LargeSegmentTileOffsetT* d_large_segments_tile_offsets)
      : temp_storage(temp_storage.Alias())
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , k_param(k_param)
      , select_directions(select_directions)
      , num_segments(num_segments)
      , d_counters(d_counters)
      , d_large_segments_ids(d_large_segments_ids)
      , d_large_segments_tile_offsets(d_large_segments_tile_offsets)
  {}

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Process()
  {
    // Identify Segment
    const int segment_id = static_cast<int>(blockIdx.x);

    // Boundary check
    // TODO (elstehle): consider skipping boundary check if we can safely assume the right grid dimensions
    if (segment_id >= num_segments.get_param(0))
    {
      return;
    }

    constexpr bool is_full_tile = params::has_single_static_value_v<SegmentSizeParameterT>
                               && params::static_min_value_v<SegmentSizeParameterT> == tile_size;

    // Resolve Segment Parameters
    const auto segment_size = segment_sizes.get_param(segment_id);
    if (!only_small_segments && segment_size > tile_size)
    {
      // Enqueue large segment
      if (threadIdx.x == 0u)
      {
        // Add to large segment queue
        const auto large_segment_queue_idx            = atomicAdd(&d_counters->large_segments_count, 1ull);
        d_large_segments_ids[large_segment_queue_idx] = static_cast<num_segments_val_t>(segment_id);
        d_large_segments_tile_offsets[large_segment_queue_idx] =
          static_cast<LargeSegmentTileOffsetT>(::cuda::ceil_div(segment_size, multi_worker_per_segment_tile_size));
      }
    }
    else
    {
      // Process small segment
      const auto k         = (::cuda::std::min) (k_param.get_param(segment_id),
                                         static_cast<decltype(k_param.get_param(segment_id))>(segment_size));
      const auto direction = select_directions.get_param(segment_id);

      // Determine padding key based on direction
      const key_t padding_key =
        (direction == detail::topk::select::max)
          ? ::cuda::std::numeric_limits<key_t>::lowest()
          : ::cuda::std::numeric_limits<key_t>::max();

      // Dereference iterator-of-iterators to get the segment specific iterator
      auto block_keys_in = d_key_segments_it[segment_id];

      // Load Keys
      key_t thread_keys[items_per_thread];
      if constexpr (is_full_tile)
      {
        // No padding needed
        block_load_keys_t(temp_storage.load_keys).Load(block_keys_in, thread_keys);
      }
      else
      {
        // Potentially partial final load with padding
        // TODO (elstehle): explore whether a runtime check for segment_size == tile_size improves performance
        block_load_keys_t(temp_storage.load_keys).Load(block_keys_in, thread_keys, segment_size);
      }

      // Load Values (if applicable)
      [[maybe_unused]] value_t thread_values[items_per_thread];

      if constexpr (!is_keys_only)
      {
        __syncthreads();
        auto block_vals_in = d_value_segments_it[segment_id];

        if constexpr (is_full_tile)
        {
          // No padding needed
          block_load_vals_t(temp_storage.load_vals).Load(block_vals_in, thread_values);
        }
        else
        {
          // Potentially partial final load with padding
          // TODO (elstehle): explore whether a runtime check for segment_size == tile_size improves performance
          block_load_vals_t(temp_storage.load_vals).Load(block_vals_in, thread_values, segment_size);
        }
      }

      __syncthreads();

      // Perform Block Top-K
      if constexpr (is_keys_only)
      {
        const bool is_successful_dispatch = cub::detail::params::dispatch_discrete(
          select_directions, segment_id, [this, &thread_keys, k, segment_size](auto direction_tag) {
            if constexpr (decltype(direction_tag)::value == detail::topk::select::max)
            {
              block_topk_t(temp_storage.topk).template max_keys<is_full_tile>(thread_keys, k, segment_size);
            }
            else
            {
              block_topk_t(temp_storage.topk).template min_keys<is_full_tile>(thread_keys, k, segment_size);
            }
          });
        _CCCL_ASSERT(is_successful_dispatch, "Error: Unsupported select direction");
      }
      else
      {
        // Pass both keys and values
        const bool is_successful_dispatch = cub::detail::params::dispatch_discrete(
          select_directions, segment_id, [this, &thread_keys, &thread_values, k, segment_size](auto direction_tag) {
            if constexpr (decltype(direction_tag)::value == detail::topk::select::max)
            {
              block_topk_t(temp_storage.topk)
                .template max_pairs<is_full_tile>(thread_keys, thread_values, k, segment_size);
            }
            else
            {
              block_topk_t(temp_storage.topk)
                .template min_pairs<is_full_tile>(thread_keys, thread_values, k, segment_size);
            }
          });
        _CCCL_ASSERT(is_successful_dispatch, "Error: Unsupported select direction");
      }

      __syncthreads();

      auto block_keys_out = d_key_segments_out_it[segment_id];

      block_store_keys_t(temp_storage.store_keys)
        .Store(block_keys_out,
               thread_keys,
               k // Only store K items
        );

      if constexpr (!is_keys_only)
      {
        __syncthreads();
        auto block_vals_out = d_value_segments_out_it[segment_id];

        block_store_vals_t(temp_storage.store_vals).Store(block_vals_out, thread_values, k);
      }
    }

    // Epilogue: Scan queued large segment sizes (in tiles not elements) for load balancing search
    // in the multi-CTA-per-segment agents. The scan additionally publishes `total_large_tiles`
    // into the trailing slot, i.e., `d_large_segments_tile_offsets[num_large_segments]`.
    if constexpr (!only_small_segments)
    {
      // Determine last block trying to retire.
      bool is_last_block = false;
      if (threadIdx.x == 0u)
      {
        __threadfence();
        const auto retirement_count = atomicAdd(&d_counters->retirement_count, 1u);
        is_last_block               = retirement_count == (gridDim.x - 1u);
      }
      // This sync also makes sure that the shared memory can be reused.
      is_last_block = static_cast<bool>(__syncthreads_or(static_cast<int>(is_last_block)));
      if (!is_last_block)
      {
        return;
      }
      const auto num_large_segments = d_counters->large_segments_count;
      // For tracking the running total across tiles (loop iterations).
      // Caution: The functor is only invoked by the first warp in the block, and the value returned by lane 0 in that
      // warp is used as the initial value.
      auto prefix_callback_op =
        [running_total = segment_size_val_t{0}](segment_size_val_t block_aggregate) mutable {
          auto old_running_total = running_total;
          running_total += block_aggregate;
          return old_running_total;
        };
      // Loop one item past `num_large_segments` to also produce a total aggregate. The trailing iteration's `BlockLoad` uses `valid_items` still capped to a total of `num_large_segments`, substituting the out-of-bounds item we asked it for with `0`. 
      const int num_large_segments_with_sentinel = static_cast<int>(num_large_segments) + 1;
      _CCCL_PRAGMA_NOUNROLL()
      for (int large_segment_offset = 0; large_segment_offset < num_large_segments_with_sentinel;
           large_segment_offset += epilogue_tile_size)
      {
        segment_size_val_t segment_tile_offsets[epilogue_items_per_thread];
        // `valid_items` for the load excludes the item for the aggregate item (it would otherwise be an OOB
        // read from uninitialised memory). The default `0` populates the aggregate position in
        // the thread-local items array.
        block_load_epilogue_t(temp_storage.load_epilogue)
          .Load(d_large_segments_tile_offsets + large_segment_offset,
                segment_tile_offsets,
                num_large_segments - large_segment_offset,
                0);
        __syncthreads();
        block_scan_epilogue_t(temp_storage.scan_epilogue)
          .ExclusiveSum(segment_tile_offsets, segment_tile_offsets, prefix_callback_op);
        __syncthreads();
        block_store_epilogue_t(temp_storage.store_epilogue)
          .Store(d_large_segments_tile_offsets + large_segment_offset,
                 segment_tile_offsets,
                 num_large_segments_with_sentinel - large_segment_offset);
        __syncthreads();
      }
    }
  }
};
//---------------------------------------------------------------------
// Segmented multi-CTA-per-segment top-k agents.
//
// Each of the three agents below is the segmented analog of one of the single-problem agents
// in `cub/agent/agent_topk.cuh`:
//   - `agent_batched_topk_histogram`         <-> `detail::topk::AgentTopKHistogram`
//   - `agent_batched_topk_filter_partition`  <-> `detail::topk::agent_topk_filter_partition`
//   - `agent_batched_topk_last_filter`       <-> `detail::topk::agent_topk_last_filter`
//
// Compared to their single-problem counterparts, each agent holds only per-launch state as
// members:
//   - iterators-of-iterators (`KeyInputItItT`, `KeyOutputItItT`, ...);
//   - parameter packs (`SegmentSizeParameterT`, `KParameterT`, `SelectDirectionParameterT`,
//     `NumSegmentsParameterT`);
//   - the segment-id provider (`SegmentIdProviderT`) that maps a queue index to the original
//     segment id (an identity `cuda::counting_iterator` on the all-large path, an index into
//     `d_large_segments_ids` on the mixed path);
//   - the per-segment tile-offset table (the exclusive scan of per-segment tile counts
//     produced by `agent_batched_topk_worker_per_segment`'s epilogue or by the all-large
//     transform-scan path); together with `total_large_tiles`, this drives block-to-segment
//     mapping;
//   - per-segment arrays of counters, histograms, and (for the filter agents) back buffers,
//     indexed by `queue_idx` rather than by original segment id.
//
// Per-launch single-problem state is moved out of the constructor and computed locally inside
// `run()` from `(global_tile_id, segment_id)` (where `global_tile_id` is supplied by the
// calling kernel's grid-stride loop and `segment_id` is resolved by the on-device binary
// search over the per-segment tile-offset table):
//   - the per-segment input/output iterators are obtained by dereferencing the outer iterators
//     at `segment_id`;
//   - `current_k`, `current_len`, `load_from_candidates_buffer`, `kth_key_bits`, ... come from
//     `d_segment_counters[queue_idx]`;
//   - the per-segment back buffers are slabs at `d_segment_*_key_buf + queue_idx *
//     candidate_buffer_length` (similarly for the value channel).
//
// Block-to-segment mapping (plan §3.5): each block handles exactly one tile of one segment,
// so `gridDim.x` covers the total large-tile count and the local tile id is derived from a
// binary search on the tile-offset table.
//
// Direction lowering (plan §3.6): for the first cut the dispatch ensures
// `SelectDirectionParameterT` is uniform across segments (compile-time or runtime-uniform), and
// passes `SelectDirection` as a template NTTP. Per-segment direction is deferred and would lower
// to a `dispatch_discrete` inside the agent, mirroring the small-segment agent's pattern.
//---------------------------------------------------------------------


//---------------------------------------------------------------------
// agent_batched_topk_histogram: segmented analog of `AgentTopKHistogram`.
//
// Each block processes exactly one tile (a contiguous chunk of `block_threads * items_per_thread`
// keys) of exactly one segment, building a per-tile histogram in smem and atomic-adding it into
// the per-segment global histogram slab at `d_segment_histograms + queue_idx * num_buckets`. The
// last block to retire on each segment runs the prefix-sum / bucket-finder epilogue, writing the
// next-pass `counter` state via `counter_update_fn` and `on_kth_bucket` callbacks (these are
// host-constructed at kernel scope, indexed per segment via the `queue_idx`).
//
// `FilterOpT` defaults to `topk_pass_through_filter_op` (pass 0) and is wrapped in
// `topk_candidate_filter_op` by the kernel when used as the "unbuffered scout" pass (the
// candidate set exceeds the back buffer at the current pass). See the single-source invariant
// comment on `AgentTopKHistogram` for why the unbuffered scout always loads from the original
// `d_keys_in` per segment.
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputItItT,
          typename ExtractBinOpT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename OffsetT,
          typename OutOffsetT,
          typename FilterOpT = detail::topk::topk_pass_through_filter_op>
struct agent_batched_topk_histogram
{
  using inner_key_it_t = it_value_t<KeyInputItItT>;
  using key_in_t       = it_value_t<inner_key_it_t>;
  using counter_t      = detail::topk::counter<key_in_t, OffsetT, OutOffsetT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;

  using keys_source_t = detail::topk::tile_data_source_t<
    inner_key_it_t,
    AgentTopKPolicyT::keys_tile_load_kind,
    block_threads,
    items_per_thread,
    OffsetT>;

  using block_identify_kth_bucket_t = detail::topk::block_identify_kth_bucket<
    block_threads,
    bits_per_pass,
    AgentTopKPolicyT::scan_algorithm,
    OffsetT,
    OutOffsetT>;

  // Smem layout mirrors `AgentTopKHistogram`'s phase-1+2 / phase-3 alias (see the comment there
  // for the reuse-plan rationale). One named arm of an outer union holds the histogram +
  // keys-source state used during the tile processing; the other holds the prefix-sum scratch
  // used only by the last block of the segment.
  struct phase_load_t
  {
    OffsetT histogram[num_buckets];
    typename keys_source_t::TempStorage keys_source_state;
    typename keys_source_t::ScratchStorage keys_source_scratch;
  };

  struct _TempStorage
  {
    union
    {
      phase_load_t phase_load;
      typename block_identify_kth_bucket_t::TempStorage prefix_sum;
    };
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // -------------------------------------------------------------------------
  // Members -- per-launch state only
  // -------------------------------------------------------------------------
  _TempStorage& temp_storage;
  KeyInputItItT d_key_segments_it;
  SegmentSizeParameterT segment_sizes;
  KParameterT k_param;
  NumSegmentsParameterT num_segments;
  SegmentIdProviderT segment_id_provider;
  const LargeSegmentTileOffsetT* d_large_segments_tile_offsets;
  counter_t* d_segment_counters;
  OffsetT* d_segment_histograms;
  ExtractBinOpT extract_bin_op;
  FilterOpT filter_op;

  // Number of enqueued large segments (queue slots). Drives the binary search over the offset
  // table. (`total_large_tiles` is owned by the calling kernel for the grid-stride loop bound;
  // it is not needed by the agent.)
  typename NumSegmentsParameterT::value_type num_large_segments;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_batched_topk_histogram(
    TempStorage& ts,
    KeyInputItItT d_key_segments_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    NumSegmentsParameterT num_segments,
    SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* d_large_segments_tile_offsets,
    counter_t* d_segment_counters,
    OffsetT* d_segment_histograms,
    ExtractBinOpT extract_bin_op,
    typename NumSegmentsParameterT::value_type num_large_segments,
    FilterOpT filter_op = {})
      : temp_storage(ts.Alias())
      , d_key_segments_it(d_key_segments_it)
      , segment_sizes(segment_sizes)
      , k_param(k_param)
      , num_segments(num_segments)
      , segment_id_provider(segment_id_provider)
      , d_large_segments_tile_offsets(d_large_segments_tile_offsets)
      , d_segment_counters(d_segment_counters)
      , d_segment_histograms(d_segment_histograms)
      , extract_bin_op(extract_bin_op)
      , filter_op(filter_op)
      , num_large_segments(num_large_segments)
  {}

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile_full(const key_in_t (&items)[items_per_thread])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      if (filter_op(items[j]))
      {
        const int bucket = extract_bin_op(items[j]);
        atomicAdd(temp_storage.phase_load.histogram + bucket, OffsetT{1});
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
        atomicAdd(temp_storage.phase_load.histogram + bucket, OffsetT{1});
      }
    }
  }

public:
  // `pass` selects which radix digit the histogram bins; `reset_histogram` is set by the
  // dispatch when the next pass also lands on this agent (i.e. the unbuffered chain continues
  // and the same global histogram slab is reused).
  //
  // `global_tile_id` is provided by the calling kernel's grid-stride loop (a single CTA may
  // process multiple tiles per launch when the grid is sized by `MaxSmOccupancy * num_sms`
  // rather than by `total_large_tiles`). The agent processes exactly one tile per call; the
  // per-segment counter epilogue is sized by the segment's tile count, so multiple calls from
  // the same physical CTA on tiles of the same segment retire that segment correctly.
  //
  // The per-segment last-block callbacks (`counter_update_fn`, `on_kth_bucket`) are constructed
  // *inside* `run()` -- the segmented kernel cannot capture per-segment state on the host because
  // the binary search that maps `global_tile_id -> queue_idx` runs only on-device. This is the
  // key structural divergence from the single-problem `AgentTopKHistogram::invoke`, where the
  // dispatch builds those closures with the (single) per-problem `counter_t*` already in hand.
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(LargeSegmentTileOffsetT global_tile_id, int pass, bool reset_histogram)
  {
    // Map (global_tile_id) -> (queue_idx, local_tile_id, segment_id). All threads do the same
    // binary search; the result is uniform across the block.
    const LargeSegmentTileOffsetT queue_idx =
      UpperBound(d_large_segments_tile_offsets, num_large_segments, global_tile_id) - 1;
    const LargeSegmentTileOffsetT slab_base    = d_large_segments_tile_offsets[queue_idx];
    const LargeSegmentTileOffsetT local_tile   = global_tile_id - slab_base;
    const auto segment_id                      = segment_id_provider[queue_idx];

    // Resolve per-segment state.
    const auto d_keys_in = d_key_segments_it[segment_id];
    const OffsetT num_items = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
    // Clip `k` to the segment's input size, mirroring the worker-per-segment agent (see the
    // `(::cuda::std::min)(k_param, segment_size)` clip in `agent_batched_topk_worker_per_segment`).
    // The radix-style top-k contract is "find the kth-largest key": when `k > segment_size`, all
    // items are trivially in the top-k and the kernels should emit exactly `segment_size` items.
    // Without this clip, `find_kth_bucket` over-counts (the running sum never reaches the unclipped
    // `k`), the kth-bucket resolution lands on a wrong bin index, and the last-filter agent
    // mis-sizes the per-segment output reservation in `reserve_cand{k_total, ...}`, scrambling the
    // key/value pairing across adjacent segment slots in the compacted output buffer.
    const OutOffsetT k = (::cuda::std::min) (
      static_cast<OutOffsetT>(k_param.get_param(segment_id)), static_cast<OutOffsetT>(num_items));

    OffsetT* segment_histogram     = d_segment_histograms + queue_idx * num_buckets;
    counter_t* segment_counter     = d_segment_counters + queue_idx;
    const OffsetT segment_tiles    = static_cast<OffsetT>(::cuda::ceil_div(num_items, OffsetT{tile_items}));

    // Initialize the smem histogram for this block's tile.
    detail::topk::init_histogram<block_threads, num_buckets>(temp_storage.phase_load.histogram);
    __syncthreads();

    keys_source_t keys_source{d_keys_in, temp_storage.phase_load.keys_source_state};

    const OffsetT num_full_tiles      = num_items / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items       = num_items - num_full_tiles * static_cast<OffsetT>(tile_items);

    // Per-block tile body. Each block has exactly one tile (`local_tile`) in its segment. The
    // full-tile path is the dominant case for any segment large enough to enter the multi-CTA
    // pipeline (it occupies the first `num_full_tiles` of every segment's `segment_tiles_input`
    // tiles); we expose that bias to the compiler so its branch-layout / instruction-scheduling
    // heuristics favor it.
    if (_CCCL_BUILTIN_EXPECT(local_tile < num_full_tiles, 1))
    {
      const OffsetT tile_base = static_cast<OffsetT>(local_tile) * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(temp_storage.phase_load.keys_source_scratch);
      h.complete_load(items);
      process_tile_full(items);
    }
    else if (local_tile == num_full_tiles && partial_items > 0)
    {
      const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(temp_storage.phase_load.keys_source_scratch, partial_items);
      h.complete_load(items);
      const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
      const int num_thread_items =
        (thread_offset >= num_items)
          ? 0
          : static_cast<int>((::cuda::std::min) (static_cast<OffsetT>(items_per_thread), num_items - thread_offset));
      process_tile_partial(items, num_thread_items);
    }
    // Else this block's `local_tile` is past the end of the segment -- a defensive no-op for
    // any over-allocated grid slot inside an enqueued slab (`segment_tiles` is computed by
    // ceil-div'ing `num_items` and exactly matches the number of blocks the host launched for
    // this segment, so this `else` is unreachable in practice; the bounds check is left here
    // for robustness against off-by-one errors in the offset table).

    __syncthreads();
    detail::topk::merge_histogram<block_threads, num_buckets>(
      temp_storage.phase_load.histogram, segment_histogram);

    // Build per-segment closures inside `run()` so they can capture the per-segment counter
    // pointer (only knowable after the on-device binary search). These mirror the single-problem
    // `DeviceTopKHistogramKernel`'s closures verbatim.
    auto counter_update_fn = [segment_counter, num_items] {
      segment_counter->num_candidates_in      = num_items;
      segment_counter->num_candidates_written = 0;
    };
    auto on_kth_bucket =
      [segment_counter, pass](OutOffsetT current_k, int bin_index, OffsetT num_selected, OffsetT num_candidates) {
        segment_counter->k                  = static_cast<OutOffsetT>(current_k - num_selected);
        segment_counter->num_candidates_out = num_candidates;
        detail::topk::set_kth_key_bits<bits_per_pass>(
          segment_counter->kth_key_bits, pass, static_cast<unsigned int>(bin_index));
      };

    // Last-block-per-segment epilogue: thread 0 runs the per-segment counter update, the unique
    // thread that owns the kth bucket invokes `on_kth_bucket` with the resolved bin index, and
    // the global histogram for this segment is reset for the next pass if requested. The reset
    // counter (`expected_block_count = segment_tiles`) drives `finalize_pass` to elect exactly
    // one block per segment to run the epilogue.
    auto epilogue_op = [this, &on_kth_bucket, &counter_update_fn, k, segment_histogram, reset_histogram] {
      if (threadIdx.x == 0)
      {
        counter_update_fn();
      }
      block_identify_kth_bucket_t{temp_storage.prefix_sum}.find_kth_bucket(segment_histogram, k, on_kth_bucket);
      if (reset_histogram)
      {
        detail::topk::init_histogram<block_threads, num_buckets>(segment_histogram);
      }
    };

    detail::topk::finalize_pass(
      &segment_counter->finished_block_cnt, static_cast<unsigned int>(segment_tiles), epilogue_op);
  }
};
} // namespace detail::batched_topk
CUB_NAMESPACE_END
