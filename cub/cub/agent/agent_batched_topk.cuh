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
#include <cub/detail/warpspeed/make_warp_uniform.cuh>
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
// Each CTA processes a *chunk* of `tiles_per_chunk` consecutive tiles per grid-stride iteration
// (vs. one tile per stride in the previous design). Inside the chunk, the agent groups tiles by
// segment: it initialises an smem histogram once when it first sees a segment, atomic-adds every
// tile's keys into it, and merges into the per-segment global histogram only when the segment
// changes (or the chunk ends). For workloads dominated by a single large segment -- the common
// shape for the multi-CTA-per-segment path -- this amortises one init + one merge across all
// tiles a CTA processes for that segment, instead of paying init/merge on every tile.
//
// The agent does *not* run the per-segment prefix-sum / bucket-finder epilogue any more. That
// work (last-block election, prefix-sum, k-th-bucket scan, counter update, optional histogram
// reset) is fully offloaded to a separate `device_segmented_topk_finalize_histogram_kernel`
// that runs after the histogram kernel completes. Splitting the histogramming from the
// finalisation removes the per-tile `finalize_pass` (and its `__threadfence` + `__syncthreads_or`
// chain) entirely on this path; the cost is one extra device-side kernel launch per pass that
// uses this agent.
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

  // The histogram agent no longer carries the prefix-sum scratch used by the per-segment
  // last-block epilogue: that work has been hoisted out into the standalone
  // `device_segmented_topk_finalize_histogram_kernel`. Smem here is the smem histogram + the
  // keys-source state / scratch only.
  struct _TempStorage
  {
    OffsetT histogram[num_buckets];
    typename keys_source_t::TempStorage keys_source_state;
    typename keys_source_t::ScratchStorage keys_source_scratch;
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

  // Resolve `global_tile_id -> queue_idx` via the on-device binary search through
  // `d_large_segments_tile_offsets`. All threads of the warp pass the same `global_tile_id` (the
  // calling kernel's grid-stride loop is identical across lanes), so the result is warp-uniform
  // by construction. The lane-0 + `__shfl_sync` shape gives ptxas a single producer for the
  // uniform value and lets downstream `queue_idx`-indexed derivations be UR-promoted (see the
  // register-pressure investigation notes).
  _CCCL_DEVICE _CCCL_FORCEINLINE LargeSegmentTileOffsetT resolve_queue_idx(LargeSegmentTileOffsetT global_tile_id)
  {
    LargeSegmentTileOffsetT queue_idx_lane0 = 0;
    if ((threadIdx.x & 31) == 0)
    {
      queue_idx_lane0 = UpperBound(d_large_segments_tile_offsets, num_large_segments, global_tile_id) - 1;
    }
    return __shfl_sync(0xffffffff, queue_idx_lane0, 0);
  }

public:
  // Process up to `tiles_per_chunk` consecutive tiles starting at `chunk_start`, grouped by
  // segment. The chunk is a contiguous range in the *queue_idx-space*: tile `t` belongs to the
  // segment found by `UpperBound(d_large_segments_tile_offsets, ..., t) - 1`, and a single chunk
  // may straddle one or more segment boundaries. The agent maintains a single smem histogram and
  // re-initialises it whenever the segment changes (or at the start of the chunk), merging into
  // the per-segment global slab right before the change. `total_large_tiles` is the upper bound
  // of the queue_idx-space; tiles past it are no-ops.
  //
  // `pass` selects which radix digit the histogram bins (consumed by `extract_bin_op`).
  //
  // The per-segment epilogue (prefix-sum + bucket-finder + counter update + optional global
  // histogram reset) lives in a separate kernel that runs after the histogram kernel; this agent
  // is intentionally write-only with respect to `d_segment_histograms` -- it never reads the
  // global slab and never runs `finalize_pass`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_chunk(
    LargeSegmentTileOffsetT chunk_start, int tiles_per_chunk, LargeSegmentTileOffsetT total_large_tiles, int pass)
  {
    // Per-segment state cached across tiles of the same segment within the chunk. The "no active
    // segment" sentinel is the all-ones value (`-1` for signed `LargeSegmentTileOffsetT`,
    // `MAX_UINT` for unsigned). `num_large_segments` is bounded by the kernel's host-side
    // configuration and never reaches the sentinel.
    constexpr LargeSegmentTileOffsetT kNoActiveSegment = static_cast<LargeSegmentTileOffsetT>(-1);
    LargeSegmentTileOffsetT active_queue_idx           = kNoActiveSegment;

    // Cached per-segment derivations. Refreshed only when `active_queue_idx` changes.
    inner_key_it_t active_d_keys_in{};
    OffsetT* active_segment_histogram = nullptr;
    OffsetT active_num_items          = 0;
    LargeSegmentTileOffsetT active_slab_base = 0;
    OffsetT active_num_full_tiles    = 0;
    OffsetT active_partial_items     = 0;

    _CCCL_PRAGMA_NOUNROLL()
    for (int i = 0; i < tiles_per_chunk; ++i)
    {
      const LargeSegmentTileOffsetT global_tile_id = chunk_start + static_cast<LargeSegmentTileOffsetT>(i);
      if (global_tile_id >= total_large_tiles)
      {
        break;
      }

      const LargeSegmentTileOffsetT queue_idx = resolve_queue_idx(global_tile_id);

      if (queue_idx != active_queue_idx)
      {
        // Flush the previously-active segment's smem histogram into its global slab.
        if (active_queue_idx != kNoActiveSegment)
        {
          __syncthreads();
          detail::topk::merge_histogram<block_threads, num_buckets>(temp_storage.histogram, active_segment_histogram);
          // `merge_histogram` writes only via global atomics; `init_histogram` below writes to
          // smem only. The `__syncthreads()` between them ensures no thread re-initialises a
          // smem bucket while another thread is still reading it during the merge load.
        }

        // Refresh the per-segment caches for the new segment.
        active_queue_idx          = queue_idx;
        const auto segment_id     = segment_id_provider[queue_idx];
        active_d_keys_in          = d_key_segments_it[segment_id];
        active_num_items          = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
        active_segment_histogram  = d_segment_histograms + queue_idx * num_buckets;
        active_slab_base          = d_large_segments_tile_offsets[queue_idx];
        active_num_full_tiles     = active_num_items / static_cast<OffsetT>(tile_items);
        active_partial_items      = active_num_items - active_num_full_tiles * static_cast<OffsetT>(tile_items);

        __syncthreads();
        detail::topk::init_histogram<block_threads, num_buckets>(temp_storage.histogram);
        __syncthreads();
      }

      const LargeSegmentTileOffsetT local_tile = global_tile_id - active_slab_base;

      // `keys_source_t` captures the segment's input iterator + its smem state. The state lives
      // in `temp_storage.keys_source_state` and is reused across all tiles of all segments the
      // chunk visits (the source's per-tile state is reset by `set_tile_base()` below). The
      // local view-object is re-constructed every iteration to refresh the captured iterator
      // when the segment changes; on same-segment iterations the constructor folds away.
      keys_source_t keys_source{active_d_keys_in, temp_storage.keys_source_state};

      // Full-tile path is the dominant case for any segment large enough to enter the multi-CTA
      // pipeline; we expose that bias to the compiler.
      if (_CCCL_BUILTIN_EXPECT(local_tile < active_num_full_tiles, 1))
      {
        const OffsetT tile_base = static_cast<OffsetT>(local_tile) * static_cast<OffsetT>(tile_items);
        keys_source.set_tile_base(tile_base);

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(temp_storage.keys_source_scratch);
        h.complete_load(items);
        process_tile_full(items);
      }
      else if (local_tile == active_num_full_tiles && active_partial_items > 0)
      {
        const OffsetT tile_base = active_num_full_tiles * static_cast<OffsetT>(tile_items);
        keys_source.set_tile_base(tile_base);

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(temp_storage.keys_source_scratch, active_partial_items);
        h.complete_load(items);
        const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
        const int num_thread_items =
          (thread_offset >= active_num_items)
            ? 0
            : static_cast<int>((::cuda::std::min) (
              static_cast<OffsetT>(items_per_thread), active_num_items - thread_offset));
        process_tile_partial(items, num_thread_items);
      }
      // Else this block's `local_tile` is past the end of the segment -- a defensive no-op for
      // any over-allocated grid slot inside an enqueued slab (the offset table is sized by
      // ceil-div'ing `num_items` so the last queue_idx's tile count is exact; this `else` is
      // unreachable in practice).
    }

    // Flush the final segment's smem histogram into its global slab. `kNoActiveSegment` here
    // means the chunk was entirely past `total_large_tiles` (defensive: the dispatch's grid
    // sizing usually keeps a chunk inside the bound for at least its first tile).
    if (active_queue_idx != kNoActiveSegment)
    {
      __syncthreads();
      detail::topk::merge_histogram<block_threads, num_buckets>(temp_storage.histogram, active_segment_histogram);
    }
  }
};

//---------------------------------------------------------------------
// agent_batched_topk_filter_partition: segmented analog of `agent_topk_filter_partition`.
//
// Handles both `sink_mode::early_stop` and `sink_mode::buffered` in one agent type (same as the
// single-problem version), with the mode selected at runtime per segment via the segment's
// counter state. Each block processes exactly one tile of one segment. The buffered branch
// accumulates a per-segment histogram in smem, atomically merging into
// `d_segment_histograms + queue_idx * num_buckets`; the last block to retire on each segment
// runs the prefix-sum + bucket-finder epilogue via `finalize_pass`.
//
// Per-segment double-buffering: the global `DoubleBuffer<key_in_t>` `selector` is flipped once
// per pass on the host (plan §5.5 -- safe because `num_passes` is uniform across all segments).
// The per-segment back buffers are slabs of `candidate_buffer_length` items at
// `d_segment_*_key_buf + queue_idx * candidate_buffer_length` (similarly for the value channel).
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          typename DecomposerT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename OffsetT,
          typename OutOffsetT,
          detail::topk::block_partition_strategy BufferedPartStrat =
            detail::topk::block_partition_strategy::atomics,
          detail::topk::block_filter_strategy EarlyStopFilterStrat =
            detail::topk::block_filter_strategy::atomics,
          bool LazyValueLoad   = false,
          bool InlinedClassify = false>
struct agent_batched_topk_filter_partition
{
  using inner_key_it_t   = it_value_t<KeyInputItItT>;
  using inner_value_it_t = it_value_t<ValueInputItItT>;
  using inner_value_out_it_t = it_value_t<ValueOutputItItT>;
  using inner_key_out_it_t   = it_value_t<KeyOutputItItT>;

  using key_in_t   = it_value_t<inner_key_it_t>;
  using value_in_t = it_value_t<inner_value_it_t>;
  using counter_t  = detail::topk::counter<key_in_t, OffsetT, OutOffsetT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<value_in_t, cub::NullType>;

  using block_identify_kth_bucket_t = detail::topk::block_identify_kth_bucket<
    block_threads,
    bits_per_pass,
    AgentTopKPolicyT::scan_algorithm,
    OffsetT,
    OutOffsetT>;

  static constexpr bool effective_lazy_value_load = LazyValueLoad && !keys_only;

  // Multi-source key / value channels mirror the single-problem agent. The "buffer" source is
  // the candidate slab carried over from the previous pass.
  using key_source_input_t = detail::topk::tile_data_source_t<
    inner_key_it_t,
    AgentTopKPolicyT::keys_tile_load_kind,
    block_threads,
    items_per_thread,
    OffsetT>;
  using key_source_buffer_t = detail::topk::tile_data_source_t<
    key_in_t*,
    AgentTopKPolicyT::keys_tile_load_kind,
    block_threads,
    items_per_thread,
    OffsetT>;
  using keys_source_t = detail::topk::multi_source_data_source<key_source_input_t, key_source_buffer_t, OffsetT>;

  using value_source_input_t =
    detail::topk::direct_data_source<inner_value_it_t, block_threads, items_per_thread, OffsetT>;
  using value_source_buffer_t =
    detail::topk::direct_data_source<value_in_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t =
    detail::topk::multi_source_data_source<value_source_input_t, value_source_buffer_t, OffsetT>;

  using val_out_t              = inner_value_out_it_t;
  using buffered_cand_val_out_t = value_in_t*;
  using buffered_cand_key_out_t = key_in_t*;

  using buffered_value_channel_sinks_concrete_t = detail::topk::value_channel_sinks_t<
    val_out_t,
    buffered_cand_val_out_t,
    ::cuda::std::identity,
    ::cuda::std::identity>;
  using buffered_value_channel_sinks_t =
    ::cuda::std::conditional_t<keys_only, NullType, buffered_value_channel_sinks_concrete_t>;

  using early_stop_value_channel_sinks_concrete_t =
    detail::topk::value_channel_sinks_filter_t<val_out_t, ::cuda::std::identity>;
  using early_stop_value_channel_sinks_t =
    ::cuda::std::conditional_t<keys_only, NullType, early_stop_value_channel_sinks_concrete_t>;

  using agent_value_t = ::cuda::std::conditional_t<keys_only, NullType, value_in_t>;
  using agent_value_data_source_scratch_t =
    ::cuda::std::conditional_t<keys_only, NullType, typename value_source_t::ScratchStorage>;

  using selected_offset_t  = OutOffsetT;
  using candidate_offset_t = OffsetT;

  using selected_reserve_op_t  = detail::topk::atomic_reserve_range_op<selected_offset_t>;
  using candidate_reserve_op_t = detail::topk::atomic_reserve_range_op<candidate_offset_t>;

  using key_xform_t = ::cuda::std::identity;

  using histogram_callback_op_t = detail::topk::topk_histogram_callback_op<ExtractBinOpT, OffsetT>;
  using identify_selected_op_t  = detail::topk::topk_identify_selected_op<IdentifyCandidatesOpT>;

  using buffered_partition_t = detail::topk::strategy_to_partition_class_t<
    BufferedPartStrat,
    block_threads,
    items_per_thread,
    AgentTopKPolicyT::accumulating_buffer_capacity,
    AgentTopKPolicyT::speculative_selected_buffer_capacity,
    key_in_t,
    selected_offset_t,
    candidate_offset_t,
    selected_reserve_op_t,
    candidate_reserve_op_t,
    key_xform_t,
    key_xform_t,
    inner_key_out_it_t,
    buffered_cand_key_out_t,
    IdentifyCandidatesOpT,
    histogram_callback_op_t,
    buffered_value_channel_sinks_t,
    agent_value_t,
    agent_value_data_source_scratch_t,
    effective_lazy_value_load,
    InlinedClassify>;

  using early_stop_filter_t = detail::topk::strategy_to_filter_class_t<
    EarlyStopFilterStrat,
    block_threads,
    items_per_thread,
    AgentTopKPolicyT::accumulating_buffer_capacity,
    key_in_t,
    selected_offset_t,
    selected_reserve_op_t,
    key_xform_t,
    inner_key_out_it_t,
    identify_selected_op_t,
    early_stop_value_channel_sinks_t,
    agent_value_t,
    agent_value_data_source_scratch_t,
    effective_lazy_value_load,
    InlinedClassify>;

  // Same `empty_prefix_sum_t` placeholder pattern as the single-problem agent: hoist `prefix_sum`
  // out of the per-mode arena into the outer `_TempStorage` union so it can alias with the
  // (larger) phase-1+2 footprint.
  struct empty_prefix_sum_t
  {};

  using buffered_storage_layout_t = detail::topk::partition_storage_layout_for_t<
    buffered_partition_t,
    typename keys_source_t::ScratchStorage,
    empty_prefix_sum_t>;
  using early_stop_storage_layout_t = detail::topk::partition_storage_layout_for_t<
    early_stop_filter_t,
    typename keys_source_t::ScratchStorage,
    empty_prefix_sum_t>;

  struct _TempStorage
  {
    union arms_t
    {
      struct buffered_t
      {
        OffsetT histogram[num_buckets];
        typename keys_source_t::TempStorage keys_source_state;
        buffered_storage_layout_t arena;
      } buffered;

      typename block_identify_kth_bucket_t::TempStorage prefix_sum;

      struct early_stop_t
      {
        typename keys_source_t::TempStorage keys_source_state;
        early_stop_storage_layout_t arena;
      } early_stop;

      _CCCL_HOST_DEVICE arms_t() {}
      _CCCL_HOST_DEVICE ~arms_t() {}
    } arms;
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // -------------------------------------------------------------------------
  // Members -- per-launch state only
  // -------------------------------------------------------------------------
  _TempStorage& storage;
  KeyInputItItT d_key_segments_it;
  KeyOutputItItT d_key_segments_out_it;
  ValueInputItItT d_value_segments_it;
  ValueOutputItItT d_value_segments_out_it;
  SegmentSizeParameterT segment_sizes;
  KParameterT k_param;
  NumSegmentsParameterT num_segments;
  SegmentIdProviderT segment_id_provider;
  const LargeSegmentTileOffsetT* d_large_segments_tile_offsets;
  counter_t* d_segment_counters;
  OffsetT* d_segment_histograms;
  key_in_t* d_segment_in_key_buf;
  value_in_t* d_segment_in_val_buf;
  key_in_t* d_segment_out_key_buf;
  value_in_t* d_segment_out_val_buf;
  ExtractBinOpT extract_bin_op;
  int total_bits;
  DecomposerT decomposer;
  OffsetT candidate_buffer_length;
  // Cost-savings threshold for entering the buffered chain (see `run()`). Buffering is only
  // beneficial when `num_candidates_out` is at most `segment_num_items / coefficient`; otherwise
  // the extra write-side cost of populating the candidate buffer outweighs the read-side savings
  // for the next pass. Threaded from the dispatch in lock-step with `candidate_buffer_length` so
  // the per-segment buffer-sizing assumption and the runtime gating heuristic stay consistent.
  OffsetT candidate_buffer_coefficient;
  // (See `agent_batched_topk_histogram` for the rationale behind dropping `total_large_tiles`.)
  typename NumSegmentsParameterT::value_type num_large_segments;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_batched_topk_filter_partition(
    TempStorage& ts,
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    NumSegmentsParameterT num_segments,
    SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* d_large_segments_tile_offsets,
    counter_t* d_segment_counters,
    OffsetT* d_segment_histograms,
    key_in_t* d_segment_in_key_buf,
    value_in_t* d_segment_in_val_buf,
    key_in_t* d_segment_out_key_buf,
    value_in_t* d_segment_out_val_buf,
    ExtractBinOpT extract_bin_op,
    int total_bits,
    DecomposerT decomposer,
    OffsetT candidate_buffer_length,
    OffsetT candidate_buffer_coefficient,
    typename NumSegmentsParameterT::value_type num_large_segments)
      : storage(ts.Alias())
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , k_param(k_param)
      , num_segments(num_segments)
      , segment_id_provider(segment_id_provider)
      , d_large_segments_tile_offsets(d_large_segments_tile_offsets)
      , d_segment_counters(d_segment_counters)
      , d_segment_histograms(d_segment_histograms)
      , d_segment_in_key_buf(d_segment_in_key_buf)
      , d_segment_in_val_buf(d_segment_in_val_buf)
      , d_segment_out_key_buf(d_segment_out_key_buf)
      , d_segment_out_val_buf(d_segment_out_val_buf)
      , extract_bin_op(extract_bin_op)
      , total_bits(total_bits)
      , decomposer(decomposer)
      , candidate_buffer_length(candidate_buffer_length)
      , candidate_buffer_coefficient(candidate_buffer_coefficient)
      , num_large_segments(num_large_segments)
  {}

private:
  // Build the per-segment buffered-mode sinks. The candidate values are deposited into the
  // per-segment slab of the back-buffer-out (selected via the host-side double-buffer flip).
  template <typename ValueOutSinkT>
  _CCCL_DEVICE _CCCL_FORCEINLINE auto
  make_buffered_value_channel_sinks(ValueOutSinkT values_out_sink, [[maybe_unused]] buffered_cand_val_out_t cand_val_out)
  {
    if constexpr (keys_only)
    {
      return NullType{};
    }
    else
    {
      return buffered_value_channel_sinks_concrete_t{
        values_out_sink, cand_val_out, ::cuda::std::identity{}, ::cuda::std::identity{}};
    }
  }

  template <typename ValueOutSinkT>
  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_early_stop_value_channel_sinks(ValueOutSinkT values_out_sink)
  {
    if constexpr (keys_only)
    {
      return NullType{};
    }
    else
    {
      return early_stop_value_channel_sinks_concrete_t{values_out_sink, ::cuda::std::identity{}};
    }
  }

public:
  // `pass` selects the radix digit; `reset_histogram` is set on every non-final filter pass.
  //
  // `global_tile_id` is supplied by the calling kernel's grid-stride loop; see the matching
  // comment on `agent_batched_topk_histogram::run()` for why the agent processes one tile per
  // call and the kernel iterates.
  //
  // The per-segment last-block callbacks (`counter_update_fn`, `on_kth_bucket`) are built inside
  // `run()` -- only here, after the on-device binary search, do we know the segment's
  // `counter_t*`. The closures capture per-segment `current_len`, `early_stop`, `will_buffer`,
  // `pass`, and the segment counter pointer.
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(LargeSegmentTileOffsetT global_tile_id, int pass, bool reset_histogram)
  {
    // See the matching comment on `agent_batched_topk_histogram::run()`: only lane 0 runs
    // the binary search; `__shfl_sync` then broadcasts the result to the whole warp. This
    // makes the warp-uniformity of `queue_idx` and every value indexed by it explicit at
    // the SASS level (uniform broadcast from a single producer) instead of relying on
    // ptxas's heuristic to recover uniformity from 32 redundant computations.
    LargeSegmentTileOffsetT queue_idx_lane0 = 0;
    if ((threadIdx.x & 31) == 0)
    {
      queue_idx_lane0 = UpperBound(d_large_segments_tile_offsets, num_large_segments, global_tile_id) - 1;
    }
    const LargeSegmentTileOffsetT queue_idx  = __shfl_sync(0xffffffff, queue_idx_lane0, 0);
    const LargeSegmentTileOffsetT slab_base  = d_large_segments_tile_offsets[queue_idx];
    const LargeSegmentTileOffsetT local_tile = global_tile_id - slab_base;
    const auto segment_id                    = segment_id_provider[queue_idx];

    auto d_keys_in    = d_key_segments_it[segment_id];
    auto d_keys_out   = d_key_segments_out_it[segment_id];
    [[maybe_unused]] auto d_values_in  = [&] {
      if constexpr (!keys_only)
      {
        return d_value_segments_it[segment_id];
      }
      else
      {
        return inner_value_it_t{};
      }
    }();
    [[maybe_unused]] auto d_values_out = [&] {
      if constexpr (!keys_only)
      {
        return d_value_segments_out_it[segment_id];
      }
      else
      {
        return inner_value_out_it_t{};
      }
    }();

    counter_t* segment_counter = d_segment_counters + queue_idx;
    OffsetT* segment_histogram = d_segment_histograms + queue_idx * num_buckets;

    const OutOffsetT current_k             = segment_counter->k;
    const OffsetT current_len              = segment_counter->num_candidates_out;
    const OffsetT counter_input_length     = segment_counter->num_candidates_in;
    const bool load_from_candidates_buffer = segment_counter->load_from_candidates_buffer;

    if (counter_input_length == 0)
    {
      return;
    }

    // Construct the per-segment `identify_candidates_op` with the per-segment counter's
    // `kth_key_bits` pointer. The kernel-level construction is not possible because the
    // (queue_idx -> segment) mapping is resolved by the on-device binary search.
    IdentifyCandidatesOpT identify_candidates_op{&segment_counter->kth_key_bits, pass, total_bits, decomposer};

    const OffsetT segment_num_items   = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
    // `counter_input_length` is the segment's input length for this pass (single-problem analog
    // `dispatch_topk::DeviceTopKFilterKernel`'s `input_length = counter->num_candidates_in`). In
    // the unbuffered chain it equals `segment_num_items` (set by the histogram pass's
    // counter_update_fn); in the buffered chain it equals the count of candidates written to
    // the candidate buffer by the previous pass (set by the previous buffered pass's
    // counter_update_fn `num_candidates_in = current_len`). `current_len` is the previous
    // pass's `num_candidates_out` -- used for branch selection (`early_stop` / `will_buffer`)
    // but NOT for sizing the tile loop.
    const OffsetT input_length_actual = counter_input_length;

    const bool early_stop  = (current_len == static_cast<OffsetT>(current_k));
    // Entry condition for the buffered chain.
    //
    //   (a) `current_len <= candidate_buffer_length`   -- the candidate set fits in the
    //       per-segment back buffer slab the dispatch carved out for this segment.
    //
    //   (b) `current_len <= segment_num_items / candidate_buffer_coefficient` -- the candidate
    //       set is small enough relative to the segment that the write-side cost of materializing
    //       the back buffer pays off via the read-side savings in the next pass. The threshold is
    //       threaded from the dispatch in lock-step with `candidate_buffer_length` (which is
    //       sized as `static_max_seg_size / coefficient`). For small segments where
    //       `segment_num_items / coefficient < current_len`, we keep iterating the unbuffered
    //       chain instead. Integer-divide on `segment_num_items` is exact here -- both operands
    //       are `OffsetT`.
    const bool will_buffer = !early_stop && (current_len <= candidate_buffer_length)
                          && (current_len <= segment_num_items / candidate_buffer_coefficient);

    // Per-segment back-buffer slabs. The host-side global double-buffer flip determines whether
    // `d_segment_in_*_buf` is the previous pass's candidate buffer or the original input buffer
    // pair; the agent reads from `in_key_buf` only when the per-segment counter says so.
    key_in_t* in_key_buf = d_segment_in_key_buf + queue_idx * candidate_buffer_length;
    key_in_t* out_key_buf = will_buffer ? (d_segment_out_key_buf + queue_idx * candidate_buffer_length) : nullptr;
    value_in_t* in_val_buf = nullptr;
    value_in_t* out_val_buf = nullptr;
    if constexpr (!keys_only)
    {
      in_val_buf  = load_from_candidates_buffer ? (d_segment_in_val_buf + queue_idx * candidate_buffer_length) : nullptr;
      out_val_buf = will_buffer ? (d_segment_out_val_buf + queue_idx * candidate_buffer_length) : nullptr;
    }

    // Number of tiles this segment processes for this pass, computed from `input_length_actual`.
    // The per-segment tile-offset table was scanned over the (host-side) upper bound on each
    // segment's tile count (`ceil_div(segment_size, multi_worker_tile_size)`), so on the
    // buffered path -- where the actual input length is `current_len` and strictly smaller than
    // `segment_num_items` -- some of the per-segment tile slots fall past `segment_tiles_input`
    // and must be ignored. The retirement counter (`expected_block_count = segment_tiles_input`,
    // below) is sized to the meaningful tile count, so the early-exit here keeps the elected
    // last block on the buffered path consistent with the (now-shorter) tile slab.
    const OffsetT segment_tiles_input =
      static_cast<OffsetT>(::cuda::ceil_div(input_length_actual, OffsetT{tile_items}));

    if (static_cast<OffsetT>(local_tile) >= segment_tiles_input)
    {
      return;
    }

    selected_reserve_op_t reserve_sel{&segment_counter->num_selected_written};
    key_xform_t sel_key_xform{};
    key_xform_t cand_key_xform{};

    const OffsetT num_full_tiles = input_length_actual / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items  = input_length_actual - num_full_tiles * static_cast<OffsetT>(tile_items);

    // Three mutually-exclusive sub-modes (mirrors the single-problem dispatch's host-side branch
    // `if (early_stop || will_buffer) {...} else {...}`):
    //   - `early_stop`            : filter-only; selected items go directly to `d_keys_out`.
    //   - `buffered`              : partition; selected -> `d_keys_out`, candidates -> `out_key_buf`.
    //   - `unbuffered` (else)     : histogram-only "scout" pass, with a candidate filter applied.
    //                               No items are written; only the global histogram is updated and
    //                               the per-segment counter (kth-key bits, num_candidates_out, k)
    //                               is advanced via `on_kth_bucket`. This matches the single-
    //                               problem `agent_ub_t = AgentTopKHistogram` invocation.

    if (early_stop)
    {
      typename key_source_input_t::TempStorage* state_a_ptr = &storage.arms.early_stop.keys_source_state.a;
      typename key_source_buffer_t::TempStorage* state_b_ptr = &storage.arms.early_stop.keys_source_state.b;
      key_source_input_t key_src_input{d_keys_in, *state_a_ptr};
      key_source_buffer_t key_src_buffer{in_key_buf, *state_b_ptr};
      keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/load_from_candidates_buffer};

      identify_selected_op_t identify_selected{identify_candidates_op};
      auto value_channel_sinks = make_early_stop_value_channel_sinks(d_values_out);

      early_stop_filter_t filter{
        storage.arms.early_stop.arena.get_partition_state(),
        reserve_sel,
        sel_key_xform,
        d_keys_out,
        value_channel_sinks,
        identify_selected};

      // Single tile body for early_stop.
      if (local_tile < num_full_tiles)
      {
        const OffsetT tile_base = static_cast<OffsetT>(local_tile) * static_cast<OffsetT>(tile_items);
        keys_source.set_tile_base(tile_base);

        auto value_source = [&] {
          if constexpr (keys_only)
          {
            return NullType{};
          }
          else
          {
            typename value_source_input_t::TempStorage val_state_input{};
            typename value_source_buffer_t::TempStorage val_state_buffer{};
            value_source_input_t val_input{d_values_in, val_state_input};
            value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};
            value_source_t val_src{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
            val_src.set_tile_base(tile_base);
            return val_src;
          }
        }();

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.arms.early_stop.arena.get_keys_source_scratch());
        h.complete_load(items);
        __syncthreads();
        filter.partition(storage.arms.early_stop.arena.get_partition_scratch(), items, value_source);
      }
      else if (local_tile == num_full_tiles && partial_items > 0)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);
        keys_source.set_tile_base(tile_base);

        auto value_source = [&] {
          if constexpr (keys_only)
          {
            return NullType{};
          }
          else
          {
            typename value_source_input_t::TempStorage val_state_input{};
            typename value_source_buffer_t::TempStorage val_state_buffer{};
            value_source_input_t val_input{d_values_in, val_state_input};
            value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};
            value_source_t val_src{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
            val_src.set_tile_base(tile_base);
            return val_src;
          }
        }();

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.arms.early_stop.arena.get_keys_source_scratch(), partial_items);
        h.complete_load(items);
        __syncthreads();
        filter.partition(storage.arms.early_stop.arena.get_partition_scratch(), items, partial_items, value_source);
      }

      filter.epilogue();
    }
    else if (will_buffer)
    {
      key_source_input_t key_src_input{d_keys_in, storage.arms.buffered.keys_source_state.a};
      key_source_buffer_t key_src_buffer{in_key_buf, storage.arms.buffered.keys_source_state.b};
      keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/load_from_candidates_buffer};

      detail::topk::init_histogram<block_threads, num_buckets>(storage.arms.buffered.histogram);
      __syncthreads();

      buffered_cand_key_out_t cand_key_out = out_key_buf;
      buffered_cand_val_out_t cand_val_out = out_val_buf;
      candidate_reserve_op_t reserve_cand{&segment_counter->num_candidates_written};
      histogram_callback_op_t histogram_cb{extract_bin_op, storage.arms.buffered.histogram};
      auto value_channel_sinks = make_buffered_value_channel_sinks(d_values_out, cand_val_out);

      buffered_partition_t partition{
        storage.arms.buffered.arena.get_partition_state(),
        reserve_sel,
        reserve_cand,
        sel_key_xform,
        cand_key_xform,
        d_keys_out,
        cand_key_out,
        value_channel_sinks,
        identify_candidates_op,
        histogram_cb};

      if (local_tile < num_full_tiles)
      {
        const OffsetT tile_base = static_cast<OffsetT>(local_tile) * static_cast<OffsetT>(tile_items);
        keys_source.set_tile_base(tile_base);

        auto value_source = [&] {
          if constexpr (keys_only)
          {
            return NullType{};
          }
          else
          {
            typename value_source_input_t::TempStorage val_state_input{};
            typename value_source_buffer_t::TempStorage val_state_buffer{};
            value_source_input_t val_input{d_values_in, val_state_input};
            value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};
            value_source_t val_src{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
            val_src.set_tile_base(tile_base);
            return val_src;
          }
        }();

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch());
        h.complete_load(items);
        __syncthreads();
        partition.partition(storage.arms.buffered.arena.get_partition_scratch(), items, value_source);
      }
      else if (local_tile == num_full_tiles && partial_items > 0)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);
        keys_source.set_tile_base(tile_base);

        auto value_source = [&] {
          if constexpr (keys_only)
          {
            return NullType{};
          }
          else
          {
            typename value_source_input_t::TempStorage val_state_input{};
            typename value_source_buffer_t::TempStorage val_state_buffer{};
            value_source_input_t val_input{d_values_in, val_state_input};
            value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};
            value_source_t val_src{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
            val_src.set_tile_base(tile_base);
            return val_src;
          }
        }();

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch(), partial_items);
        h.complete_load(items);
        __syncthreads();
        partition.partition(
          storage.arms.buffered.arena.get_partition_scratch(), items, partial_items, value_source);
      }

      partition.epilogue();

      __syncthreads();
      detail::topk::merge_histogram<block_threads, num_buckets>(
        storage.arms.buffered.histogram, segment_histogram);
    }
    else
    {
      // Unbuffered scout pass: candidate count exceeded the per-segment candidate buffer
      // capacity. Mirrors the single-problem `agent_ub_t = AgentTopKHistogram` invocation in
      // the unbuffered branch -- we update only the per-segment histogram (with a candidate
      // filter applied) and let the next pass narrow the kth_key_bits further. The unbuffered
      // chain implies `load_from_candidates_buffer == false`, so we read directly from
      // `d_keys_in` and never touch the candidate buffer.
      using filter_op_t = detail::topk::topk_candidate_filter_op<IdentifyCandidatesOpT>;
      filter_op_t filter_op{identify_candidates_op};

      key_source_input_t key_src_input{d_keys_in, storage.arms.buffered.keys_source_state.a};
      key_source_buffer_t key_src_buffer{in_key_buf, storage.arms.buffered.keys_source_state.b};
      keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/false};

      detail::topk::init_histogram<block_threads, num_buckets>(storage.arms.buffered.histogram);
      __syncthreads();

      if (local_tile < num_full_tiles)
      {
        const OffsetT tile_base = static_cast<OffsetT>(local_tile) * static_cast<OffsetT>(tile_items);
        keys_source.set_tile_base(tile_base);

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch());
        h.complete_load(items);
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < items_per_thread; ++j)
        {
          if (filter_op(items[j]))
          {
            const int bucket = extract_bin_op(items[j]);
            atomicAdd(storage.arms.buffered.histogram + bucket, OffsetT{1});
          }
        }
      }
      else if (local_tile == num_full_tiles && partial_items > 0)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);
        keys_source.set_tile_base(tile_base);

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch(), partial_items);
        h.complete_load(items);
        const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
        const int num_thread_items =
          (thread_offset >= input_length_actual)
            ? 0
            : static_cast<int>((::cuda::std::min) (
                static_cast<OffsetT>(items_per_thread), input_length_actual - thread_offset));
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < items_per_thread; ++j)
        {
          if (j < num_thread_items && filter_op(items[j]))
          {
            const int bucket = extract_bin_op(items[j]);
            atomicAdd(storage.arms.buffered.histogram + bucket, OffsetT{1});
          }
        }
      }

      __syncthreads();
      detail::topk::merge_histogram<block_threads, num_buckets>(
        storage.arms.buffered.histogram, segment_histogram);
    }

    // Build per-segment closures inside `run()` so they can capture the per-segment counter
    // pointer (only knowable after the on-device binary search). These mirror the single-problem
    // `DeviceTopKFilterKernel`'s closures verbatim, with `counter` replaced by the
    // per-segment `segment_counter`. The unbuffered branch makes no counter changes in
    // `counter_update_fn` -- `num_candidates_in` stays at the previous pass's
    // `segment_num_items` and `load_from_candidates_buffer` stays `false`.
    auto counter_update_fn = [segment_counter, current_len, early_stop, will_buffer] {
      if (early_stop)
      {
        segment_counter->num_candidates_in = 0;
      }
      else if (will_buffer)
      {
        segment_counter->num_candidates_in           = current_len;
        segment_counter->load_from_candidates_buffer = true;
        segment_counter->num_candidates_written      = 0;
      }
    };
    auto on_kth_bucket =
      [segment_counter, pass](OutOffsetT current_k_cb, int bin_index, OffsetT num_selected, OffsetT num_candidates) {
        segment_counter->k                  = static_cast<OutOffsetT>(current_k_cb - num_selected);
        segment_counter->num_candidates_out = num_candidates;
        detail::topk::set_kth_key_bits<bits_per_pass>(
          segment_counter->kth_key_bits, pass, static_cast<unsigned int>(bin_index));
      };

    // Per-segment last-block epilogue. `expected_block_count` is the number of tiles in this
    // segment's input stream (computed above as `segment_tiles_input`). Both the buffered and
    // unbuffered branches updated the per-segment histogram, so both must run the kth-bucket
    // scan in the epilogue. Only the `early_stop` branch skips the histogram update and the
    // kth-bucket scan.
    auto epilogue_op = [this,
                        &counter_update_fn,
                        &on_kth_bucket,
                        current_k,
                        early_stop,
                        segment_histogram,
                        reset_histogram] {
      if (threadIdx.x == 0)
      {
        counter_update_fn();
      }
      if (!early_stop)
      {
        block_identify_kth_bucket_t{storage.arms.prefix_sum}.find_kth_bucket(
          segment_histogram, current_k, on_kth_bucket);
        if (reset_histogram)
        {
          detail::topk::init_histogram<block_threads, num_buckets>(segment_histogram);
        }
      }
    };

    detail::topk::finalize_pass(
      &segment_counter->finished_block_cnt, static_cast<unsigned int>(segment_tiles_input), epilogue_op);
  }
};

//---------------------------------------------------------------------
// agent_batched_topk_last_filter: segmented analog of `agent_topk_last_filter`.
//
// No histogram accumulation, no `finalize_pass`. Each block processes one tile of one segment;
// the partition primitive scatters surviving "selected" candidates to the front of
// `d_key_segments_out_it[segment_id]` via `p_num_selected_written` and ties (kth-class) to the
// back via a `back_grow_capped_reserve_op` (cap = `num_of_kth_needed`, anchor = `k_total`).
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename IdentifyCandidatesOpT,
          typename DecomposerT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename OffsetT,
          typename OutOffsetT,
          detail::topk::block_partition_strategy PartStrat =
            detail::topk::block_partition_strategy::atomics,
          bool LazyValueLoad   = false,
          bool InlinedClassify = false>
struct agent_batched_topk_last_filter
{
  using inner_key_it_t       = it_value_t<KeyInputItItT>;
  using inner_value_it_t     = it_value_t<ValueInputItItT>;
  using inner_value_out_it_t = it_value_t<ValueOutputItItT>;
  using inner_key_out_it_t   = it_value_t<KeyOutputItItT>;

  using key_in_t   = it_value_t<inner_key_it_t>;
  using value_in_t = it_value_t<inner_value_it_t>;
  using counter_t  = detail::topk::counter<key_in_t, OffsetT, OutOffsetT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<value_in_t, cub::NullType>;

  static constexpr bool effective_lazy_value_load = LazyValueLoad && !keys_only;

  using selected_offset_t  = OutOffsetT;
  using candidate_offset_t = OutOffsetT;

  using key_source_input_t = detail::topk::tile_data_source_t<
    inner_key_it_t,
    AgentTopKPolicyT::keys_tile_load_kind,
    block_threads,
    items_per_thread,
    OffsetT>;
  using key_source_buffer_t = detail::topk::tile_data_source_t<
    key_in_t*,
    AgentTopKPolicyT::keys_tile_load_kind,
    block_threads,
    items_per_thread,
    OffsetT>;
  using keys_source_t = detail::topk::multi_source_data_source<key_source_input_t, key_source_buffer_t, OffsetT>;

  using value_source_input_t =
    detail::topk::direct_data_source<inner_value_it_t, block_threads, items_per_thread, OffsetT>;
  using value_source_buffer_t =
    detail::topk::direct_data_source<value_in_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t =
    detail::topk::multi_source_data_source<value_source_input_t, value_source_buffer_t, OffsetT>;

  using val_out_t      = inner_value_out_it_t;
  using cand_val_out_t = inner_value_out_it_t;

  using value_channel_sinks_concrete_t = detail::topk::value_channel_sinks_t<
    val_out_t,
    cand_val_out_t,
    ::cuda::std::identity,
    ::cuda::std::identity>;
  using value_channel_sinks_or_null_t =
    ::cuda::std::conditional_t<keys_only, NullType, value_channel_sinks_concrete_t>;

  using agent_value_t = ::cuda::std::conditional_t<keys_only, NullType, value_in_t>;
  using agent_value_data_source_scratch_t =
    ::cuda::std::conditional_t<keys_only, NullType, typename value_source_t::ScratchStorage>;

  using selected_reserve_op_t  = detail::topk::atomic_reserve_range_op<selected_offset_t>;
  using candidate_reserve_op_t = detail::topk::back_grow_capped_reserve_op<candidate_offset_t>;

  using key_xform_t = ::cuda::std::identity;

  using partition_t = detail::topk::strategy_to_partition_class_t<
    PartStrat,
    block_threads,
    items_per_thread,
    AgentTopKPolicyT::accumulating_buffer_capacity,
    AgentTopKPolicyT::speculative_selected_buffer_capacity,
    key_in_t,
    selected_offset_t,
    candidate_offset_t,
    selected_reserve_op_t,
    candidate_reserve_op_t,
    key_xform_t,
    key_xform_t,
    inner_key_out_it_t,
    inner_key_out_it_t,
    IdentifyCandidatesOpT,
    detail::topk::topk_noop_candidate_callback_op,
    value_channel_sinks_or_null_t,
    agent_value_t,
    agent_value_data_source_scratch_t,
    effective_lazy_value_load,
    InlinedClassify>;

  struct empty_prefix_sum_t
  {};

  using storage_layout_t = detail::topk::partition_storage_layout_for_t<
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

  // -------------------------------------------------------------------------
  // Members -- per-launch state only
  // -------------------------------------------------------------------------
  _TempStorage& storage;
  KeyInputItItT d_key_segments_it;
  KeyOutputItItT d_key_segments_out_it;
  ValueInputItItT d_value_segments_it;
  ValueOutputItItT d_value_segments_out_it;
  SegmentSizeParameterT segment_sizes;
  KParameterT k_param;
  NumSegmentsParameterT num_segments;
  SegmentIdProviderT segment_id_provider;
  const LargeSegmentTileOffsetT* d_large_segments_tile_offsets;
  counter_t* d_segment_counters;
  key_in_t* d_segment_in_key_buf;
  value_in_t* d_segment_in_val_buf;
  int pass;
  int total_bits;
  DecomposerT decomposer;
  OffsetT candidate_buffer_length;
  // (See `agent_batched_topk_histogram` for the rationale behind dropping `total_large_tiles`.)
  typename NumSegmentsParameterT::value_type num_large_segments;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_batched_topk_last_filter(
    TempStorage& ts,
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    NumSegmentsParameterT num_segments,
    SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* d_large_segments_tile_offsets,
    counter_t* d_segment_counters,
    key_in_t* d_segment_in_key_buf,
    value_in_t* d_segment_in_val_buf,
    int pass,
    int total_bits,
    DecomposerT decomposer,
    OffsetT candidate_buffer_length,
    typename NumSegmentsParameterT::value_type num_large_segments)
      : storage(ts.Alias())
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , k_param(k_param)
      , num_segments(num_segments)
      , segment_id_provider(segment_id_provider)
      , d_large_segments_tile_offsets(d_large_segments_tile_offsets)
      , d_segment_counters(d_segment_counters)
      , d_segment_in_key_buf(d_segment_in_key_buf)
      , d_segment_in_val_buf(d_segment_in_val_buf)
      , pass(pass)
      , total_bits(total_bits)
      , decomposer(decomposer)
      , candidate_buffer_length(candidate_buffer_length)
      , num_large_segments(num_large_segments)
  {}

  // `global_tile_id` is supplied by the calling kernel's grid-stride loop; see the matching
  // comment on `agent_batched_topk_histogram::run()` for why the agent processes one tile per
  // call and the kernel iterates. The last-filter pass has no histogram and no `finalize_pass`,
  // so each tile's work is fully independent of the others within a segment.
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(LargeSegmentTileOffsetT global_tile_id)
  {
    // See the matching comment on `agent_batched_topk_histogram::run()`: only lane 0
    // runs the binary search; `__shfl_sync` broadcasts the result to the whole warp.
    LargeSegmentTileOffsetT queue_idx_lane0 = 0;
    if ((threadIdx.x & 31) == 0)
    {
      queue_idx_lane0 = UpperBound(d_large_segments_tile_offsets, num_large_segments, global_tile_id) - 1;
    }
    const LargeSegmentTileOffsetT queue_idx  = __shfl_sync(0xffffffff, queue_idx_lane0, 0);
    const LargeSegmentTileOffsetT slab_base  = d_large_segments_tile_offsets[queue_idx];
    const LargeSegmentTileOffsetT local_tile = global_tile_id - slab_base;
    const auto segment_id                    = segment_id_provider[queue_idx];

    auto d_keys_in    = d_key_segments_it[segment_id];
    auto d_keys_out   = d_key_segments_out_it[segment_id];
    [[maybe_unused]] auto d_values_in = [&] {
      if constexpr (!keys_only)
      {
        return d_value_segments_it[segment_id];
      }
      else
      {
        return inner_value_it_t{};
      }
    }();
    [[maybe_unused]] auto d_values_out = [&] {
      if constexpr (!keys_only)
      {
        return d_value_segments_out_it[segment_id];
      }
      else
      {
        return inner_value_out_it_t{};
      }
    }();

    counter_t* segment_counter             = d_segment_counters + queue_idx;
    const OffsetT input_length             = segment_counter->num_candidates_in;
    const bool load_from_candidates_buffer = segment_counter->load_from_candidates_buffer;

    if (input_length == 0)
    {
      return;
    }

    // Construct the per-segment `identify_candidates_op` with the per-segment counter's
    // `kth_key_bits` pointer. Same rationale as `agent_batched_topk_filter_partition::run()`.
    IdentifyCandidatesOpT identify_candidates_op{&segment_counter->kth_key_bits, pass, total_bits, decomposer};

    // Mirrors the histogram agent's clip: when `k > segment_size`, all items in the segment are
    // in the top-k. `reserve_cand` sizes the per-segment output reservation from `k_total`, so we
    // must use the same clipped value here to keep the candidate reservation in lock-step with
    // what the prior multi-CTA passes actually placed in the segment counter.
    const OffsetT segment_num_items = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
    const OutOffsetT k_total = (::cuda::std::min) (
      static_cast<OutOffsetT>(k_param.get_param(segment_id)), static_cast<OutOffsetT>(segment_num_items));
    const OutOffsetT num_of_kth_needed  = static_cast<OutOffsetT>(segment_counter->k);

    key_in_t* in_key_buf = d_segment_in_key_buf + queue_idx * candidate_buffer_length;
    value_in_t* in_val_buf = nullptr;
    if constexpr (!keys_only)
    {
      in_val_buf = load_from_candidates_buffer ? (d_segment_in_val_buf + queue_idx * candidate_buffer_length) : nullptr;
    }

    const OffsetT segment_tiles_input =
      static_cast<OffsetT>(::cuda::ceil_div(input_length, OffsetT{tile_items}));
    if (static_cast<OffsetT>(local_tile) >= segment_tiles_input)
    {
      return;
    }

    key_source_input_t key_src_input{d_keys_in, storage.keys_source_state.a};
    key_source_buffer_t key_src_buffer{in_key_buf, storage.keys_source_state.b};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/load_from_candidates_buffer};

    selected_reserve_op_t reserve_sel{&segment_counter->num_selected_written};
    candidate_reserve_op_t reserve_cand{
      &segment_counter->num_ties_written_to_back,
      static_cast<candidate_offset_t>(k_total),
      static_cast<candidate_offset_t>(num_of_kth_needed)};
    key_xform_t sel_key_xform{};
    key_xform_t cand_key_xform{};

    auto value_channel_sinks = [&] {
      if constexpr (keys_only)
      {
        return NullType{};
      }
      else
      {
        return value_channel_sinks_concrete_t{
          d_values_out, d_values_out, ::cuda::std::identity{}, ::cuda::std::identity{}};
      }
    }();
    detail::topk::topk_noop_candidate_callback_op callback_op{};

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

    const OffsetT num_full_tiles = input_length / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items  = input_length - num_full_tiles * static_cast<OffsetT>(tile_items);

    if (local_tile < num_full_tiles)
    {
      const OffsetT tile_base = static_cast<OffsetT>(local_tile) * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);

      auto value_source = [&] {
        if constexpr (keys_only)
        {
          return NullType{};
        }
        else
        {
          typename value_source_input_t::TempStorage val_state_input{};
          typename value_source_buffer_t::TempStorage val_state_buffer{};
          value_source_input_t val_input{d_values_in, val_state_input};
          value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};
          value_source_t val_src{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
          val_src.set_tile_base(tile_base);
          return val_src;
        }
      }();

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch());
      h.complete_load(items);
      __syncthreads();
      partition.partition(storage.partition_arena.get_partition_scratch(), items, value_source);
    }
    else if (local_tile == num_full_tiles && partial_items > 0)
    {
      const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);

      auto value_source = [&] {
        if constexpr (keys_only)
        {
          return NullType{};
        }
        else
        {
          typename value_source_input_t::TempStorage val_state_input{};
          typename value_source_buffer_t::TempStorage val_state_buffer{};
          value_source_input_t val_input{d_values_in, val_state_input};
          value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};
          value_source_t val_src{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
          val_src.set_tile_base(tile_base);
          return val_src;
        }
      }();

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch(), partial_items);
      h.complete_load(items);
      __syncthreads();
      partition.partition(storage.partition_arena.get_partition_scratch(), items, partial_items, value_source);
    }

    partition.epilogue();
  }
};

} // namespace detail::batched_topk
CUB_NAMESPACE_END
