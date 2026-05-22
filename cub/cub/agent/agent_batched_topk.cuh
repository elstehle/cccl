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
          typename LargeSegmentsCountItT,
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

  // Iterator yielding the number of enqueued large segments (queue slots) when dereferenced.
  // Stored as the iterator (a kernel parameter) rather than the dereferenced scalar so the
  // agent matches the kernel's parameter shape one-for-one. The sentinel-slot read for
  // `total_large_tiles` is `d_large_segments_tile_offsets[*large_segments_count_it]` and the
  // `UpperBound` upper bound is `*large_segments_count_it` -- both deferred to use sites
  // inside the agent body.
  LargeSegmentsCountItT large_segments_count_it;

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
    LargeSegmentsCountItT large_segments_count_it,
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
      , large_segments_count_it(large_segments_count_it)
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
      queue_idx_lane0 = UpperBound(d_large_segments_tile_offsets, *large_segments_count_it, global_tile_id) - 1;
    }
    return __shfl_sync(0xffffffff, queue_idx_lane0, 0);
  }

public:
  // Drive this CTA's entire grid-strided histogram pass. The agent walks chunks of
  // `tiles_per_chunk` consecutive tiles in queue_idx-space (chunks are `gridDim.x *
  // tiles_per_chunk` apart in the global tile sequence) and maintains per-segment cached
  // state *across* chunks. As long as consecutive chunks of this CTA fall in the same
  // segment (the common case for the single-large-segment dispatch workload), the agent
  // does **one** `init_histogram` at the start of the CTA's work, accumulates into smem
  // across every chunk's tiles, and runs **one** `merge_histogram` into the per-segment
  // global slab when this CTA crosses out of the segment (or when the grid-stride loop
  // ends). For workloads with multiple segments, the init/merge pair runs once per
  // (CTA, segment-stretch) pair, where a stretch is a contiguous run of tiles this CTA
  // owns inside the same segment.
  //
  // Per-segment cache (alive across the outer chunk loop):
  //   - `active_queue_idx`        : the segment being accumulated into ("no active" sentinel = -1)
  //   - `active_segment_end`      : the segment's half-open upper bound in queue_idx-space
  //                                 (`slab_base + segment_tile_count`); the cheap check
  //                                 `chunk_cursor < active_segment_end` is what decides whether
  //                                 the cache still applies for the next chunk's tiles
  //   - other per-segment derivations (`d_keys_in`, `slab_base`, `num_full_tiles`,
  //     `partial_items`, `num_items`, `segment_histogram`)
  //
  // The per-segment finalize epilogue (prefix-sum + bucket-finder + counter update + optional
  // global histogram reset) is owned by `device_segmented_topk_finalize_histogram_kernel`, run
  // by the dispatch after this kernel completes. The radix-digit extraction logic (`pass`,
  // `total_bits`, `decomposer`) is absorbed into `extract_bin_op`, constructed host-side by
  // the dispatch and passed as the agent's `ExtractBinOpT` member -- the agent itself does
  // not depend on the pass index.
  // The agent derives `total_large_tiles` from its own members
  // (`d_large_segments_tile_offsets[*large_segments_count_it]`) at use sites. The kernel no
  // longer pre-resolves either `num_large_segments` or the sentinel-slot pointer; only the
  // raw kernel parameters (the `large_segments_count_it` iterator + the tile-offsets array)
  // flow into the agent.
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(int tiles_per_chunk)
  {
    // Pointer to the sentinel slot of the per-segment tile-offset table. Computed once at
    // entry from the agent's own members so the inner-loop bound checks below can dereference
    // a single pointer rather than re-deriving the address every iteration.
    const LargeSegmentTileOffsetT* const d_total_large_tiles =
      &d_large_segments_tile_offsets[*large_segments_count_it];
    // Sentinel meaning "no segment loaded yet". `active_queue_idx` is set to a real value the
    // first time this CTA touches a tile, and never reverts to the sentinel.
    constexpr LargeSegmentTileOffsetT kNoActiveSegment = static_cast<LargeSegmentTileOffsetT>(-1);
    LargeSegmentTileOffsetT active_queue_idx           = kNoActiveSegment;
    LargeSegmentTileOffsetT active_segment_end         = 0;
    LargeSegmentTileOffsetT active_slab_base           = 0;
    inner_key_it_t active_d_keys_in{};
    OffsetT* active_segment_histogram                  = nullptr;
    OffsetT active_num_items                           = 0;
    OffsetT active_num_full_tiles                      = 0;
    OffsetT active_partial_items                       = 0;

    const LargeSegmentTileOffsetT chunk_size_v = static_cast<LargeSegmentTileOffsetT>(tiles_per_chunk);
    const LargeSegmentTileOffsetT stride       = static_cast<LargeSegmentTileOffsetT>(gridDim.x) * chunk_size_v;

    for (LargeSegmentTileOffsetT chunk_start = static_cast<LargeSegmentTileOffsetT>(blockIdx.x) * chunk_size_v;
         chunk_start < *d_total_large_tiles;
         chunk_start += stride)
    {
      // `total_large_tiles_local` is the value seen by this iteration -- the loop condition above
      // already loaded it once for the bound check, and the chunk-end computation below needs the
      // same value. Naming the load explicitly here makes ptxas's CSE within the iteration body
      // straightforward; the upper-bound comparison uses the freshly-loaded `*d_total_large_tiles`
      // so the loop guard is self-contained.
      const LargeSegmentTileOffsetT total_large_tiles_local = *d_total_large_tiles;
      const LargeSegmentTileOffsetT chunk_end =
        (chunk_start + chunk_size_v < total_large_tiles_local) ? chunk_start + chunk_size_v : total_large_tiles_local;

      LargeSegmentTileOffsetT chunk_cursor = chunk_start;
      while (chunk_cursor < chunk_end)
      {
        // ----- Segment-state refresh, only when the cached segment doesn't cover chunk_cursor.
        // For consecutive chunks of the same segment (the common case in the single-segment
        // workload that drives the multi-CTA dispatch path), this branch is taken **once** per
        // CTA total -- right at the start of the first chunk this CTA owns. After that, the
        // condition `chunk_cursor < active_segment_end` holds for every subsequent iteration
        // until the CTA stops grid-striding or crosses a segment boundary.
        if (active_queue_idx == kNoActiveSegment || chunk_cursor >= active_segment_end)
        {
          // Flush the previously-active segment's smem histogram into its global slab.
          if (active_queue_idx != kNoActiveSegment)
          {
            __syncthreads();
            detail::topk::merge_histogram<block_threads, num_buckets>(
              temp_storage.histogram, active_segment_histogram);
          }

          // Resolve the new segment for `chunk_cursor` via lane-0 + `__shfl_sync` broadcast.
          const LargeSegmentTileOffsetT queue_idx = resolve_queue_idx(chunk_cursor);
          const auto segment_id                   = segment_id_provider[queue_idx];
          active_queue_idx                        = queue_idx;
          active_d_keys_in                        = d_key_segments_it[segment_id];
          active_num_items                        = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
          active_segment_histogram                = d_segment_histograms + queue_idx * num_buckets;
          active_slab_base                        = d_large_segments_tile_offsets[queue_idx];
          active_num_full_tiles                   = active_num_items / static_cast<OffsetT>(tile_items);
          active_partial_items =
            active_num_items - active_num_full_tiles * static_cast<OffsetT>(tile_items);
          // Half-open upper bound in queue_idx-space: full tiles + 1 partial tile (if any).
          const OffsetT seg_tile_count =
            active_num_full_tiles + (active_partial_items > 0 ? OffsetT{1} : OffsetT{0});
          active_segment_end = active_slab_base + static_cast<LargeSegmentTileOffsetT>(seg_tile_count);

          __syncthreads();
          detail::topk::init_histogram<block_threads, num_buckets>(temp_storage.histogram);
          __syncthreads();
        }

        // ----- Tile-space accounting using the cached segment state. -----------------------
        const OffsetT local_tile_start = static_cast<OffsetT>(chunk_cursor - active_slab_base);
        const LargeSegmentTileOffsetT remaining_in_chunk = chunk_end - chunk_cursor;
        const OffsetT full_tiles_remaining_in_seg =
          (local_tile_start < active_num_full_tiles) ? (active_num_full_tiles - local_tile_start) : OffsetT{0};
        const OffsetT full_tiles_to_process =
          (::cuda::std::min) (static_cast<OffsetT>(remaining_in_chunk), full_tiles_remaining_in_seg);

        // At-most-one trailing partial tile, claimed iff the loop ends at the segment's
        // partial-tile slot AND the segment has a partial AND chunk budget remains.
        const OffsetT next_local_tile = local_tile_start + full_tiles_to_process;
        const bool process_partial    = (next_local_tile == active_num_full_tiles) && (active_partial_items > 0)
                                  && (full_tiles_to_process + OffsetT{1} <= static_cast<OffsetT>(remaining_in_chunk));

        // The keys-source view captures the segment's input iterator + its smem state. State
        // lives in smem and persists across the segment-stretch; we construct a fresh view-
        // object on segment-state refresh (when iterator may have changed) and reuse it across
        // same-segment iterations.
        keys_source_t keys_source{active_d_keys_in, temp_storage.keys_source_state};

        // ----- Full-tile loop -- the tight hot path. ----------------------------------------
        for (OffsetT i = 0; i < full_tiles_to_process; ++i)
        {
          const OffsetT tile_base = (local_tile_start + i) * static_cast<OffsetT>(tile_items);
          keys_source.set_tile_base(tile_base);

          __syncthreads();
          key_in_t items[items_per_thread];
          auto h = keys_source.submit_load(temp_storage.keys_source_scratch);
          h.complete_load(items);
          process_tile_full(items);
        }

        // ----- At-most-one trailing partial tile. -------------------------------------------
        if (process_partial)
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
              : static_cast<int>(
                  (::cuda::std::min) (static_cast<OffsetT>(items_per_thread), active_num_items - thread_offset));
          process_tile_partial(items, num_thread_items);
        }

        const OffsetT tiles_consumed = full_tiles_to_process + (process_partial ? OffsetT{1} : OffsetT{0});
        // Defensive: `tiles_consumed == 0` should be unreachable -- the offset table is sized
        // by ceil-div'ing `num_items` per segment, so `local_tile_start` is always in
        // `[0, num_full_tiles + (partial_items > 0 ? 1 : 0))`. If an off-by-one in the offsets
        // ever produces a stretch with no work, break rather than spin.
        if (tiles_consumed == OffsetT{0})
        {
          break;
        }
        chunk_cursor += static_cast<LargeSegmentTileOffsetT>(tiles_consumed);
      }
    }

    // Final flush: merge the last active segment-stretch's smem histogram into its global
    // slab. Skipped only when this CTA had no work at all (empty grid-stride iterator).
    if (active_queue_idx != kNoActiveSegment)
    {
      __syncthreads();
      detail::topk::merge_histogram<block_threads, num_buckets>(
        temp_storage.histogram, active_segment_histogram);
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

  // The filter agent no longer carries `block_identify_kth_bucket_t` (the per-segment
  // prefix-sum + kth-bucket scan that drives the next pass's counter state). That work has
  // been hoisted into a dedicated `device_segmented_topk_finalize_filter_kernel` that runs
  // after the filter kernel finishes, removing the per-tile `finalize_pass` cost from this
  // agent.

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

  // Same `empty_prefix_sum_t` placeholder pattern as the single-problem agent.
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

  // The prefix-sum scratch that used to sit in this union (the `prefix_sum` arm aliased with
  // `buffered`/`early_stop`) has been removed; the per-segment prefix-sum + kth-bucket scan
  // now lives in `device_segmented_topk_finalize_filter_kernel`. Smem here is just the
  // per-mode arms used during the tile body itself.
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

private:
  // Per-segment derived state cached across tiles of the same segment within a chunk. Same
  // pattern as the histogram agent's per-segment cache: re-derived only when the chunk crosses
  // a segment boundary; held in registers across same-segment tiles.
  struct per_segment_state_t
  {
    bool empty;       // counter_input_length == 0 -> all tiles of this segment are no-ops
    bool early_stop;  // current_len == current_k
    bool will_buffer; // !early_stop && fits-in-back-buffer && cost-justified
    bool load_from_candidates_buffer;

    inner_key_it_t d_keys_in{};
    inner_key_out_it_t d_keys_out{};
    [[maybe_unused]] inner_value_it_t d_values_in{};
    [[maybe_unused]] inner_value_out_it_t d_values_out{};

    key_in_t* in_key_buf;
    key_in_t* out_key_buf;
    [[maybe_unused]] value_in_t* in_val_buf;
    [[maybe_unused]] value_in_t* out_val_buf;

    OffsetT* segment_histogram;
    counter_t* segment_counter;

    // `identify_candidates_op_t` (for some `T`) has no default ctor, so we cache the inputs to
    // its ctor here and construct it on demand in the per-mode tile body. The ctor is cheap
    // (a `key_prefix_storage_t*` copy + an `int` shift).
    int pass;

    OutOffsetT current_k;
    OffsetT current_len;
    OffsetT input_length_actual;
    OffsetT num_full_tiles;
    OffsetT partial_items;
    OffsetT segment_tiles_input;
    LargeSegmentTileOffsetT slab_base;
  };

  // Lane-0 + `__shfl_sync` `UpperBound`, same idiom as the histogram agent.
  _CCCL_DEVICE _CCCL_FORCEINLINE LargeSegmentTileOffsetT resolve_queue_idx(LargeSegmentTileOffsetT global_tile_id)
  {
    LargeSegmentTileOffsetT queue_idx_lane0 = 0;
    if ((threadIdx.x & 31) == 0)
    {
      queue_idx_lane0 = UpperBound(d_large_segments_tile_offsets, num_large_segments, global_tile_id) - 1;
    }
    return __shfl_sync(0xffffffff, queue_idx_lane0, 0);
  }

  // Build the per-segment cached state for `queue_idx`. Pure function of `queue_idx` and the
  // per-launch agent state. Same logic as the prologue of the pre-refactor `run()`.
  _CCCL_DEVICE _CCCL_FORCEINLINE per_segment_state_t resolve_segment_state(LargeSegmentTileOffsetT queue_idx, int pass)
  {
    per_segment_state_t s{};
    s.slab_base       = d_large_segments_tile_offsets[queue_idx];
    const auto segment_id = segment_id_provider[queue_idx];

    s.d_keys_in  = d_key_segments_it[segment_id];
    s.d_keys_out = d_key_segments_out_it[segment_id];
    if constexpr (!keys_only)
    {
      s.d_values_in  = d_value_segments_it[segment_id];
      s.d_values_out = d_value_segments_out_it[segment_id];
    }

    s.segment_counter   = d_segment_counters + queue_idx;
    s.segment_histogram = d_segment_histograms + queue_idx * num_buckets;

    s.current_k                        = s.segment_counter->k;
    s.current_len                      = s.segment_counter->num_candidates_out;
    const OffsetT counter_input_length = s.segment_counter->num_candidates_in;
    s.load_from_candidates_buffer      = s.segment_counter->load_from_candidates_buffer;

    s.pass = pass;

    s.empty = (counter_input_length == 0);
    if (s.empty)
    {
      return s;
    }

    const OffsetT segment_num_items = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
    s.input_length_actual           = counter_input_length;

    s.early_stop  = (s.current_len == static_cast<OffsetT>(s.current_k));
    s.will_buffer = !s.early_stop && (s.current_len <= candidate_buffer_length)
                 && (s.current_len <= segment_num_items / candidate_buffer_coefficient);

    s.in_key_buf  = d_segment_in_key_buf + queue_idx * candidate_buffer_length;
    s.out_key_buf = s.will_buffer ? (d_segment_out_key_buf + queue_idx * candidate_buffer_length) : nullptr;
    if constexpr (!keys_only)
    {
      s.in_val_buf =
        s.load_from_candidates_buffer ? (d_segment_in_val_buf + queue_idx * candidate_buffer_length) : nullptr;
      s.out_val_buf = s.will_buffer ? (d_segment_out_val_buf + queue_idx * candidate_buffer_length) : nullptr;
    }

    s.num_full_tiles = s.input_length_actual / static_cast<OffsetT>(tile_items);
    s.partial_items  = s.input_length_actual - s.num_full_tiles * static_cast<OffsetT>(tile_items);
    s.segment_tiles_input =
      static_cast<OffsetT>(::cuda::ceil_div(s.input_length_actual, OffsetT{tile_items}));
    return s;
  }

  // Per-mode tile bodies. Each takes the per-segment cached state and a tile-local index, runs
  // exactly the same code the pre-refactor `run()` ran inside its `if (early_stop) {} else if
  // (will_buffer) {} else {}` branches, minus the surrounding init / merge / finalize_pass --
  // those are managed by `process_chunk` (init/merge) and the finalize kernel (finalize_pass).

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  process_tile_early_stop(const per_segment_state_t& s, LargeSegmentTileOffsetT local_tile)
  {
    key_source_input_t key_src_input{s.d_keys_in, storage.arms.early_stop.keys_source_state.a};
    key_source_buffer_t key_src_buffer{s.in_key_buf, storage.arms.early_stop.keys_source_state.b};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/s.load_from_candidates_buffer};

    // The filter primitive's ctor takes the identify-selected op by non-const reference,
    // so we build a local op (its ctor is cheap: a `key_prefix_storage_t*` copy + an
    // `int` shift) from the cached per-segment fields.
    IdentifyCandidatesOpT identify_candidates_op{
      &s.segment_counter->kth_key_bits, s.pass, total_bits, decomposer};
    identify_selected_op_t identify_selected{identify_candidates_op};
    auto value_channel_sinks = make_early_stop_value_channel_sinks(s.d_values_out);

    selected_reserve_op_t reserve_sel{&s.segment_counter->num_selected_written};
    key_xform_t sel_key_xform{};

    early_stop_filter_t filter{
      storage.arms.early_stop.arena.get_partition_state(),
      reserve_sel,
      sel_key_xform,
      s.d_keys_out,
      value_channel_sinks,
      identify_selected};

    if (local_tile < s.num_full_tiles)
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
          value_source_input_t val_input{s.d_values_in, val_state_input};
          value_source_buffer_t val_buffer{s.in_val_buf, val_state_buffer};
          value_source_t val_src{val_input, val_buffer, /*pick_b=*/s.load_from_candidates_buffer};
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
    else if (local_tile == s.num_full_tiles && s.partial_items > 0)
    {
      const OffsetT tile_base = s.num_full_tiles * static_cast<OffsetT>(tile_items);
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
          value_source_input_t val_input{s.d_values_in, val_state_input};
          value_source_buffer_t val_buffer{s.in_val_buf, val_state_buffer};
          value_source_t val_src{val_input, val_buffer, /*pick_b=*/s.load_from_candidates_buffer};
          val_src.set_tile_base(tile_base);
          return val_src;
        }
      }();

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.early_stop.arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      __syncthreads();
      filter.partition(storage.arms.early_stop.arena.get_partition_scratch(), items, s.partial_items, value_source);
    }

    filter.epilogue();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  process_tile_buffered(const per_segment_state_t& s, LargeSegmentTileOffsetT local_tile)
  {
    key_source_input_t key_src_input{s.d_keys_in, storage.arms.buffered.keys_source_state.a};
    key_source_buffer_t key_src_buffer{s.in_key_buf, storage.arms.buffered.keys_source_state.b};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/s.load_from_candidates_buffer};

    selected_reserve_op_t reserve_sel{&s.segment_counter->num_selected_written};
    candidate_reserve_op_t reserve_cand{&s.segment_counter->num_candidates_written};
    key_xform_t sel_key_xform{};
    key_xform_t cand_key_xform{};

    buffered_cand_key_out_t cand_key_out = s.out_key_buf;
    [[maybe_unused]] buffered_cand_val_out_t cand_val_out = s.out_val_buf;
    histogram_callback_op_t histogram_cb{extract_bin_op, storage.arms.buffered.histogram};
    auto value_channel_sinks = make_buffered_value_channel_sinks(s.d_values_out, cand_val_out);

    // The partition primitive's ctor takes `IdentifyCandidatesOp&` (non-const); build a
    // local op (cheap ctor) from the cached per-segment fields.
    IdentifyCandidatesOpT identify_candidates_op{
      &s.segment_counter->kth_key_bits, s.pass, total_bits, decomposer};

    buffered_partition_t partition{
      storage.arms.buffered.arena.get_partition_state(),
      reserve_sel,
      reserve_cand,
      sel_key_xform,
      cand_key_xform,
      s.d_keys_out,
      cand_key_out,
      value_channel_sinks,
      identify_candidates_op,
      histogram_cb};

    if (local_tile < s.num_full_tiles)
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
          value_source_input_t val_input{s.d_values_in, val_state_input};
          value_source_buffer_t val_buffer{s.in_val_buf, val_state_buffer};
          value_source_t val_src{val_input, val_buffer, /*pick_b=*/s.load_from_candidates_buffer};
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
    else if (local_tile == s.num_full_tiles && s.partial_items > 0)
    {
      const OffsetT tile_base = s.num_full_tiles * static_cast<OffsetT>(tile_items);
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
          value_source_input_t val_input{s.d_values_in, val_state_input};
          value_source_buffer_t val_buffer{s.in_val_buf, val_state_buffer};
          value_source_t val_src{val_input, val_buffer, /*pick_b=*/s.load_from_candidates_buffer};
          val_src.set_tile_base(tile_base);
          return val_src;
        }
      }();

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      __syncthreads();
      partition.partition(
        storage.arms.buffered.arena.get_partition_scratch(), items, s.partial_items, value_source);
    }

    partition.epilogue();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  process_tile_unbuffered(const per_segment_state_t& s, LargeSegmentTileOffsetT local_tile)
  {
    using filter_op_t = detail::topk::topk_candidate_filter_op<IdentifyCandidatesOpT>;
    IdentifyCandidatesOpT identify_candidates_op{
      &s.segment_counter->kth_key_bits, s.pass, total_bits, decomposer};
    filter_op_t filter_op{identify_candidates_op};

    key_source_input_t key_src_input{s.d_keys_in, storage.arms.buffered.keys_source_state.a};
    key_source_buffer_t key_src_buffer{s.in_key_buf, storage.arms.buffered.keys_source_state.b};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/false};

    if (local_tile < s.num_full_tiles)
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
    else if (local_tile == s.num_full_tiles && s.partial_items > 0)
    {
      const OffsetT tile_base = s.num_full_tiles * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
      const int num_thread_items =
        (thread_offset >= s.input_length_actual)
          ? 0
          : static_cast<int>((::cuda::std::min) (
            static_cast<OffsetT>(items_per_thread), s.input_length_actual - thread_offset));
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
  }

public:
  // Process up to `tiles_per_chunk` consecutive tiles starting at `chunk_start`, grouped by
  // segment -- the same chunk-grouping shape as the histogram agent. Each segment encountered
  // in the chunk dispatches to one of three per-mode tile bodies (early_stop / buffered /
  // unbuffered); the smem histogram (only used by the two non-early_stop modes) is initialised
  // once at segment entry and merged into the per-segment global slab once at segment exit,
  // amortising the init / merge across all same-segment tiles in the chunk.
  //
  // `pass` is only used to construct the per-segment `identify_candidates_op` (which captures
  // the segment counter's `kth_key_bits` pointer); the kernel-side grid-stride loop forwards it
  // unchanged from the dispatch.
  //
  // The per-segment epilogue (counter update + prefix-sum + bucket-finder + optional histogram
  // reset) lives in `device_segmented_topk_finalize_filter_kernel` and runs after the filter
  // kernel finishes on the same stream. This agent does not run `finalize_pass`.
  // See the matching comment on `agent_batched_topk_histogram::run` for why
  // `d_total_large_tiles` is a pointer rather than a value.
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_chunk(
    LargeSegmentTileOffsetT chunk_start,
    int tiles_per_chunk,
    const LargeSegmentTileOffsetT* d_total_large_tiles,
    int pass)
  {
    constexpr LargeSegmentTileOffsetT kNoActiveSegment = static_cast<LargeSegmentTileOffsetT>(-1);
    LargeSegmentTileOffsetT active_queue_idx           = kNoActiveSegment;
    per_segment_state_t state{};

    // Whether the smem histogram has been initialised for the currently-active segment. Both
    // the buffered and unbuffered modes use it; early_stop does not touch it. Tracked
    // separately from `state.empty` so we don't flush a never-initialised histogram into
    // global on segment changes (which would be a wasted scan-loop over the buckets).
    bool smem_histogram_active = false;

    _CCCL_PRAGMA_NOUNROLL()
    for (int i = 0; i < tiles_per_chunk; ++i)
    {
      const LargeSegmentTileOffsetT global_tile_id = chunk_start + static_cast<LargeSegmentTileOffsetT>(i);
      if (global_tile_id >= *d_total_large_tiles)
      {
        break;
      }

      const LargeSegmentTileOffsetT queue_idx = resolve_queue_idx(global_tile_id);

      if (queue_idx != active_queue_idx)
      {
        // Flush the previous segment's smem histogram into its global slab (if it was active).
        if (smem_histogram_active)
        {
          __syncthreads();
          detail::topk::merge_histogram<block_threads, num_buckets>(
            storage.arms.buffered.histogram, state.segment_histogram);
          smem_histogram_active = false;
        }

        // Refresh per-segment caches for the new segment.
        active_queue_idx = queue_idx;
        state            = resolve_segment_state(queue_idx, pass);

        // Init smem histogram for the new segment if the mode uses it (buffered/unbuffered).
        if (!state.empty && !state.early_stop)
        {
          __syncthreads();
          detail::topk::init_histogram<block_threads, num_buckets>(storage.arms.buffered.histogram);
          smem_histogram_active = true;
        }
        __syncthreads();
      }

      if (state.empty)
      {
        continue;
      }
      const LargeSegmentTileOffsetT local_tile = global_tile_id - state.slab_base;
      if (static_cast<OffsetT>(local_tile) >= state.segment_tiles_input)
      {
        continue;
      }

      if (state.early_stop)
      {
        process_tile_early_stop(state, local_tile);
      }
      else if (state.will_buffer)
      {
        process_tile_buffered(state, local_tile);
      }
      else
      {
        process_tile_unbuffered(state, local_tile);
      }
    }

    // Flush the final segment's smem histogram into its global slab.
    if (smem_histogram_active)
    {
      __syncthreads();
      detail::topk::merge_histogram<block_threads, num_buckets>(
        storage.arms.buffered.histogram, state.segment_histogram);
    }
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

private:
  // Per-segment cached state, mirrors the filter agent's pattern. Re-derived only when the
  // chunk crosses a segment boundary; held in registers across same-segment tiles.
  struct per_segment_state_t
  {
    bool empty;
    bool load_from_candidates_buffer;

    inner_key_it_t d_keys_in{};
    inner_key_out_it_t d_keys_out{};
    [[maybe_unused]] inner_value_it_t d_values_in{};
    [[maybe_unused]] inner_value_out_it_t d_values_out{};

    counter_t* segment_counter;
    key_in_t* in_key_buf;
    [[maybe_unused]] value_in_t* in_val_buf;

    // `identify_candidates_op_t` has no default ctor for some `T`; cache ctor inputs and
    // build the op on demand in `process_tile`.
    int pass;

    OutOffsetT k_total;
    OutOffsetT num_of_kth_needed;
    OffsetT input_length;
    OffsetT num_full_tiles;
    OffsetT partial_items;
    OffsetT segment_tiles_input;
    LargeSegmentTileOffsetT slab_base;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE LargeSegmentTileOffsetT resolve_queue_idx(LargeSegmentTileOffsetT global_tile_id)
  {
    LargeSegmentTileOffsetT queue_idx_lane0 = 0;
    if ((threadIdx.x & 31) == 0)
    {
      queue_idx_lane0 = UpperBound(d_large_segments_tile_offsets, num_large_segments, global_tile_id) - 1;
    }
    return __shfl_sync(0xffffffff, queue_idx_lane0, 0);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE per_segment_state_t resolve_segment_state(LargeSegmentTileOffsetT queue_idx)
  {
    per_segment_state_t s{};
    s.slab_base       = d_large_segments_tile_offsets[queue_idx];
    const auto segment_id = segment_id_provider[queue_idx];

    s.d_keys_in  = d_key_segments_it[segment_id];
    s.d_keys_out = d_key_segments_out_it[segment_id];
    if constexpr (!keys_only)
    {
      s.d_values_in  = d_value_segments_it[segment_id];
      s.d_values_out = d_value_segments_out_it[segment_id];
    }

    s.segment_counter = d_segment_counters + queue_idx;
    s.input_length    = s.segment_counter->num_candidates_in;
    s.load_from_candidates_buffer = s.segment_counter->load_from_candidates_buffer;
    s.pass            = pass;

    s.empty = (s.input_length == 0);
    if (s.empty)
    {
      return s;
    }

    // Mirrors the histogram agent's clip: when `k > segment_size`, all items are in the top-k.
    // `reserve_cand` is sized from `k_total`; keeping it in lock-step with the prior passes'
    // counter writes is required for correct per-segment output reservation.
    const OffsetT segment_num_items = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
    s.k_total = (::cuda::std::min) (
      static_cast<OutOffsetT>(k_param.get_param(segment_id)), static_cast<OutOffsetT>(segment_num_items));
    s.num_of_kth_needed = static_cast<OutOffsetT>(s.segment_counter->k);

    s.in_key_buf = d_segment_in_key_buf + queue_idx * candidate_buffer_length;
    if constexpr (!keys_only)
    {
      s.in_val_buf =
        s.load_from_candidates_buffer ? (d_segment_in_val_buf + queue_idx * candidate_buffer_length) : nullptr;
    }

    s.num_full_tiles = s.input_length / static_cast<OffsetT>(tile_items);
    s.partial_items  = s.input_length - s.num_full_tiles * static_cast<OffsetT>(tile_items);
    s.segment_tiles_input =
      static_cast<OffsetT>(::cuda::ceil_div(s.input_length, OffsetT{tile_items}));
    return s;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile(const per_segment_state_t& s, LargeSegmentTileOffsetT local_tile)
  {
    key_source_input_t key_src_input{s.d_keys_in, storage.keys_source_state.a};
    key_source_buffer_t key_src_buffer{s.in_key_buf, storage.keys_source_state.b};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/s.load_from_candidates_buffer};

    selected_reserve_op_t reserve_sel{&s.segment_counter->num_selected_written};
    candidate_reserve_op_t reserve_cand{
      &s.segment_counter->num_ties_written_to_back,
      static_cast<candidate_offset_t>(s.k_total),
      static_cast<candidate_offset_t>(s.num_of_kth_needed)};
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
          s.d_values_out, s.d_values_out, ::cuda::std::identity{}, ::cuda::std::identity{}};
      }
    }();
    detail::topk::topk_noop_candidate_callback_op callback_op{};

    // The partition primitive's ctor takes `IdentifyCandidatesOp&` (non-const); build a
    // local op (cheap ctor) from the cached per-segment fields.
    IdentifyCandidatesOpT identify_candidates_op{
      &s.segment_counter->kth_key_bits, s.pass, total_bits, decomposer};

    partition_t partition{
      storage.partition_arena.get_partition_state(),
      reserve_sel,
      reserve_cand,
      sel_key_xform,
      cand_key_xform,
      s.d_keys_out,
      s.d_keys_out,
      value_channel_sinks,
      identify_candidates_op,
      callback_op};

    if (local_tile < s.num_full_tiles)
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
          value_source_input_t val_input{s.d_values_in, val_state_input};
          value_source_buffer_t val_buffer{s.in_val_buf, val_state_buffer};
          value_source_t val_src{val_input, val_buffer, /*pick_b=*/s.load_from_candidates_buffer};
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
    else if (local_tile == s.num_full_tiles && s.partial_items > 0)
    {
      const OffsetT tile_base = s.num_full_tiles * static_cast<OffsetT>(tile_items);
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
          value_source_input_t val_input{s.d_values_in, val_state_input};
          value_source_buffer_t val_buffer{s.in_val_buf, val_state_buffer};
          value_source_t val_src{val_input, val_buffer, /*pick_b=*/s.load_from_candidates_buffer};
          val_src.set_tile_base(tile_base);
          return val_src;
        }
      }();

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      __syncthreads();
      partition.partition(storage.partition_arena.get_partition_scratch(), items, s.partial_items, value_source);
    }

    partition.epilogue();
  }

public:
  // Process up to `tiles_per_chunk` consecutive tiles starting at `chunk_start`, grouped by
  // segment. Last-filter has no histogram and no `finalize_pass`, so the chunk loop is the
  // simplest of the three multi-CTA agents: per-segment-state cache + per-tile
  // `partition.run`. Same-segment tiles share the cached state.
  // See the matching comment on `agent_batched_topk_histogram::run` for why
  // `d_total_large_tiles` is a pointer rather than a value.
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_chunk(
    LargeSegmentTileOffsetT chunk_start, int tiles_per_chunk, const LargeSegmentTileOffsetT* d_total_large_tiles)
  {
    constexpr LargeSegmentTileOffsetT kNoActiveSegment = static_cast<LargeSegmentTileOffsetT>(-1);
    LargeSegmentTileOffsetT active_queue_idx           = kNoActiveSegment;
    per_segment_state_t state{};

    _CCCL_PRAGMA_NOUNROLL()
    for (int i = 0; i < tiles_per_chunk; ++i)
    {
      const LargeSegmentTileOffsetT global_tile_id = chunk_start + static_cast<LargeSegmentTileOffsetT>(i);
      if (global_tile_id >= *d_total_large_tiles)
      {
        break;
      }
      const LargeSegmentTileOffsetT queue_idx = resolve_queue_idx(global_tile_id);
      if (queue_idx != active_queue_idx)
      {
        active_queue_idx = queue_idx;
        state            = resolve_segment_state(queue_idx);
      }
      if (state.empty)
      {
        continue;
      }
      const LargeSegmentTileOffsetT local_tile = global_tile_id - state.slab_base;
      if (static_cast<OffsetT>(local_tile) >= state.segment_tiles_input)
      {
        continue;
      }
      process_tile(state, local_tile);
    }
  }
};

} // namespace detail::batched_topk
CUB_NAMESPACE_END
