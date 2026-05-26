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

// Opt-in per-pass / per-segment debug printf for batched TopK. Enabled by adding
//   -DCUB_DETAIL_BATCHED_TOPK_DEBUG_PRINTF=1
// to the compile command (e.g. via -DCMAKE_CUDA_FLAGS). When enabled, exactly one
// thread per kernel launch (`blockIdx.x == 0 && threadIdx.x == 0`) prints a single
// line summarising the per-segment state resolved at the top of each agent `run()`.
// The intent is to expose pass-to-pass shrinking of the candidate set, the
// load-from-candidates-buffer state, and the per-pass num_selected_written counter.
#ifndef CUB_DETAIL_BATCHED_TOPK_DEBUG_PRINTF
#  define CUB_DETAIL_BATCHED_TOPK_DEBUG_PRINTF 0
#endif

#if CUB_DETAIL_BATCHED_TOPK_DEBUG_PRINTF
#  define CUB_DETAIL_BATCHED_TOPK_DPRINTF(...)                              \
    do                                                                      \
    {                                                                       \
      if (blockIdx.x == 0 && threadIdx.x == 0)                              \
      {                                                                     \
        ::printf(__VA_ARGS__);                                              \
      }                                                                     \
    } while (false)
#else
#  define CUB_DETAIL_BATCHED_TOPK_DPRINTF(...) ((void) 0)
#endif

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
          typename SegmentCountT,
          // Experimental switch (mirrors `multi_worker_policy::full_tiles_only_histogram`).
          // When `true`, the agent's `run()` skips the partial-tile path entirely: only full
          // tiles flow through the inner loop, no `process_partial` predicate is computed,
          // and `process_partial_tile_at_segment_end` is never instantiated. The trailing
          // partial tile of each segment is the responsibility of
          // `device_segmented_topk_finalize_histogram_kernel` in that mode.
          bool FullTilesOnly = false,
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

  // Per-segment cache. Lives in smem rather than per-thread registers so all use sites read
  // through the same canonical handle. Thread 0 of the CTA writes this on each segment
  // boundary; every other thread reads through it. Each scalar / pointer / iterator is
  // dereferenced at the use site from `temp_storage.active_segment` (not cached into per-
  // thread register locals) so the control-flow boundaries stay explicit and the compiler
  // doesn't end up replicating the same scalar 32 times in the register file.
  struct active_segment_state_t
  {
    // Half-open tile-space window of the segment owned by this `active_segment` slot:
    // `[slab_base, segment_end)`. `chunk_cursor < segment_end` is the cheap "still in the
    // active segment" check that gates the segment-state refresh.
    LargeSegmentTileOffsetT slab_base;
    LargeSegmentTileOffsetT segment_end;

    // Per-segment tile-shape state.
    OffsetT num_items;
    OffsetT num_full_tiles;
    OffsetT partial_items;

    // Per-segment global-slab pointer for the merge / per-segment input iterator for the
    // tile load.
    OffsetT* segment_histogram;
    inner_key_it_t d_keys_in;
  };

  // The histogram agent no longer carries the prefix-sum scratch used by the per-segment
  // last-block epilogue: that work has been hoisted out into the standalone
  // `device_segmented_topk_finalize_histogram_kernel`. Smem here is the smem histogram + the
  // keys-source state / scratch + the smem-resident `active_segment` cache.
  struct _TempStorage
  {
    OffsetT histogram[num_buckets];
    typename keys_source_t::TempStorage keys_source_state;
    typename keys_source_t::ScratchStorage keys_source_scratch;
    active_segment_state_t active_segment;
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
  // Process one full tile's items by binning into the smem histogram.
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

  // Process the trailing partial tile's `num_thread_items` items per thread.
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

  // Thread 0 resolves the segment containing `cursor` and publishes the result to smem. Other
  // threads only need to participate in the surrounding `__syncthreads()`. The caller is
  // responsible for the publish barrier after this returns.
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_segment_state(LargeSegmentTileOffsetT cursor)
  {
    if (threadIdx.x == 0)
    {
      const LargeSegmentTileOffsetT queue_idx =
        UpperBound(
          d_large_segments_tile_offsets, static_cast<SegmentCountT>(*large_segments_count_it), cursor)
        - 1;
      const auto segment_id                = segment_id_provider[queue_idx];
      const OffsetT num_items              = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
      const OffsetT num_full_tiles         = num_items / static_cast<OffsetT>(tile_items);
      const OffsetT partial_items          = num_items - num_full_tiles * static_cast<OffsetT>(tile_items);
      const OffsetT seg_tile_count = num_full_tiles + (partial_items > 0 ? OffsetT{1} : OffsetT{0});
      const LargeSegmentTileOffsetT slab_base = d_large_segments_tile_offsets[queue_idx];

      temp_storage.active_segment.slab_base   = slab_base;
      temp_storage.active_segment.segment_end = slab_base + static_cast<LargeSegmentTileOffsetT>(seg_tile_count);
      temp_storage.active_segment.num_items         = num_items;
      temp_storage.active_segment.num_full_tiles    = num_full_tiles;
      temp_storage.active_segment.partial_items     = partial_items;
      temp_storage.active_segment.segment_histogram = d_segment_histograms + queue_idx * num_buckets;
      temp_storage.active_segment.d_keys_in         = d_key_segments_it[segment_id];
    }
  }

  // Flush the smem histogram into the active segment's global slab. All threads participate;
  // the caller must `__syncthreads()` beforehand so all atomic-adds from the previous tile are
  // visible.
  _CCCL_DEVICE _CCCL_FORCEINLINE void flush_active_segment()
  {
    detail::topk::merge_histogram<block_threads, num_buckets>(
      temp_storage.histogram, temp_storage.active_segment.segment_histogram);
  }

  // Bring up a freshly-resolved segment-stretch: thread 0 writes the new `active_segment`,
  // then everyone zeros the smem histogram. The two `__syncthreads()` bracket the smem
  // writes against the reads/writes that surround them.
  _CCCL_DEVICE _CCCL_FORCEINLINE void enter_segment(LargeSegmentTileOffsetT cursor)
  {
    load_segment_state(cursor);
    __syncthreads();
    detail::topk::init_histogram<block_threads, num_buckets>(temp_storage.histogram);
    __syncthreads();
  }

  // Combination "leave the current segment, enter the next one" used when a chunk straddles a
  // segment boundary (or when grid-striding lands the CTA on a new segment). Flushes the
  // current smem histogram, refreshes `active_segment` for `cursor`, and re-inits the smem
  // histogram for the new segment. The interior `__syncthreads()` brackets the smem-active-
  // segment slot against concurrent reads (from the just-completed `merge_histogram` call,
  // which reads `active_segment.segment_histogram` to drive its atomic adds) and the
  // upcoming thread-0 write (`load_segment_state` inside `enter_segment`).
  _CCCL_DEVICE _CCCL_FORCEINLINE void switch_to_segment(LargeSegmentTileOffsetT cursor)
  {
    __syncthreads();
    flush_active_segment();
    __syncthreads();
    enter_segment(cursor);
  }

  // `BlockLoad` algorithms that drive their inter-thread transfer through the
  // `keys_source_scratch` smem region (TRANSPOSE / WARP_TRANSPOSE / WARP_TRANSPOSE_TIMESLICED)
  // and the async-to-shared TMA path stage data in shared memory. For those, two consecutive
  // tile loads need a `__syncthreads()` between them so the next tile's writes don't clobber
  // the previous tile's reads. The DIRECT and VECTORIZE algorithms, by contrast, issue
  // per-thread `LDG.E.{,2,4}` straight into the destination registers and never touch the
  // shared scratch; the inter-tile barrier is dead work for those configurations. The
  // intra-tile atomicAdds into the smem histogram are independent across tiles and don't
  // need fencing either way -- atomics are program-ordered per thread and the histogram
  // bucket addresses are data-dependent (different keys -> different buckets in the common
  // case), so an early-arriving thread can start its next tile's LDG without waiting.
  static constexpr bool tile_load_kind_uses_smem =
    AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_direct
    && AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_vectorize;

  // Process one full tile of the active segment at local index `local_tile`. The caller
  // owns the long-lived `keys_source_t` (constructed once per middle-loop iteration so the
  // underlying `BlockLoadToShared` mbarrier is initialized **once** per segment-stretch
  // -- re-constructing it per tile would re-init the persistent mbarrier and deadlock).
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_full_tile_at(keys_source_t& keys_source, OffsetT local_tile)
  {
    const OffsetT tile_base = local_tile * static_cast<OffsetT>(tile_items);
    keys_source.set_tile_base(tile_base);

    if constexpr (tile_load_kind_uses_smem)
    {
      __syncthreads();
    }
    key_in_t items[items_per_thread];
    auto h = keys_source.submit_load(temp_storage.keys_source_scratch);
    h.complete_load(items);
    process_tile_full(items);
  }

  // Process the active segment's trailing partial tile (exactly `partial_items` items spread
  // across the block). Reads `num_full_tiles`, `partial_items`, `num_items` from the smem
  // `active_segment` slot. Uses the caller-owned `keys_source_t` (see
  // `process_full_tile_at`).
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_partial_tile_at_segment_end(keys_source_t& keys_source)
  {
    const OffsetT num_full_tiles = temp_storage.active_segment.num_full_tiles;
    const OffsetT partial_items  = temp_storage.active_segment.partial_items;
    const OffsetT num_items      = temp_storage.active_segment.num_items;

    const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);
    keys_source.set_tile_base(tile_base);

    if constexpr (tile_load_kind_uses_smem)
    {
      __syncthreads();
    }
    key_in_t items[items_per_thread];
    auto h = keys_source.submit_load(temp_storage.keys_source_scratch, partial_items);
    h.complete_load(items);
    const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
    const int num_thread_items =
      (thread_offset >= num_items)
        ? 0
        : static_cast<int>(
            (::cuda::std::min) (static_cast<OffsetT>(items_per_thread), num_items - thread_offset));
    process_tile_partial(items, num_thread_items);
  }

public:
  // Drive this CTA's entire grid-strided histogram pass. Structurally identical to the
  // pre-refactor loop -- the only material change is that per-segment cached state lives in
  // shared memory (`temp_storage.active_segment`) instead of per-thread register locals, and
  // the segment-state refresh / smem-histogram bookkeeping is factored into the helpers
  // declared above. The sentinel-based "no active segment yet" pattern is retained so the
  // initial segment load happens lazily inside the middle loop (rather than as a separate
  // call before the outer loop) -- this matches the original ordering of `__syncthreads()`
  // and `merge_histogram` calls bit-for-bit.
  //
  // The partial-tile schema preserved: full tiles flow through the inner `for` loop, then a
  // conditional *after* the loop handles the at-most-one trailing partial tile.
  //
  // `TilesPerChunk` is taken as a compile-time non-type template parameter rather than as
  // a runtime argument so ptxas can reason about the per-chunk stride and the middle while
  // loop's bound statically. Whether that pays off in codegen is the experiment driven by
  // this commit (see the `profile_round_*` baselines in the repo's investigation tree).
  template <int TilesPerChunk>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run()
  {
    // The slow-path's bit-decomposition is hard-wired for power-of-two chunk sizes up to 8.
    // For larger or non-power-of-two chunk sizes the decomposition would need extra `if`s
    // (and we'd lose the property that exactly one tile-count branch fires per stretch).
    static_assert(TilesPerChunk == 2 || TilesPerChunk == 4 || TilesPerChunk == 8,
                  "agent_batched_topk_histogram::run<TilesPerChunk> requires "
                  "TilesPerChunk to be a power of two in {2, 4, 8}.");

    const LargeSegmentTileOffsetT* const d_total_large_tiles =
      &d_large_segments_tile_offsets[static_cast<SegmentCountT>(*large_segments_count_it)];

    constexpr LargeSegmentTileOffsetT chunk_size_v = static_cast<LargeSegmentTileOffsetT>(TilesPerChunk);
    const LargeSegmentTileOffsetT stride           = static_cast<LargeSegmentTileOffsetT>(gridDim.x) * chunk_size_v;
    const LargeSegmentTileOffsetT first_chunk_start =
      static_cast<LargeSegmentTileOffsetT>(blockIdx.x) * chunk_size_v;

    // CTA whose first chunk lands past the queue's last tile has no work to do.
    if (first_chunk_start >= *d_total_large_tiles)
    {
      return;
    }

    // First segment-state load is hoisted out of the outer loop. After this `enter_segment`
    // call, `temp_storage.active_segment` is valid for `first_chunk_start`, and every
    // subsequent iteration only needs the cheaper `switch_to_segment` check on a segment-
    // boundary crossing.
    enter_segment(first_chunk_start);

    for (LargeSegmentTileOffsetT chunk_start = first_chunk_start; chunk_start < *d_total_large_tiles;
         chunk_start += stride)
    {
      const LargeSegmentTileOffsetT chunk_end =
        (chunk_start + chunk_size_v < *d_total_large_tiles) ? chunk_start + chunk_size_v : *d_total_large_tiles;

      // Segment-state refresh -- only when the cached segment no longer covers `chunk_start`.
      // On the very first iteration `chunk_start == first_chunk_start` is by construction
      // inside the segment we just loaded above, so this branch is taken at most once per
      // CTA, when grid-striding crosses a segment boundary.
      if (chunk_start >= temp_storage.active_segment.segment_end)
      {
        switch_to_segment(chunk_start);
      }

      // Step 2: fast-path check -- the chunk fits entirely inside the active segment's
      // full-tile range. When this fires, the whole chunk is exactly `TilesPerChunk` full
      // tiles drawn from one segment; no segment-switching, no partial-tile bookkeeping,
      // no `chunk_end` clipping. The tile loop below sees only the chunk's `local_tile_start`
      // and a fully unrolled run of `TilesPerChunk` full-tile loads.
      const LargeSegmentTileOffsetT slab_base    = temp_storage.active_segment.slab_base;
      const OffsetT num_full_tiles_in_seg        = temp_storage.active_segment.num_full_tiles;
      const LargeSegmentTileOffsetT full_tile_boundary =
        slab_base + static_cast<LargeSegmentTileOffsetT>(num_full_tiles_in_seg);

      if (chunk_start + chunk_size_v <= full_tile_boundary)
      {
        // ----- Fast path: TilesPerChunk full tiles, one segment, no switching. ------------
        // Note: `chunk_start + chunk_size_v <= full_tile_boundary <= *d_total_large_tiles`,
        // so the chunk also can't have been clipped at the end of the queue -- the implicit
        // `chunk_end == chunk_start + chunk_size_v` is what enables the fully-unrolled loop.
        const OffsetT local_tile_start = static_cast<OffsetT>(chunk_start - slab_base);
        keys_source_t keys_source{temp_storage.active_segment.d_keys_in, temp_storage.keys_source_state};

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < TilesPerChunk; ++i)
        {
          process_full_tile_at(keys_source, static_cast<OffsetT>(local_tile_start + i));
        }
        continue;
      }

      // ----- Slow path: chunk straddles a segment boundary, hits a partial-tile slot,
      // or is clipped at the queue's tail. Walk segment-stretches; per-stretch tile work
      // uses a power-of-two bit decomposition. Each stretch processes strictly fewer than
      // `TilesPerChunk` full tiles (otherwise we'd have taken the fast path), so the
      // decomposition only needs to cover the range `[0, TilesPerChunk - 1]`. The chunk
      // walk visits at most `TilesPerChunk` segments (since each segment occupies >= 1
      // tile slot in queue-idx space).
      LargeSegmentTileOffsetT chunk_cursor = chunk_start;
      while (chunk_cursor < chunk_end)
      {
        if (chunk_cursor >= temp_storage.active_segment.segment_end)
        {
          switch_to_segment(chunk_cursor);
        }

        const LargeSegmentTileOffsetT seg_slab_base  = temp_storage.active_segment.slab_base;
        const OffsetT seg_num_full                   = temp_storage.active_segment.num_full_tiles;
        const OffsetT local_tile_start               = static_cast<OffsetT>(chunk_cursor - seg_slab_base);
        const LargeSegmentTileOffsetT remaining_in_chunk = chunk_end - chunk_cursor;
        const OffsetT full_tiles_remaining_in_seg =
          (local_tile_start < seg_num_full) ? (seg_num_full - local_tile_start) : OffsetT{0};
        const OffsetT full_tiles_in_stretch =
          (::cuda::std::min) (static_cast<OffsetT>(remaining_in_chunk), full_tiles_remaining_in_seg);

        // Power-of-two bit-decomposition of `full_tiles_in_stretch ∈ [0, TilesPerChunk-1]`.
        // Each `if` covers one bit of the count and is the only branch in this stretch's
        // tile path; the inner `for` is statically sized so it unrolls cleanly.
        keys_source_t keys_source{temp_storage.active_segment.d_keys_in, temp_storage.keys_source_state};
        OffsetT remaining_full_tiles = full_tiles_in_stretch;
        OffsetT local                = local_tile_start;
        if constexpr (TilesPerChunk >= 8)
        {
          if (remaining_full_tiles >= OffsetT{4})
          {
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int i = 0; i < 4; ++i)
            {
              process_full_tile_at(keys_source, static_cast<OffsetT>(local + i));
            }
            local += OffsetT{4};
            remaining_full_tiles -= OffsetT{4};
          }
        }
        if constexpr (TilesPerChunk >= 4)
        {
          if (remaining_full_tiles >= OffsetT{2})
          {
            _CCCL_PRAGMA_UNROLL_FULL()
            for (int i = 0; i < 2; ++i)
            {
              process_full_tile_at(keys_source, static_cast<OffsetT>(local + i));
            }
            local += OffsetT{2};
            remaining_full_tiles -= OffsetT{2};
          }
        }
        // The `>= 1` branch is always present (TilesPerChunk >= 2). The trailing
        // decrement / `local` bump are dead at this point and intentionally omitted.
        if (remaining_full_tiles >= OffsetT{1})
        {
          process_full_tile_at(keys_source, static_cast<OffsetT>(local));
        }

        // Partial-tile slot bookkeeping. In `FullTilesOnly` mode the partial-tile load + bin
        // is owned by the finalize-histogram kernel, so the call site is `if constexpr`-
        // eliminated; the slot is still "consumed" by `tiles_consumed` below so the chunk
        // walk doesn't stall on the segment's partial-tile index.
        const OffsetT next_local_tile = local_tile_start + full_tiles_in_stretch;
        const bool reaches_partial_slot =
          (next_local_tile == seg_num_full) && (temp_storage.active_segment.partial_items > 0)
          && (full_tiles_in_stretch + OffsetT{1} <= static_cast<OffsetT>(remaining_in_chunk));
        if constexpr (!FullTilesOnly)
        {
          if (reaches_partial_slot)
          {
            process_partial_tile_at_segment_end(keys_source);
          }
        }

        const OffsetT tiles_consumed = full_tiles_in_stretch + (reaches_partial_slot ? OffsetT{1} : OffsetT{0});
        if (tiles_consumed == OffsetT{0})
        {
          break;
        }
        chunk_cursor += static_cast<LargeSegmentTileOffsetT>(tiles_consumed);
      }
    }

    // Final flush: merge the last active segment-stretch's smem histogram into its global
    // slab. Unconditional -- the early-return above already filtered out CTAs with no work,
    // so by the time we reach this point we've entered at least one segment.
    __syncthreads();
    flush_active_segment();
  }

  // Sentinel-based, monolithic alternative to `run()` (kept around as `run2` so a future
  // policy / kernel can opt into it when a workload shows the fast/slow-path split's
  // register cost outweighing its throughput gains -- I16 took +11 R going from the
  // monolithic shape to the split, while floats / wide ints saved 4 -- 8 us per histogram
  // pass at 2^28 elements; see `profile_round_11_*` / `profile_round_13_*` /
  // `profile_round_14_*` for the side-by-side data).
  //
  // Structurally identical to the round-11 / round-12 (post-revert) `run()`:
  //   - single while-loop over `chunk_cursor`, no fast/slow peel;
  //   - `active_queue_idx == kNoActiveSegment` sentinel gates the first-time
  //     `enter_segment` inside the loop body (no hoist, no pre-loop early return);
  //   - per-stretch full tiles flow through one straight `for (i = 0;
  //     i < full_tiles_to_process; ++i)` (no power-of-two bit decomposition).
  //
  // Blast radius: zero outside this function -- every name referenced
  // (`d_large_segments_tile_offsets`, `*large_segments_count_it`, the
  // `temp_storage.active_segment.*` fields, the `enter_segment` /
  // `switch_to_segment` / `flush_active_segment` / `process_full_tile_at` /
  // `process_partial_tile_at_segment_end` helpers, the `keys_source_t` alias) is
  // already a member of `agent_batched_topk_histogram`. Because this method is a
  // member of a class template it is only instantiated when called, so leaving it
  // sitting here uncalled costs nothing in the default build.
  template <int TilesPerChunk>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run2()
  {
    const LargeSegmentTileOffsetT* const d_total_large_tiles =
      &d_large_segments_tile_offsets[static_cast<SegmentCountT>(*large_segments_count_it)];

    // Sentinel meaning "no segment loaded yet"; flips to the segment's queue_idx the first
    // time this CTA touches a tile and never reverts.
    constexpr LargeSegmentTileOffsetT kNoActiveSegment = static_cast<LargeSegmentTileOffsetT>(-1);
    LargeSegmentTileOffsetT active_queue_idx           = kNoActiveSegment;

    constexpr LargeSegmentTileOffsetT chunk_size_v = static_cast<LargeSegmentTileOffsetT>(TilesPerChunk);
    const LargeSegmentTileOffsetT stride           = static_cast<LargeSegmentTileOffsetT>(gridDim.x) * chunk_size_v;

    for (LargeSegmentTileOffsetT chunk_start = static_cast<LargeSegmentTileOffsetT>(blockIdx.x) * chunk_size_v;
         chunk_start < *d_total_large_tiles;
         chunk_start += stride)
    {
      const LargeSegmentTileOffsetT chunk_end =
        (chunk_start + chunk_size_v < *d_total_large_tiles) ? chunk_start + chunk_size_v : *d_total_large_tiles;

      LargeSegmentTileOffsetT chunk_cursor = chunk_start;
      while (chunk_cursor < chunk_end)
      {
        // Segment-state refresh -- only when the cached segment doesn't cover `chunk_cursor`.
        // The first refresh on this CTA's run uses `enter_segment` (no smem-histogram to
        // flush yet); subsequent refreshes use `switch_to_segment` which flushes first.
        if (active_queue_idx == kNoActiveSegment)
        {
          enter_segment(chunk_cursor);
          // Mark "have an active segment" -- the actual queue_idx value isn't read again
          // (the agent reads everything from `temp_storage.active_segment`); the sentinel
          // toggle just gates the flush-vs-no-flush refresh choice.
          active_queue_idx = LargeSegmentTileOffsetT{0};
        }
        else if (chunk_cursor >= temp_storage.active_segment.segment_end)
        {
          switch_to_segment(chunk_cursor);
        }

        // Tile-space bounds of this segment-stretch inside the chunk.
        const LargeSegmentTileOffsetT slab_base   = temp_storage.active_segment.slab_base;
        const OffsetT local_tile_start            = static_cast<OffsetT>(chunk_cursor - slab_base);
        const LargeSegmentTileOffsetT remaining_in_chunk = chunk_end - chunk_cursor;
        const OffsetT num_full_tiles              = temp_storage.active_segment.num_full_tiles;
        const OffsetT full_tiles_remaining_in_seg =
          (local_tile_start < num_full_tiles) ? (num_full_tiles - local_tile_start) : OffsetT{0};
        const OffsetT full_tiles_to_process =
          (::cuda::std::min) (static_cast<OffsetT>(remaining_in_chunk), full_tiles_remaining_in_seg);

        // At-most-one trailing partial tile, claimed iff the full-tile loop ends at the
        // segment's partial-tile slot AND the segment has a partial AND chunk budget remains.
        // The cursor advance is always the same -- the partial-tile slot is "consumed"
        // either way. In `FullTilesOnly` mode the slot is *only* stepped over; the actual
        // partial-tile load + bin is delegated to the finalize-histogram kernel. In the
        // default mode the partial tile is processed inline as before.
        const OffsetT next_local_tile = local_tile_start + full_tiles_to_process;
        const bool reaches_partial_slot =
          (next_local_tile == num_full_tiles) && (temp_storage.active_segment.partial_items > 0)
          && (full_tiles_to_process + OffsetT{1} <= static_cast<OffsetT>(remaining_in_chunk));

        // Construct the per-segment-stretch keys-source view once -- reused across the full-
        // tile loop and the trailing partial.
        keys_source_t keys_source{temp_storage.active_segment.d_keys_in, temp_storage.keys_source_state};

        // Inner: full-tile loop.
        for (OffsetT i = 0; i < full_tiles_to_process; ++i)
        {
          process_full_tile_at(keys_source, local_tile_start + i);
        }

        // Partial-tile conditional *after* the loop. In `FullTilesOnly` mode the call is
        // `if constexpr`-eliminated -- the slot is still consumed by `tiles_consumed` below
        // so the chunk walk doesn't stall on the segment's partial-tile index.
        if constexpr (!FullTilesOnly)
        {
          if (reaches_partial_slot)
          {
            process_partial_tile_at_segment_end(keys_source);
          }
        }

        const OffsetT tiles_consumed = full_tiles_to_process + (reaches_partial_slot ? OffsetT{1} : OffsetT{0});
        if (tiles_consumed == OffsetT{0})
        {
          break;
        }
        chunk_cursor += static_cast<LargeSegmentTileOffsetT>(tiles_consumed);
      }
    }

    // Final flush: merge the last active segment-stretch's smem histogram into its global
    // slab. Skipped only when this CTA had no work at all.
    if (active_queue_idx != kNoActiveSegment)
    {
      __syncthreads();
      flush_active_segment();
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
          bool InlinedClassify = false,
          // Experimental switch (mirrors `multi_worker_policy::full_tiles_only_filter`).
          // When `true`, the agent's `run()` skips the slow-path partial-tile
          // `dispatch_tile<false>` call; the partial tile of each segment is processed
          // by `device_segmented_topk_finalize_filter_kernel` via
          // `agent.process_partial_for_segment(queue_idx, pass)` before its prefix-sum +
          // bucket-finder runs.
          bool FullTilesOnly   = false>
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

  // Mirrors the histogram agent's constexpr -- see the docstring on
  // `agent_batched_topk_histogram::tile_load_kind_uses_smem`. For DIRECT / VECTORIZE the
  // `BlockLoad` runs as per-thread `LDG.E.{,2,4}` straight into registers without touching
  // the shared scratch, so the pre-`submit_load` `__syncthreads()` is dead work. The
  // post-`complete_load` sync is *kept* in every mode because `keys_source_scratch` and
  // `partition_scratch` alias through the smem union in
  // `partition_storage_layout_for_t` -- without that sync, the next tile's `partition`
  // could clobber the bytes the just-completed load still owned (in the smem-using case)
  // or that the previous tile's `partition` still owned (in either case).
  static constexpr bool tile_load_kind_uses_smem =
    AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_direct
    && AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_vectorize;

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

  using buffered_value_channel_sinks_concrete_t =
    detail::topk::value_channel_sinks_t<val_out_t, buffered_cand_val_out_t>;
  using buffered_value_channel_sinks_t =
    ::cuda::std::conditional_t<keys_only, NullType, buffered_value_channel_sinks_concrete_t>;

  using early_stop_value_channel_sinks_concrete_t = detail::topk::value_channel_sinks_filter_t<val_out_t>;
  using early_stop_value_channel_sinks_t =
    ::cuda::std::conditional_t<keys_only, NullType, early_stop_value_channel_sinks_concrete_t>;

  using agent_value_t = ::cuda::std::conditional_t<keys_only, NullType, value_in_t>;
  using agent_value_data_source_scratch_t =
    ::cuda::std::conditional_t<keys_only, NullType, typename value_source_t::ScratchStorage>;

  using selected_offset_t  = OutOffsetT;
  using candidate_offset_t = OffsetT;

  using selected_reserve_op_t  = detail::topk::atomic_reserve_range_op<selected_offset_t>;
  using candidate_reserve_op_t = detail::topk::atomic_reserve_range_op<candidate_offset_t>;


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
      return buffered_value_channel_sinks_concrete_t{values_out_sink, cand_val_out};
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
      return early_stop_value_channel_sinks_concrete_t{values_out_sink};
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
    // Width of the segment in queue-tile-space, read from
    // `d_large_segments_tile_offsets[queue_idx + 1]`. Sized at segment-enqueue time from the
    // original segment size, so independent of which pass we're in. See the matching
    // `last_filter` doc for the slow-path cursor-jump motivation.
    LargeSegmentTileOffsetT queue_segment_end;
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
    s.slab_base         = d_large_segments_tile_offsets[queue_idx];
    // The offset table is sized `num_large_segments + 1` (sentinel at the end stores
    // `total_large_tiles`), so the next-slot read is in-bounds for every valid `queue_idx`.
    s.queue_segment_end = d_large_segments_tile_offsets[queue_idx + 1];
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

  // Debug-only: print a one-line summary of the per-segment state resolved at the top of
  // `run()` (and each segment-boundary refresh). Compiled out when
  // `CUB_DETAIL_BATCHED_TOPK_DEBUG_PRINTF == 0`. Only block 0 / thread 0 prints, so for the
  // single-segment benchmark we get exactly one line per kernel launch.
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  debug_print_state(const char* stage, int pass, LargeSegmentTileOffsetT queue_idx, const per_segment_state_t& s) const
  {
#if CUB_DETAIL_BATCHED_TOPK_DEBUG_PRINTF
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      ::printf("[batched_topk %s pass=%d seg=%lld grid=%d tile_items=%d] "
               "empty=%d early_stop=%d will_buffer=%d load_from_cand=%d "
               "k=%lld in_len=%lld curr_len=%lld num_sel_wr=%lld num_ties_back=%lld "
               "num_full_tiles=%lld partial=%lld slab_base=%lld queue_end=%lld\n",
               stage,
               pass,
               static_cast<long long>(queue_idx),
               static_cast<int>(gridDim.x),
               static_cast<int>(tile_items),
               static_cast<int>(s.empty),
               static_cast<int>(s.early_stop),
               static_cast<int>(s.will_buffer),
               static_cast<int>(s.load_from_candidates_buffer),
               static_cast<long long>(s.current_k),
               static_cast<long long>(s.input_length_actual),
               static_cast<long long>(s.current_len),
               static_cast<long long>(s.empty ? 0 : s.segment_counter->num_selected_written),
               static_cast<long long>(s.empty ? 0 : s.segment_counter->num_ties_written_to_back),
               static_cast<long long>(s.num_full_tiles),
               static_cast<long long>(s.partial_items),
               static_cast<long long>(s.slab_base),
               static_cast<long long>(s.queue_segment_end));
    }
#else
    (void) stage;
    (void) pass;
    (void) queue_idx;
    (void) s;
#endif
  }

  // Per-mode tile bodies. Each takes the per-segment cached state and a tile-local index, runs
  // exactly the same code the pre-refactor `run()` ran inside its `if (early_stop) {} else if
  // (will_buffer) {} else {}` branches, minus the surrounding init / merge / finalize_pass --
  // those are managed by `process_chunk` (init/merge) and the finalize kernel (finalize_pass).

  // Templated on `IsFullTile` so the fast / slow-full-tile paths can skip the runtime
  // partial-vs-full branch. Callers must guarantee:
  //   - `IsFullTile == true`  -> `local_tile < s.num_full_tiles`
  //   - `IsFullTile == false` -> `local_tile == s.num_full_tiles && s.partial_items > 0`
  template <bool IsFullTile>
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

    early_stop_filter_t filter{
      storage.arms.early_stop.arena.get_partition_state(),
      reserve_sel,
      s.d_keys_out,
      value_channel_sinks,
      identify_selected};

    if constexpr (IsFullTile)
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

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.early_stop.arena.get_keys_source_scratch());
      h.complete_load(items);
      __syncthreads();
      filter.partition(storage.arms.early_stop.arena.get_partition_scratch(), items, value_source);
    }
    else
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

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.early_stop.arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      __syncthreads();
      filter.partition(storage.arms.early_stop.arena.get_partition_scratch(), items, s.partial_items, value_source);
    }

    filter.epilogue();
  }

  // See the `process_tile_early_stop` doc for the `IsFullTile` contract.
  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  process_tile_buffered(const per_segment_state_t& s, LargeSegmentTileOffsetT local_tile)
  {
    key_source_input_t key_src_input{s.d_keys_in, storage.arms.buffered.keys_source_state.a};
    key_source_buffer_t key_src_buffer{s.in_key_buf, storage.arms.buffered.keys_source_state.b};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/s.load_from_candidates_buffer};

    selected_reserve_op_t reserve_sel{&s.segment_counter->num_selected_written};
    candidate_reserve_op_t reserve_cand{&s.segment_counter->num_candidates_written};

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
      s.d_keys_out,
      cand_key_out,
      value_channel_sinks,
      identify_candidates_op,
      histogram_cb};

    if constexpr (IsFullTile)
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

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch());
      h.complete_load(items);
      __syncthreads();
      partition.partition(storage.arms.buffered.arena.get_partition_scratch(), items, value_source);
    }
    else
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

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      __syncthreads();
      partition.partition(
        storage.arms.buffered.arena.get_partition_scratch(), items, s.partial_items, value_source);
    }

    partition.epilogue();
  }

  // See the `process_tile_early_stop` doc for the `IsFullTile` contract.
  template <bool IsFullTile>
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

    if constexpr (IsFullTile)
    {
      const OffsetT tile_base = static_cast<OffsetT>(local_tile) * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
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
    else
    {
      const OffsetT tile_base = s.num_full_tiles * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
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

  // Per-mode tile dispatcher. Loop-invariant `s.early_stop` / `s.will_buffer` is what
  // separates the three tile bodies; the agent reads them on every call but their value is
  // fixed for the whole segment-stretch the caller is processing, so within a single
  // unrolled / bit-decomposed run ptxas can hoist the mode branch above the tile loop.
  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void dispatch_tile(const per_segment_state_t& s, LargeSegmentTileOffsetT local_tile)
  {
    if (s.early_stop)
    {
      process_tile_early_stop<IsFullTile>(s, local_tile);
    }
    else if (s.will_buffer)
    {
      process_tile_buffered<IsFullTile>(s, local_tile);
    }
    else
    {
      process_tile_unbuffered<IsFullTile>(s, local_tile);
    }
  }

  // Whether the segment's mode uses the smem histogram. The two non-early_stop modes
  // (buffered / unbuffered) both accumulate into `storage.arms.buffered.histogram`;
  // early_stop touches none of it. Empty segments don't touch it either.
  _CCCL_DEVICE _CCCL_FORCEINLINE static bool segment_uses_smem_histogram(const per_segment_state_t& s)
  {
    return !s.empty && !s.early_stop;
  }

  // Init the smem histogram for the given segment iff its mode uses it. Caller must
  // `__syncthreads()` after to publish the writes.
  _CCCL_DEVICE _CCCL_FORCEINLINE void init_segment_histogram(const per_segment_state_t& s)
  {
    if (segment_uses_smem_histogram(s))
    {
      detail::topk::init_histogram<block_threads, num_buckets>(storage.arms.buffered.histogram);
    }
  }

  // Merge the smem histogram into the segment's global slab iff its mode used it. Caller
  // must `__syncthreads()` beforehand so all atomic-adds from the segment's tile loop are
  // visible.
  _CCCL_DEVICE _CCCL_FORCEINLINE void merge_segment_histogram(const per_segment_state_t& s)
  {
    if (segment_uses_smem_histogram(s))
    {
      detail::topk::merge_histogram<block_threads, num_buckets>(
        storage.arms.buffered.histogram, s.segment_histogram);
    }
  }

public:
  // Drive this CTA's entire grid-strided filter pass.
  //
  // Design choice: this is a *flat* grid-stride loop -- one tile per
  // iteration, stride = `gridDim.x`. The earlier chunked variant kept
  // `TilesPerChunk` (4-8) tiles per CTA iteration and split into a
  // fully-unrolled "fast path" (chunk fits inside a segment's full tiles)
  // plus a power-of-two bit-decomposition "slow path" (chunk crosses a
  // segment boundary or hits a partial-tile slot). The fast-path
  // unrolling was preserved chunk-level instruction-level parallelism;
  // the slow path was a tail handler.
  //
  // Why the flat shape now:
  //
  //   * Register pressure. SASS liveness analysis on the int8/int8 filter
  //     kernel (`topk_perf_tracking/reports/int8_int8_per_segment_state_breakdown.md`)
  //     showed ~7-10 *additional* persistent registers tied up just by
  //     the chunk-walk machinery (`chunk_start` / `chunk_end` /
  //     `local_tile_start` / `local_stretch_end` / `local_full_end` /
  //     `full_tiles_in_stretch`, plus the bit-decomposition `remaining`
  //     / `local`). All of those are CTA-uniform yet land in per-thread
  //     general registers because the compiler can't UR-promote them
  //     through the chunk-walk dataflow. Removing the chunk concept makes
  //     them disappear.
  //   * Loss of chunk-level ILP is small on Blackwell. The intra-tile
  //     `BlockLoad -> classify -> write` is already deeply unrolled
  //     (`items_per_thread`); occupancy from `__launch_bounds__` gives
  //     the warp scheduler enough work to hide tile-to-tile latency.
  //
  // Shape:
  //   * Early-return for CTAs whose first tile lands past the queue.
  //   * One `resolve_segment_state(blockIdx.x)` hoisted before the loop,
  //     with the smem histogram initialised inline.
  //   * Per tile:
  //       - Refresh `state` if we crossed a segment boundary
  //         (`tile_id >= state.queue_segment_end`) -- the
  //         flush-old-hist / resolve-new / init-new-hist handshake.
  //       - Skip empty segments and wasted-tail / past-data tiles.
  //       - Dispatch a single `dispatch_tile<true>` (full tile) or, when
  //         the policy lets the filter kernel handle partials, a
  //         `dispatch_tile<false>` on the tile right after the last full
  //         one.
  //   * Final flush of the last segment's smem histogram (no-op for
  //     early_stop / empty / never-entered).
  //
  // The per-segment epilogue (counter update + prefix-sum + bucket-finder
  // + optional histogram reset) still lives in
  // `device_segmented_topk_finalize_filter_kernel` and runs on the same
  // stream after this kernel.
  //
  // `TilesPerChunk` is kept on the template signature for ABI/source
  // compatibility with the histogram / last_filter agents but is unused
  // inside the body; the static_assert preserves the policy contract.
  template <int TilesPerChunk>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(int pass)
  {
    static_assert(TilesPerChunk == 2 || TilesPerChunk == 4 || TilesPerChunk == 8,
                  "agent_batched_topk_filter_partition::run<TilesPerChunk> requires "
                  "TilesPerChunk to be a power of two in {2, 4, 8}.");

    const LargeSegmentTileOffsetT* const d_total_large_tiles =
      &d_large_segments_tile_offsets[num_large_segments];
    const LargeSegmentTileOffsetT total = *d_total_large_tiles;

    const LargeSegmentTileOffsetT first_tile = static_cast<LargeSegmentTileOffsetT>(blockIdx.x);
    const LargeSegmentTileOffsetT stride     = static_cast<LargeSegmentTileOffsetT>(gridDim.x);

    if (first_tile >= total)
    {
      return;
    }

    // Hoist first segment-state resolve + smem-hist init.
    const LargeSegmentTileOffsetT first_queue_idx = resolve_queue_idx(first_tile);
    per_segment_state_t state                     = resolve_segment_state(first_queue_idx, pass);
    debug_print_state("filter_run_first", pass, first_queue_idx, state);
    __syncthreads();
    init_segment_histogram(state);
    __syncthreads();

    for (LargeSegmentTileOffsetT tile_id = first_tile; tile_id < total; tile_id += stride)
    {
      // Segment refresh -- only when the cached segment no longer covers `tile_id`.
      if (tile_id >= state.queue_segment_end)
      {
        __syncthreads();
        merge_segment_histogram(state);
        const LargeSegmentTileOffsetT next_queue_idx = resolve_queue_idx(tile_id);
        state                                        = resolve_segment_state(next_queue_idx, pass);
        debug_print_state("filter_run_refresh", pass, next_queue_idx, state);
        __syncthreads();
        init_segment_histogram(state);
        __syncthreads();
      }

      if (state.empty)
      {
        continue;
      }

      const OffsetT local_tile = static_cast<OffsetT>(tile_id - state.slab_base);
      if (local_tile < state.num_full_tiles)
      {
        dispatch_tile<true>(state, static_cast<LargeSegmentTileOffsetT>(local_tile));
      }
      else if constexpr (!FullTilesOnly)
      {
        // local_tile >= num_full_tiles: this is either the trailing partial slot
        // (if `partial_items > 0`) or a wasted-tail slot past the segment's data
        // end. Only the former drives a `dispatch_tile<false>` call; the latter
        // falls through and the grid-stride loop skips it. In `FullTilesOnly`
        // mode this whole branch is `if constexpr`-eliminated -- the partial
        // tile is owned by `device_segmented_topk_finalize_filter_kernel`.
        if (local_tile == state.num_full_tiles && state.partial_items > OffsetT{0})
        {
          dispatch_tile<false>(state, static_cast<LargeSegmentTileOffsetT>(state.num_full_tiles));
        }
      }
    }

    // Final flush: merge the last active segment's smem histogram (no-op for
    // early_stop / empty). Always reached -- the early-return filtered out CTAs with
    // no work; the predicate inside `merge_segment_histogram` decides whether to
    // actually merge.
    __syncthreads();
    merge_segment_histogram(state);
  }

  // Process the trailing partial tile of `queue_idx`'s segment for the current pass, using
  // whatever per-mode tile body the segment's runtime state selects. Invoked by
  // `device_segmented_topk_finalize_filter_kernel` (one CTA per segment in its grid-stride)
  // when the policy's `full_tiles_only_filter` knob is on -- in that mode the filter
  // kernel's `run()` skips the slow-path `dispatch_tile<false>` call entirely, so each
  // segment's partial-tile contribution must be re-injected here before the prefix-sum +
  // bucket-finder runs.
  //
  // Smem-histogram handshake (for buffered / unbuffered modes only):
  //   - Caller `__syncthreads()` before entry (so prior smem state is settled).
  //   - This method `init_segment_histogram(state)`s the smem hist, processes the partial
  //     via `dispatch_tile<false>` (which atomicAdds into the smem hist), then
  //     `merge_segment_histogram(state)`s it into the per-segment global slab.
  //   - Caller `__syncthreads()` after -- the smem buffer is now ready for the
  //     prefix-sum scratch.
  //
  // early_stop mode: no smem-histogram touched (the partition primitive only writes the
  // selected channel); the per-mode `process_tile_early_stop<false>` body runs as-is.
  //
  // Empty segments and segments with no partial (`partial_items == 0`) are no-ops.
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_partial_for_segment(LargeSegmentTileOffsetT queue_idx, int pass)
  {
    per_segment_state_t state = resolve_segment_state(queue_idx, pass);
    debug_print_state("filter_partial", pass, queue_idx, state);
    if (state.empty || state.partial_items == 0)
    {
      return;
    }

    if (segment_uses_smem_histogram(state))
    {
      init_segment_histogram(state);
      __syncthreads();
    }

    dispatch_tile<false>(state, static_cast<LargeSegmentTileOffsetT>(state.num_full_tiles));

    if (segment_uses_smem_histogram(state))
    {
      __syncthreads();
      merge_segment_histogram(state);
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

  // Mirrors the histogram / filter agents' constexpr -- DIRECT / VECTORIZE `BlockLoad`
  // doesn't touch the shared scratch, so the pre-`submit_load` `__syncthreads()` is dead
  // work for those algos. The post-`complete_load` sync stays in (it serializes
  // consecutive `partition` calls through the smem union that aliases
  // `keys_source_scratch` with `partition_scratch`).
  static constexpr bool tile_load_kind_uses_smem =
    AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_direct
    && AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_vectorize;

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

  using value_channel_sinks_concrete_t = detail::topk::value_channel_sinks_t<val_out_t, cand_val_out_t>;
  using value_channel_sinks_or_null_t =
    ::cuda::std::conditional_t<keys_only, NullType, value_channel_sinks_concrete_t>;

  using agent_value_t = ::cuda::std::conditional_t<keys_only, NullType, value_in_t>;
  using agent_value_data_source_scratch_t =
    ::cuda::std::conditional_t<keys_only, NullType, typename value_source_t::ScratchStorage>;

  using selected_reserve_op_t  = detail::topk::atomic_reserve_range_op<selected_offset_t>;
  using candidate_reserve_op_t = detail::topk::back_grow_capped_reserve_op<candidate_offset_t>;


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
  //
  // Note on `slab_base` / `queue_segment_end` vs `segment_tiles_input`:
  //   - `[slab_base, queue_segment_end)` is the segment's tile-space window in the global
  //     queue. The width is `d_large_segments_tile_offsets[queue_idx + 1] - slab_base`,
  //     fixed at segment-enqueue time from the *original* segment size and so independent
  //     of which pass we're in.
  //   - `segment_tiles_input` is the number of those slots that actually carry data this
  //     pass (= `ceil(input_length / tile_items)`); this is per-pass and can be 0 for
  //     "empty" segments (`num_candidates_in == 0`).
  //   - `[slab_base, slab_base + segment_tiles_input)` contains the live tiles; any
  //     `[slab_base + segment_tiles_input, queue_segment_end)` tail is "wasted" queue
  //     slots that the agent must walk past.
  // Tracking the wider `queue_segment_end` lets `run`'s slow-path cursor jump past empty
  // segments / wasted-slot tails in one step rather than via per-tile `UpperBound`.
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
    LargeSegmentTileOffsetT queue_segment_end;
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

  // Debug-only: print a one-line summary of the per-segment state resolved at the top of
  // `run()` (and each segment-boundary refresh) for `last_filter`. Compiled out when
  // `CUB_DETAIL_BATCHED_TOPK_DEBUG_PRINTF == 0`. Only block 0 / thread 0 prints, so for the
  // single-segment benchmark we get exactly one line per kernel launch.
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  debug_print_state(const char* stage, LargeSegmentTileOffsetT queue_idx, const per_segment_state_t& s) const
  {
#if CUB_DETAIL_BATCHED_TOPK_DEBUG_PRINTF
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      ::printf("[batched_topk %s pass=%d seg=%lld grid=%d tile_items=%d] "
               "empty=%d load_from_cand=%d "
               "k_total=%lld num_of_kth_needed=%lld in_len=%lld "
               "num_sel_wr=%lld num_ties_back=%lld "
               "num_full_tiles=%lld partial=%lld slab_base=%lld queue_end=%lld\n",
               stage,
               static_cast<int>(s.pass),
               static_cast<long long>(queue_idx),
               static_cast<int>(gridDim.x),
               static_cast<int>(tile_items),
               static_cast<int>(s.empty),
               static_cast<int>(s.load_from_candidates_buffer),
               static_cast<long long>(s.k_total),
               static_cast<long long>(s.num_of_kth_needed),
               static_cast<long long>(s.input_length),
               static_cast<long long>(s.empty ? 0 : s.segment_counter->num_selected_written),
               static_cast<long long>(s.empty ? 0 : s.segment_counter->num_ties_written_to_back),
               static_cast<long long>(s.num_full_tiles),
               static_cast<long long>(s.partial_items),
               static_cast<long long>(s.slab_base),
               static_cast<long long>(s.queue_segment_end));
    }
#else
    (void) stage;
    (void) queue_idx;
    (void) s;
#endif
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE per_segment_state_t resolve_segment_state(LargeSegmentTileOffsetT queue_idx)
  {
    per_segment_state_t s{};
    s.slab_base         = d_large_segments_tile_offsets[queue_idx];
    // `queue_segment_end` is the next segment's `slab_base` -- the table is sized
    // `num_large_segments + 1` (sentinel at the end stores `total_large_tiles`), so the
    // read is in-bounds for every valid `queue_idx`.
    s.queue_segment_end = d_large_segments_tile_offsets[queue_idx + 1];
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

  // Build the keys-source for the current segment. Lives across all tiles of this segment.
  _CCCL_DEVICE _CCCL_FORCEINLINE keys_source_t make_keys_source_for_segment(const per_segment_state_t& s)
  {
    key_source_input_t key_src_input{s.d_keys_in, storage.keys_source_state.a};
    key_source_buffer_t key_src_buffer{s.in_key_buf, storage.keys_source_state.b};
    return keys_source_t{key_src_input, key_src_buffer, /*pick_b=*/s.load_from_candidates_buffer};
  }

  // Build the partition object for the current segment. Lives across all tiles of this segment so
  // its per-thread `cand_reserve_open` flag (the back-grow-cap exit hint that drops per-item
  // atomics after the first observed grant=0) persists, just like `agent_topk_last_filter::run`
  // already does in the single-problem dispatch. Called fresh at every segment-boundary crossing
  // in `run()`.
  _CCCL_DEVICE _CCCL_FORCEINLINE partition_t make_partition_for_segment(const per_segment_state_t& s)
  {
    selected_reserve_op_t reserve_sel{&s.segment_counter->num_selected_written};
    candidate_reserve_op_t reserve_cand{
      &s.segment_counter->num_ties_written_to_back,
      static_cast<candidate_offset_t>(s.k_total),
      static_cast<candidate_offset_t>(s.num_of_kth_needed)};

    auto value_channel_sinks = [&] {
      if constexpr (keys_only)
      {
        return NullType{};
      }
      else
      {
        return value_channel_sinks_concrete_t{s.d_values_out, s.d_values_out};
      }
    }();
    detail::topk::topk_noop_candidate_callback_op callback_op{};

    // The identify-candidates op carries the segment's `kth_key_bits` (value, after the
    // value-holding sibling lands), and the partition holds it by value too, so the partition
    // ctor copy is what binds it to this segment's state.
    IdentifyCandidatesOpT identify_candidates_op{
      &s.segment_counter->kth_key_bits, s.pass, total_bits, decomposer};

    return partition_t{
      storage.partition_arena.get_partition_state(),
      reserve_sel,
      reserve_cand,
      s.d_keys_out,
      s.d_keys_out,
      value_channel_sinks,
      identify_candidates_op,
      callback_op};
  }

  // Templated on `IsFullTile` so the fast / slow-full path can skip the runtime
  // partial-vs-full branch. Callers must guarantee:
  //   - `IsFullTile == true`  -> `local_tile < s.num_full_tiles`
  //   - `IsFullTile == false` -> `local_tile == s.num_full_tiles && s.partial_items > 0`
  //
  // `partition` and `keys_source` are owned by the caller (`run()`) and reused across all tiles
  // of the same segment. The per-thread `cand_reserve_open` flag inside `partition` therefore
  // persists across calls, so once any thread observes a grant=0 from the back-grow-capped
  // candidate reserve the subsequent tiles dispatch through the cheaper
  // `HasCandidateStream=false` classifier specialisation that drops the per-item atomic
  // entirely (see `block_partition.cuh` doc on `cand_reserve_open`).
  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile(
    const per_segment_state_t& s,
    partition_t& partition,
    keys_source_t& keys_source,
    LargeSegmentTileOffsetT local_tile)
  {
    if constexpr (IsFullTile)
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

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch());
      h.complete_load(items);
      __syncthreads();
      partition.partition(storage.partition_arena.get_partition_scratch(), items, value_source);
    }
    else
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

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      __syncthreads();
      partition.partition(storage.partition_arena.get_partition_scratch(), items, s.partial_items, value_source);
    }
  }

public:
  // Drive this CTA's entire grid-strided last-filter pass.
  //
  // Flat shape: one tile per grid-stride iteration, stride = `gridDim.x`.
  // Mirrors the filter agent's flat-walk rewrite -- same motivation
  // (drop the chunk-walk machinery's persistent registers).
  //
  // Shape:
  //   * Early-return for CTAs whose first tile lands past the queue.
  //   * One `resolve_segment_state(blockIdx.x)` up front.
  //   * Construct one `partition_t` per segment, *outside* the tile loop, so its per-thread
  //     `cand_reserve_open` flag survives across tiles of the same segment. This is the
  //     mechanism that drops the per-item candidate-reserve atomic on subsequent tiles once
  //     the back-grow cap is hit (see `block_partition.cuh` doc). Without this, every tile
  //     of an entropy=0 (all-equal-keys) workload re-fires the per-item atomic, costing 30x
  //     vs main on `KeyT=int, Elements=2^24, Entropy=0.000`.
  //   * Per tile:
  //       - Refresh `state` when we cross a segment boundary
  //         (`tile_id >= state.queue_segment_end`). The refresh flushes the previous
  //         segment's partition via `partition.epilogue()` (no-op on the atomics strategy,
  //         a real flush on the accumulating sister classes), and rebuilds the partition
  //         + keys_source for the new segment, resetting `cand_reserve_open` to `true`.
  //       - Skip empty segments / wasted-tail tiles past data end.
  //       - Dispatch `process_tile<true>` for full tiles or
  //         `process_tile<false>` for the at-most-one trailing partial.
  //   * Final `partition.epilogue()` after the loop terminates the last active segment.
  //
  // `TilesPerChunk` is kept on the template signature for ABI/source
  // compatibility with the filter / histogram agents but is unused inside
  // the body; the static_assert preserves the policy contract.
  template <int TilesPerChunk>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run()
  {
    static_assert(TilesPerChunk == 2 || TilesPerChunk == 4 || TilesPerChunk == 8,
                  "agent_batched_topk_last_filter::run<TilesPerChunk> requires "
                  "TilesPerChunk to be a power of two in {2, 4, 8}.");

    const LargeSegmentTileOffsetT* const d_total_large_tiles =
      &d_large_segments_tile_offsets[num_large_segments];
    const LargeSegmentTileOffsetT total = *d_total_large_tiles;

    const LargeSegmentTileOffsetT first_tile = static_cast<LargeSegmentTileOffsetT>(blockIdx.x);
    const LargeSegmentTileOffsetT stride     = static_cast<LargeSegmentTileOffsetT>(gridDim.x);

    if (first_tile >= total)
    {
      return;
    }

    // Hoist first segment-state resolve + the partition / keys-source construction out of the
    // loop. Both `partition` and `keys_source` live across tiles of the same segment so
    // per-thread cross-tile state (notably `cand_reserve_open`) is preserved.
    const LargeSegmentTileOffsetT first_queue_idx = resolve_queue_idx(first_tile);
    per_segment_state_t state                     = resolve_segment_state(first_queue_idx);
    debug_print_state("last_filter_run_first", first_queue_idx, state);
    partition_t partition     = make_partition_for_segment(state);
    keys_source_t keys_source = make_keys_source_for_segment(state);

    for (LargeSegmentTileOffsetT tile_id = first_tile; tile_id < total; tile_id += stride)
    {
      // Segment refresh -- only when the cached segment no longer covers `tile_id`. Flush the
      // previous segment's partition (terminal accumulation flush; no-op on atomics) and
      // rebuild for the new segment so `cand_reserve_open` resets to `true`.
      if (tile_id >= state.queue_segment_end)
      {
        partition.epilogue();
        const LargeSegmentTileOffsetT next_queue_idx = resolve_queue_idx(tile_id);
        state                                        = resolve_segment_state(next_queue_idx);
        debug_print_state("last_filter_run_refresh", next_queue_idx, state);
        partition   = make_partition_for_segment(state);
        keys_source = make_keys_source_for_segment(state);
      }

      if (state.empty)
      {
        continue;
      }

      const OffsetT local_tile = static_cast<OffsetT>(tile_id - state.slab_base);
      if (local_tile < state.num_full_tiles)
      {
        process_tile<true>(state, partition, keys_source, static_cast<LargeSegmentTileOffsetT>(local_tile));
      }
      else if (local_tile == state.num_full_tiles && state.partial_items > OffsetT{0})
      {
        process_tile<false>(
          state, partition, keys_source, static_cast<LargeSegmentTileOffsetT>(state.num_full_tiles));
      }
      // else: wasted-tail tile beyond segment data; skip implicitly via the grid-stride loop.
    }

    // Final flush of the last active segment.
    partition.epilogue();
  }
};

} // namespace detail::batched_topk
CUB_NAMESPACE_END
