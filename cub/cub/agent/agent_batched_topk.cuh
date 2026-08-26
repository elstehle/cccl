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
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/partition_storage_layout.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/detail/warpspeed/make_warp_uniform.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_topk_common.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/argument>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// Atomic counters used by the small-segment kernel to (a) enqueue large segments into the large-segment work queue
// and (b) elect the last block to run the epilogue scan over the queued tile counts. `alignas(128)` isolates each
// counter on its own cache line for performance.
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

// Queue length / queue index type for the multi-CTA-per-segment agents and kernels. Unconditionally 32-bit and
// unsigned: the public entry rejects `num_segments > INT_MAX` outright (see `dispatch`, which returns
// cudaErrorInvalidValue), so the queue can never be longer than that. Keeping it 32 bits keeps `resolve_queue_idx`'s
// `UpperBound` search bound and the offset-table indexing 32-bit, which matters because that binary search sits in the
// per-tile inner loop of every multi-CTA kernel.
//
// This is only the *length / index* type. The queue's element array (`baseline_kernel_args::d_large_segments_ids`) and
// `batched_topk_counters` keep the segment-count argument's own element type, which is the existing kernel ABI the
// worker agent's enqueue writes through -- narrowing those would mistype the queue itself.
using queue_segment_count_t = ::cuda::std::uint32_t;

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

  using segment_size_val_t = typename ::cuda::args::__traits<SegmentSizeParameterT>::element_type;
  using num_segments_val_t = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;
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
  static constexpr bool only_small_segments = ::cuda::args::__traits<SegmentSizeParameterT>::highest <= tile_size;

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
  // Scans the queued per-segment tile counts. Typed on `segment_size_val_t` to match the epilogue's load/store and the
  // thread-local array: a tile count is bounded by its segment's size, so this is always wide enough, whereas a
  // hardcoded `int` breaks for any segment-size type wider than `int`.
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
    if (segment_id >= params::get_param(num_segments, 0))
    {
      return;
    }

    constexpr bool is_full_tile = ::cuda::args::__traits<SegmentSizeParameterT>::is_constant
                               && ::cuda::args::__traits<SegmentSizeParameterT>::lowest == tile_size;

    // Resolve Segment Parameters
    const auto segment_size = params::__get_and_clamp_param_to_nonnegative(segment_sizes, segment_id);
    if (!only_small_segments && segment_size > tile_size)
    {
      // Enqueue large segment
      // TODO(topk): once the large-segment worker is wired up, skip enqueue when the effective k is 0 (nothing to
      // select) so an empty/zero-k large segment does not schedule pointless work.
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
      // Process small segment. Clamp `k` (already floored to >= 0) to the segment size in a width holding both operands
      // so a `k` type narrower than the segment size cannot wrap; the result fits the segment-size type.
      const auto k = static_cast<decltype(segment_size)>(
        (::cuda::std::min) (static_cast<::cuda::std::uint64_t>(
                              params::__get_and_clamp_param_to_nonnegative(k_param, segment_id)),
                            static_cast<::cuda::std::uint64_t>(segment_size)));
      // Nothing to select for an empty segment (including a negative size or a negative `k`, both clamped to 0) or a
      // zero k: skip the block work, leaving its output untouched (also keeps the block primitive's `valid_items in
      // [1, tile_items]` precondition). We must not `return` here -- the large-segment epilogue below is unconditional
      // participation, so bailing out would drop this block from the retirement count and stall the epilogue scan.
      if (k != 0)
      {
        const auto direction = select_directions.get_param(segment_id);

        // Determine padding key based on direction
        const key_t padding_key =
          (direction == detail::topk::select::max)
            ? ::cuda::std::numeric_limits<key_t>::lowest()
            : (::cuda::std::numeric_limits<key_t>::max)();

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
      } // if (k != 0)
    }

    // Epilogue: Scan queued large segment sizes (in tiles not elements) for load balancing search in the large segment
    // agent. The scan additionally publishes `total_large_tiles` into the trailing slot, i.e.
    // `d_large_segments_tile_offsets[num_large_segments]`, which is how the multi-CTA-per-segment kernels learn the
    // total tile count without a separate device-side counter.
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
      // Not `const`: `BlockScan::ExclusiveSum` takes the prefix callback by non-const reference and invokes it, which a
      // `mutable` lambda cannot satisfy through a const object.
      auto prefix_callback_op = [running_total = segment_size_val_t{0}](segment_size_val_t block_aggregate) mutable {
        auto old_running_total = running_total;
        running_total += block_aggregate;
        return old_running_total;
      };
      // Loop one item past `num_large_segments` so the exclusive scan also publishes the total aggregate into the
      // trailing (sentinel) slot. The trailing iteration's `BlockLoad` keeps `valid_items` capped at
      // `num_large_segments`, so the out-of-bounds item it would otherwise read from uninitialised memory is
      // substituted with the default `0`; only the `BlockStore` extends to the sentinel.
      const int num_large_segments_with_sentinel = static_cast<int>(num_large_segments) + 1;
      _CCCL_PRAGMA_NOUNROLL()
      for (int large_segment_offset = 0; large_segment_offset < num_large_segments_with_sentinel;
           large_segment_offset += epilogue_tile_size)
      {
        segment_size_val_t segment_tile_offsets[epilogue_items_per_thread];
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
// Each agent holds only per-launch state as members: iterators-of-iterators, parameter packs, the
// `SegmentIdProviderT` (maps a queue index to the original segment id -- identity on the all-large path, an index
// into `d_large_segments_ids` on the mixed path), the per-segment tile-offset table (exclusive scan of per-segment
// tile counts; together with `total_large_tiles` it drives block-to-segment mapping), and the per-segment
// counter/histogram/back-buffer arrays indexed by `queue_idx`. Per-segment single-problem state (input/output
// iterators, counter fields, back-buffer slabs) is computed locally inside `run()` from `(global_tile_id,
// segment_id)`, where `segment_id` is resolved by an on-device binary search over the tile-offset table.
//
// Block-to-segment mapping: each block handles exactly one tile of one segment, so `gridDim.x` covers the total
// large-tile count and the local tile id is derived from a binary search on the tile-offset table.
//
// Direction lowering: the dispatch keeps `SelectDirectionParameterT` uniform across segments and passes
// `SelectDirection` as a template NTTP; per-segment direction is deferred.
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// agent_batched_topk_histogram: segmented analog of `AgentTopKHistogram`.
//
// Each CTA processes a *chunk* of `tiles_per_chunk` consecutive tiles per grid-stride iteration. Inside the chunk,
// the agent groups tiles by segment: it initialises an smem histogram once when it first sees a segment, atomic-adds
// every tile's keys into it, and merges into the per-segment global histogram only when the segment changes (or the
// chunk ends). For workloads dominated by a single large segment this amortises one init + one merge across all of a
// CTA's tiles for that segment.
//
// The per-segment prefix-sum / bucket-finder epilogue (last-block election, prefix-sum, k-th-bucket scan, counter
// update, optional histogram reset) lives in a separate `device_segmented_topk_finalize_histogram_kernel` that runs
// after this kernel, so no per-tile finalize step runs on this path.
//
// `FilterOpT` defaults to `topk_pass_through_filter_op` (pass 0) and is wrapped in `topk_candidate_filter_op` by the
// kernel for the "unbuffered scout" pass (candidate set exceeds the back buffer). The unbuffered scout always loads
// from the original `d_keys_in` per segment (see the single-source invariant on `AgentTopKHistogram`).
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputItItT,
          typename ExtractBinOpT,
          typename SegmentSizeParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename OffsetT,
          typename LargeSegmentsCountItT,
          typename SegmentCountT,
          typename FilterOpT = detail::batched_topk::topk_pass_through_filter_op>
struct agent_batched_topk_histogram
{
  using keys_in_it_t = it_value_t<KeyInputItItT>;
  using key_t        = it_value_t<keys_in_it_t>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;

  using keys_source_t = detail::topk::
    tile_data_source_t<keys_in_it_t, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;

  // Block-wide histogram primitive: owns the smem bins plus the per-tile binning and histogram
  // lifecycle. Tile iteration, segment handling, and data loading stay in this agent.
  using tile_histogram_t = detail::batched_topk::tile_histogram<block_threads, num_buckets, OffsetT, ExtractBinOpT>;

  // Per-segment cache in smem (not per-thread registers) so all use sites read through one canonical handle.
  // Thread 0 writes it on each segment boundary; every other thread reads it. Dereferenced at the use site (not
  // cached into register locals) so the compiler doesn't replicate the same scalar across all threads.
  struct active_segment_state_t
  {
    // Half-open tile-space window of the segment owned by this `active_segment` slot:
    // `[slab_base, segment_end)`. `chunk_cursor < segment_end` is the cheap "still in the
    // active segment" check that gates the segment-state refresh.
    LargeSegmentTileOffsetT slab_base;
    LargeSegmentTileOffsetT segment_end;

    // Per-segment tile-shape state.
    OffsetT num_full_tiles;
    OffsetT partial_items;

    // Per-segment global-slab pointer for the merge / per-segment input iterator for the
    // tile load.
    OffsetT* segment_histogram;
    keys_in_it_t d_keys_in;
  };

  // Smem holds the histogram, the keys-source state / scratch, and the smem-resident `active_segment` cache.
  // (The per-segment prefix-sum scratch lives in `device_segmented_topk_finalize_histogram_kernel`.)
  struct _TempStorage
  {
    typename tile_histogram_t::TempStorage histogram;
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
  SegmentIdProviderT segment_id_provider;
  const LargeSegmentTileOffsetT* d_large_segments_tile_offsets;
  OffsetT* d_segment_histograms;
  FilterOpT filter_op;

  // Iterator yielding the number of enqueued large segments when dereferenced. Stored as the iterator (a kernel
  // parameter) so the agent matches the kernel's parameter shape; the `total_large_tiles` sentinel read
  // (`d_large_segments_tile_offsets[*large_segments_count_it]`) and the `UpperBound` bound are deferred to use sites.
  LargeSegmentsCountItT large_segments_count_it;

  // Owns the smem histogram plus per-tile binning and lifecycle; holds `extract_bin_op`.
  tile_histogram_t hist;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_batched_topk_histogram(
    TempStorage& ts,
    KeyInputItItT d_key_segments_it,
    SegmentSizeParameterT segment_sizes,
    SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* d_large_segments_tile_offsets,
    OffsetT* d_segment_histograms,
    ExtractBinOpT extract_bin_op,
    LargeSegmentsCountItT large_segments_count_it,
    FilterOpT filter_op = {})
      : temp_storage(ts.Alias())
      , d_key_segments_it(d_key_segments_it)
      , segment_sizes(segment_sizes)
      , segment_id_provider(segment_id_provider)
      , d_large_segments_tile_offsets(d_large_segments_tile_offsets)
      , d_segment_histograms(d_segment_histograms)
      , filter_op(filter_op)
      , large_segments_count_it(large_segments_count_it)
      , hist(temp_storage.histogram, extract_bin_op)
  {}

private:
  // Thread 0 resolves the segment containing `cursor` and publishes the result to smem. Other
  // threads only need to participate in the surrounding `__syncthreads()`. The caller is
  // responsible for the publish barrier after this returns.
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_segment_state(LargeSegmentTileOffsetT cursor)
  {
    if (threadIdx.x == 0)
    {
      const LargeSegmentTileOffsetT queue_idx =
        UpperBound(d_large_segments_tile_offsets, static_cast<SegmentCountT>(*large_segments_count_it), cursor) - 1;
      const auto segment_id                   = segment_id_provider[queue_idx];
      const OffsetT num_items =
        static_cast<OffsetT>(params::__get_and_clamp_param_to_nonnegative(segment_sizes, segment_id));
      const OffsetT num_full_tiles            = num_items / static_cast<OffsetT>(tile_items);
      const OffsetT partial_items             = num_items - num_full_tiles * static_cast<OffsetT>(tile_items);
      const OffsetT seg_tile_count            = num_full_tiles + (partial_items > 0 ? OffsetT{1} : OffsetT{0});
      const LargeSegmentTileOffsetT slab_base = d_large_segments_tile_offsets[queue_idx];

      temp_storage.active_segment.slab_base         = slab_base;
      temp_storage.active_segment.segment_end       = slab_base + static_cast<LargeSegmentTileOffsetT>(seg_tile_count);
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
    // This is the first and only point at which this kernel touches the global histogram slabs, so it is where the
    // dependency on the init kernel that zeroed them has to be honored -- everything before it (resolving segments,
    // loading tiles, binning into shared memory) is independent and overlaps with that kernel under PDL. Same
    // placement as `agent_radix_sort_histogram`, which syncs only before `AccumulateGlobalHistograms`. Calling this
    // once per flush is fine: the intrinsic is idempotent, and all threads reach it uniformly.
    _CCCL_PDL_GRID_DEPENDENCY_SYNC();
    hist.flush(temp_storage.active_segment.segment_histogram);
  }

  // Bring up a freshly-resolved segment-stretch: thread 0 writes the new `active_segment`,
  // then everyone zeros the smem histogram. The two `__syncthreads()` bracket the smem
  // writes against the reads/writes that surround them.
  _CCCL_DEVICE _CCCL_FORCEINLINE void enter_segment(LargeSegmentTileOffsetT cursor)
  {
    load_segment_state(cursor);
    __syncthreads();
    hist.reset();
    __syncthreads();
  }

  // "Leave the current segment, enter the next one" when a chunk crosses a segment boundary. Flushes the current
  // smem histogram, refreshes `active_segment` for `cursor`, and re-inits the smem histogram. The interior
  // `__syncthreads()` brackets the `active_segment` slot against the prior flush and the upcoming thread-0 write.
  _CCCL_DEVICE _CCCL_FORCEINLINE void switch_to_segment(LargeSegmentTileOffsetT cursor)
  {
    __syncthreads();
    flush_active_segment();
    __syncthreads();
    enter_segment(cursor);
  }

  // `BlockLoad` algorithms that route their inter-thread transfer through the `keys_source_scratch` smem region
  // (TRANSPOSE / WARP_TRANSPOSE / WARP_TRANSPOSE_TIMESLICED) need a `__syncthreads()` between two consecutive tile
  // loads so the next tile's writes don't clobber the previous tile's reads. DIRECT / VECTORIZE issue per-thread
  // `LDG.E.{,2,4}` straight into registers and never touch the scratch, so that inter-tile barrier is dead work.
  static constexpr bool tile_load_kind_uses_smem =
    AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_direct
    && AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_vectorize;

  // Process one full tile of the active segment at local index `local_tile`. The caller owns the long-lived
  // `keys_source_t` (constructed once per segment-stretch rather than per tile).
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_full_tile_at(keys_source_t& keys_source, OffsetT local_tile)
  {
    const OffsetT tile_base = local_tile * static_cast<OffsetT>(tile_items);
    keys_source.set_tile_base(tile_base);

    if constexpr (tile_load_kind_uses_smem)
    {
      __syncthreads();
    }
    key_t items[items_per_thread];
    auto h = keys_source.submit_load(temp_storage.keys_source_scratch);
    h.complete_load(items);
    hist.add_full(items, filter_op);
  }

public:
  // Drive this CTA's entire grid-strided histogram pass. Per-segment cached state lives in shared memory
  // (`temp_storage.active_segment`); the segment-state refresh / smem-histogram bookkeeping is in the helpers above.
  // `TilesPerChunk` is a compile-time NTTP so ptxas can reason about the per-chunk stride and loop bound statically.
  template <int TilesPerChunk>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run()
  {
    // The slow-path bit-decomposition is hard-wired for power-of-two chunk sizes up to 8; other sizes would need
    // extra branches (and lose the "exactly one tile-count branch per stretch" property).
    static_assert(TilesPerChunk == 2 || TilesPerChunk == 4 || TilesPerChunk == 8,
                  "agent_batched_topk_histogram::run<TilesPerChunk> requires "
                  "TilesPerChunk to be a power of two in {2, 4, 8}.");

    const LargeSegmentTileOffsetT* const d_total_large_tiles =
      &d_large_segments_tile_offsets[static_cast<SegmentCountT>(*large_segments_count_it)];

    constexpr LargeSegmentTileOffsetT chunk_size_v  = static_cast<LargeSegmentTileOffsetT>(TilesPerChunk);
    const LargeSegmentTileOffsetT stride            = static_cast<LargeSegmentTileOffsetT>(gridDim.x) * chunk_size_v;
    const LargeSegmentTileOffsetT first_chunk_start = static_cast<LargeSegmentTileOffsetT>(blockIdx.x) * chunk_size_v;

    // CTA whose first chunk lands past the queue's last tile has no work to do.
    if (first_chunk_start >= *d_total_large_tiles)
    {
      return;
    }

    // First segment-state load hoisted out of the loop; subsequent iterations only need the cheaper
    // `switch_to_segment` check on a segment-boundary crossing.
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

      // Fast-path check: the chunk fits entirely inside the active segment's full-tile range -- exactly
      // `TilesPerChunk` full tiles from one segment, no switching / partial / clipping. The tile loop below sees
      // only `local_tile_start` and a fully unrolled run of `TilesPerChunk` loads.
      const LargeSegmentTileOffsetT slab_base = temp_storage.active_segment.slab_base;
      const OffsetT num_full_tiles_in_seg     = temp_storage.active_segment.num_full_tiles;
      const LargeSegmentTileOffsetT full_tile_boundary =
        slab_base + static_cast<LargeSegmentTileOffsetT>(num_full_tiles_in_seg);

      if (chunk_start + chunk_size_v <= full_tile_boundary)
      {
        // ----- Fast path: TilesPerChunk full tiles, one segment, no switching. -----
        // `chunk_start + chunk_size_v <= full_tile_boundary <= *d_total_large_tiles`, so the chunk can't have been
        // clipped at the queue tail either -- `chunk_end == chunk_start + chunk_size_v`, enabling the unrolled loop.
        const OffsetT local_tile_start = static_cast<OffsetT>(chunk_start - slab_base);
        keys_source_t keys_source{temp_storage.active_segment.d_keys_in, temp_storage.keys_source_state};

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < TilesPerChunk; ++i)
        {
          process_full_tile_at(keys_source, static_cast<OffsetT>(local_tile_start + i));
        }
        keys_source.invalidate();
        continue;
      }

      // ----- Slow path: chunk straddles a segment boundary, hits a partial-tile slot, or is clipped at the queue
      // tail. Walk segment-stretches; each processes < `TilesPerChunk` full tiles, so the per-stretch power-of-two
      // bit decomposition only needs to cover `[0, TilesPerChunk - 1]`. The walk visits at most `TilesPerChunk`
      // segments (each occupies >= 1 queue-idx slot).
      LargeSegmentTileOffsetT chunk_cursor = chunk_start;
      while (chunk_cursor < chunk_end)
      {
        if (chunk_cursor >= temp_storage.active_segment.segment_end)
        {
          switch_to_segment(chunk_cursor);
        }

        const LargeSegmentTileOffsetT seg_slab_base      = temp_storage.active_segment.slab_base;
        const OffsetT seg_num_full                       = temp_storage.active_segment.num_full_tiles;
        const OffsetT local_tile_start                   = static_cast<OffsetT>(chunk_cursor - seg_slab_base);
        const LargeSegmentTileOffsetT remaining_in_chunk = chunk_end - chunk_cursor;
        const OffsetT full_tiles_remaining_in_seg =
          (local_tile_start < seg_num_full) ? (seg_num_full - local_tile_start) : OffsetT{0};
        const OffsetT full_tiles_in_stretch =
          (::cuda::std::min) (static_cast<OffsetT>(remaining_in_chunk), full_tiles_remaining_in_seg);

        // Power-of-two bit-decomposition of `full_tiles_in_stretch ∈ [0, TilesPerChunk-1]`: each `if` covers one
        // bit and is the only branch in the stretch; the inner `for` is statically sized so it unrolls cleanly.
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

        // Partial-tile slot bookkeeping. The partial-tile load + bin is owned by the finalize-histogram kernel;
        // here the slot is only "consumed" by `tiles_consumed` below so the chunk walk doesn't stall on it.
        const OffsetT next_local_tile = local_tile_start + full_tiles_in_stretch;
        const bool reaches_partial_slot =
          (next_local_tile == seg_num_full) && (temp_storage.active_segment.partial_items > 0)
          && (full_tiles_in_stretch + OffsetT{1} <= static_cast<OffsetT>(remaining_in_chunk));
        const OffsetT tiles_consumed = full_tiles_in_stretch + (reaches_partial_slot ? OffsetT{1} : OffsetT{0});
        if (tiles_consumed == OffsetT{0})
        {
          keys_source.invalidate();
          break;
        }
        chunk_cursor += static_cast<LargeSegmentTileOffsetT>(tiles_consumed);
        keys_source.invalidate();
      }
    }

    // Final flush: merge the last active segment-stretch's smem histogram into its global slab. Unconditional --
    // the early-return above already filtered out CTAs with no work, so we've entered at least one segment.
    __syncthreads();
    flush_active_segment();
  }
};

//---------------------------------------------------------------------
// agent_batched_topk_filter_partition: segmented analog of `agent_topk_filter_partition`.
//
// Handles both the early-stop and buffered modes in one agent type (same as the single-problem version), selected at
// runtime per segment via the segment's counter state. Each block processes exactly one tile of one segment. The
// buffered branch accumulates a per-segment histogram in smem, atomically merging into
// `d_segment_histograms + queue_idx * num_buckets`; the prefix-sum + bucket-finder epilogue runs in
// `device_segmented_topk_finalize_filter_kernel`.
//
// Per-segment double-buffering: the global `DoubleBuffer<key_t>` `selector` is flipped once per pass on the host
// (safe because `num_passes` is uniform across all segments). The per-segment back buffers are slabs of
// `candidate_buffer_length` items at `d_segment_*_key_buf + queue_idx * candidate_buffer_length` (and the
// value channel).
//---------------------------------------------------------------------

// How the filter/partition agent processes one segment on a given pass. Mutually exclusive, recomputed per segment
// in `resolve_segment_state` from the counter's length vs k. Orthogonal to the mode is the
// `load_from_candidates_buffer` source selector (any non-empty mode may read the input or the prior pass's buffer).
enum class segment_processing_mode
{
  empty, // counter's `num_candidates_in == 0` -> every tile of this segment is a no-op
  early_stop, // `current_len == current_k` -> scatter survivors straight out, no histogram
  buffered, // !early_stop && fits-in-back-buffer && cost-justified -> partition into the candidates buffer
  unbuffered, // scout/default -> re-histogram the full input without buffering
};

template <typename AgentTopKPolicyT,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          typename DecomposerT,
          typename SegmentSizeParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename OffsetT,
          typename OutOffsetT,
          bool LazyValueLoad   = false,
          bool InlinedClassify = false>
struct agent_batched_topk_filter_partition
{
  using keys_in_it_t    = it_value_t<KeyInputItItT>;
  using values_in_it_t  = it_value_t<ValueInputItItT>;
  using values_out_it_t = it_value_t<ValueOutputItItT>;
  using keys_out_it_t   = it_value_t<KeyOutputItItT>;

  using key_t     = it_value_t<keys_in_it_t>;
  using value_t   = it_value_t<values_in_it_t>;
  using counter_t = detail::batched_topk::counter<key_t, OffsetT, OutOffsetT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<value_t, cub::NullType>;

  // Mirrors `agent_batched_topk_histogram::tile_load_kind_uses_smem`: for DIRECT / VECTORIZE the `BlockLoad` goes
  // straight into registers without touching the shared scratch, so the pre-`submit_load` `__syncthreads()` is dead
  // work. The post-`complete_load` sync is *kept* in every mode because `keys_source_scratch` and `partition_scratch`
  // alias through the smem union in `partition_storage_layout_for_t`, so without it the next tile's `partition` could
  // clobber bytes the just-completed load (or the previous tile's `partition`) still owned.
  static constexpr bool tile_load_kind_uses_smem =
    AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_direct
    && AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_vectorize;

  // The per-segment prefix-sum + kth-bucket scan (driving the next pass's counter state) lives in
  // `device_segmented_topk_finalize_filter_kernel`, not in this agent.

  static constexpr bool effective_lazy_value_load = LazyValueLoad && !keys_only;

  // Multi-source key / value channels mirror the single-problem agent. The "buffer" source is
  // the candidate slab carried over from the previous pass.
  using key_source_input_t = detail::topk::
    tile_data_source_t<keys_in_it_t, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using key_source_buffer_t = detail::topk::
    tile_data_source_t<key_t*, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using keys_source_t = detail::topk::multi_source_data_source<key_source_input_t, key_source_buffer_t, OffsetT>;

  using value_source_input_t =
    detail::topk::direct_data_source<values_in_it_t, block_threads, items_per_thread, OffsetT>;
  using value_source_buffer_t = detail::topk::direct_data_source<value_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t = detail::topk::multi_source_data_source<value_source_input_t, value_source_buffer_t, OffsetT>;

  // Value multi-source held for the whole CTA run (the value analog of `keys_source_t`). `keys_only` collapses it to
  // `NullType` (built but never loaded from); pairs hold the real source. Built once and passed by reference.
  using held_value_source_t = ::cuda::std::conditional_t<keys_only, NullType, value_source_t>;

  using val_out_t               = values_out_it_t;
  using buffered_cand_val_out_t = value_t*;
  using buffered_cand_key_out_t = key_t*;

  using buffered_value_channel_sinks_concrete_t =
    detail::topk::value_channel_sinks_t<val_out_t, buffered_cand_val_out_t>;
  using buffered_value_channel_sinks_t =
    ::cuda::std::conditional_t<keys_only, NullType, buffered_value_channel_sinks_concrete_t>;

  using early_stop_value_channel_sinks_concrete_t = detail::topk::value_channel_sinks_filter_t<val_out_t>;
  using early_stop_value_channel_sinks_t =
    ::cuda::std::conditional_t<keys_only, NullType, early_stop_value_channel_sinks_concrete_t>;

  using agent_value_t = ::cuda::std::conditional_t<keys_only, NullType, value_t>;
  using agent_value_data_source_scratch_t =
    ::cuda::std::conditional_t<keys_only, NullType, typename value_source_t::ScratchStorage>;

  using selected_offset_t  = OutOffsetT;
  using candidate_offset_t = OffsetT;

  using selected_reserve_op_t  = detail::topk::atomic_reserve_range_op<selected_offset_t>;
  using candidate_reserve_op_t = detail::topk::atomic_reserve_range_op<candidate_offset_t>;

  using histogram_callback_op_t = detail::batched_topk::topk_histogram_callback_op<ExtractBinOpT, OffsetT>;
  using identify_selected_op_t  = detail::batched_topk::topk_identify_selected_op<IdentifyCandidatesOpT>;

  // Block-wide histogram primitive shared with the histogram agent: owns the smem bins plus the
  // per-tile binning and lifecycle. Only the buffered / unbuffered (non-early_stop) modes use it.
  using tile_histogram_t = detail::batched_topk::tile_histogram<block_threads, num_buckets, OffsetT, ExtractBinOpT>;

  using buffered_partition_t = detail::topk::block_partition_atomics<
    block_threads,
    items_per_thread,
    InlinedClassify,
    key_t,
    selected_offset_t,
    candidate_offset_t,
    selected_reserve_op_t,
    candidate_reserve_op_t,
    keys_out_it_t,
    buffered_cand_key_out_t,
    IdentifyCandidatesOpT,
    histogram_callback_op_t,
    buffered_value_channel_sinks_t,
    agent_value_t,
    agent_value_data_source_scratch_t,
    effective_lazy_value_load>;

  using early_stop_filter_t = detail::topk::block_filter_atomics<
    block_threads,
    items_per_thread,
    InlinedClassify,
    key_t,
    selected_offset_t,
    selected_reserve_op_t,
    keys_out_it_t,
    identify_selected_op_t,
    early_stop_value_channel_sinks_t,
    agent_value_t,
    agent_value_data_source_scratch_t,
    effective_lazy_value_load>;

  using buffered_storage_layout_t =
    detail::topk::partition_storage_layout_for_t<buffered_partition_t, typename keys_source_t::ScratchStorage>;
  using early_stop_storage_layout_t =
    detail::topk::partition_storage_layout_for_t<early_stop_filter_t, typename keys_source_t::ScratchStorage>;

  // The per-segment prefix-sum + kth-bucket scan lives in `device_segmented_topk_finalize_filter_kernel`, so smem
  // here is just the per-mode arms used during the tile body. `keys_source_t` (multi-source) doesn't publish its own
  // `TempStorage` (see `tile_data_source.cuh`), so the agent holds one `TempStorage` per child source.
  struct _TempStorage
  {
    // Persistent keys-source state, hoisted out of the per-mode `arms` union so the keys source can persist across
    // segments of different modes: `run()` builds it once and re-targets it via `set_inputs` on each segment
    // boundary instead of reconstructing it. The per-load staging scratch stays in each arm's `arena`. For the
    // current sync loads this state is empty, so the hoist costs no extra smem.
    typename key_source_input_t::TempStorage key_src_input_state;
    typename key_source_buffer_t::TempStorage key_src_buffer_state;
    // Persistent value-source state, hoisted for the same reason as the keys-source state above so the value source
    // can also persist / be re-targeted instead of rebuilt. The current `direct` value source has empty state (and
    // it's empty for `keys_only` too), so these slots add no smem.
    typename value_source_input_t::TempStorage val_src_input_state;
    typename value_source_buffer_t::TempStorage val_src_buffer_state;
    union arms_t
    {
      struct buffered_t
      {
        typename tile_histogram_t::TempStorage histogram;
        buffered_storage_layout_t arena;
      } buffered;

      struct early_stop_t
      {
        early_stop_storage_layout_t arena;
      } early_stop;
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
  SegmentIdProviderT segment_id_provider;
  const LargeSegmentTileOffsetT* d_large_segments_tile_offsets;
  counter_t* d_segment_counters;
  OffsetT* d_segment_histograms;
  key_t* d_segment_in_key_buf;
  value_t* d_segment_in_val_buf;
  key_t* d_segment_out_key_buf;
  value_t* d_segment_out_val_buf;
  int total_bits;
  DecomposerT decomposer;
  OffsetT candidate_buffer_length;
  // Cost-savings threshold for entering the buffered chain (see `run()`): buffering pays off only when
  // `num_candidates_out <= segment_num_items / coefficient`, else the candidate-buffer write cost outweighs the next
  // pass's read savings. Threaded from the dispatch in lock-step with `candidate_buffer_length`.
  OffsetT candidate_buffer_coefficient;
  // Narrowed segment count (32-bit when the count fits) so `resolve_queue_idx`'s `UpperBound` and the
  // offset-table indexing stay 32-bit.
  queue_segment_count_t num_large_segments;

  // Owns the smem histogram plus per-tile binning and lifecycle, over the buffered arm's bins.
  // Only the buffered / unbuffered modes touch it (guarded by `segment_uses_smem_histogram`).
  tile_histogram_t hist;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_batched_topk_filter_partition(
    TempStorage& ts,
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* d_large_segments_tile_offsets,
    counter_t* d_segment_counters,
    OffsetT* d_segment_histograms,
    key_t* d_segment_in_key_buf,
    value_t* d_segment_in_val_buf,
    key_t* d_segment_out_key_buf,
    value_t* d_segment_out_val_buf,
    ExtractBinOpT extract_bin_op,
    int total_bits,
    DecomposerT decomposer,
    OffsetT candidate_buffer_length,
    OffsetT candidate_buffer_coefficient,
    queue_segment_count_t num_large_segments)
      : storage(ts.Alias())
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , segment_id_provider(segment_id_provider)
      , d_large_segments_tile_offsets(d_large_segments_tile_offsets)
      , d_segment_counters(d_segment_counters)
      , d_segment_histograms(d_segment_histograms)
      , d_segment_in_key_buf(d_segment_in_key_buf)
      , d_segment_in_val_buf(d_segment_in_val_buf)
      , d_segment_out_key_buf(d_segment_out_key_buf)
      , d_segment_out_val_buf(d_segment_out_val_buf)
      , total_bits(total_bits)
      , decomposer(decomposer)
      , candidate_buffer_length(candidate_buffer_length)
      , candidate_buffer_coefficient(candidate_buffer_coefficient)
      , num_large_segments(num_large_segments)
      , hist(storage.arms.buffered.histogram, extract_bin_op)
  {}

private:
  // Build the per-segment buffered-mode sinks. The candidate values are deposited into the
  // per-segment slab of the back-buffer-out (selected via the host-side double-buffer flip).
  template <typename ValueOutSinkT>
  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_buffered_value_channel_sinks(
    ValueOutSinkT values_out_sink, [[maybe_unused]] buffered_cand_val_out_t cand_val_out)
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
  // Per-segment derived state cached across tiles of the same segment (same pattern as the histogram agent's cache):
  // re-derived only when the chunk crosses a segment boundary; held in registers across same-segment tiles.
  struct per_segment_state_t
  {
    // How this segment is processed this pass (empty / early_stop / buffered / unbuffered).
    segment_processing_mode mode;
    // Orthogonal to `mode`: whether this pass's input for the segment comes from the prior pass's
    // candidates buffer (source B) rather than the original input (source A). Read from the counter.
    bool load_from_candidates_buffer;

    keys_in_it_t d_keys_in{};
    keys_out_it_t d_keys_out{};
    [[maybe_unused]] values_in_it_t d_values_in{};
    [[maybe_unused]] values_out_it_t d_values_out{};

    key_t* in_key_buf;
    key_t* out_key_buf;
    [[maybe_unused]] value_t* in_val_buf;
    [[maybe_unused]] value_t* out_val_buf;

    OffsetT* segment_histogram;
    counter_t* segment_counter;

    // `identify_candidates_op_t` has no default ctor for some `T`, so cache its ctor inputs here and build it on
    // demand in the per-mode tile body (cheap: a `key_prefix_storage_t*` copy + an `int` shift).
    int pass;
    OffsetT input_length_actual;
    OffsetT num_full_tiles;
    OffsetT partial_items;
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
    // Broadcast lane-0's result and route it through `makeWarpUniform` so ptxas recognises `queue_idx` as warp-
    // uniform; that keeps downstream `queue_idx`-dependent atomics (the `atomicAdd` on
    // `&segment_counter->num_ties_written_to_back` in `back_grow_capped_reserve_op`) eligible for warp-aggregation.
    // Narrow the broadcast to the queue-index type before the call. A queue index is bounded by the queue length, and
    // hence by `num_segments <= INT_MAX`, so this cannot truncate; it also picks `makeWarpUniform`'s 32-bit overload
    // unambiguously, which is the one that lowers to a single CREDUX.
    return static_cast<LargeSegmentTileOffsetT>(
      detail::warpspeed::makeWarpUniform(static_cast<queue_segment_count_t>(__shfl_sync(0xffffffff, queue_idx_lane0, 0))));
  }

  // Build the per-segment cached state for `queue_idx`. Pure function of `queue_idx` and the per-launch agent state.
  _CCCL_DEVICE _CCCL_FORCEINLINE per_segment_state_t resolve_segment_state(LargeSegmentTileOffsetT queue_idx, int pass)
  {
    per_segment_state_t s{};
    s.slab_base = d_large_segments_tile_offsets[queue_idx];
    // The offset table is sized `num_large_segments + 1` (sentinel at the end stores
    // `total_large_tiles`), so the next-slot read is in-bounds for every valid `queue_idx`.
    s.queue_segment_end   = d_large_segments_tile_offsets[queue_idx + 1];
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

    const OffsetT counter_input_length = s.segment_counter->num_candidates_in;
    s.load_from_candidates_buffer      = s.segment_counter->load_from_candidates_buffer;

    s.pass = pass;

    if (counter_input_length == 0)
    {
      s.mode = segment_processing_mode::empty;
      return s;
    }

    const OffsetT segment_num_items =
      static_cast<OffsetT>(params::__get_and_clamp_param_to_nonnegative(segment_sizes, segment_id));
    s.input_length_actual           = counter_input_length;

    const OutOffsetT current_k = s.segment_counter->k;
    const OffsetT current_len  = s.segment_counter->num_candidates_out;
    const bool early_stop      = (current_len == static_cast<OffsetT>(current_k));
    const bool will_buffer     = !early_stop && (current_len <= candidate_buffer_length)
                          && (current_len <= segment_num_items / candidate_buffer_coefficient);
    s.mode = early_stop  ? segment_processing_mode::early_stop
           : will_buffer ? segment_processing_mode::buffered
                         : segment_processing_mode::unbuffered;

    s.in_key_buf  = d_segment_in_key_buf + queue_idx * candidate_buffer_length;
    s.out_key_buf = will_buffer ? (d_segment_out_key_buf + queue_idx * candidate_buffer_length) : nullptr;
    if constexpr (!keys_only)
    {
      s.in_val_buf =
        s.load_from_candidates_buffer ? (d_segment_in_val_buf + queue_idx * candidate_buffer_length) : nullptr;
      s.out_val_buf = will_buffer ? (d_segment_out_val_buf + queue_idx * candidate_buffer_length) : nullptr;
    }

    s.num_full_tiles = s.input_length_actual / static_cast<OffsetT>(tile_items);
    s.partial_items  = s.input_length_actual - s.num_full_tiles * static_cast<OffsetT>(tile_items);
    return s;
  }

  // Per-mode tile bodies. Each takes the per-segment cached state and a tile-local index and runs that mode's tile
  // work; the surrounding smem-histogram init/merge is managed by `run()` and the prefix-sum by the finalize kernel.

  // Templated on `IsFullTile` so the fast / slow-full-tile paths can skip the runtime
  // partial-vs-full branch. Callers must guarantee:
  //   - `IsFullTile == true`  -> `local_tile < s.num_full_tiles`
  //   - `IsFullTile == false` -> `local_tile == s.num_full_tiles && s.partial_items > 0`
  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile_early_stop(
    const per_segment_state_t& s,
    LargeSegmentTileOffsetT local_tile,
    keys_source_t& keys_source,
    held_value_source_t& value_source)
  {
    // `keys_source` / `value_source` are owned by `run()` and re-targeted to this segment via `set_inputs`; here we
    // only set the per-tile base + submit the load.

    // The filter primitive's ctor takes the identify-selected op by non-const reference, so build a local op (cheap
    // ctor: a `key_prefix_storage_t*` copy + an `int` shift) from the cached per-segment fields.
    IdentifyCandidatesOpT identify_candidates_op{&s.segment_counter->kth_key_bits, s.pass, total_bits, decomposer};
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
      if constexpr (!keys_only)
      {
        value_source.set_tile_base(tile_base);
      }

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.early_stop.arena.get_keys_source_scratch());
      h.complete_load(items);
      // Fence the just-completed load's smem writes (smem-using BlockLoad) *and* the previous tile's `partition`
      // smem writes against the next `partition` call -- both alias `partition_arena` via the smem union. Dead work
      // when neither wrote to smem (DIRECT/VECTORIZE load + empty `partition_t::ScratchStorage`).
      if constexpr (tile_load_kind_uses_smem
                    || !detail::topk::is_empty_storage_v<typename early_stop_filter_t::ScratchStorage>)
      {
        __syncthreads();
      }
      filter.partition(storage.arms.early_stop.arena.get_partition_scratch(), items, value_source);
    }
    else
    {
      const OffsetT tile_base = s.num_full_tiles * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);
      if constexpr (!keys_only)
      {
        value_source.set_tile_base(tile_base);
      }

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.early_stop.arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      if constexpr (tile_load_kind_uses_smem
                    || !detail::topk::is_empty_storage_v<typename early_stop_filter_t::ScratchStorage>)
      {
        __syncthreads();
      }
      filter.partition(storage.arms.early_stop.arena.get_partition_scratch(), items, s.partial_items, value_source);
    }

    filter.epilogue();
  }

  // See the `process_tile_early_stop` doc for the `IsFullTile` contract.
  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile_buffered(
    const per_segment_state_t& s,
    LargeSegmentTileOffsetT local_tile,
    keys_source_t& keys_source,
    held_value_source_t& value_source)
  {
    selected_reserve_op_t reserve_sel{&s.segment_counter->num_selected_written};
    candidate_reserve_op_t reserve_cand{&s.segment_counter->num_candidates_written};

    buffered_cand_key_out_t cand_key_out                  = s.out_key_buf;
    [[maybe_unused]] buffered_cand_val_out_t cand_val_out = s.out_val_buf;
    histogram_callback_op_t histogram_cb                  = hist.make_callback();
    auto value_channel_sinks = make_buffered_value_channel_sinks(s.d_values_out, cand_val_out);

    // The partition primitive's ctor takes `IdentifyCandidatesOp&` (non-const); build a
    // local op (cheap ctor) from the cached per-segment fields.
    IdentifyCandidatesOpT identify_candidates_op{&s.segment_counter->kth_key_bits, s.pass, total_bits, decomposer};

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
      if constexpr (!keys_only)
      {
        value_source.set_tile_base(tile_base);
      }

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch());
      h.complete_load(items);
      // Kept unconditionally (unlike the early-stop arm, which can elide it when neither the load nor the partition
      // touches smem): the buffered partition fires per-tile `atomicAdd` bursts into the smem histogram (via
      // `histogram_callback_op`), so dropping the barrier increases smem-atomic contention with the next tile.
      __syncthreads();
      partition.partition(storage.arms.buffered.arena.get_partition_scratch(), items, value_source);
    }
    else
    {
      const OffsetT tile_base = s.num_full_tiles * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);
      if constexpr (!keys_only)
      {
        value_source.set_tile_base(tile_base);
      }

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      // See the matching comment on the full-tile arm above for why this
      // sync is kept unconditionally on the buffered partition path.
      __syncthreads();
      partition.partition(storage.arms.buffered.arena.get_partition_scratch(), items, s.partial_items, value_source);
    }

    partition.epilogue();
  }

  // See the `process_tile_early_stop` doc for the `IsFullTile` contract.
  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  process_tile_unbuffered(const per_segment_state_t& s, LargeSegmentTileOffsetT local_tile, keys_source_t& keys_source)
  {
    using filter_op_t = detail::batched_topk::topk_candidate_filter_op<IdentifyCandidatesOpT>;
    IdentifyCandidatesOpT identify_candidates_op{&s.segment_counter->kth_key_bits, s.pass, total_bits, decomposer};
    filter_op_t filter_op{identify_candidates_op};

    if constexpr (IsFullTile)
    {
      const OffsetT tile_base = static_cast<OffsetT>(local_tile) * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch());
      h.complete_load(items);
      hist.add_full(items, filter_op);
    }
    else
    {
      const OffsetT tile_base = s.num_full_tiles * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.arms.buffered.arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
      const int num_thread_items =
        (thread_offset >= s.input_length_actual)
          ? 0
          : static_cast<int>(
              (::cuda::std::min) (static_cast<OffsetT>(items_per_thread), s.input_length_actual - thread_offset));
      hist.add_partial(items, num_thread_items, filter_op);
    }
  }

  // Per-mode tile dispatcher. `s.mode` is fixed for the whole segment-stretch the caller processes, so within a
  // single unrolled run ptxas can hoist the mode branch above the tile loop.
  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void dispatch_tile(
    const per_segment_state_t& s,
    LargeSegmentTileOffsetT local_tile,
    keys_source_t& keys_source,
    held_value_source_t& value_source)
  {
    if (s.mode == segment_processing_mode::early_stop)
    {
      process_tile_early_stop<IsFullTile>(s, local_tile, keys_source, value_source);
    }
    else if (s.mode == segment_processing_mode::buffered)
    {
      process_tile_buffered<IsFullTile>(s, local_tile, keys_source, value_source);
    }
    else
    {
      // The unbuffered (scout) arm is keys-only -- it accumulates the histogram and never touches the
      // value channel -- so the hoisted `value_source` isn't threaded into it.
      process_tile_unbuffered<IsFullTile>(s, local_tile, keys_source);
    }
  }

  // Active-arm selector for the keys multi-source, per segment mode:
  //   - unbuffered (scout): always re-reads the original input -> source A (`false`).
  //   - early_stop / buffered: follow the counter's `load_from_candidates_buffer`.
  _CCCL_DEVICE _CCCL_FORCEINLINE static bool keys_pick_source_b(const per_segment_state_t& s)
  {
    return (s.mode == segment_processing_mode::early_stop || s.mode == segment_processing_mode::buffered)
           ? static_cast<bool>(s.load_from_candidates_buffer)
           : false;
  }

  // Whether the segment's mode uses the smem histogram. The two non-early_stop modes
  // (buffered / unbuffered) both accumulate into `storage.arms.buffered.histogram`;
  // early_stop touches none of it. Empty segments don't touch it either.
  _CCCL_DEVICE _CCCL_FORCEINLINE static bool segment_uses_smem_histogram(const per_segment_state_t& s)
  {
    return s.mode == segment_processing_mode::buffered || s.mode == segment_processing_mode::unbuffered;
  }

  // Init the smem histogram for the given segment iff its mode uses it. Caller must
  // `__syncthreads()` after to publish the writes.
  _CCCL_DEVICE _CCCL_FORCEINLINE void init_segment_histogram(const per_segment_state_t& s)
  {
    if (segment_uses_smem_histogram(s))
    {
      hist.reset();
    }
  }

  // Merge the smem histogram into the segment's global slab iff its mode used it. Caller
  // must `__syncthreads()` beforehand so all atomic-adds from the segment's tile loop are
  // visible.
  _CCCL_DEVICE _CCCL_FORCEINLINE void merge_segment_histogram(const per_segment_state_t& s)
  {
    if (segment_uses_smem_histogram(s))
    {
      hist.flush(s.segment_histogram);
    }
  }

public:
  // Drive this CTA's entire grid-strided filter pass: a flat grid-stride loop, one tile per iteration,
  // stride = `gridDim.x`.
  //
  // Shape:
  //   * Early-return for CTAs whose first tile lands past the queue.
  //   * One `resolve_segment_state(blockIdx.x)` hoisted before the loop, with the smem histogram initialised inline.
  //   * Per tile: refresh `state` on a segment-boundary crossing (`tile_id >= state.queue_segment_end`) via the
  //     flush-old-hist / resolve-new / init-new-hist handshake; skip empty segments and wasted-tail tiles; dispatch
  //     `dispatch_tile<true>` for full tiles.
  //   * Final flush of the last segment's smem histogram (no-op for early_stop / empty / never-entered).
  //
  // The per-segment epilogue (counter update + prefix-sum + bucket-finder + optional histogram reset) lives in
  // `device_segmented_topk_finalize_filter_kernel`, run after this kernel.
  //
  // `TilesPerChunk` is kept on the signature for parity with the histogram / last_filter agents but is unused in the
  // body; the static_assert preserves the policy contract.
  template <int TilesPerChunk>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run(int pass)
  {
    static_assert(TilesPerChunk == 2 || TilesPerChunk == 4 || TilesPerChunk == 8,
                  "agent_batched_topk_filter_partition::run<TilesPerChunk> requires "
                  "TilesPerChunk to be a power of two in {2, 4, 8}.");

    // Everything up to the grid-dependency sync below is independent of the finalize kernel that precedes this one:
    // the tile-offset table (including its trailing total) is published by the producer -- the worker epilogue or the
    // all-large scan -- not by any finalize pass. So under PDL this CTA's scheduling, its load of `total`, its index
    // arithmetic and its no-work early-exit all overlap with the primary, which is a single CTA running a
    // `num_buckets`-wide prefix scan while the rest of the machine is idle.
    const LargeSegmentTileOffsetT* const d_total_large_tiles = &d_large_segments_tile_offsets[num_large_segments];
    const LargeSegmentTileOffsetT total                      = *d_total_large_tiles;

    const LargeSegmentTileOffsetT first_tile = static_cast<LargeSegmentTileOffsetT>(blockIdx.x);
    const LargeSegmentTileOffsetT stride     = static_cast<LargeSegmentTileOffsetT>(gridDim.x);

    if (first_tile >= total)
    {
      // Exits without syncing: the intrinsic guards dependent *reads*, and this CTA performs none.
      return;
    }

    // The per-segment state resolved below reads each segment's counter, which the preceding finalize kernel writes.
    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    // Hoist first segment-state resolve + smem-hist init.
    per_segment_state_t state = resolve_segment_state(resolve_queue_idx(first_tile), pass);
    // Fast path: if the first segment this CTA sees is empty AND its grid-stride run never crosses into another
    // segment, there is no work anywhere -- skip the tile-loop body (which would otherwise burn O(total/stride)
    // iterations doing `continue;` plus no-op smem-histogram handshakes). The CTA's last tile is
    // `first_tile + stride * floor((total - first_tile - 1) / stride)`; if that's still inside `queue_segment_end`
    // we never cross a boundary. Catches the common case where a universal early-stop pass left
    // `num_candidates_in == 0`.
    if (state.mode == segment_processing_mode::empty)
    {
      // `total > first_tile` is guaranteed by the early-return above; `stride > 0`
      // because grid is non-empty.
      const LargeSegmentTileOffsetT last_tile_for_cta =
        first_tile + stride * static_cast<LargeSegmentTileOffsetT>((total - first_tile - 1) / stride);
      if (last_tile_for_cta < state.queue_segment_end)
      {
        return;
      }
    }
    __syncthreads();
    init_segment_histogram(state);
    __syncthreads();

    // Build the keys multi-source ONCE for this CTA against the hoisted (mode-independent) per-child state. The
    // children outlive the source; on each segment boundary below we re-target it via `set_inputs` (iterators +
    // active arm) rather than reconstructing it. The per-load staging scratch still comes from the active mode's arena.
    key_source_input_t key_src_input{state.d_keys_in, storage.key_src_input_state};
    key_source_buffer_t key_src_buffer{state.in_key_buf, storage.key_src_buffer_state};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/keys_pick_source_b(state)};

    // Build the value multi-source ONCE too, the same way (re-targeted per segment below). For `keys_only` it
    // collapses to `NullType{}` and is never loaded from. It's consumed only by the early_stop / buffered arms, whose
    // active-arm selector matches the keys' `keys_pick_source_b`.
    [[maybe_unused]] value_source_input_t val_src_input{state.d_values_in, storage.val_src_input_state};
    [[maybe_unused]] value_source_buffer_t val_src_buffer{state.in_val_buf, storage.val_src_buffer_state};
    held_value_source_t value_source = [&]() -> held_value_source_t {
      if constexpr (keys_only)
      {
        return NullType{};
      }
      else
      {
        return value_source_t{val_src_input, val_src_buffer, /*pick_b=*/keys_pick_source_b(state)};
      }
    }();

    for (LargeSegmentTileOffsetT tile_id = first_tile; tile_id < total; tile_id += stride)
    {
      // Segment refresh -- only when the cached segment no longer covers `tile_id`.
      if (tile_id >= state.queue_segment_end)
      {
        __syncthreads();
        merge_segment_histogram(state);
        state = resolve_segment_state(resolve_queue_idx(tile_id), pass);
        // Re-target the long-lived keys source to the new segment (iterators + active arm), no reconstruction.
        // `set_inputs` only mutates per-thread iterator/selector state, so it needs no barrier of its own.
        keys_source.set_inputs(state.d_keys_in, state.in_key_buf, keys_pick_source_b(state));
        if constexpr (!keys_only)
        {
          // Re-target the value source to the new segment too (iterators + active arm). Same no-barrier
          // contract as the keys source: `set_inputs` only mutates per-thread iterator/selector state.
          value_source.set_inputs(state.d_values_in, state.in_val_buf, keys_pick_source_b(state));
        }
        __syncthreads();
        init_segment_histogram(state);
        __syncthreads();
      }

      if (state.mode == segment_processing_mode::empty)
      {
        continue;
      }

      const OffsetT local_tile = static_cast<OffsetT>(tile_id - state.slab_base);
      if (local_tile < state.num_full_tiles)
      {
        dispatch_tile<true>(state, static_cast<LargeSegmentTileOffsetT>(local_tile), keys_source, value_source);
      }
    }

    // Final flush: merge the last active segment's smem histogram (no-op for early_stop / empty). Always reached --
    // the early-return filtered out CTAs with no work; `merge_segment_histogram` decides whether to actually merge.
    __syncthreads();
    merge_segment_histogram(state);

    keys_source.invalidate();
    if constexpr (!keys_only)
    {
      value_source.invalidate();
    }
  }

  // Process the trailing partial tile of `queue_idx`'s segment for the current pass, using whatever per-mode tile
  // body the segment's runtime state selects. Invoked by `device_segmented_topk_finalize_filter_kernel` (one CTA per
  // segment) so each segment's partial-tile contribution is injected before its prefix-sum + bucket-finder runs.
  //
  // Smem-histogram handshake (buffered / unbuffered modes only): caller `__syncthreads()` before entry; this method
  // inits the smem hist, processes the partial via `dispatch_tile<false>` (atomicAdds into it), then merges it into
  // the per-segment global slab; caller `__syncthreads()` after. early_stop touches no smem histogram. Empty
  // segments and segments with no partial (`partial_items == 0`) are no-ops.
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_partial_for_segment(LargeSegmentTileOffsetT queue_idx, int pass)
  {
    per_segment_state_t state = resolve_segment_state(queue_idx, pass);
    if (state.mode == segment_processing_mode::empty || state.partial_items == 0)
    {
      return;
    }

    if (segment_uses_smem_histogram(state))
    {
      init_segment_histogram(state);
      __syncthreads();
    }

    // One CTA per segment here, so build the keys + value sources for this single segment and process its partial
    // tile. Same hoisted state slots as `run()`.
    key_source_input_t key_src_input{state.d_keys_in, storage.key_src_input_state};
    key_source_buffer_t key_src_buffer{state.in_key_buf, storage.key_src_buffer_state};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/keys_pick_source_b(state)};

    [[maybe_unused]] value_source_input_t val_src_input{state.d_values_in, storage.val_src_input_state};
    [[maybe_unused]] value_source_buffer_t val_src_buffer{state.in_val_buf, storage.val_src_buffer_state};
    held_value_source_t value_source = [&]() -> held_value_source_t {
      if constexpr (keys_only)
      {
        return NullType{};
      }
      else
      {
        return value_source_t{val_src_input, val_src_buffer, /*pick_b=*/keys_pick_source_b(state)};
      }
    }();

    dispatch_tile<false>(state, static_cast<LargeSegmentTileOffsetT>(state.num_full_tiles), keys_source, value_source);

    keys_source.invalidate();
    if constexpr (!keys_only)
    {
      value_source.invalidate();
    }

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
// No histogram accumulation. Each block processes one tile of one segment; the partition primitive scatters
// surviving "selected" candidates to the front of `d_key_segments_out_it[segment_id]` via `p_num_selected_written`
// and ties (kth-class) to the back via a `back_grow_capped_reserve_op` (cap = `num_of_kth_needed`, anchor = `k_total`).
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
          bool LazyValueLoad   = false,
          bool InlinedClassify = false>
struct agent_batched_topk_last_filter
{
  using keys_in_it_t    = it_value_t<KeyInputItItT>;
  using values_in_it_t  = it_value_t<ValueInputItItT>;
  using values_out_it_t = it_value_t<ValueOutputItItT>;
  using keys_out_it_t   = it_value_t<KeyOutputItItT>;

  using key_t     = it_value_t<keys_in_it_t>;
  using value_t   = it_value_t<values_in_it_t>;
  using counter_t = detail::batched_topk::counter<key_t, OffsetT, OutOffsetT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<value_t, cub::NullType>;

  // Mirrors the histogram / filter agents' constexpr: DIRECT / VECTORIZE `BlockLoad` doesn't touch the shared
  // scratch, so the pre-`submit_load` `__syncthreads()` is dead work. The post-`complete_load` sync stays in (it
  // serializes consecutive `partition` calls through the smem union aliasing `keys_source_scratch` /
  // `partition_scratch`).
  static constexpr bool tile_load_kind_uses_smem =
    AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_direct
    && AgentTopKPolicyT::keys_tile_load_kind != detail::topk::tile_load_kind::block_load_vectorize;

  static constexpr bool effective_lazy_value_load = LazyValueLoad && !keys_only;

  using selected_offset_t  = OutOffsetT;
  using candidate_offset_t = OutOffsetT;

  using key_source_input_t = detail::topk::
    tile_data_source_t<keys_in_it_t, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using key_source_buffer_t = detail::topk::
    tile_data_source_t<key_t*, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using keys_source_t = detail::topk::multi_source_data_source<key_source_input_t, key_source_buffer_t, OffsetT>;

  using value_source_input_t =
    detail::topk::direct_data_source<values_in_it_t, block_threads, items_per_thread, OffsetT>;
  using value_source_buffer_t = detail::topk::direct_data_source<value_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t = detail::topk::multi_source_data_source<value_source_input_t, value_source_buffer_t, OffsetT>;

  // Value multi-source held for the whole CTA run (the value analog of `keys_source_t`). `keys_only` collapses it to
  // `NullType`; pairs hold the real source. Built once and passed by reference into `process_tile`.
  using held_value_source_t = ::cuda::std::conditional_t<keys_only, NullType, value_source_t>;

  using val_out_t      = values_out_it_t;
  using cand_val_out_t = values_out_it_t;

  using value_channel_sinks_concrete_t = detail::topk::value_channel_sinks_t<val_out_t, cand_val_out_t>;
  using value_channel_sinks_or_null_t = ::cuda::std::conditional_t<keys_only, NullType, value_channel_sinks_concrete_t>;

  using agent_value_t = ::cuda::std::conditional_t<keys_only, NullType, value_t>;
  using agent_value_data_source_scratch_t =
    ::cuda::std::conditional_t<keys_only, NullType, typename value_source_t::ScratchStorage>;

  using selected_reserve_op_t  = detail::topk::atomic_reserve_range_op<selected_offset_t>;
  using candidate_reserve_op_t = detail::topk::back_grow_capped_reserve_op<candidate_offset_t>;

  using partition_t = detail::topk::block_partition_atomics<
    block_threads,
    items_per_thread,
    InlinedClassify,
    key_t,
    selected_offset_t,
    candidate_offset_t,
    selected_reserve_op_t,
    candidate_reserve_op_t,
    keys_out_it_t,
    keys_out_it_t,
    IdentifyCandidatesOpT,
    detail::batched_topk::topk_noop_candidate_callback_op,
    value_channel_sinks_or_null_t,
    agent_value_t,
    agent_value_data_source_scratch_t,
    effective_lazy_value_load>;

  using storage_layout_t =
    detail::topk::partition_storage_layout_for_t<partition_t, typename keys_source_t::ScratchStorage>;

  // Per-child persistent state. `keys_source_t` (multi-source) doesn't publish a `TempStorage`, so the agent holds
  // one `TempStorage` per child source.
  struct _TempStorage
  {
    typename key_source_input_t::TempStorage key_src_input_state;
    typename key_source_buffer_t::TempStorage key_src_buffer_state;
    // Persistent value-source state, hoisted alongside the keys-source state so the value source can also persist
    // across segments / be re-targeted (via `set_inputs`) instead of rebuilt. Empty for the current `direct` value
    // source (and for `keys_only`), so no extra smem.
    typename value_source_input_t::TempStorage val_src_input_state;
    typename value_source_buffer_t::TempStorage val_src_buffer_state;
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
  SegmentIdProviderT segment_id_provider;
  const LargeSegmentTileOffsetT* d_large_segments_tile_offsets;
  counter_t* d_segment_counters;
  key_t* d_segment_in_key_buf;
  value_t* d_segment_in_val_buf;
  int pass;
  int total_bits;
  DecomposerT decomposer;
  OffsetT candidate_buffer_length;
  // Narrowed segment count: keeps `resolve_queue_idx`'s `UpperBound` + indexing 32-bit.
  queue_segment_count_t num_large_segments;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_batched_topk_last_filter(
    TempStorage& ts,
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* d_large_segments_tile_offsets,
    counter_t* d_segment_counters,
    key_t* d_segment_in_key_buf,
    value_t* d_segment_in_val_buf,
    int pass,
    int total_bits,
    DecomposerT decomposer,
    OffsetT candidate_buffer_length,
    queue_segment_count_t num_large_segments)
      : storage(ts.Alias())
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , k_param(k_param)
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
  // Per-segment cached state, mirrors the filter agent's pattern. Re-derived only when the chunk crosses a segment
  // boundary; held in registers across same-segment tiles.
  //
  // `[slab_base, queue_segment_end)` is the segment's tile-space window in the global queue, sized at
  // segment-enqueue time from the *original* segment size (so pass-independent). Only
  // `[slab_base, slab_base + ceil(num_candidates_in / tile_items))` carry data this pass; the tail up to
  // `queue_segment_end` is "wasted" slots. Tracking the wider `queue_segment_end` lets `run`'s slow path jump past
  // empty segments / wasted-slot tails in one step instead of a per-tile `UpperBound`.
  struct per_segment_state_t
  {
    bool empty;
    bool load_from_candidates_buffer;

    keys_in_it_t d_keys_in{};
    keys_out_it_t d_keys_out{};
    [[maybe_unused]] values_in_it_t d_values_in{};
    [[maybe_unused]] values_out_it_t d_values_out{};

    counter_t* segment_counter;
    key_t* in_key_buf;
    [[maybe_unused]] value_t* in_val_buf;

    // `identify_candidates_op_t` has no default ctor for some `T`; cache ctor inputs and
    // build the op on demand in `process_tile`.
    int pass;

    OutOffsetT k_total;
    OutOffsetT num_of_kth_needed;
    OffsetT num_full_tiles;
    OffsetT partial_items;
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
    // See `agent_batched_topk_filter_partition::resolve_queue_idx`. Routing the broadcast through `makeWarpUniform`
    // restores warp-aggregated atomics on `&segment_counter->num_ties_written_to_back` (used by
    // `back_grow_capped_reserve_op` inside `partition.partition()`).
    // Narrow the broadcast to the queue-index type before the call. A queue index is bounded by the queue length, and
    // hence by `num_segments <= INT_MAX`, so this cannot truncate; it also picks `makeWarpUniform`'s 32-bit overload
    // unambiguously, which is the one that lowers to a single CREDUX.
    return static_cast<LargeSegmentTileOffsetT>(
      detail::warpspeed::makeWarpUniform(static_cast<queue_segment_count_t>(__shfl_sync(0xffffffff, queue_idx_lane0, 0))));
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE per_segment_state_t resolve_segment_state(LargeSegmentTileOffsetT queue_idx)
  {
    per_segment_state_t s{};
    s.slab_base = d_large_segments_tile_offsets[queue_idx];
    // `queue_segment_end` is the next segment's `slab_base` -- the table is sized
    // `num_large_segments + 1` (sentinel at the end stores `total_large_tiles`), so the
    // read is in-bounds for every valid `queue_idx`.
    s.queue_segment_end   = d_large_segments_tile_offsets[queue_idx + 1];
    const auto segment_id = segment_id_provider[queue_idx];

    s.d_keys_in  = d_key_segments_it[segment_id];
    s.d_keys_out = d_key_segments_out_it[segment_id];
    if constexpr (!keys_only)
    {
      s.d_values_in  = d_value_segments_it[segment_id];
      s.d_values_out = d_value_segments_out_it[segment_id];
    }

    s.segment_counter             = d_segment_counters + queue_idx;
    const OffsetT input_length    = s.segment_counter->num_candidates_in;
    s.load_from_candidates_buffer = s.segment_counter->load_from_candidates_buffer;
    s.pass                        = pass;

    s.empty = (input_length == 0);
    if (s.empty)
    {
      return s;
    }

    // Mirrors the histogram agent's clip: when `k > segment_size`, all items are in the top-k. `reserve_cand` is
    // sized from `k_total`, kept in lock-step with prior passes' counter writes for correct output reservation.
    const OffsetT segment_num_items =
      static_cast<OffsetT>(params::__get_and_clamp_param_to_nonnegative(segment_sizes, segment_id));
    s.k_total =
      (::cuda::std::min) (static_cast<OutOffsetT>(params::__get_and_clamp_param_to_nonnegative(k_param, segment_id)),
                          static_cast<OutOffsetT>(segment_num_items));
    s.num_of_kth_needed             = static_cast<OutOffsetT>(s.segment_counter->k);

    s.in_key_buf = d_segment_in_key_buf + queue_idx * candidate_buffer_length;
    if constexpr (!keys_only)
    {
      s.in_val_buf =
        s.load_from_candidates_buffer ? (d_segment_in_val_buf + queue_idx * candidate_buffer_length) : nullptr;
    }

    s.num_full_tiles = input_length / static_cast<OffsetT>(tile_items);
    s.partial_items  = input_length - s.num_full_tiles * static_cast<OffsetT>(tile_items);
    return s;
  }

  // Build the partition object for the current segment. Lives across all tiles of this segment so its per-thread
  // `cand_reserve_open` flag (the back-grow-cap exit hint that drops per-item atomics after the first observed
  // grant=0) persists. Rebuilt at every segment-boundary crossing in `run()`.
  _CCCL_DEVICE _CCCL_FORCEINLINE partition_t make_partition_for_segment(const per_segment_state_t& s)
  {
    selected_reserve_op_t reserve_sel{&s.segment_counter->num_selected_written};
    // The back-grow-capped reserve op carries the precomputed `region_start = k_total - num_of_kth_needed` rather
    // than the back-region end anchor, so its per-call math collapses from two subtracts to one add.
    candidate_reserve_op_t reserve_cand{
      &s.segment_counter->num_ties_written_to_back,
      static_cast<candidate_offset_t>(s.k_total - s.num_of_kth_needed),
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
    detail::batched_topk::topk_noop_candidate_callback_op callback_op{};

    // The identify-candidates op carries the segment's `kth_key_bits` and the partition holds it by value, so the
    // partition ctor copy binds it to this segment's state.
    IdentifyCandidatesOpT identify_candidates_op{&s.segment_counter->kth_key_bits, s.pass, total_bits, decomposer};

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

  // Templated on `IsFullTile` so the fast / slow path skips the runtime partial-vs-full branch. Callers must guarantee:
  //   - `IsFullTile == true`  -> `local_tile < s.num_full_tiles`
  //   - `IsFullTile == false` -> `local_tile == s.num_full_tiles && s.partial_items > 0`
  // `partition` / `keys_source` / `value_source` are owned by `run()` and reused across tiles, so `partition`'s
  // per-thread `cand_reserve_open` flag persists: once a thread sees grant=0 from the back-grow-capped reserve, later
  // tiles use the cheaper `HasCandidateStream=false` classifier that drops the per-item atomic
  // (see `block_partition.cuh`).
  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile(
    const per_segment_state_t& s,
    partition_t& partition,
    keys_source_t& keys_source,
    held_value_source_t& value_source,
    LargeSegmentTileOffsetT local_tile)
  {
    // `keys_source` / `value_source` are owned by `run()` and re-targeted to this segment via `set_inputs`; here we
    // only set the per-tile base + submit the load.

    if constexpr (IsFullTile)
    {
      const OffsetT tile_base = static_cast<OffsetT>(local_tile) * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);
      if constexpr (!keys_only)
      {
        value_source.set_tile_base(tile_base);
      }

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch());
      h.complete_load(items);
      // See the matching note on the filter agent's early-stop arm.
      if constexpr (tile_load_kind_uses_smem || !detail::topk::is_empty_storage_v<typename partition_t::ScratchStorage>)
      {
        __syncthreads();
      }
      partition.partition(storage.partition_arena.get_partition_scratch(), items, value_source);
    }
    else
    {
      const OffsetT tile_base = s.num_full_tiles * static_cast<OffsetT>(tile_items);
      keys_source.set_tile_base(tile_base);
      if constexpr (!keys_only)
      {
        value_source.set_tile_base(tile_base);
      }

      if constexpr (tile_load_kind_uses_smem)
      {
        __syncthreads();
      }
      key_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch(), s.partial_items);
      h.complete_load(items);
      if constexpr (tile_load_kind_uses_smem || !detail::topk::is_empty_storage_v<typename partition_t::ScratchStorage>)
      {
        __syncthreads();
      }
      partition.partition(storage.partition_arena.get_partition_scratch(), items, s.partial_items, value_source);
    }
  }

public:
  // Drive this CTA's entire grid-strided last-filter pass: a flat grid-stride loop, one tile per iteration,
  // stride = `gridDim.x` (same flat shape as the filter agent).
  //
  // Shape:
  //   * Early-return for CTAs whose first tile lands past the queue.
  //   * One `resolve_segment_state(blockIdx.x)` up front.
  //   * Construct one `partition_t` per segment, *outside* the tile loop, so its per-thread `cand_reserve_open` flag
  //     survives across tiles of the same segment -- the mechanism that drops the per-item candidate-reserve atomic
  //     on subsequent tiles once the back-grow cap is hit (see `block_partition.cuh`). This matters most for
  //     entropy=0 (all-equal-keys) workloads, where every tile would otherwise re-fire the per-item atomic.
  //   * Per tile: refresh `state` on a segment-boundary crossing (`tile_id >= state.queue_segment_end`) -- flush the
  //     previous segment's partition via `partition.epilogue()` (no-op on the atomics strategy) and rebuild the
  //     partition + re-target keys_source for the new segment, resetting `cand_reserve_open` to `true`; skip empty /
  //     wasted-tail tiles; dispatch `process_tile<true>` for full tiles or `process_tile<false>` for the
  //     trailing partial.
  //   * Final `partition.epilogue()` after the loop terminates the last active segment.
  //
  // `TilesPerChunk` is kept on the signature for parity with the filter / histogram agents but is unused in the body;
  // the static_assert preserves the policy contract.
  template <int TilesPerChunk>
  _CCCL_DEVICE _CCCL_FORCEINLINE void run()
  {
    static_assert(TilesPerChunk == 2 || TilesPerChunk == 4 || TilesPerChunk == 8,
                  "agent_batched_topk_last_filter::run<TilesPerChunk> requires "
                  "TilesPerChunk to be a power of two in {2, 4, 8}.");

    const LargeSegmentTileOffsetT* const d_total_large_tiles = &d_large_segments_tile_offsets[num_large_segments];
    const LargeSegmentTileOffsetT total                      = *d_total_large_tiles;

    const LargeSegmentTileOffsetT first_tile = static_cast<LargeSegmentTileOffsetT>(blockIdx.x);
    const LargeSegmentTileOffsetT stride     = static_cast<LargeSegmentTileOffsetT>(gridDim.x);

    if (first_tile >= total)
    {
      // Exits without syncing: the intrinsic guards dependent *reads*, and this CTA performs none.
      return;
    }

    // As in the filter agent: the tile-offset table read above comes from the producer, not from the finalize kernel
    // that immediately precedes this launch, so it overlaps. The per-segment state below reads the counters that
    // finalize wrote.
    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    // Hoist the first segment-state resolve + partition / keys-source construction out of the loop. Both `partition`
    // and `keys_source` live across tiles of the same segment so per-thread cross-tile state
    // (`cand_reserve_open`) persists.
    per_segment_state_t state = resolve_segment_state(resolve_queue_idx(first_tile));
    // Fast-empty-exit -- same motivation as `agent_batched_topk_filter_partition::run`. When a prior pass set
    // `num_candidates_in = 0` universally, every CTA sees an empty segment; bail out when this CTA's whole
    // grid-stride run is inside the same empty segment.
    if (state.empty)
    {
      const LargeSegmentTileOffsetT last_tile_for_cta =
        first_tile + stride * static_cast<LargeSegmentTileOffsetT>((total - first_tile - 1) / stride);
      if (last_tile_for_cta < state.queue_segment_end)
      {
        return;
      }
    }
    partition_t partition = make_partition_for_segment(state);
    // Build the keys + value multi-sources ONCE for this CTA against the hoisted (mode-independent) per-child state.
    // The children outlive the sources; on each segment boundary below we re-target via `set_inputs` (iterators +
    // active arm) rather than reconstructing. (The `partition` still *is* rebuilt per segment, resetting its
    // `cand_reserve_open` flag, which the sources don't carry.) For `keys_only` the value source collapses to
    // `NullType{}`.
    key_source_input_t key_src_input{state.d_keys_in, storage.key_src_input_state};
    key_source_buffer_t key_src_buffer{state.in_key_buf, storage.key_src_buffer_state};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/state.load_from_candidates_buffer};

    [[maybe_unused]] value_source_input_t val_src_input{state.d_values_in, storage.val_src_input_state};
    [[maybe_unused]] value_source_buffer_t val_src_buffer{state.in_val_buf, storage.val_src_buffer_state};
    held_value_source_t value_source = [&]() -> held_value_source_t {
      if constexpr (keys_only)
      {
        return NullType{};
      }
      else
      {
        return value_source_t{val_src_input, val_src_buffer, /*pick_b=*/state.load_from_candidates_buffer};
      }
    }();

    for (LargeSegmentTileOffsetT tile_id = first_tile; tile_id < total; tile_id += stride)
    {
      // Segment refresh -- only when the cached segment no longer covers `tile_id`. Flush the previous segment's
      // partition (terminal accumulation flush; no-op on atomics) and rebuild it for the new segment so
      // `cand_reserve_open` resets to `true`. The keys / value sources are *not* rebuilt -- they're re-targeted in
      // place via `set_inputs` (iterators + active arm). `set_inputs` only mutates per-thread state, so no
      // barrier needed.
      if (tile_id >= state.queue_segment_end)
      {
        partition.epilogue();
        state     = resolve_segment_state(resolve_queue_idx(tile_id));
        partition = make_partition_for_segment(state);
        keys_source.set_inputs(state.d_keys_in, state.in_key_buf, /*pick_b=*/state.load_from_candidates_buffer);
        if constexpr (!keys_only)
        {
          value_source.set_inputs(state.d_values_in, state.in_val_buf, /*pick_b=*/state.load_from_candidates_buffer);
        }
      }

      if (state.empty)
      {
        continue;
      }

      const OffsetT local_tile = static_cast<OffsetT>(tile_id - state.slab_base);
      if (local_tile < state.num_full_tiles)
      {
        process_tile<true>(
          state, partition, keys_source, value_source, static_cast<LargeSegmentTileOffsetT>(local_tile));
      }
      else if (local_tile == state.num_full_tiles && state.partial_items > OffsetT{0})
      {
        process_tile<false>(
          state, partition, keys_source, value_source, static_cast<LargeSegmentTileOffsetT>(state.num_full_tiles));
      }
      // else: wasted-tail tile beyond segment data; skip implicitly via the grid-stride loop.
    }

    // Final flush of the last active segment.
    partition.epilogue();

    keys_source.invalidate();
    if constexpr (!keys_only)
    {
      value_source.invalidate();
    }
  }
};
} // namespace detail::batched_topk
CUB_NAMESPACE_END
