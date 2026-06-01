// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Kernel entry point for device-wide batched top-k.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_batched_topk.cuh>
#include <cub/agent/agent_topk_common.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_topk_common.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_arch.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__type_traits/conditional.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// Finds the smallest worker_per_segment policy whose tile size still covers the upper bound on
// segment size AND whose instantiated agent's shared memory usage fits within the static shared
// memory limit (max_smem_per_block). When such a policy exists, `found == true` and `policy` /
// `agent_t` refer to it; otherwise `found == false` and callers must fall back to
// `find_largest_fitting_smem_policy`.
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_smallest_covering_policy
{
private:
  struct policy_t
  {
    worker_policy worker_per_segment_policy;
    multi_worker_policy multi_worker_per_segment_policy;
  };
  static constexpr ::cuda::std::int64_t max_segment_size = params::static_max_value_v<SegmentSizeParameterT>;
  static constexpr batched_topk_policy active_policy     = current_policy<PolicySelector>();

  template <int Index>
  [[nodiscard]] static constexpr int find_index()
  {
    if constexpr (Index >= active_policy.worker_per_segment_policies.size())
    {
      return -1;
    }
    else
    {
      constexpr worker_policy wp = active_policy.worker_per_segment_policies[Index];
      constexpr auto tile_size   = ::cuda::std::int64_t{wp.threads_per_block} * wp.items_per_thread;

      struct policy_getter_17 // TODO(bgruber): drop this in C++17 and pass wp directly
      {
        _CCCL_HOST_DEVICE_API constexpr auto operator()() const
        {
          return policy_t{active_policy.worker_per_segment_policies[Index],
                          active_policy.multi_worker_per_segment_policy};
        }
      };
      using candidate_agent_t  = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
      constexpr bool covers    = tile_size >= max_segment_size;
      constexpr bool fits_smem = sizeof(typename candidate_agent_t::TempStorage) <= max_smem_per_block;
      constexpr int next       = find_index<Index + 1>();
      if constexpr (covers && fits_smem)
      {
        return next >= 0 ? next : Index;
      }
      else
      {
        return next;
      }
    }
  }

  static constexpr int raw_selected_index = find_index<0>();
  // Defaulted to 0 for the not-found case so the array accesses below stay well-formed
  static constexpr int safe_index = raw_selected_index >= 0 ? raw_selected_index : 0;

public:
  // -1 when no covering+fitting policy exists.
  static constexpr int selected_index = raw_selected_index;
  static constexpr bool found         = (raw_selected_index >= 0);

  // Only meaningful when `found == true`.
  static constexpr policy_t policy = {
    active_policy.worker_per_segment_policies[safe_index], active_policy.multi_worker_per_segment_policy};

  struct policy_getter_17 // TODO(bgruber): drop this in C++17 and pass policy directly
  {
    _CCCL_HOST_DEVICE_API constexpr auto operator()() const
    {
      return policy;
    }
  };
  using agent_t = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
};

// Finds the largest worker_per_segment policy whose instantiated agent's shared memory usage
// fits within the static shared memory limit (max_smem_per_block). Used as the fallback when the upper bound on segment
// size exceeds every worker policy's tile size). In that case the worker treats any segment with
// `segment_size > tile_size` as "large" at runtime and enqueues it onto the large-segment queue
// (the multi-CTA-per-segment kernels then consume that queue).
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_largest_fitting_smem_policy
{
private:
  struct policy_t
  {
    worker_policy worker_per_segment_policy;
    multi_worker_policy multi_worker_per_segment_policy;
  };
  static constexpr batched_topk_policy active_policy = current_policy<PolicySelector>();

  template <int Index>
  [[nodiscard]] static constexpr int find_index()
  {
    if constexpr (Index >= active_policy.worker_per_segment_policies.size())
    {
      return -1;
    }
    else
    {
      struct policy_getter_17
      {
        _CCCL_HOST_DEVICE_API constexpr auto operator()() const
        {
          return policy_t{active_policy.worker_per_segment_policies[Index],
                          active_policy.multi_worker_per_segment_policy};
        }
      };
      using candidate_agent_t  = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
      constexpr bool fits_smem = sizeof(typename candidate_agent_t::TempStorage) <= max_smem_per_block;
      if constexpr (fits_smem)
      {
        // `worker_per_segment_policies` is ordered by decreasing tile size, so the first one
        // that fits in smem is automatically the largest fitting one.
        return Index;
      }
      else
      {
        return find_index<Index + 1>();
      }
    }
  }

  static constexpr int raw_selected_index = find_index<0>();
  static constexpr int safe_index         = raw_selected_index >= 0 ? raw_selected_index : 0;

public:
  static constexpr int selected_index = raw_selected_index;
  static constexpr bool found         = (raw_selected_index >= 0);

  static constexpr policy_t policy = {
    active_policy.worker_per_segment_policies[safe_index], active_policy.multi_worker_per_segment_policy};

  struct policy_getter_17 // TODO(bgruber): drop this in C++17 and pass policy directly
  {
    _CCCL_HOST_DEVICE_API constexpr auto operator()() const
    {
      return policy;
    }
  };
  using agent_t = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
};

// Resolves the worker_per_segment policy used by the kernel + dispatch by preferring the
// smallest covering+fits-smem policy and falling back to the largest fits-smem policy.
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
using resolved_worker_per_segment_policy = ::cuda::std::conditional_t<
  find_smallest_covering_policy<PolicySelector, SegmentSizeParameterT, AgentParamsT...>::found,
  find_smallest_covering_policy<PolicySelector, SegmentSizeParameterT, AgentParamsT...>,
  find_largest_fitting_smem_policy<PolicySelector, SegmentSizeParameterT, AgentParamsT...>>;

// -----------------------------------------------------------------------------
// Global Kernel Entry Point
// -----------------------------------------------------------------------------
template <typename PolicySelector,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename LargeSegmentTileOffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires batched_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(
  resolved_worker_per_segment_policy<
    PolicySelector,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT,
    LargeSegmentTileOffsetT>::policy.worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_kernel(
    const KeyInputItItT d_key_segments_it,
    const KeyOutputItItT d_key_segments_out_it,
    const ValueInputItItT d_value_segments_it,
    const ValueOutputItItT d_value_segments_out_it,
    const SegmentSizeParameterT segment_sizes,
    const KParameterT k,
    const SelectDirectionParameterT select_directions,
    const NumSegmentsParameterT num_segments,
    // These device pointers are passed plain (not `_CCCL_GRID_CONSTANT`): a grid-constant pointer
    // param lowers to generic addressing (`LD.E`/`ST.E`/`ATOM.E`) kernel-wide, under which ptxas
    // cannot prove the per-segment reserve-counter address warp-uniform once it is carried across
    // the tile loop, so the scatter `atomicAdd`s degrade to one-per-lane. Plain pointers resolve to
    // the global space (`LDG`/`STG`/`ATOMG`), which keeps those atomics warp-aggregated (one per
    // warp) and also avoids the per-access generic-addressing overhead.
    batched_topk_counters<narrow_segment_count_t<NumSegmentsParameterT>>* const d_counters,
    narrow_segment_count_t<NumSegmentsParameterT>* const d_large_segments_ids,
    LargeSegmentTileOffsetT* const d_large_segments_tile_offsets)
{
  using resolved_t = resolved_worker_per_segment_policy<
    PolicySelector,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT,
    LargeSegmentTileOffsetT>;

  // Static Assertions (Constraints)
  static_assert(resolved_t::found, "No valid policy found for one-worker-per-segment approach");
  using agent_t = typename resolved_t::agent_t;
  static_assert(sizeof(typename agent_t::TempStorage) <= max_smem_per_block,
                "Static shared memory per block must not exceed 48KB limit.");

  // Temporary storage allocation
  __shared__ typename agent_t::TempStorage temp_storage;

  // Instantiate agent
  agent_t agent(
    temp_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    k,
    select_directions,
    num_segments,
    d_counters,
    d_large_segments_ids,
    d_large_segments_tile_offsets);

  // Process segments
  agent.Process();
}

//---------------------------------------------------------------------
// Segmented multi-CTA-per-segment top-k kernels.
//
// Three kernels mirror the single-problem `DeviceTopK{Histogram,Filter,LastFilter}Kernel` trio
// in `cub/device/dispatch/dispatch_topk.cuh`. Each kernel:
//   1. Resolves `multi_worker_per_segment_policy` from the active `batched_topk_policy`.
//   2. Lifts that into an `AgentTopKPolicy<...>` instantiation.
//   3. Instantiates the matching segmented agent (`agent_batched_topk_histogram`,
//      `agent_batched_topk_filter_partition`, `agent_batched_topk_last_filter`).
//   4. Forwards every per-launch arg to the agent's constructor and calls `agent.run(...)`.
//
// The `SelectDirection` template NTTP comes from the host-side `dispatch_discrete` over the
// uniform `SelectDirectionParameterT` (plan §3.6 / §5.5); the kernels are otherwise direction-
// agnostic.
//---------------------------------------------------------------------

namespace topk_seg_kernel_detail
{
// Helper: build an `AgentTopKPolicy<...>` from the multi-worker policy of a given selector.
template <typename PolicySelector>
struct multi_worker_agent_policy_lift
{
  static constexpr batched_topk_policy bp = current_policy<PolicySelector>();
  static constexpr multi_worker_policy mw = bp.multi_worker_per_segment_policy;
  using type                              = detail::batched_topk::AgentTopKPolicy<
                                 mw.threads_per_block,
                                 mw.items_per_thread,
                                 mw.bits_per_pass,
                                 mw.scan_algorithm,
                                 mw.keys_tile_load_kind,
                                 mw.accumulating_buffer_capacity,
                                 mw.speculative_selected_buffer_capacity>;
};

// Lift `multi_worker_per_segment_policy.tiles_per_chunk` to a compile-time integral constant
// the multi-CTA-per-segment kernels (histogram / filter / last_filter) can use as a
// `static constexpr` loop bound. Kept separate from `multi_worker_agent_policy_lift` because the
// agent's compile-time policy struct (`AgentTopKPolicy`) does not carry this knob -- it is
// consumed only by the kernels' outer/inner grid-stride loop, not by the agents' smem layouts
// or template logic.
template <typename PolicySelector>
struct tiles_per_chunk
{
  static constexpr int value = current_policy<PolicySelector>().multi_worker_per_segment_policy.tiles_per_chunk;
};

// Lift `multi_worker_per_segment_policy.full_tiles_only_histogram` to a compile-time boolean.
// When `true`, the histogram kernel skips the partial-tile path (only full tiles are loaded /
// binned) and the finalize-histogram kernel grows a partial-tile epilogue that loads + bins
// the trailing partial of each segment directly into the segment's global histogram before the
// prefix-sum + bucket-finder runs.
template <typename PolicySelector>
struct full_tiles_only_histogram
{
  static constexpr bool value =
    current_policy<PolicySelector>().multi_worker_per_segment_policy.full_tiles_only_histogram;
};

// Same lift, for the filter kernel. When `true`, the filter kernel skipped the trailing
// partial tile of every segment; the finalize-filter kernel re-injects each segment's
// partial via `agent_batched_topk_filter_partition::process_partial_for_segment` before
// running the prefix-sum + bucket-finder.
template <typename PolicySelector>
struct full_tiles_only_filter
{
  static constexpr bool value = current_policy<PolicySelector>().multi_worker_per_segment_policy.full_tiles_only_filter;
};
} // namespace topk_seg_kernel_detail

template <typename PolicySelector,
          typename KeyInputItItT,
          typename SegmentSizeParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename LargeSegmentsCountItT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT,
          typename SegmentCountT>
#if _CCCL_HAS_CONCEPTS()
  requires batched_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_histogram_kernel(
    const KeyInputItItT d_key_segments_it,
    const SegmentSizeParameterT segment_sizes,
    const SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* const d_large_segments_tile_offsets,
    OffsetT* const d_segment_histograms,
    const LargeSegmentsCountItT large_segments_count_it,
    const ExtractBinOpT extract_bin_op)
{
  // `large_segments_count_it` is either a raw pointer into the mixed-path
  // `batched_topk_counters::large_segments_count` (written by the worker-per-segment kernel's
  // atomicAdd enqueue) or a `transform_iterator` returning the host-known `num_segments_val`
  // for the all-large path; the kernel does not need to know which. `total_large_tiles` lives
  // in the sentinel slot of the offset table -- a `+1`-allocated entry the worker-per-segment
  // epilogue (mixed) or the host-side scan (all-large) populates with the inclusive total of
  // large-segment tile counts. The kernel no longer pre-resolves either value: only the raw
  // parameters (the `large_segments_count_it` iterator + the tile-offsets array) flow into
  // the agent, which dereferences them inside `run` / `resolve_queue_idx`.
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;

  // Compile-time switch for the experimental "histogram only walks full tiles" mode. When
  // `true`, the agent drops the partial-tile path; the trailing partial of each segment is
  // handled by `device_segmented_topk_finalize_histogram_kernel`.
  static constexpr bool full_tiles_only = topk_seg_kernel_detail::full_tiles_only_histogram<PolicySelector>::value;

  using agent_t = agent_batched_topk_histogram<
    agent_topk_policy_t,
    KeyInputItItT,
    ExtractBinOpT,
    SegmentSizeParameterT,
    SegmentIdProviderT,
    LargeSegmentTileOffsetT,
    OffsetT,
    OutOffsetT,
    LargeSegmentsCountItT,
    SegmentCountT,
    full_tiles_only>;

  __shared__ typename agent_t::TempStorage temp_storage;

  agent_t agent(
    temp_storage,
    d_key_segments_it,
    segment_sizes,
    segment_id_provider,
    d_large_segments_tile_offsets,
    d_segment_histograms,
    extract_bin_op,
    large_segments_count_it);

  // The chunk-level grid-stride loop lives inside the agent (`agent.run`) so that per-segment
  // cached state (smem histogram, segment-end bound, segment pointers / scalars) can persist
  // across chunks. In the common case where a CTA's grid-stride run stays inside a single
  // segment, this collapses to exactly **one** `init_histogram` at the start of the CTA's
  // work and **one** `merge_histogram` when the CTA finishes -- matching the single-problem
  // agent's cost model. Multi-segment workloads pay init/merge per (CTA, segment-stretch).
  //
  // This kernel no longer runs the per-segment prefix-sum / bucket-finder epilogue; that work
  // is done by `device_segmented_topk_finalize_histogram_kernel` after this kernel completes.
  // The pass index / total_bits / decomposer that drove the radix-digit extraction in the old
  // signature are now absorbed into `extract_bin_op` (constructed by the dispatch) -- the
  // kernel itself does not need to know about the pass.
  // `TilesPerChunk` is now lifted to a compile-time non-type template parameter on
  // `agent.run` so the middle while-loop's `chunk_end - chunk_start` bound and the per-CTA
  // stride/chunk_start arithmetic are known at codegen time. The runtime `int` overload that
  // the agent previously exposed has been removed; the compile-time `tiles_per_chunk` helper
  // remains the single source of truth.
  static constexpr int tiles_per_chunk = topk_seg_kernel_detail::tiles_per_chunk<PolicySelector>::value;
  agent.template run<tiles_per_chunk>();
}

// Per-segment epilogue kernel for the histogram pass. Runs after
// `device_segmented_topk_histogram_kernel` finishes (host-side launch ordering on the same
// stream ensures all CTAs of the histogram kernel retire before this kernel starts). One CTA per
// large segment in a grid-strided loop: prefix-sums that segment's global histogram, finds the
// bucket containing the k-th key, updates the per-segment counter, and (optionally) zeros the
// histogram slab for the next pass.
//
// Splitting this out of the histogram kernel removes the per-tile `finalize_pass` cost (a
// `__threadfence` + `__syncthreads_or` chain plus the prefix-sum scratch's smem footprint) from
// the histogram CTAs. The trade-off is one extra cheap kernel launch per pass on this path.
template <typename PolicySelector,
          typename KeyInputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentsCountItT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyT>
#if _CCCL_HAS_CONCEPTS()
  requires batched_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_finalize_histogram_kernel(
    const KeyInputItItT d_key_segments_it,
    const SegmentSizeParameterT segment_sizes,
    const KParameterT k_param,
    const SegmentIdProviderT segment_id_provider,
    // See the `device_segmented_topk_histogram_kernel` doc for why we mark the pointer
    // (not the pointee) grid-constant.
    detail::batched_topk::counter<KeyT, OffsetT, OutOffsetT>* const d_segment_counters,
    OffsetT* const d_segment_histograms,
    const LargeSegmentsCountItT large_segments_count_it,
    const ExtractBinOpT extract_bin_op,
    const int pass,
    const bool reset_histogram)
{
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;
  static constexpr int block_threads    = agent_topk_policy_t::block_threads;
  static constexpr int items_per_thread = agent_topk_policy_t::items_per_thread;
  static constexpr int bits_per_pass    = agent_topk_policy_t::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;

  // Mirrors `topk_seg_kernel_detail::full_tiles_only_histogram<PolicySelector>::value`.
  // When `true`, the companion histogram kernel skipped the trailing partial tile of every
  // segment; this kernel's responsibility, before running the prefix-sum + bucket-finder, is
  // to load + bin that partial tile directly into the segment's global histogram slab.
  static constexpr bool process_partial = topk_seg_kernel_detail::full_tiles_only_histogram<PolicySelector>::value;

  using counter_t                   = detail::batched_topk::counter<KeyT, OffsetT, OutOffsetT>;
  using block_identify_kth_bucket_t = detail::batched_topk::
    block_identify_kth_bucket<block_threads, bits_per_pass, agent_topk_policy_t::scan_algorithm, OffsetT, OutOffsetT>;

  // In `process_partial` mode the trailing partial tile is staged into an smem histogram (primed
  // from the global slab), read into blocked registers, and fed to the bucket-finder from registers
  // -- avoiding the global atomic-adds and the global histogram re-read. The staged histogram is
  // dead before the bucket-finder reuses the smem for its scan, so the two alias via a union. In
  // the default mode only the bucket-finder storage is needed.
  using tile_histogram_t = detail::batched_topk::tile_histogram<block_threads, num_buckets, OffsetT, ExtractBinOpT>;
  union staged_storage_t
  {
    typename tile_histogram_t::TempStorage staged_histogram;
    typename block_identify_kth_bucket_t::TempStorage prefix_sum;
  };
  union plain_storage_t
  {
    typename block_identify_kth_bucket_t::TempStorage prefix_sum;
  };
  __shared__ ::cuda::std::conditional_t<process_partial, staged_storage_t, plain_storage_t> temp_storage;

  const narrow_segment_count_t<NumSegmentsParameterT> num_large_segments =
    static_cast<narrow_segment_count_t<NumSegmentsParameterT>>(*large_segments_count_it);

  // Grid-stride loop over queue slots. One CTA owns one segment for the duration of that
  // segment's epilogue; CTAs are independent and write to disjoint counter / histogram slabs.
  using queue_idx_t = narrow_segment_count_t<NumSegmentsParameterT>;
  for (queue_idx_t queue_idx = static_cast<queue_idx_t>(blockIdx.x); queue_idx < num_large_segments;
       queue_idx += static_cast<queue_idx_t>(gridDim.x))
  {
    const auto segment_id      = segment_id_provider[queue_idx];
    const OffsetT num_items    = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
    counter_t* segment_counter = d_segment_counters + queue_idx;
    OffsetT* segment_histogram = d_segment_histograms + queue_idx * num_buckets;

    // Clip `k` to the segment's input size (same as the histogram agent did pre-refactor; see
    // the comment on that clip for why).
    const OutOffsetT k =
      (::cuda::std::min) (static_cast<OutOffsetT>(k_param.get_param(segment_id)), static_cast<OutOffsetT>(num_items));

    // Per-segment counter update + kth-bucket find. `on_kth_bucket` writes the kth bucket's
    // bin index into the counter's `kth_key_bits` for the next pass to consume, and decrements
    // `k` by the count of already-selected items.
    auto on_kth_bucket =
      [segment_counter, pass](OutOffsetT current_k, int bin_index, OffsetT num_selected, OffsetT num_candidates) {
        segment_counter->k                  = static_cast<OutOffsetT>(current_k - num_selected);
        segment_counter->num_candidates_out = num_candidates;
        detail::batched_topk::set_kth_key_bits<bits_per_pass>(
          segment_counter->kth_key_bits, pass, static_cast<unsigned int>(bin_index));
      };

    if (threadIdx.x == 0)
    {
      segment_counter->num_candidates_in      = num_items;
      segment_counter->num_candidates_written = 0;
    }

    if constexpr (process_partial)
    {
      // "Histogram processes full tiles only" mode: the companion histogram kernel skipped each
      // segment's trailing partial tile. Rather than atomic-adding the partial into the global
      // slab and re-reading the global histogram, stage the complete histogram in smem: prime it
      // from the global slab (the full-tile counts), add the partial tile via fast smem atomics,
      // then drain it into blocked registers and run the bucket-finder against registers.
      tile_histogram_t hist{temp_storage.staged_histogram, extract_bin_op};
      hist.load_from(segment_histogram);
      __syncthreads();

      const OffsetT num_full_tiles = num_items / static_cast<OffsetT>(tile_items);
      const OffsetT partial_items  = num_items - num_full_tiles * static_cast<OffsetT>(tile_items);
      if (partial_items > OffsetT{0})
      {
        const auto inner_key_it = d_key_segments_it[segment_id];
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);
        // Blocked per-thread load of the trailing partial tile (matches the histogram agent's
        // `add_partial` arrangement); out-of-range lanes load nothing and are not binned.
        KeyT items[items_per_thread];
        const OffsetT thread_base = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < items_per_thread; ++j)
        {
          const OffsetT idx = thread_base + static_cast<OffsetT>(j);
          if (idx < num_items)
          {
            items[j] = inner_key_it[idx];
          }
        }
        const int num_thread_items =
          (thread_base >= num_items)
            ? 0
            : static_cast<int>((::cuda::std::min) (static_cast<OffsetT>(items_per_thread), num_items - thread_base));
        hist.add_partial(items, num_thread_items);
        __syncthreads();
      }

      // Drain the staged histogram into this thread's blocked register chunk (direct smem read, no
      // transpose), then run the bucket-finder against registers. The staged-histogram smem aliases
      // the bucket-finder's scan storage, so the read must complete before the scan overwrites it.
      OffsetT thread_histogram[block_identify_kth_bucket_t::bins_per_thread];
      block_identify_kth_bucket_t::load_blocked(hist.data(), thread_histogram);
      __syncthreads();
      block_identify_kth_bucket_t{temp_storage.prefix_sum}.find_kth_bucket(thread_histogram, k, on_kth_bucket);
    }
    else
    {
      block_identify_kth_bucket_t{temp_storage.prefix_sum}.find_kth_bucket(segment_histogram, k, on_kth_bucket);
    }

    if (reset_histogram)
    {
      // Zero the per-segment histogram slab so the next pass starts clean. The two
      // `__syncthreads()` bracket the reset against the kth-bucket primitive's smem reuse and
      // against the next iteration's load.
      __syncthreads();
      detail::batched_topk::init_histogram<block_threads, num_buckets>(segment_histogram);
    }

    // Separate iterations work on independent counter / histogram slabs but share the smem
    // `temp_storage.prefix_sum` arena; barrier between iterations.
    __syncthreads();
  }
}

template <typename PolicySelector,
          detail::topk::select SelectDirection,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename LargeSegmentsCountItT,
          typename DecomposerT,
          typename OffsetT,
          typename OutOffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires batched_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_filter_kernel(
    const KeyInputItItT d_key_segments_it,
    const KeyOutputItItT d_key_segments_out_it,
    const ValueInputItItT d_value_segments_it,
    const ValueOutputItItT d_value_segments_out_it,
    const SegmentSizeParameterT segment_sizes,
    const SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* const d_large_segments_tile_offsets,
    // See the `device_segmented_topk_histogram_kernel` doc for why we mark the pointer
    // (not the pointee) grid-constant.
    detail::batched_topk::counter<it_value_t<it_value_t<KeyInputItItT>>, OffsetT, OutOffsetT>* const d_segment_counters,
    OffsetT* const d_segment_histograms,
    it_value_t<it_value_t<KeyInputItItT>>* const d_segment_in_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_in_val_buf,
    it_value_t<it_value_t<KeyInputItItT>>* const d_segment_out_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_out_val_buf,
    const OffsetT candidate_buffer_length,
    const OffsetT candidate_buffer_coefficient,
    const LargeSegmentsCountItT large_segments_count_it,
    const int pass,
    const int total_bits,
    const bool reset_histogram,
    const DecomposerT decomposer)
{
  using key_t = it_value_t<it_value_t<KeyInputItItT>>;
  // See the histogram kernel for the rationale behind reading `total_large_tiles` from the
  // sentinel slot and `large_segments_count` through an iterator. Narrowed to `narrow_segment_count_t`
  // so the agent's `resolve_queue_idx` `UpperBound` + indexing stay 32-bit when the count fits.
  const narrow_segment_count_t<NumSegmentsParameterT> num_large_segments =
    static_cast<narrow_segment_count_t<NumSegmentsParameterT>>(*large_segments_count_it);
  // Pointer to the sentinel slot of the per-segment tile-offset table; the agent dereferences
  // it lazily at the grid-stride loop boundary instead of materialising the value into a
  // long-lived register at kernel entry. See the agent's `run` doc for the register-pressure
  // motivation.
  const LargeSegmentTileOffsetT* const d_total_large_tiles = &d_large_segments_tile_offsets[num_large_segments];
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;

  static constexpr batched_topk_policy bp                                      = current_policy<PolicySelector>();
  static constexpr multi_worker_policy mw                                      = bp.multi_worker_per_segment_policy;
  static constexpr detail::topk::block_partition_strategy buffered_part_strat  = mw.buffered_partition_strategy;
  static constexpr detail::topk::block_filter_strategy early_stop_filter_strat = mw.early_stop_filter_strategy;
  static constexpr bool lazy_value_load                                        = mw.lazy_value_load;
  static constexpr bool inlined_classify                                       = mw.inlined_classify;
  // Compile-time switch for the experimental "filter only walks full tiles" mode. When
  // `true`, the agent drops the slow-path `dispatch_tile<false>` call; the trailing
  // partial tile of each segment is processed by
  // `device_segmented_topk_finalize_filter_kernel` via
  // `agent.process_partial_for_segment(queue_idx, pass)`.
  static constexpr bool full_tiles_only_filter = topk_seg_kernel_detail::full_tiles_only_filter<PolicySelector>::value;

  using extract_bin_op_t =
    detail::batched_topk::extract_bin_op_t<key_t, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;
  using identify_candidates_op_t = detail::batched_topk::
    identify_candidates_op_t<key_t, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;

  using agent_t = agent_batched_topk_filter_partition<
    agent_topk_policy_t,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    extract_bin_op_t,
    identify_candidates_op_t,
    DecomposerT,
    SegmentSizeParameterT,
    NumSegmentsParameterT,
    SegmentIdProviderT,
    LargeSegmentTileOffsetT,
    OffsetT,
    OutOffsetT,
    buffered_part_strat,
    early_stop_filter_strat,
    lazy_value_load,
    inlined_classify,
    full_tiles_only_filter>;

  __shared__ typename agent_t::TempStorage temp_storage;
  const extract_bin_op_t extract_bin_op{pass, total_bits, decomposer};
  // `identify_candidates_op_t` is constructed inside the agent's `run()` after the on-device
  // binary search resolves the per-segment counter (and thus the per-segment `kth_key_bits`
  // pointer). The agent stores `(pass, total_bits, decomposer)` plus the per-segment counter
  // pointer to rebuild it.

  agent_t agent(
    temp_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    segment_id_provider,
    d_large_segments_tile_offsets,
    d_segment_counters,
    d_segment_histograms,
    d_segment_in_key_buf,
    d_segment_in_val_buf,
    d_segment_out_key_buf,
    d_segment_out_val_buf,
    extract_bin_op,
    total_bits,
    decomposer,
    candidate_buffer_length,
    candidate_buffer_coefficient,
    num_large_segments);

  // Grid-stride loop now lives inside `agent.run<TilesPerChunk>(pass)` -- same shape the
  // histogram / last_filter agents use. The kernel materialises the policy's
  // `tiles_per_chunk` knob and hands off.
  //
  // The per-segment epilogue (counter update + prefix-sum + bucket-finder + optional global
  // histogram reset) is done by `device_segmented_topk_finalize_filter_kernel`, which the
  // dispatch launches on the same stream right after this kernel; `reset_histogram` flows
  // to that kernel rather than this one.
  (void) reset_histogram;
  (void) d_total_large_tiles;
  static constexpr int tiles_per_chunk = topk_seg_kernel_detail::tiles_per_chunk<PolicySelector>::value;
  agent.template run<tiles_per_chunk>(pass);
}

// Per-segment epilogue kernel for the filter pass. Runs after
// `device_segmented_topk_filter_kernel` finishes (host-side launch ordering on the same stream).
// One CTA per large segment: prefix-sums the per-segment global histogram (skipping early_stop
// segments), finds the bucket containing the k-th key, updates the per-segment counter, and
// (optionally) zeros the histogram slab for the next pass.
//
// The `early_stop` / `will_buffer` mode discovered by the filter pass per segment is recomputed
// here from the same counter fields the filter agent read at entry; the filter kernel does not
// modify those fields, so the two kernels stay in lock-step without an extra device-side flag.
template <typename PolicySelector,
          detail::topk::select SelectDirection,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename LargeSegmentsCountItT,
          typename DecomposerT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyT>
#if _CCCL_HAS_CONCEPTS()
  requires batched_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_finalize_filter_kernel(
    const KeyInputItItT d_key_segments_it,
    const KeyOutputItItT d_key_segments_out_it,
    const ValueInputItItT d_value_segments_it,
    const ValueOutputItItT d_value_segments_out_it,
    const SegmentSizeParameterT segment_sizes,
    const SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* const d_large_segments_tile_offsets,
    // See the `device_segmented_topk_histogram_kernel` doc for why we mark the pointer
    // (not the pointee) grid-constant.
    detail::batched_topk::counter<KeyT, OffsetT, OutOffsetT>* const d_segment_counters,
    OffsetT* const d_segment_histograms,
    KeyT* const d_segment_in_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_in_val_buf,
    KeyT* const d_segment_out_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_out_val_buf,
    const LargeSegmentsCountItT large_segments_count_it,
    const OffsetT candidate_buffer_length,
    const OffsetT candidate_buffer_coefficient,
    const int pass,
    const int total_bits,
    const DecomposerT decomposer,
    const bool reset_histogram)
{
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;
  static constexpr int block_threads = agent_topk_policy_t::block_threads;
  static constexpr int bits_per_pass = agent_topk_policy_t::bits_per_pass;
  static constexpr int num_buckets   = 1 << bits_per_pass;

  // Mirrors `topk_seg_kernel_detail::full_tiles_only_filter<PolicySelector>::value`. When
  // `true`, the companion filter kernel skipped the trailing partial tile of every
  // segment; this kernel's responsibility, before running the per-segment prefix-sum +
  // bucket-finder, is to call `agent.process_partial_for_segment(queue_idx, pass)` to
  // re-inject the partial-tile contribution via the appropriate mode-specific partition
  // primitive.
  static constexpr bool process_partial = topk_seg_kernel_detail::full_tiles_only_filter<PolicySelector>::value;

  using counter_t                   = detail::batched_topk::counter<KeyT, OffsetT, OutOffsetT>;
  using block_identify_kth_bucket_t = detail::batched_topk::
    block_identify_kth_bucket<block_threads, bits_per_pass, agent_topk_policy_t::scan_algorithm, OffsetT, OutOffsetT>;

  // Partial-tile processing instantiates the same filter agent the filter kernel uses --
  // same `extract_bin_op_t` / `identify_candidates_op_t`, same partition primitives, same
  // smem layout. Only the entry method differs (`process_partial_for_segment` vs `run`).
  static constexpr batched_topk_policy bp                                      = current_policy<PolicySelector>();
  static constexpr multi_worker_policy mw                                      = bp.multi_worker_per_segment_policy;
  static constexpr detail::topk::block_partition_strategy buffered_part_strat  = mw.buffered_partition_strategy;
  static constexpr detail::topk::block_filter_strategy early_stop_filter_strat = mw.early_stop_filter_strategy;
  static constexpr bool lazy_value_load                                        = mw.lazy_value_load;
  static constexpr bool inlined_classify                                       = mw.inlined_classify;

  using extract_bin_op_t =
    detail::batched_topk::extract_bin_op_t<KeyT, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;
  using identify_candidates_op_t = detail::batched_topk::
    identify_candidates_op_t<KeyT, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;

  using filter_agent_t = agent_batched_topk_filter_partition<
    agent_topk_policy_t,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    extract_bin_op_t,
    identify_candidates_op_t,
    DecomposerT,
    SegmentSizeParameterT,
    NumSegmentsParameterT,
    SegmentIdProviderT,
    LargeSegmentTileOffsetT,
    OffsetT,
    OutOffsetT,
    buffered_part_strat,
    early_stop_filter_strat,
    lazy_value_load,
    inlined_classify,
    /*FullTilesOnly=*/process_partial>;

  // Union the agent's per-tile smem (~9 KB for the buffered arm: smem histogram + keys
  // source state + partition arena) with the prefix-sum scratch. Partial processing
  // touches `agent_storage` first; after `__syncthreads()` the bytes are reused for
  // `prefix_sum`.
  __shared__ union
  {
    typename filter_agent_t::TempStorage agent_storage;
    typename block_identify_kth_bucket_t::TempStorage prefix_sum;
  } temp_storage;

  // The agent's constructor is cheap (member-init of pointers + iterators) so we always
  // build it; ptxas drops the unused args when `process_partial == false`.
  const narrow_segment_count_t<NumSegmentsParameterT> num_large_segments =
    static_cast<narrow_segment_count_t<NumSegmentsParameterT>>(*large_segments_count_it);
  const extract_bin_op_t extract_bin_op{pass, total_bits, decomposer};
  filter_agent_t agent{
    temp_storage.agent_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    segment_id_provider,
    d_large_segments_tile_offsets,
    d_segment_counters,
    d_segment_histograms,
    d_segment_in_key_buf,
    d_segment_in_val_buf,
    d_segment_out_key_buf,
    d_segment_out_val_buf,
    extract_bin_op,
    total_bits,
    decomposer,
    candidate_buffer_length,
    candidate_buffer_coefficient,
    num_large_segments};

  using queue_idx_t = narrow_segment_count_t<NumSegmentsParameterT>;
  for (queue_idx_t queue_idx = static_cast<queue_idx_t>(blockIdx.x); queue_idx < num_large_segments;
       queue_idx += static_cast<queue_idx_t>(gridDim.x))
  {
    const auto segment_id      = segment_id_provider[queue_idx];
    const OffsetT num_items    = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
    counter_t* segment_counter = d_segment_counters + queue_idx;
    OffsetT* segment_histogram = d_segment_histograms + queue_idx * num_buckets;

    const OutOffsetT current_k         = segment_counter->k;
    const OffsetT current_len          = segment_counter->num_candidates_out;
    const OffsetT counter_input_length = segment_counter->num_candidates_in;

    // Skip empty segments (universal early-exit) -- match the filter agent's same check.
    if (counter_input_length == 0)
    {
      __syncthreads();
      continue;
    }

    // Phase 1: trailing partial-tile work (only when the filter kernel skipped it). The
    // agent reads the per-pass counter fields itself; this must run BEFORE the per-mode
    // counter update below so it sees the same `current_k` / `num_candidates_out` /
    // `load_from_candidates_buffer` the filter agent did, and so picks the same mode.
    if constexpr (process_partial)
    {
      __syncthreads();
      agent.process_partial_for_segment(queue_idx, pass);
      __syncthreads();
    }

    // Recompute the mode the filter pass took for this segment, from the same counter fields
    // the filter agent read at entry. Same expressions, same operands, so we stay in lock-step
    // without an extra device-side flag.
    const bool early_stop  = (current_len == static_cast<OffsetT>(current_k));
    const bool will_buffer = !early_stop && (current_len <= candidate_buffer_length)
                          && (current_len <= num_items / candidate_buffer_coefficient);

    // Per-mode counter update (mirror of the filter agent's pre-refactor `counter_update_fn`):
    //   - early_stop : write `num_candidates_in = 0` (universal early-exit for the next pass).
    //   - buffered   : write `num_candidates_in = current_len`, flip
    //                  `load_from_candidates_buffer` to true, reset `num_candidates_written`.
    //   - unbuffered : no counter writes.
    if (threadIdx.x == 0)
    {
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
    }

    // For early_stop the histogram is meaningless (the agent did not touch it). Skip the
    // prefix-sum + bucket-finder entirely. For buffered / unbuffered, both branches updated
    // the per-segment global histogram, so both run the kth-bucket scan.
    if (!early_stop)
    {
      auto on_kth_bucket =
        [segment_counter, pass](OutOffsetT k_cb, int bin_index, OffsetT num_selected, OffsetT num_candidates) {
          segment_counter->k                  = static_cast<OutOffsetT>(k_cb - num_selected);
          segment_counter->num_candidates_out = num_candidates;
          detail::batched_topk::set_kth_key_bits<bits_per_pass>(
            segment_counter->kth_key_bits, pass, static_cast<unsigned int>(bin_index));
        };

      __syncthreads();
      block_identify_kth_bucket_t{temp_storage.prefix_sum}.find_kth_bucket(segment_histogram, current_k, on_kth_bucket);

      if (reset_histogram)
      {
        __syncthreads();
        detail::batched_topk::init_histogram<block_threads, num_buckets>(segment_histogram);
      }
    }

    // Separate iterations work on independent counter / histogram slabs but share the smem
    // `temp_storage.prefix_sum` arena; barrier between iterations.
    __syncthreads();
  }
}

template <typename PolicySelector,
          detail::topk::select SelectDirection,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename LargeSegmentsCountItT,
          typename DecomposerT,
          typename OffsetT,
          typename OutOffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires batched_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_topk_last_filter_kernel(
    const KeyInputItItT d_key_segments_it,
    const KeyOutputItItT d_key_segments_out_it,
    const ValueInputItItT d_value_segments_it,
    const ValueOutputItItT d_value_segments_out_it,
    const SegmentSizeParameterT segment_sizes,
    const KParameterT k_param,
    const SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* const d_large_segments_tile_offsets,
    // See the `device_segmented_topk_histogram_kernel` doc for why we mark the pointer
    // (not the pointee) grid-constant.
    detail::batched_topk::counter<it_value_t<it_value_t<KeyInputItItT>>, OffsetT, OutOffsetT>* const d_segment_counters,
    it_value_t<it_value_t<KeyInputItItT>>* const d_segment_in_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_in_val_buf,
    const OffsetT candidate_buffer_length,
    const LargeSegmentsCountItT large_segments_count_it,
    const int pass,
    const int total_bits,
    const DecomposerT decomposer)
{
  using key_t = it_value_t<it_value_t<KeyInputItItT>>;
  // Materialise the queue-shape `num_large_segments` so the agent can hold it as a member
  // (the agent re-derives `d_total_large_tiles` from `d_large_segments_tile_offsets +
  // num_large_segments` itself on entry to `run`). Narrowed to `narrow_segment_count_t` to keep the
  // agent's `resolve_queue_idx` `UpperBound` + indexing 32-bit when the count fits.
  const narrow_segment_count_t<NumSegmentsParameterT> num_large_segments =
    static_cast<narrow_segment_count_t<NumSegmentsParameterT>>(*large_segments_count_it);
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;

  static constexpr batched_topk_policy bp                            = current_policy<PolicySelector>();
  static constexpr multi_worker_policy mw                            = bp.multi_worker_per_segment_policy;
  static constexpr detail::topk::block_partition_strategy part_strat = mw.last_filter_partition_strategy;
  static constexpr bool lazy_value_load                              = mw.lazy_value_load;
  static constexpr bool inlined_classify                             = mw.inlined_classify;

  using identify_candidates_op_t = detail::batched_topk::
    identify_candidates_op_t<key_t, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;

  using agent_t = agent_batched_topk_last_filter<
    agent_topk_policy_t,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    identify_candidates_op_t,
    DecomposerT,
    SegmentSizeParameterT,
    KParameterT,
    NumSegmentsParameterT,
    SegmentIdProviderT,
    LargeSegmentTileOffsetT,
    OffsetT,
    OutOffsetT,
    part_strat,
    lazy_value_load,
    inlined_classify>;

  __shared__ typename agent_t::TempStorage temp_storage;

  agent_t agent(
    temp_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    k_param,
    segment_id_provider,
    d_large_segments_tile_offsets,
    d_segment_counters,
    d_segment_in_key_buf,
    d_segment_in_val_buf,
    pass,
    total_bits,
    decomposer,
    candidate_buffer_length,
    num_large_segments);

  // Grid-stride loop now lives inside `agent.run<TilesPerChunk>()` -- same shape the
  // histogram agent uses. The kernel materialises the policy's `tiles_per_chunk` knob and
  // hands off.
  static constexpr int tiles_per_chunk = topk_seg_kernel_detail::tiles_per_chunk<PolicySelector>::value;
  agent.template run<tiles_per_chunk>();
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
