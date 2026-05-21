// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. SPDX-License-Identifier:
// Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items from
//! sequences of unordered data items residing within device-accessible memory.

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
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/dispatch_topk_common.cuh>
#include <cub/device/dispatch/kernels/kernel_batched_topk.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__iterator/constant_iterator.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/__iterator/transform_output_iterator.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// -----------------------------------------------------------------------------
// Segmented Top-K-Specific Parameter Types
// -----------------------------------------------------------------------------

// ------------ SELECTION DIRECTION PARAMETER TYPES ------------

// Selection direction known at compile time, same value applies to all segments
template <detail::topk::select SelectDirection>
using select_direction_static = params::uniform_discrete_param<detail::topk::select, SelectDirection>;

// Selection direction is a runtime value, same value applies to all segments
using select_direction_uniform =
  params::uniform_discrete_param<detail::topk::select, detail::topk::select::max, detail::topk::select::min>;

// Per-segment selection direction via iterator
template <typename SelectionDirectionIt, detail::topk::select... SelectDirectionOptions>
using select_direction_per_segment =
  params::per_segment_discrete_param<SelectionDirectionIt, detail::topk::select, SelectDirectionOptions...>;

// ------------ SEGMENT SIZE PARAMETER TYPES ------------

// Segment size known at compile time, same value applies to all segments
template <::cuda::std::int64_t SegmentSize>
using segment_size_static = params::static_constant_param<::cuda::std::int64_t, SegmentSize>;

// Segment size is a runtime value, same value applies to all segments
template <::cuda::std::int64_t MinSegmentSize = 0,
          ::cuda::std::int64_t MaxSegmentSize = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using segment_size_uniform = params::uniform_param<::cuda::std::int64_t, MinSegmentSize, MaxSegmentSize>;

// Segment size via iterator
template <typename SegmentSizesItT,
          ::cuda::std::int64_t MinSegmentSize = 1,
          ::cuda::std::int64_t MaxSegmentSize = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using segment_size_per_segment =
  params::per_segment_param<SegmentSizesItT, ::cuda::std::int64_t, MinSegmentSize, MaxSegmentSize>;

// ------------ K PARAMETER TYPES ------------

// K known at compile time, same value applies to all segments
template <::cuda::std::int64_t K>
using k_static = params::static_constant_param<::cuda::std::int64_t, K>;

// K is a runtime value, same value applies to all segments
template <::cuda::std::int64_t MinK = 1,
          ::cuda::std::int64_t MaxK = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using k_uniform = params::uniform_param<::cuda::std::int64_t, MinK, MaxK>;

// K via iterator
template <typename KItT,
          ::cuda::std::int64_t MinK = 1,
          ::cuda::std::int64_t MaxK = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using k_per_segment = params::per_segment_param<KItT, ::cuda::std::int64_t, MinK, MaxK>;

// ------------ TOTAL NUMBER OF SEGMENTS ------------
// Number of segments known at compile time
template <::cuda::std::int64_t StaticNumSegments>
using num_segments_static = params::static_constant_param<::cuda::std::int64_t, StaticNumSegments>;

// Number of segments is a runtime value
template <::cuda::std::int64_t MinNumSegments = 1,
          ::cuda::std::int64_t MaxNumSegments = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using num_segments_uniform = params::uniform_param<::cuda::std::int64_t, MinNumSegments, MaxNumSegments>;

// Number of segments via iterator
template <typename NumSegmentsItT,
          ::cuda::std::int64_t MinNumSegments = 1,
          ::cuda::std::int64_t MaxNumSegments = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using num_segments_indirect =
  params::per_segment_param<NumSegmentsItT, ::cuda::std::int64_t, MinNumSegments, MaxNumSegments>;

// ------------ TOTAL NUMBER OF ITEMS PARAMETER TYPES ------------

// Number of items guarantee
template <::cuda::std::int64_t MinNumItems = 1,
          ::cuda::std::int64_t MaxNumItems = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
struct total_num_items_guarantee
{
  using value_type                                 = ::cuda::std::int64_t;
  static constexpr value_type static_min_num_items = MinNumItems;
  static constexpr value_type static_max_num_items = MaxNumItems;

  value_type min_num_items = MinNumItems;
  value_type max_num_items = MaxNumItems;

  // Create default ctor, 1 param ctor taking min, 2 param ctor taking min/max
  total_num_items_guarantee() = default;

  _CCCL_HOST_DEVICE total_num_items_guarantee(value_type num_items)
      : min_num_items(num_items)
      , max_num_items(num_items)
  {}

  _CCCL_HOST_DEVICE total_num_items_guarantee(value_type min_items, value_type max_items)
      : min_num_items(min_items)
      , max_num_items(max_items)
  {}
};

// -----------------------------------------------------------------------------
// Helper: compile-time predicate "does this (non-negative) integer value fit in `uint32_t`?".
//
// Used by the `OffsetT` / `OutOffsetT` deduction below to pick `uint32_t` whenever any of the
// available upper bounds justifies it. The cast through `unsigned long long` lets us compare
// values of arbitrary integral types against `numeric_limits<uint32_t>::max()` without running
// into narrow-type truncation; negative inputs (not expected for size/count bounds) wrap to a
// large value and report `false`.
// -----------------------------------------------------------------------------
template <auto Value>
inline constexpr bool fits_in_uint32_v =
  static_cast<unsigned long long>(Value)
  <= static_cast<unsigned long long>(::cuda::std::numeric_limits<::cuda::std::uint32_t>::max());

// -----------------------------------------------------------------------------
// Helper: turn a segment ID into the number of large-segment-agent tiles needed
// to cover that segment. Wrapped in a transform_iterator, this produces the
// per-segment tile counts that we exclusive-scan to obtain per-segment tile
// offsets.
// -----------------------------------------------------------------------------
template <class SegmentSizeParameterT, class TotalNumItemsValueType, class NumSegmentsParameterT>
struct segment_size_to_tile_count_op
{
  SegmentSizeParameterT segment_sizes;
  int large_segment_agent_tile_size;
  
  // Stored as a params object so this keeps the all-large-segments scan working
  // even when `num_segments` is a device-accessible-only param.
  // This cutoff used to make the op safe to evaluate at the index (at position `num_segments`) for the total aggregate. The
  // all-large-segments path scans over `num_segments + 1` inputs so the inclusive total ends up in the
  // trailing slot of the offset table; for that to work without indexing past the end of `segment_sizes`, the op short-circuits to 0 at the sentinel.
  NumSegmentsParameterT num_segments;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr TotalNumItemsValueType operator()(SegmentIndexT segment_id) const
  {
    using num_segments_val_t = typename NumSegmentsParameterT::value_type;
    if (static_cast<num_segments_val_t>(segment_id) >= num_segments.get_param(0))
    {
      return TotalNumItemsValueType{0};
    }
    return static_cast<TotalNumItemsValueType>(
      ::cuda::ceil_div(segment_sizes.get_param(segment_id), large_segment_agent_tile_size));
  }
};

// -----------------------------------------------------------------------------
// Helper: constant-value transform op over a `params`-like object.
//
// Wrapped in a `cuda::transform_iterator` over a `counting_iterator`, this turns into an
// "iterator that always dereferences to the same value": the first element of the wrapped
// `params` object (i.e. `params.get_param(0)`). The all-large-segments path uses it to feed
// `num_segments` to the multi-CTA kernels through the same `LargeSegmentsCountItT` interface
// the mixed path uses to feed `&d_counters->large_segments_count`.
//
// The functor stores the params object rather than its first value so the dereference happens
// on-device at kernel-entry time. That way we do not bake a host-side `num_segments.get_param(0)`
// read into the all-large path; the dispatch only needs that value for places that genuinely
// require a host-resident scalar (allocation sizing, scan extent, grid dim of the
// worker-per-segment kernel). Kept local to the dispatch until a second user appears.
// -----------------------------------------------------------------------------
template <typename ParamObjT>
struct constant_value_op
{
  ParamObjT params;

  template <typename IndexT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(IndexT) const
  {
    return params.get_param(0);
  }
};

// -----------------------------------------------------------------------------
// Helper: per-segment indexed-mode output-iterator builder.
//
// When the multi-CTA-per-segment path runs in `value_materialization_mode::indexed`, the
// candidate buffer stores `OffsetT` indices instead of full values. To translate "agent writes
// index" into "values_out[pos] = values_in[idx]" we wrap each segment's value-output iterator in a
// `cuda::transform_output_iterator` whose transform op is `topk_index_gather_op{user_in[i]}`.
//
// This functor, wrapped in a `cuda::transform_iterator` over a `counting_iterator<segment_id>`,
// gives us an iterator-of-iterators that produces the per-segment wrapped output iterator on
// `operator[](segment_id)`. The captured outer iterators must be trivially copyable -- they
// travel by value into the kernel argument area, same as the unwrapped `d_value_segments_*`
// iterators do today.
// -----------------------------------------------------------------------------
template <typename ValueInputItItT, typename ValueOutputItItT>
struct per_segment_indexed_out_op
{
  ValueInputItItT d_value_segments_it;
  ValueOutputItItT d_value_segments_out_it;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto operator()(SegmentIndexT segment_id) const
  {
    using inner_value_in_it_t = it_value_t<ValueInputItItT>;
    using gather_op_t         = detail::topk::topk_index_gather_op<inner_value_in_it_t>;
    return ::cuda::make_transform_output_iterator(
      d_value_segments_out_it[segment_id], gather_op_t{d_value_segments_it[segment_id]});
  }
};

// -----------------------------------------------------------------------------
// Segmented Top-K Dispatch
// -----------------------------------------------------------------------------

//! @param d_temp_storage Device-accessible allocation of temporary storage. When `nullptr`, the required allocation
//!        size is written to `temp_storage_bytes` and no work is done.
//! @param temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation
//! @param d_key_segments_it d_key_segments_it[segment_index] -> iterator to the input sequence of key data for segment
//!        `segment_index`
//! @param d_key_segments_out_it d_key_segments_out_it[segment_index] -> iterator to the output sequence of key data for
//!        segment `segment_index`
//! @param d_value_segments_it d_value_segments_it[segment_index] -> iterator to the input sequence of associated value
//!        items for segment `segment_index`. When cub::NullType**, only keys are provided.
//! @param d_value_segments_out_it d_value_segments_out_it[segment_index] -> iterator to the output sequence of
//!        associated value items for segment `segment_index`
//! @param segment_sizes Parameter providing segment sizes for each segment
//! @param k Parameter providing K for each segment
//! @param select_directions Parameter providing the selection direction for each segment
//! @param num_segments Number of segments. May be supplied as a host-resident value (e.g.
//!        `num_segments_static<...>`, `num_segments_uniform<...>{actual}`) or as a
//!        device-accessible-only value (`num_segments_indirect<It, Min, Max>{iter, min_v,
//!        max_v}`). For the device-accessible-only form, the dispatch over-provisions every
//!        host-resident sizing quantity (allocation extents, `cudaMemsetAsync` extents,
//!        worker-per-segment `grid_dim`, all-large scan extent, multi-CTA grid cap) using
//!        `num_segments.max_value` -- the runtime upper bound on the params object, which the
//!        framework guarantees is at most the static `Max` template arg. Callers using
//!        `num_segments_indirect` must therefore supply a tight `Max` (template) or
//!        `max_value` (ctor arg); a default `Max == numeric_limits<int64_t>::max()` would
//!        result in impractically large allocations.
//! @param total_num_items_guarantee Allows the user to provide a guarantee on the upper bound of the total number of
//!        items
template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT,
          typename PolicySelector = policy_selector_from_types<it_value_t<it_value_t<KeyInputItItT>>,
                                                               it_value_t<it_value_t<ValueInputItItT>>,
                                                               ::cuda::std::int64_t,
                                                               params::static_max_value_v<KParameterT>>>
#if _CCCL_HAS_CONCEPTS()
  requires batched_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  [[maybe_unused]] TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream                             = nullptr,
  [[maybe_unused]] PolicySelector policy_selector = {})
{
  // Index type for the per-segment tile-offset table and for global tile ids in the multi-CTA
  // path. Pinned at `uint32_t` to keep the table dense and the multi-CTA agents' binary search
  // cheap.
  //
  // Supported upper bound: `multi_worker_per_segment_tile_size * numeric_limits<uint32_t>::max()`
  // items in aggregate across all segments (plus the per-segment partial-tile slack -- at most
  // `num_segments_upper_bound` extra tiles). For the default `2048`-item multi-worker tile
  // size that is roughly `8.8 * 10^12` items, which is well beyond any realistic batched
  // workload. Workloads that exceed this contract will silently overflow `total_large_tiles`
  // and `d_large_segments_tile_offsets` entries; the dispatch does not currently validate it
  // at runtime.
  using large_segment_tile_offset_t = ::cuda::std::uint32_t;
  // Resolver that determines (a) whether there's any one-worker-per-segment policy supporting the
  // range of segment sizes, and (b) if so, which of the one-worker-per-segment policies to
  // use. Prefers the smallest covering+fits-smem policy (`find_smallest_covering_policy`) and
  // falls back to the largest fits-smem policy (`find_largest_fitting_smem_policy`) when no
  // covering policy was found. In the fallback case `only_small_segments == false` below,
  // and any segment exceeding the chosen tile size is routed onto the multi-CTA-per-segment path.
  using resolved_worker_per_segment_t = resolved_worker_per_segment_policy<
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
    large_segment_tile_offset_t>;
    
  // If there's no fitting one-worker-per-segment policy, this kernel should be bypassed altogether in favor of a multi-CTA-per-segment-only kernel
  static_assert(resolved_worker_per_segment_t::found,
                "No valid policy found for one-worker-per-segment approach");

  constexpr auto policy = resolved_worker_per_segment_t::policy;
  constexpr worker_policy worker_per_segment_policy             = policy.worker_per_segment_policy;
  constexpr multi_worker_policy multi_worker_per_segment_policy = policy.multi_worker_per_segment_policy;

  static constexpr int worker_per_segment_tile_size =
    worker_per_segment_policy.threads_per_block * worker_per_segment_policy.items_per_thread;
  static constexpr bool any_small_segments =
    params::static_min_value_v<SegmentSizeParameterT> <= worker_per_segment_tile_size;
  static constexpr bool only_small_segments =
    params::static_max_value_v<SegmentSizeParameterT> <= worker_per_segment_tile_size;

  // Derived value-channel types -- the segmented dispatch is iterator-of-iterators, so peel the
  // inner iterator type once and then re-derive the inner value type. `keys_only` propagates
  // from `ValueInputItItT == NullType**`.
  using key_in_t        = it_value_t<it_value_t<KeyInputItItT>>;
  using value_in_t      = it_value_t<it_value_t<ValueInputItItT>>;
  static constexpr bool keys_only = ::cuda::std::is_same_v<value_in_t, cub::NullType>;

  using num_segments_val_t         = typename NumSegmentsParameterT::value_type;
  using counters_t                 = batched_topk_counters<num_segments_val_t>;
  using segment_size_scan_offset_t = detail::choose_offset_t<num_segments_val_t>;
  using segment_size_scan_input_op_t =
    segment_size_to_tile_count_op<SegmentSizeParameterT, large_segment_tile_offset_t, NumSegmentsParameterT>;
  static constexpr auto multi_worker_per_segment_tile_size =
    multi_worker_per_segment_policy.threads_per_block * multi_worker_per_segment_policy.items_per_thread;
  // `num_segments` is captured by value into the op so its sentinel cutoff can be resolved
  // on-device via `num_segments.get_param(0)`. The op is host-callable but only ever invoked
  // on-device (inside the all-large-path scan's transform iterator).
  const segment_size_scan_input_op_t segment_size_scan_input_op{
    segment_sizes, multi_worker_per_segment_tile_size, num_segments};
  // Transform iterator over [0, num_segments + 1) producing the tile-count for each segment (and
  // 0 for the sentinel index at position `num_segments`). The scans on both the all-large and
  // mixed paths consume one extra input compared to the previous design so the resulting offset
  // table is naturally `num_segments + 1` entries wide, with the last entry equal to
  // `total_large_tiles`.
  [[maybe_unused]] const auto segment_size_scan_input_it = ::cuda::transform_iterator(
    ::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}}, segment_size_scan_input_op);

  // OffsetT / OutOffsetT chosen as the smaller of three host-known upper bounds. For each, we
  // ask "is a 32-bit type enough?" via three independent sources; if *any* source says yes we
  // pin `uint32_t`, otherwise we fall back to `unsigned long long`. The agents instantiate
  // their atomics on these types, so picking `uint32_t` saves both register pressure and the
  // device-side counter struct's footprint when the workload genuinely fits.
  //
  // OffsetT (per-segment offsets / counters):
  //   (1) static upper bound on `SegmentSizeParameterT` -- the natural bound on a single
  //       segment's offset (`params::static_max_value_v<SegmentSizeParameterT>`).
  //   (2) the user-declared underlying type of `SegmentSizeParameterT::value_type` -- if the
  //       user picked `int32_t`/`uint32_t` for segment sizes, segment-local offsets cannot
  //       exceed `UINT32_MAX` regardless of the static bound.
  //   (3) static upper bound on `TotalNumItemsGuaranteeT` -- a conservative bound: each
  //       per-segment offset is bounded by the cross-segment total.
  static constexpr bool offset_fits_u32 =
       fits_in_uint32_v<params::static_max_value_v<SegmentSizeParameterT>>
    || (sizeof(typename SegmentSizeParameterT::value_type) <= 4)
    || fits_in_uint32_v<TotalNumItemsGuaranteeT::static_max_num_items>;
  using OffsetT = ::cuda::std::conditional_t<offset_fits_u32, ::cuda::std::uint32_t, unsigned long long>;

  // OutOffsetT (per-segment `k` counters / "num selected" / "num ties to back").  Same three
  // sources, with `KParameterT` substituted for `SegmentSizeParameterT`.  K is bounded by the
  // per-segment size and (transitively) by the total item count, so sources (1) and (3) carry
  // over verbatim; source (2) reflects the user's declared type for `k`.
  static constexpr bool out_offset_fits_u32 =
       fits_in_uint32_v<params::static_max_value_v<KParameterT>>
    || (sizeof(typename KParameterT::value_type) <= 4)
    || fits_in_uint32_v<TotalNumItemsGuaranteeT::static_max_num_items>;
  using OutOffsetT = ::cuda::std::conditional_t<out_offset_fits_u32, ::cuda::std::uint32_t, unsigned long long>;
  // Per-segment top-k counter type. The dispatch allocates an `N_slabs`-sized array of these.
  using seg_counter_t = detail::topk::counter<key_in_t, OffsetT, OutOffsetT>;

  // Value-channel materialization mode for the multi-CTA-per-segment path. Mirrors the
  // single-problem dispatch (`dispatch_topk.cuh`):
  //
  //   indexed      -- the value channel is rewired so the per-segment candidate buffer stores
  //                   `OffsetT` indices and the value-output iterator is wrapped per-segment in
  //                   a `cuda::transform_output_iterator{user_out[i], topk_index_gather_op{user_in[i]}}`.
  //                   Smem / temp-storage footprint shrinks when values are wider than offsets.
  //   materialized -- the per-segment candidate buffer stores full `value_in_t` records and the
  //                   kernels read/write the user's iterators directly.
  //
  // Only consulted on the multi-CTA path; the worker_per_segment kernel has no candidate
  // buffer and uses the user iterators unchanged. Forced to `materialized` (effectively no-op)
  // on the keys-only path so the value-channel types keep pointing at the original `NullType*`
  // iterators.
  static constexpr bool indexed =
    !only_small_segments && !keys_only
    && multi_worker_per_segment_policy.value_materialization == detail::topk::value_materialization_mode::indexed;
  // Per-record type of the per-segment candidate value buffer. `OffsetT` in indexed mode, the
  // user's `value_in_t` otherwise.
  using effective_value_in_t = ::cuda::std::conditional_t<indexed, OffsetT, value_in_t>;

  // ---------------------------------------------------------------------
  // Allocation layout (see plan §5.1).
  //
  // Mixed (any_small && !only_small):
  //   [0] = per-segment tile-offset table (sized to N_seg + 1; sentinel slot holds the inclusive
  //         total of large-segment tile counts, i.e. `total_large_tiles`)
  //   [1] = mixed-path `batched_topk_counters` (large-segment queue length + retirement counter;
  //         consumed only by the worker-per-segment kernel and its epilogue)
  //   [2] = enqueued large-segment ids (sized to N_seg, addressed by queue_idx)
  // All-large (!any_small):
  //   [0] = per-segment tile-offset table (sized to N_seg + 1; sentinel slot holds the inclusive
  //         total of large-segment tile counts)
  //   [1] = transform_scan temp storage
  //
  // Multi-CTA path slabs (appended in both !only_small cases, in the same order):
  //   [+0] = per-segment counter array        (N_slabs * sizeof(seg_counter_t))
  //   [+1] = per-segment histogram slab       (N_slabs * num_buckets * sizeof(OffsetT))
  //   [+2] = per-segment candidate-key buf A  (N_slabs * candidate_buffer_length * sizeof(key_in_t))
  //   [+3] = per-segment candidate-key buf B  (N_slabs * candidate_buffer_length * sizeof(key_in_t))
  //   [+4] = per-segment candidate-val buf A  (only when !keys_only;
  //          element type is `effective_value_in_t`: `OffsetT` in indexed mode, `value_in_t` otherwise)
  //   [+5] = per-segment candidate-val buf B  (only when !keys_only;
  //          element type is `effective_value_in_t`: `OffsetT` in indexed mode, `value_in_t` otherwise)
  //
  // N_slabs is the host upper bound on the number of large segments, which equals
  // `num_segments_upper_bound` (any segment could in principle land in the large queue, and the
  // upper bound is the only host-resident sizing quantity that also accommodates a
  // device-accessible-only `num_segments`). The slabs are indexed by `queue_idx`, not by
  // original `segment_id`; the agents resolve `queue_idx -> segment_id` via the on-device
  // binary search + segment-id provider.
  //
  // `total_large_tiles` and `large_segments_count` are *not* stored in a global counters struct
  // any more. `total_large_tiles` lives in the trailing slot of the offset table; the multi-CTA
  // kernels read it as `d_large_segments_tile_offsets[num_large_segments]`. `large_segments_count`
  // is fed to those kernels through an iterator parameter: a raw pointer to
  // `batched_topk_counters::large_segments_count` on the mixed path; a `transform_iterator` over
  // a `counting_iterator` returning `num_segments.get_param(0)` (resolved on-device by
  // `constant_value_op`) on the all-large path.
  // ---------------------------------------------------------------------
  static constexpr int bits_per_pass            = multi_worker_per_segment_policy.bits_per_pass;
  [[maybe_unused]] static constexpr int num_buckets   = 1 << bits_per_pass;
  static constexpr int per_seg_allocs           = keys_only ? 4 : 6;
  // Mixed path: [0] offsets, [1] counters, [2] queue ids -> 3 pre-slots.
  // All-large path: [0] offsets, [1] scan temp -> 2 pre-slots.
  // Only-small path: no multi-CTA work -> 0 pre-slots.
  static constexpr int pre_multi_cta_allocs =
    only_small_segments ? 0 : (any_small_segments ? 3 : 2);
  static constexpr int allocations_array_size   = only_small_segments ? 1 : (pre_multi_cta_allocs + per_seg_allocs);

  // Indices into `allocations` for the multi-CTA slabs.
  [[maybe_unused]] static constexpr int idx_seg_counters_arr   = pre_multi_cta_allocs + 0;
  [[maybe_unused]] static constexpr int idx_seg_histograms_arr = pre_multi_cta_allocs + 1;
  [[maybe_unused]] static constexpr int idx_seg_key_buf_a      = pre_multi_cta_allocs + 2;
  [[maybe_unused]] static constexpr int idx_seg_key_buf_b      = pre_multi_cta_allocs + 3;
  [[maybe_unused]] static constexpr int idx_seg_val_buf_a      = pre_multi_cta_allocs + 4;
  [[maybe_unused]] static constexpr int idx_seg_val_buf_b      = pre_multi_cta_allocs + 5;

  size_t allocation_sizes[allocations_array_size] = {1};

  // Host-readable upper bound on `num_segments`, used to size every downstream host-resident
  // quantity (allocation extents, `cudaMemsetAsync` extents, worker-per-segment `grid_dim`,
  // scan extent on the all-large path, `total_large_tiles_upper_bound`).
  //
  // For host-known params (`uniform_param`, `static_constant_param`) we keep using
  // `get_param(0)` -- the exact actual value -- so the non-device-accessible-only path is not
  // pessimized. For device-accessible-only `per_segment_param` we fall back to the runtime
  // upper bound `num_segments.max_value` (a host field on the params object); the framework
  // contract is that this is at most the static `Max` template arg, so a single host load
  // covers both bounds.
  //
  // Callers of `num_segments_indirect` are responsible for supplying a tight `Max` (template)
  // or `max_value` (ctor arg); a default `Max == numeric_limits<int64_t>::max()` would yield
  // impractically large allocations.
  [[maybe_unused]] const num_segments_val_t num_segments_upper_bound = [&]() {
    if constexpr (params::is_per_segment_param_v<NumSegmentsParameterT>)
    {
      return static_cast<num_segments_val_t>(num_segments.max_value);
    }
    else
    {
      return static_cast<num_segments_val_t>(num_segments.get_param(0));
    }
  }();

  // Upper bound on the per-segment candidate buffer length (per the plan §5.4; v1 uses a flat
  // per-slab cap). For static-segment-size cases, `static_max_value_v` is the tight value; for
  // runtime-sized cases, we fall back on `total_num_items_guarantee.max_num_items`.
  //
  // TODO (elstehle): the per-segment candidate buffer length is currently sized by the static
  // upper bound on segment size and applied uniformly to every queue slot. This is acceptable
  // when the upper bound is tight (~1M items) but wasteful when it is loose or when most
  // segments are far smaller than the bound. Investigate (a) sizing each segment's buffer by
  // its actual `segment_size / coefficient`, (b) skipping the per-segment buffer allocation
  // entirely for segments that never enqueue onto the large-segment queue, and (c) gating
  // entry into the buffered chain on whether buffering actually saves work for that segment
  // (see the `will_buffer` heuristic in `agent_batched_topk_filter_partition::run()`).
  static constexpr ::cuda::std::int64_t coefficient_for_candidate_buffer = 128;
  [[maybe_unused]] const ::cuda::std::int64_t max_segment_size_upper_bound = (::cuda::std::min) (
    static_cast<::cuda::std::int64_t>(params::static_max_value_v<SegmentSizeParameterT>),
    total_num_items_guarantee.max_num_items);
  [[maybe_unused]] const OffsetT candidate_buffer_length = static_cast<OffsetT>(
    (::cuda::std::max) (::cuda::std::int64_t{1},
                        max_segment_size_upper_bound / coefficient_for_candidate_buffer));

  if constexpr (!only_small_segments)
  {
    // Scan output: per-segment tile offsets (exclusive scan, sized to
    // `num_segments_upper_bound + 1`). The trailing sentinel slot `[num_segments_actual]` holds
    // the inclusive total of large-segment tile counts after the scan; the multi-CTA kernels
    // read it as `total_large_tiles` (in lieu of a separate device-side counter). When
    // `num_segments` is host-known, `num_segments_upper_bound == num_segments_actual`, so the
    // sentinel sits at the array tail. When `num_segments` is device-only and
    // `num_segments_upper_bound > num_segments_actual`, the scan inputs past
    // `num_segments_actual` evaluate to 0 via `segment_size_to_tile_count_op`'s sentinel
    // short-circuit, so the inclusive total ends up at slot `num_segments_actual` and is
    // propagated unchanged through the trailing padding entries. The slot is *not*
    // pre-initialised here: in the mixed path the worker-per-segment epilogue's `BlockLoad`
    // substitutes 0 for the OOB index on its final pass.
    allocation_sizes[0] = (num_segments_upper_bound + 1) * sizeof(large_segment_tile_offset_t);

    if constexpr (any_small_segments)
    {
      // Mixed: counters struct + large-segment ids.
      allocation_sizes[1] = sizeof(counters_t);
      allocation_sizes[2] = num_segments_upper_bound * sizeof(num_segments_val_t);
    }
    else
    {
      // All-large: scan temp storage at [1] only. The exclusive scan runs over
      // `num_segments_upper_bound + 1` inputs (the trailing one being the sentinel that the op
      // short-circuits to 0); the output occupies the full `num_segments_upper_bound + 1`-entry
      // tile-offset table allocated at [0], with slot `num_segments_actual` holding the
      // inclusive total.
      if (const auto error = CubDebug(detail::scan::dispatch(
            nullptr,
            allocation_sizes[1],
            segment_size_scan_input_it,
            static_cast<large_segment_tile_offset_t*>(nullptr),
            ::cuda::std::plus<>{},
            detail::InputValue<large_segment_tile_offset_t>(large_segment_tile_offset_t{0}),
            static_cast<segment_size_scan_offset_t>(num_segments_upper_bound + num_segments_val_t{1}),
            stream)))
      {
        return error;
      }
    }

    // Multi-CTA per-segment slabs. N_slabs = num_segments_upper_bound.
    allocation_sizes[idx_seg_counters_arr]   = num_segments_upper_bound * sizeof(seg_counter_t);
    allocation_sizes[idx_seg_histograms_arr] = num_segments_upper_bound * static_cast<size_t>(num_buckets) * sizeof(OffsetT);
    allocation_sizes[idx_seg_key_buf_a]      = num_segments_upper_bound * candidate_buffer_length * sizeof(key_in_t);
    allocation_sizes[idx_seg_key_buf_b]      = num_segments_upper_bound * candidate_buffer_length * sizeof(key_in_t);
    if constexpr (!keys_only)
    {
      allocation_sizes[idx_seg_val_buf_a] = num_segments_upper_bound * candidate_buffer_length * sizeof(effective_value_in_t);
      allocation_sizes[idx_seg_val_buf_b] = num_segments_upper_bound * candidate_buffer_length * sizeof(effective_value_in_t);
    }
  }

  // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
  void* allocations[allocations_array_size] = {};
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  if constexpr (any_small_segments)
  {
    if constexpr (!only_small_segments)
    {
      // Zero-initialize the counters struct that holds the large-segment queue length and the block retirement
      // counter; both are read by the agent's atomic operations and must start at 0.
      if (const auto error = CubDebug(cudaMemsetAsync(allocations[1], 0, sizeof(counters_t), stream)))
      {
        return error;
      }
    }
    // Launch one block per `num_segments_upper_bound` slot; blocks past the actual count
    // early-exit on the `segment_id >= num_segments.get_param(0)` check inside `Process()`.
    // For host-known params this is exactly the actual count (no over-launch); for
    // device-only params this is the user-declared upper bound.
    const int grid_dim      = static_cast<int>(num_segments_upper_bound);
    constexpr int block_dim = worker_per_segment_policy.threads_per_block;
    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(grid_dim, block_dim, 0, stream)
            .doit(
              device_segmented_topk_kernel<
                PolicySelector,
                KeyInputItItT,
                KeyOutputItItT,
                ValueInputItItT,
                ValueOutputItItT,
                SegmentSizeParameterT,
                KParameterT,
                SelectDirectionParameterT,
                NumSegmentsParameterT,
                large_segment_tile_offset_t>,
              d_key_segments_it,
              d_key_segments_out_it,
              d_value_segments_it,
              d_value_segments_out_it,
              segment_sizes,
              k,
              select_directions,
              num_segments,
              only_small_segments ? nullptr : static_cast<counters_t*>(allocations[1]),
              only_small_segments ? nullptr : static_cast<num_segments_val_t*>(allocations[2]),
              only_small_segments ? nullptr : static_cast<large_segment_tile_offset_t*>(allocations[0]))))
    {
      return error;
    }
  }
  else
  {
    // No small segments: the small-kernel epilogue (which would otherwise produce the per-segment
    // tile offsets) does not run. Compute the per-segment tile offsets directly via a
    // transform-scan over all segment sizes. The scan runs over `num_segments_upper_bound + 1`
    // inputs; the sentinel input at index `num_segments_upper_bound` is `0` (via
    // `segment_size_to_tile_count_op`'s sentinel short-circuit), and so are all trailing inputs
    // past `num_segments.get_param(0)` when `num_segments` is device-only. The inclusive total
    // therefore ends up at slot `num_segments_actual`, which is what the multi-CTA kernels read
    // as `total_large_tiles`. No separate publication step is needed on this path.
    if (const auto error = CubDebug(detail::scan::dispatch(
          allocations[1],
          allocation_sizes[1],
          segment_size_scan_input_it,
          static_cast<large_segment_tile_offset_t*>(allocations[0]),
          ::cuda::std::plus<>{},
          detail::InputValue<large_segment_tile_offset_t>(large_segment_tile_offset_t{0}),
          static_cast<segment_size_scan_offset_t>(num_segments_upper_bound + num_segments_val_t{1}),
          stream)))
    {
      return error;
    }
  }

  if constexpr (!only_small_segments)
  {
    // ---------------------------------------------------------------------
    // Multi-CTA-per-segment path.
    //
    // After the small-segment kernel epilogue (mixed) or the host-driven `transform_scan`
    // (all-large) completes, `d_large_segments_tile_offsets[0..num_segments_upper_bound + 1)`
    // holds an exclusive prefix sum of per-large-segment tile counts; in particular,
    // `d_large_segments_tile_offsets[num_large_segments]` is the inclusive total
    // (i.e. `total_large_tiles`). `num_large_segments` itself is fed to the multi-CTA kernels
    // through an iterator (a raw pointer into the mixed-path `batched_topk_counters` on the
    // mixed path; a constant `transform_iterator` dereferencing `num_segments.get_param(0)`
    // on-device on the all-large path). We now run the same three-stage radix-style top-k that
    // the single-problem dispatch runs, but with per-segment arrays everywhere.
    // ---------------------------------------------------------------------
    large_segment_tile_offset_t* const d_large_segments_tile_offsets =
      static_cast<large_segment_tile_offset_t*>(allocations[0]);

    seg_counter_t* const d_seg_counters = static_cast<seg_counter_t*>(allocations[idx_seg_counters_arr]);
    OffsetT* const d_seg_histograms     = static_cast<OffsetT*>(allocations[idx_seg_histograms_arr]);

    // Zero the per-segment counter array. The dispatch is the source of truth for each
    // segment's `load_from_candidates_buffer == false` at pass 0 (single-problem analog at
    // `dispatch_topk.cuh` -- the memset over the counter blob).
    if (const auto error = CubDebug(cudaMemsetAsync(
          d_seg_counters, 0, num_segments_upper_bound * sizeof(seg_counter_t), stream)))
    {
      return error;
    }
    // Zero the per-segment global histograms. The histogram agent's `init_histogram` clears the
    // smem histogram (not global), so the global slabs must already be zeroed before the first
    // `atomicAdd` from the per-block merge.
    if (const auto error = CubDebug(cudaMemsetAsync(
          d_seg_histograms,
          0,
          num_segments_upper_bound * static_cast<size_t>(num_buckets) * sizeof(OffsetT),
          stream)))
    {
      return error;
    }

    key_in_t* const d_seg_key_buf_a = static_cast<key_in_t*>(allocations[idx_seg_key_buf_a]);
    key_in_t* const d_seg_key_buf_b = static_cast<key_in_t*>(allocations[idx_seg_key_buf_b]);
    // The candidate-value buffer's element type tracks the value-channel materialization mode
    // (see `effective_value_in_t` above): `OffsetT` in indexed mode, `value_in_t` in materialized.
    [[maybe_unused]] effective_value_in_t* d_seg_val_buf_a = nullptr;
    [[maybe_unused]] effective_value_in_t* d_seg_val_buf_b = nullptr;
    if constexpr (!keys_only)
    {
      d_seg_val_buf_a = static_cast<effective_value_in_t*>(allocations[idx_seg_val_buf_a]);
      d_seg_val_buf_b = static_cast<effective_value_in_t*>(allocations[idx_seg_val_buf_b]);
    }

    // Effective outer value iterators consumed by the multi-CTA filter / last-filter kernels.
    //
    // In `materialized` mode (and on the keys-only path) these are just aliases for the user's
    // `d_value_segments_it` / `d_value_segments_out_it`. In `indexed` mode they are rewired so
    // that:
    //
    //   * `effective_d_value_segments_it[segment_id]` returns
    //     `cuda::counting_iterator<OffsetT>{0}` -- a stateless source of per-segment indices,
    //     letting the filter agents stamp the candidate-value buffer with `OffsetT` indices
    //     instead of full `value_in_t` records.
    //
    //   * `effective_d_value_segments_out_it[segment_id]` returns a
    //     `cuda::transform_output_iterator{user_out[i], topk_index_gather_op{user_in[i]}}` so
    //     that the last-filter kernel's "write index" turns into "user_out[i][pos] =
    //     user_in[i][idx]" via the gather op.
    //
    // The outer iterators are constructed by-value here and travel via the kernel argument
    // area; they capture the user's iterator-of-iterators by value, matching the materialized
    // path's ABI.
    auto effective_d_value_segments_it = [&]() {
      if constexpr (indexed)
      {
        return ::cuda::constant_iterator{::cuda::counting_iterator<OffsetT>{OffsetT{0}}};
      }
      else
      {
        return d_value_segments_it;
      }
    }();
    auto effective_d_value_segments_out_it = [&]() {
      if constexpr (indexed)
      {
        using indexed_out_op_t = per_segment_indexed_out_op<ValueInputItItT, ValueOutputItItT>;
        return ::cuda::transform_iterator{
          ::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}},
          indexed_out_op_t{d_value_segments_it, d_value_segments_out_it}};
      }
      else
      {
        return d_value_segments_out_it;
      }
    }();
    using effective_value_input_it_it_t  = decltype(effective_d_value_segments_it);
    using effective_value_output_it_it_t = decltype(effective_d_value_segments_out_it);

    // Compute radix-style multi-pass scheduling. `total_bits` and `num_passes` are derived from
    // `key_in_t` and `bits_per_pass`, both compile-time / dispatch-time scalars uniform across
    // all segments (single-source-invariant for pass scheduling across segments).
    const detail::identity_decomposer_t decomposer{};
    const int total_bits = detail::radix::traits_t<key_in_t>::default_end_bit(decomposer);
    const int num_passes = detail::topk::calc_num_passes<bits_per_pass>(total_bits);

    // Host-side upper bound on the total number of large tiles. Used as a *cap* on the
    // multi-CTA kernel grid sizes (the actual launch dim is `min(MaxSmOccupancy * num_sms,
    // total_large_tiles_upper_bound)`). `total_large_tiles` (the actual count) is read by each
    // block from `d_large_segments_tile_offsets[num_large_segments]` (the trailing sentinel
    // slot) and drives the grid-stride loop bound inside every multi-CTA kernel; physical CTAs
    // whose stride-loop body never executes early-exit. The upper bound is
    // `ceil(total_items / tile) + N_seg` (each segment contributes at most one extra partial
    // tile beyond the dense item count).
    const auto total_large_tiles_upper_bound = static_cast<unsigned int>(
      ::cuda::ceil_div(total_num_items_guarantee.max_num_items,
                       static_cast<::cuda::std::int64_t>(multi_worker_per_segment_tile_size))
      + static_cast<::cuda::std::int64_t>(num_segments_upper_bound));
    constexpr int multi_worker_threads_per_block = multi_worker_per_segment_policy.threads_per_block;

    // Multi-processor count for the active device; used together with each kernel's
    // `MaxSmOccupancy` to derive a max-occupancy grid that is then capped at
    // `total_large_tiles_upper_bound`. Mirrors the single-problem `dispatch_topk` approach.
    int active_device_id = 0;
    if (const auto error = CubDebug(cudaGetDevice(&active_device_id)))
    {
      return error;
    }
    int num_sms = 0;
    if (const auto error =
          CubDebug(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, active_device_id)))
    {
      return error;
    }

    // Segment-id provider: on the mixed path, `queue_idx` indexes into `d_large_segments_ids` to
    // get the original `segment_id`. On the all-large path, every segment is large so
    // `queue_idx == segment_id` and we pass a `counting_iterator` as the identity.
    auto segment_id_provider = [&]() {
      if constexpr (any_small_segments)
      {
        return static_cast<num_segments_val_t*>(allocations[2]);
      }
      else
      {
        return ::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}};
      }
    }();
    using segment_id_provider_t = decltype(segment_id_provider);

    // Iterator producing `num_large_segments` when dereferenced.
    //
    //   Mixed     -- raw pointer into `batched_topk_counters::large_segments_count` (the
    //                worker-per-segment kernel's atomicAdd enqueue is the source of truth here).
    //   All-large -- a `transform_iterator` over a `counting_iterator` that ignores its index
    //                and returns `num_segments.get_param(0)`. The `num_segments` params object is
    //                captured by value into the functor (`constant_value_op`); the actual scalar
    //                read happens on-device at kernel entry, so the dispatch does not bake a
    //                host-side read of the count into this path. No on-device write either --
    //                the count is carried into the kernel by value through the captured params
    //                object's state.
    //
    // The multi-CTA kernels dereference the iterator once at entry; they do not need to know
    // which path produced it.
    auto large_segments_count_it = [&]() {
      if constexpr (any_small_segments)
      {
        return &(static_cast<counters_t*>(allocations[1])->large_segments_count);
      }
      else
      {
        return ::cuda::transform_iterator(
          ::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}},
          constant_value_op<NumSegmentsParameterT>{num_segments});
      }
    }();
    using large_segments_count_it_t = decltype(large_segments_count_it);

    // Compile-time direction lowering. `SelectDirectionParameterT` is uniform across segments; we
    // hand-roll the dispatch on the two enumerators instead of going through
    // `params::dispatch_discrete`, which is `_CCCL_HOST_DEVICE` and therefore requires its functor
    // to be HD-callable. The body below issues host-only kernel launches, so the dispatch must run
    // from host context with a host-only functor. Per-segment direction is deferred (plan §3.6) --
    // the kernels are direction-NTTP'd.
    auto launch_passes = [&](auto direction_tag) -> cudaError_t {
      static constexpr detail::topk::select select_dir = decltype(direction_tag)::value;

      // Kernel pointers (template instantiations) shared between the `MaxSmOccupancy` query and
      // the actual `triple_chevron::doit()` launch below. Resolving them once per direction
      // avoids re-instantiating the (long) template parameter list per pass.
      auto histogram_kernel_ptr = device_segmented_topk_histogram_kernel<
        PolicySelector,
        select_dir,
        KeyInputItItT,
        SegmentSizeParameterT,
        KParameterT,
        NumSegmentsParameterT,
        segment_id_provider_t,
        large_segment_tile_offset_t,
        large_segments_count_it_t,
        detail::identity_decomposer_t,
        OffsetT,
        OutOffsetT>;
      // The filter and last-filter kernels read/write the per-segment candidate buffer through
      // the value-channel iterators, so we instantiate them on the *effective* iterator types
      // (see `effective_d_value_segments_*` above). In materialized mode these are aliases for
      // `ValueInputItItT` / `ValueOutputItItT`; in indexed mode they rewire the value channel
      // to flow `OffsetT` indices through the candidate buffer and a `topk_index_gather_op` at
      // the user-output boundary.
      auto filter_kernel_ptr = device_segmented_topk_filter_kernel<
        PolicySelector,
        select_dir,
        KeyInputItItT,
        KeyOutputItItT,
        effective_value_input_it_it_t,
        effective_value_output_it_it_t,
        SegmentSizeParameterT,
        KParameterT,
        NumSegmentsParameterT,
        segment_id_provider_t,
        large_segment_tile_offset_t,
        large_segments_count_it_t,
        detail::identity_decomposer_t,
        OffsetT,
        OutOffsetT>;
      auto last_filter_kernel_ptr = device_segmented_topk_last_filter_kernel<
        PolicySelector,
        select_dir,
        KeyInputItItT,
        KeyOutputItItT,
        effective_value_input_it_it_t,
        effective_value_output_it_it_t,
        SegmentSizeParameterT,
        KParameterT,
        NumSegmentsParameterT,
        segment_id_provider_t,
        large_segment_tile_offset_t,
        large_segments_count_it_t,
        detail::identity_decomposer_t,
        OffsetT,
        OutOffsetT>;

      // Max-occupancy grid sizes per kernel, capped at `total_large_tiles_upper_bound`. Each
      // multi-CTA kernel iterates the tile space via a grid-stride loop; physical CTAs whose
      // stride-loop body never executes early-exit. Mirrors the single-problem
      // `dispatch_topk.cuh` approach.
      int histogram_blocks_per_sm   = 0;
      int filter_blocks_per_sm      = 0;
      int last_filter_blocks_per_sm = 0;
      if (const auto error = CubDebug(
            MaxSmOccupancy(histogram_blocks_per_sm, histogram_kernel_ptr, multi_worker_threads_per_block)))
      {
        return error;
      }
      if (const auto error = CubDebug(
            MaxSmOccupancy(filter_blocks_per_sm, filter_kernel_ptr, multi_worker_threads_per_block)))
      {
        return error;
      }
      if (const auto error = CubDebug(
            MaxSmOccupancy(last_filter_blocks_per_sm, last_filter_kernel_ptr, multi_worker_threads_per_block)))
      {
        return error;
      }
      const auto histogram_grid_size = (::cuda::std::min) (
        static_cast<unsigned int>(histogram_blocks_per_sm * num_sms), total_large_tiles_upper_bound);
      const auto filter_grid_size = (::cuda::std::min) (
        static_cast<unsigned int>(filter_blocks_per_sm * num_sms), total_large_tiles_upper_bound);
      const auto last_filter_grid_size = (::cuda::std::min) (
        static_cast<unsigned int>(last_filter_blocks_per_sm * num_sms), total_large_tiles_upper_bound);

      // Pass 0: dedicated histogram-only kernel over the per-segment original inputs.
      {
        const int reset_histogram = num_passes != 1;
        if (const auto error = CubDebug(
              THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
                histogram_grid_size, multi_worker_threads_per_block, 0, stream)
                .doit(
                  histogram_kernel_ptr,
                  d_key_segments_it,
                  segment_sizes,
                  k,
                  num_segments,
                  segment_id_provider,
                  static_cast<const large_segment_tile_offset_t*>(d_large_segments_tile_offsets),
                  d_seg_counters,
                  d_seg_histograms,
                  large_segments_count_it,
                  0,
                  total_bits,
                  reset_histogram,
                  decomposer)))
        {
          return error;
        }
      }

      // Passes 1..num_passes-1: filter+histogram (or early_stop) kernel; double-buffer flips per pass.
      // The candidate value buffer's element type tracks the value-channel materialization mode
      // (see `effective_value_in_t` above), so the value-buf DoubleBuffer is templated on
      // `effective_value_in_t` rather than the user's `value_in_t`.
      DoubleBuffer<key_in_t> key_bufs(d_seg_key_buf_b, d_seg_key_buf_a);
      DoubleBuffer<effective_value_in_t> val_bufs;
      if constexpr (!keys_only)
      {
        val_bufs = DoubleBuffer<effective_value_in_t>(d_seg_val_buf_b, d_seg_val_buf_a);
      }

      int pass = 1;
      for (; pass < num_passes; ++pass)
      {
        const bool reset_histogram = pass != num_passes - 1;
        if (const auto error = CubDebug(
              THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
                filter_grid_size, multi_worker_threads_per_block, 0, stream)
                .doit(
                  filter_kernel_ptr,
                  d_key_segments_it,
                  d_key_segments_out_it,
                  effective_d_value_segments_it,
                  effective_d_value_segments_out_it,
                  segment_sizes,
                  k,
                  num_segments,
                  segment_id_provider,
                  static_cast<const large_segment_tile_offset_t*>(d_large_segments_tile_offsets),
                  d_seg_counters,
                  d_seg_histograms,
                  key_bufs.Current(),
                  val_bufs.Current(),
                  key_bufs.Alternate(),
                  val_bufs.Alternate(),
                  candidate_buffer_length,
                  static_cast<OffsetT>(coefficient_for_candidate_buffer),
                  large_segments_count_it,
                  pass,
                  total_bits,
                  reset_histogram,
                  decomposer)))
        {
          return error;
        }
        key_bufs.selector ^= 1;
        if constexpr (!keys_only)
        {
          val_bufs.selector ^= 1;
        }
      }

      // Last filter pass.
      if (const auto error = CubDebug(
            THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
              last_filter_grid_size, multi_worker_threads_per_block, 0, stream)
              .doit(
                last_filter_kernel_ptr,
                d_key_segments_it,
                d_key_segments_out_it,
                effective_d_value_segments_it,
                effective_d_value_segments_out_it,
                segment_sizes,
                k,
                num_segments,
                segment_id_provider,
                static_cast<const large_segment_tile_offset_t*>(d_large_segments_tile_offsets),
                d_seg_counters,
                key_bufs.Current(),
                val_bufs.Current(),
                candidate_buffer_length,
                large_segments_count_it,
                num_passes,
                total_bits,
                decomposer)))
      {
        return error;
      }
      return cudaSuccess;
    };

    const auto direction_value = select_directions.get_param(num_segments_val_t{0});
    cudaError_t direction_error;
    if (direction_value == detail::topk::select::min)
    {
      direction_error =
        launch_passes(::cuda::std::integral_constant<detail::topk::select, detail::topk::select::min>{});
    }
    else
    {
      _CCCL_ASSERT(direction_value == detail::topk::select::max, "select_directions value not in the supported list");
      direction_error =
        launch_passes(::cuda::std::integral_constant<detail::topk::select, detail::topk::select::max>{});
    }
    if (direction_error != cudaSuccess)
    {
      return direction_error;
    }
  }
  return CubDebug(detail::DebugSyncStream(stream));
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
