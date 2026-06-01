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

// Compile-time predicate: does this (non-negative) integer value fit in `uint32_t`? The cast through
// `unsigned long long` avoids narrow-type truncation when comparing arbitrary integral types;
// negatives (not expected for size/count bounds) wrap large and report `false`.
template <auto Value>
inline constexpr bool fits_in_uint32_v =
  static_cast<unsigned long long>(Value)
  <= static_cast<unsigned long long>(::cuda::std::numeric_limits<::cuda::std::uint32_t>::max());

// Helper: turn a segment ID into the number of large-segment-agent tiles needed to cover that
// segment. Wrapped in a transform_iterator and exclusive-scanned to obtain per-segment tile offsets.
template <class SegmentSizeParameterT, class TotalNumItemsValueType, class NumSegmentsParameterT>
struct segment_size_to_tile_count_op
{
  SegmentSizeParameterT segment_sizes;
  int large_segment_agent_tile_size;

  // Stored as a params object so the all-large-segments scan works even when `num_segments` is a
  // device-accessible-only param. That scan runs over `num_segments + 1` inputs; the op
  // short-circuits to 0 at the sentinel index so it never reads past the end of `segment_sizes`.
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

// Helper: constant-value transform op over a `params`-like object. Wrapped in a
// `cuda::transform_iterator`, it always dereferences to `params.get_param(0)`. The all-large path
// uses it to feed `num_segments` to the multi-CTA kernels through the same iterator interface the
// mixed path uses for `&d_counters->large_segments_count`. Storing the params object (not its value)
// defers the read to on-device kernel-entry time instead of baking in a host-side read.
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

// Helper: per-segment indexed-mode output-iterator builder. In `value_materialization_mode::indexed`
// the candidate buffer stores `OffsetT` indices, so each segment's value-output iterator is wrapped
// in a `cuda::transform_output_iterator` with `topk_index_gather_op{user_in[i]}` to turn "write
// index" into "values_out[pos] = values_in[idx]". `operator[](segment_id)` yields that per-segment
// iterator; captured iterators must be trivially copyable since they travel by value into the kernel
// argument area.
template <typename ValueInputItItT, typename ValueOutputItItT>
struct per_segment_indexed_out_op
{
  ValueInputItItT d_value_segments_it;
  ValueOutputItItT d_value_segments_out_it;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto operator()(SegmentIndexT segment_id) const
  {
    using values_in_it_t = it_value_t<ValueInputItItT>;
    using gather_op_t    = detail::batched_topk::topk_index_gather_op<values_in_it_t>;
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
//! @param num_segments Number of segments. May be host-resident (e.g. `num_segments_static<...>`,
//!        `num_segments_uniform<...>{actual}`) or device-accessible-only
//!        (`num_segments_indirect<It, Min, Max>{iter, min_v, max_v}`). For the device-only form the
//!        dispatch sizes all host-resident quantities from `num_segments.max_value`, so callers must
//!        supply a tight `Max`/`max_value`; the default `numeric_limits<int64_t>::max()` would yield
//!        impractically large allocations.
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
  // Index type for the per-segment tile-offset table and global tile ids in the multi-CTA path.
  // Pinned at `uint32_t` to keep the table dense and the agents' binary search cheap. Supports up to
  // `multi_worker_per_segment_tile_size * numeric_limits<uint32_t>::max()` aggregate items; larger
  // workloads silently overflow `total_large_tiles` and the offset table (not validated at runtime).
  using large_segment_tile_offset_t = ::cuda::std::uint32_t;
  // Resolver for (a) whether any one-worker-per-segment policy covers the segment-size range and
  // (b) which to use: prefers the smallest covering+fits-smem policy, else falls back to the largest
  // fits-smem policy. In the fallback case `only_small_segments == false` and segments exceeding the
  // chosen tile size route onto the multi-CTA-per-segment path.
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

  // No fitting one-worker-per-segment policy is unsupported (would require a multi-CTA-only path).
  static_assert(resolved_worker_per_segment_t::found, "No valid policy found for one-worker-per-segment approach");

  constexpr auto policy                                         = resolved_worker_per_segment_t::policy;
  constexpr worker_policy worker_per_segment_policy             = policy.worker_per_segment_policy;
  constexpr multi_worker_policy multi_worker_per_segment_policy = policy.multi_worker_per_segment_policy;

  static constexpr int worker_per_segment_tile_size =
    worker_per_segment_policy.threads_per_block * worker_per_segment_policy.items_per_thread;
  static constexpr bool any_small_segments =
    params::static_min_value_v<SegmentSizeParameterT> <= worker_per_segment_tile_size;
  static constexpr bool only_small_segments =
    params::static_max_value_v<SegmentSizeParameterT> <= worker_per_segment_tile_size;

  // Derived value-channel types: the segmented dispatch is iterator-of-iterators, so peel the inner
  // iterator type, then re-derive the inner value type. `keys_only` is `ValueInputItItT == NullType**`.
  using key_t                     = it_value_t<it_value_t<KeyInputItItT>>;
  using value_t                   = it_value_t<it_value_t<ValueInputItItT>>;
  static constexpr bool keys_only = ::cuda::std::is_same_v<value_t, cub::NullType>;

  using num_segments_val_t         = typename NumSegmentsParameterT::value_type;
  using counters_t                 = batched_topk_counters<narrow_segment_count_t<NumSegmentsParameterT>>;
  using segment_size_scan_offset_t = detail::choose_offset_t<num_segments_val_t>;
  using segment_size_scan_input_op_t =
    segment_size_to_tile_count_op<SegmentSizeParameterT, large_segment_tile_offset_t, NumSegmentsParameterT>;
  static constexpr auto multi_worker_per_segment_tile_size =
    multi_worker_per_segment_policy.threads_per_block * multi_worker_per_segment_policy.items_per_thread;
  // `num_segments` is captured by value so the op can resolve its sentinel cutoff on-device via
  // `num_segments.get_param(0)`. Host-callable but only ever invoked on-device (all-large scan).
  const segment_size_scan_input_op_t segment_size_scan_input_op{
    segment_sizes, multi_worker_per_segment_tile_size, num_segments};
  // Transform iterator over [0, num_segments + 1) producing each segment's tile-count (and 0 at the
  // sentinel index `num_segments`). The extra input makes the offset table `num_segments + 1` entries
  // wide, with the last entry equal to `total_large_tiles`.
  [[maybe_unused]] const auto segment_size_scan_input_it = ::cuda::transform_iterator(
    ::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}}, segment_size_scan_input_op);

  // OffsetT: pin `uint32_t` if any of three host-known sources proves 32 bits suffice, else
  // `unsigned long long`. Sources:
  //   (1) static max of `SegmentSizeParameterT` (`params::static_max_value_v<SegmentSizeParameterT>`);
  //   (2) its declared `value_type` being <= 4 bytes;
  //   (3) static max of `TotalNumItemsGuaranteeT` (each per-segment offset is bounded by the total).
  static constexpr bool offset_fits_u32 =
    fits_in_uint32_v<params::static_max_value_v<SegmentSizeParameterT>>
    || (sizeof(typename SegmentSizeParameterT::value_type) <= 4)
    || fits_in_uint32_v<TotalNumItemsGuaranteeT::static_max_num_items>;
  using OffsetT = ::cuda::std::conditional_t<offset_fits_u32, ::cuda::std::uint32_t, unsigned long long>;

  // OutOffsetT (per-segment `k` counters): same three sources with `KParameterT` substituted for
  // `SegmentSizeParameterT`. K is bounded by the segment size and (transitively) the total.
  static constexpr bool out_offset_fits_u32 =
    fits_in_uint32_v<params::static_max_value_v<KParameterT>> || (sizeof(typename KParameterT::value_type) <= 4)
    || fits_in_uint32_v<TotalNumItemsGuaranteeT::static_max_num_items>;
  using OutOffsetT = ::cuda::std::conditional_t<out_offset_fits_u32, ::cuda::std::uint32_t, unsigned long long>;

  // SegmentCountT: type for the count of enqueued large segments. Only two sources apply (the
  // total-items bound does not bound segment counts): (1) static max of `NumSegmentsParameterT`,
  // (2) its declared `value_type` being <= 4 bytes. Either pins `uint32_t`. Used to type-narrow the
  // dereferenced count at the agents' binary-search bound and the sentinel-slot index.
  static constexpr bool segment_count_fits_u32 =
    fits_in_uint32_v<params::static_max_value_v<NumSegmentsParameterT>>
    || (sizeof(typename NumSegmentsParameterT::value_type) <= 4);
  using SegmentCountT = ::cuda::std::conditional_t<segment_count_fits_u32, ::cuda::std::uint32_t, unsigned long long>;
  // Per-segment top-k counter type. The dispatch allocates an `N_slabs`-sized array of these.
  using seg_counter_t = detail::batched_topk::counter<key_t, OffsetT, OutOffsetT>;

  // Value-channel materialization mode for the multi-CTA path (mirrors `dispatch_topk.cuh`):
  //   indexed      -- candidate buffer stores `OffsetT` indices; the value-output iterator is wrapped
  //                   per-segment in `transform_output_iterator{user_out[i],
  //                   topk_index_gather_op{user_in[i]}}`. Shrinks footprint when values are wider
  //                   than offsets.
  //   materialized -- candidate buffer stores full `value_t`; kernels use the user iterators directly.
  // Forced to `materialized` on the keys-only path so value-channel types keep pointing at the
  // original `NullType*` iterators.
  static constexpr bool indexed =
    !only_small_segments && !keys_only
    && multi_worker_per_segment_policy.value_materialization == value_materialization_mode::indexed;
  // Per-record type of the per-segment candidate value buffer. `OffsetT` in indexed mode, the
  // user's `value_t` otherwise.
  using effective_value_t = ::cuda::std::conditional_t<indexed, OffsetT, value_t>;

  // ---------------------------------------------------------------------
  // Allocation layout.
  //
  // Mixed (any_small && !only_small):
  //   [0] = per-segment tile-offset table (sized to N_seg + 1; sentinel slot holds the inclusive
  //         total of large-segment tile counts, i.e. `total_large_tiles`)
  //   [1] = mixed-path `batched_topk_counters` (large-segment queue length + retirement counter;
  //         consumed only by the worker-per-segment kernel and its epilogue)
  //   [2] = enqueued large-segment ids (sized to N_seg, addressed by queue_idx)
  // All-large (!any_small):
  //   [0] = per-segment tile-offset table (as above)
  //   [1] = transform_scan temp storage
  //
  // Multi-CTA path slabs (appended in both !only_small cases, in the same order):
  //   [+0] = per-segment counter array        (N_slabs * sizeof(seg_counter_t))
  //   [+1] = per-segment histogram slab       (N_slabs * num_buckets * sizeof(OffsetT))
  //   [+2] = per-segment candidate-key buf A  (N_slabs * candidate_buffer_length * sizeof(key_t))
  //   [+3] = per-segment candidate-key buf B  (N_slabs * candidate_buffer_length * sizeof(key_t))
  //   [+4] = per-segment candidate-val buf A  (only when !keys_only; element `effective_value_t`)
  //   [+5] = per-segment candidate-val buf B  (only when !keys_only; element `effective_value_t`)
  //
  // N_slabs == `num_segments_upper_bound` (any segment could land in the large queue). Slabs are
  // indexed by `queue_idx`, not `segment_id`; the agents resolve `queue_idx -> segment_id` via an
  // on-device binary search + segment-id provider.
  //
  // `total_large_tiles` lives in the trailing slot of the offset table (read as
  // `d_large_segments_tile_offsets[num_large_segments]`). `large_segments_count` is fed to the
  // multi-CTA kernels via an iterator: a raw pointer into `batched_topk_counters` (mixed) or a
  // constant `transform_iterator` returning `num_segments.get_param(0)` (all-large).
  // ---------------------------------------------------------------------
  static constexpr int bits_per_pass                = multi_worker_per_segment_policy.bits_per_pass;
  [[maybe_unused]] static constexpr int num_buckets = 1 << bits_per_pass;
  static constexpr int per_seg_allocs               = keys_only ? 4 : 6;
  // Mixed path: [0] offsets, [1] counters, [2] queue ids -> 3 pre-slots.
  // All-large path: [0] offsets, [1] scan temp -> 2 pre-slots.
  // Only-small path: no multi-CTA work -> 0 pre-slots.
  static constexpr int pre_multi_cta_allocs   = only_small_segments ? 0 : (any_small_segments ? 3 : 2);
  static constexpr int allocations_array_size = only_small_segments ? 1 : (pre_multi_cta_allocs + per_seg_allocs);

  // Indices into `allocations` for the multi-CTA slabs.
  [[maybe_unused]] static constexpr int idx_seg_counters_arr   = pre_multi_cta_allocs + 0;
  [[maybe_unused]] static constexpr int idx_seg_histograms_arr = pre_multi_cta_allocs + 1;
  [[maybe_unused]] static constexpr int idx_seg_key_buf_a      = pre_multi_cta_allocs + 2;
  [[maybe_unused]] static constexpr int idx_seg_key_buf_b      = pre_multi_cta_allocs + 3;
  [[maybe_unused]] static constexpr int idx_seg_val_buf_a      = pre_multi_cta_allocs + 4;
  [[maybe_unused]] static constexpr int idx_seg_val_buf_b      = pre_multi_cta_allocs + 5;

  size_t allocation_sizes[allocations_array_size] = {1};

  // Host-readable upper bound on `num_segments`, used to size every host-resident quantity
  // (allocations, memset extents, worker-per-segment `grid_dim`, all-large scan extent). For
  // host-known params we use the exact `get_param(0)`; for device-accessible-only params we fall back
  // to the runtime `num_segments.max_value` (<= the static `Max`, per the framework contract).
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

  // Upper bound on the per-segment candidate buffer length (a flat per-slab cap). Static segment
  // sizes use `static_max_value_v`; runtime sizes fall back on `total_num_items_guarantee.max_num_items`.
  //
  // TODO (elstehle): this flat cap is wasteful when the bound is loose or most segments are far
  // smaller than it; consider sizing per-segment buffers individually or skipping never-enqueued ones.
  static constexpr ::cuda::std::int64_t coefficient_for_candidate_buffer = 128;
  [[maybe_unused]] const ::cuda::std::int64_t max_segment_size_upper_bound =
    (::cuda::std::min) (static_cast<::cuda::std::int64_t>(params::static_max_value_v<SegmentSizeParameterT>),
                        total_num_items_guarantee.max_num_items);
  [[maybe_unused]] const OffsetT candidate_buffer_length = static_cast<OffsetT>(
    (::cuda::std::max) (::cuda::std::int64_t{1}, max_segment_size_upper_bound / coefficient_for_candidate_buffer));

  if constexpr (!only_small_segments)
  {
    // Scan output: per-segment tile offsets (exclusive scan, sized `num_segments_upper_bound + 1`).
    // The sentinel slot `[num_segments_actual]` holds the inclusive total (`total_large_tiles`), which
    // the multi-CTA kernels read in lieu of a separate device-side counter. When `num_segments` is
    // device-only, inputs past `num_segments_actual` evaluate to 0 (sentinel short-circuit), so the
    // total still lands at `[num_segments_actual]`. Not pre-initialised here.
    allocation_sizes[0] = (num_segments_upper_bound + 1) * sizeof(large_segment_tile_offset_t);

    if constexpr (any_small_segments)
    {
      // Mixed: counters struct + large-segment ids.
      allocation_sizes[1] = sizeof(counters_t);
      allocation_sizes[2] = num_segments_upper_bound * sizeof(narrow_segment_count_t<NumSegmentsParameterT>);
    }
    else
    {
      // All-large: scan temp storage at [1] only. The exclusive scan over `num_segments_upper_bound
      // + 1` inputs writes the offset table at [0], with slot `num_segments_actual` holding the
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
    allocation_sizes[idx_seg_counters_arr] = num_segments_upper_bound * sizeof(seg_counter_t);
    allocation_sizes[idx_seg_histograms_arr] =
      num_segments_upper_bound * static_cast<size_t>(num_buckets) * sizeof(OffsetT);
    allocation_sizes[idx_seg_key_buf_a] = num_segments_upper_bound * candidate_buffer_length * sizeof(key_t);
    allocation_sizes[idx_seg_key_buf_b] = num_segments_upper_bound * candidate_buffer_length * sizeof(key_t);
    if constexpr (!keys_only)
    {
      allocation_sizes[idx_seg_val_buf_a] =
        num_segments_upper_bound * candidate_buffer_length * sizeof(effective_value_t);
      allocation_sizes[idx_seg_val_buf_b] =
        num_segments_upper_bound * candidate_buffer_length * sizeof(effective_value_t);
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
    // Launch one block per `num_segments_upper_bound` slot; blocks past the actual count early-exit
    // on the `segment_id >= num_segments.get_param(0)` check inside `Process()`.
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
              only_small_segments ? nullptr
                                  : static_cast<narrow_segment_count_t<NumSegmentsParameterT>*>(allocations[2]),
              only_small_segments ? nullptr : static_cast<large_segment_tile_offset_t*>(allocations[0]))))
    {
      return error;
    }
  }
  else
  {
    // No small segments: the worker-per-segment epilogue that would produce the tile offsets does
    // not run, so compute them directly via a transform-scan over all segment sizes. The inclusive
    // total lands at slot `num_segments_actual` (sentinel short-circuit zeroes trailing inputs),
    // which the multi-CTA kernels read as `total_large_tiles`.
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
    // `d_large_segments_tile_offsets` now holds an exclusive prefix sum of per-large-segment tile
    // counts, with `[num_large_segments]` the inclusive total (`total_large_tiles`).
    // `num_large_segments` is fed to the kernels through an iterator (see `large_segments_count_it`
    // below). Runs the same three-stage radix-style top-k as the single-problem dispatch, but with
    // per-segment arrays everywhere.
    // ---------------------------------------------------------------------
    large_segment_tile_offset_t* const d_large_segments_tile_offsets =
      static_cast<large_segment_tile_offset_t*>(allocations[0]);

    seg_counter_t* const d_seg_counters = static_cast<seg_counter_t*>(allocations[idx_seg_counters_arr]);
    OffsetT* const d_seg_histograms     = static_cast<OffsetT*>(allocations[idx_seg_histograms_arr]);

    // Zero the per-segment counter array (establishes each segment's `load_from_candidates_buffer
    // == false` at pass 0; same as the single-problem dispatch's counter-blob memset).
    if (const auto error =
          CubDebug(cudaMemsetAsync(d_seg_counters, 0, num_segments_upper_bound * sizeof(seg_counter_t), stream)))
    {
      return error;
    }
    // Zero the per-segment global histograms: `init_histogram` only clears the smem histogram, so
    // the global slabs must be zeroed before the first `atomicAdd` from the per-block merge.
    if (const auto error = CubDebug(cudaMemsetAsync(
          d_seg_histograms, 0, num_segments_upper_bound * static_cast<size_t>(num_buckets) * sizeof(OffsetT), stream)))
    {
      return error;
    }

    key_t* const d_seg_key_buf_a = static_cast<key_t*>(allocations[idx_seg_key_buf_a]);
    key_t* const d_seg_key_buf_b = static_cast<key_t*>(allocations[idx_seg_key_buf_b]);
    // The candidate-value buffer's element type tracks the value-channel materialization mode
    // (see `effective_value_t` above): `OffsetT` in indexed mode, `value_t` in materialized.
    [[maybe_unused]] effective_value_t* d_seg_val_buf_a = nullptr;
    [[maybe_unused]] effective_value_t* d_seg_val_buf_b = nullptr;
    if constexpr (!keys_only)
    {
      d_seg_val_buf_a = static_cast<effective_value_t*>(allocations[idx_seg_val_buf_a]);
      d_seg_val_buf_b = static_cast<effective_value_t*>(allocations[idx_seg_val_buf_b]);
    }

    // Effective outer value iterators for the multi-CTA filter / last-filter kernels. In
    // `materialized` mode (and keys-only) they alias the user's iterators. In `indexed` mode:
    //   * input  -> `counting_iterator<OffsetT>{0}`, so agents stamp the candidate buffer with
    //               `OffsetT` indices instead of full `value_t` records;
    //   * output -> `transform_output_iterator{user_out[i], topk_index_gather_op{user_in[i]}}`, so
    //               "write index" becomes "user_out[i][pos] = user_in[i][idx]".
    // Constructed by-value and captured into the kernel argument area, matching the materialized ABI.
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
        return ::cuda::transform_iterator{::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}},
                                          indexed_out_op_t{d_value_segments_it, d_value_segments_out_it}};
      }
      else
      {
        return d_value_segments_out_it;
      }
    }();
    using effective_value_input_it_it_t  = decltype(effective_d_value_segments_it);
    using effective_value_output_it_it_t = decltype(effective_d_value_segments_out_it);

    // Radix-style multi-pass scheduling. `total_bits` / `num_passes` derive from `key_t` and
    // `bits_per_pass`, uniform across all segments.
    const detail::identity_decomposer_t decomposer{};
    const int total_bits = detail::radix::traits_t<key_t>::default_end_bit(decomposer);
    const int num_passes = detail::batched_topk::calc_num_passes<bits_per_pass>(total_bits);

    // Host-side upper bound on the total number of large tiles, used to cap the multi-CTA grid sizes
    // (`min(MaxSmOccupancy * num_sms, total_large_tiles_upper_bound)`). The actual `total_large_tiles`
    // is read on-device from the offset table's sentinel slot and drives each kernel's grid-stride
    // loop. Bound is `ceil(total_items / tile) + N_seg` (one partial tile per segment).
    const auto total_large_tiles_upper_bound = static_cast<unsigned int>(
      ::cuda::ceil_div(total_num_items_guarantee.max_num_items,
                       static_cast<::cuda::std::int64_t>(multi_worker_per_segment_tile_size))
      + static_cast<::cuda::std::int64_t>(num_segments_upper_bound));
    constexpr int multi_worker_threads_per_block = multi_worker_per_segment_policy.threads_per_block;

    // SM count for the active device, combined with each kernel's `MaxSmOccupancy` to derive a
    // max-occupancy grid capped at `total_large_tiles_upper_bound`.
    int active_device_id = 0;
    if (const auto error = CubDebug(cudaGetDevice(&active_device_id)))
    {
      return error;
    }
    int num_sms = 0;
    if (const auto error = CubDebug(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, active_device_id)))
    {
      return error;
    }

    // Segment-id provider: on the mixed path, `queue_idx` indexes into `d_large_segments_ids` to
    // get the original `segment_id`. On the all-large path, every segment is large so
    // `queue_idx == segment_id` and we pass a `counting_iterator` as the identity.
    auto segment_id_provider = [&]() {
      if constexpr (any_small_segments)
      {
        return static_cast<narrow_segment_count_t<NumSegmentsParameterT>*>(allocations[2]);
      }
      else
      {
        return ::cuda::counting_iterator<narrow_segment_count_t<NumSegmentsParameterT>>{
          narrow_segment_count_t<NumSegmentsParameterT>{0}};
      }
    }();
    using segment_id_provider_t = decltype(segment_id_provider);

    // Iterator producing `num_large_segments` when dereferenced:
    //   Mixed     -- raw pointer into `batched_topk_counters::large_segments_count` (filled by the
    //                worker-per-segment kernel's atomicAdd enqueue).
    //   All-large -- a constant `transform_iterator` returning `num_segments.get_param(0)`, read
    //                on-device at kernel entry (params captured by value in `constant_value_op`).
    // The multi-CTA kernels dereference it once at entry, agnostic to which path produced it.
    auto large_segments_count_it = [&]() {
      if constexpr (any_small_segments)
      {
        return &(static_cast<counters_t*>(allocations[1])->large_segments_count);
      }
      else
      {
        return ::cuda::transform_iterator(::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}},
                                          constant_value_op<NumSegmentsParameterT>{num_segments});
      }
    }();
    using large_segments_count_it_t = decltype(large_segments_count_it);

    // Compile-time direction lowering. We hand-roll the dispatch on the two enumerators instead of
    // `params::dispatch_discrete` (which is `_CCCL_HOST_DEVICE` and needs an HD-callable functor)
    // because the body issues host-only kernel launches. Per-segment direction is deferred.
    auto launch_passes = [&](auto direction_tag) -> cudaError_t {
      static constexpr detail::topk::select select_dir = decltype(direction_tag)::value;

      // Kernel pointers shared between the `MaxSmOccupancy` query and the `doit()` launches below;
      // resolving them once per direction avoids re-instantiating the long template list per pass.
      // The histogram kernel takes `extract_bin_op` (built host-side from `pass`, `total_bits`,
      // `decomposer`) as a template+runtime parameter rather than those scalars directly.
      using extract_bin_op_t = detail::batched_topk::
        extract_bin_op_t<key_t, select_dir, multi_worker_per_segment_policy.bits_per_pass, detail::identity_decomposer_t>;
      auto histogram_kernel_ptr = device_segmented_topk_histogram_kernel<
        PolicySelector,
        KeyInputItItT,
        SegmentSizeParameterT,
        segment_id_provider_t,
        large_segment_tile_offset_t,
        large_segments_count_it_t,
        extract_bin_op_t,
        OffsetT,
        OutOffsetT,
        SegmentCountT>;
      // Per-segment epilogue for the histogram pass: one CTA per large segment, doing the
      // prefix-sum + bucket-finder + counter update + (optional) global histogram reset. When the
      // policy enables `full_tiles_only_histogram` it also loads + bins each segment's trailing
      // partial tile into `segment_histogram` before the prefix-sum.
      auto finalize_histogram_kernel_ptr = device_segmented_topk_finalize_histogram_kernel<
        PolicySelector,
        KeyInputItItT,
        SegmentSizeParameterT,
        KParameterT,
        NumSegmentsParameterT,
        segment_id_provider_t,
        large_segments_count_it_t,
        extract_bin_op_t,
        OffsetT,
        OutOffsetT,
        key_t>;
      // The filter / last-filter kernels touch the candidate buffer through the value-channel
      // iterators, so they are instantiated on the *effective* iterator types (aliases of the
      // user's in materialized mode; rewired to flow `OffsetT` indices in indexed mode).
      auto filter_kernel_ptr = device_segmented_topk_filter_kernel<
        PolicySelector,
        select_dir,
        KeyInputItItT,
        KeyOutputItItT,
        effective_value_input_it_it_t,
        effective_value_output_it_it_t,
        SegmentSizeParameterT,
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
      // Per-segment epilogue for the filter pass (passes 1..num_passes-1). One CTA per segment:
      // trailing-partial-tile processing (when the policy enables `full_tiles_only_filter`),
      // prefix-sum + bucket-finder, counter update, and optional histogram reset. The
      // trailing-partial work reuses `agent_batched_topk_filter_partition`, so the kernel takes the
      // full agent template / argument list.
      auto finalize_filter_kernel_ptr = device_segmented_topk_finalize_filter_kernel<
        PolicySelector,
        select_dir,
        KeyInputItItT,
        KeyOutputItItT,
        effective_value_input_it_it_t,
        effective_value_output_it_it_t,
        SegmentSizeParameterT,
        NumSegmentsParameterT,
        segment_id_provider_t,
        large_segment_tile_offset_t,
        large_segments_count_it_t,
        detail::identity_decomposer_t,
        OffsetT,
        OutOffsetT,
        key_t>;

      // Max-occupancy grid sizes per kernel, capped at `total_large_tiles_upper_bound`. Each kernel
      // iterates the tile space via a grid-stride loop; CTAs with no work early-exit.
      int histogram_blocks_per_sm          = 0;
      int finalize_histogram_blocks_per_sm = 0;
      int filter_blocks_per_sm             = 0;
      int finalize_filter_blocks_per_sm    = 0;
      int last_filter_blocks_per_sm        = 0;
      if (const auto error =
            CubDebug(MaxSmOccupancy(histogram_blocks_per_sm, histogram_kernel_ptr, multi_worker_threads_per_block)))
      {
        return error;
      }
      if (const auto error = CubDebug(MaxSmOccupancy(
            finalize_histogram_blocks_per_sm, finalize_histogram_kernel_ptr, multi_worker_threads_per_block)))
      {
        return error;
      }
      if (const auto error =
            CubDebug(MaxSmOccupancy(filter_blocks_per_sm, filter_kernel_ptr, multi_worker_threads_per_block)))
      {
        return error;
      }
      if (const auto error = CubDebug(
            MaxSmOccupancy(finalize_filter_blocks_per_sm, finalize_filter_kernel_ptr, multi_worker_threads_per_block)))
      {
        return error;
      }
      if (const auto error =
            CubDebug(MaxSmOccupancy(last_filter_blocks_per_sm, last_filter_kernel_ptr, multi_worker_threads_per_block)))
      {
        return error;
      }
      const auto histogram_grid_size = (::cuda::std::min) (static_cast<unsigned int>(histogram_blocks_per_sm * num_sms),
                                                           total_large_tiles_upper_bound);
      // Finalize-histogram / finalize-filter launch one CTA per large segment, so cap at
      // `num_segments_upper_bound` rather than `total_large_tiles_upper_bound`. Their grid-stride
      // loops bound against the device-side `num_large_segments`, so the tighter cap is just an
      // optimization.
      const auto finalize_histogram_grid_size =
        (::cuda::std::min) (static_cast<unsigned int>(finalize_histogram_blocks_per_sm * num_sms),
                            static_cast<unsigned int>(num_segments_upper_bound));
      const auto filter_grid_size =
        (::cuda::std::min) (static_cast<unsigned int>(filter_blocks_per_sm * num_sms), total_large_tiles_upper_bound);
      const auto finalize_filter_grid_size =
        (::cuda::std::min) (static_cast<unsigned int>(finalize_filter_blocks_per_sm * num_sms),
                            static_cast<unsigned int>(num_segments_upper_bound));
      const auto last_filter_grid_size =
        (::cuda::std::min) (static_cast<unsigned int>(last_filter_blocks_per_sm * num_sms),
                            total_large_tiles_upper_bound);

      // Pass 0: histogram-only kernel over the per-segment original inputs, followed by the
      // per-segment epilogue (prefix-sum + bucket-finder + counter update + optional histogram
      // reset).
      {
        const bool reset_histogram = num_passes != 1;
        const extract_bin_op_t extract_bin_op{0, total_bits, decomposer};
        if (const auto error = CubDebug(
              THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
                histogram_grid_size, multi_worker_threads_per_block, 0, stream)
                .doit(histogram_kernel_ptr,
                      d_key_segments_it,
                      segment_sizes,
                      segment_id_provider,
                      static_cast<const large_segment_tile_offset_t*>(d_large_segments_tile_offsets),
                      d_seg_histograms,
                      large_segments_count_it,
                      extract_bin_op)))
        {
          return error;
        }
        if (const auto error = CubDebug(
              THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
                finalize_histogram_grid_size, multi_worker_threads_per_block, 0, stream)
                .doit(finalize_histogram_kernel_ptr,
                      d_key_segments_it,
                      segment_sizes,
                      k,
                      segment_id_provider,
                      d_seg_counters,
                      d_seg_histograms,
                      large_segments_count_it,
                      extract_bin_op,
                      0,
                      reset_histogram)))
        {
          return error;
        }
      }

      // Passes 1..num_passes-1: filter+histogram (or early_stop) kernel; double-buffer flips per pass.
      // The value-buf `DoubleBuffer` is templated on `effective_value_t` (tracks the materialization
      // mode), not the user's `value_t`.
      DoubleBuffer<key_t> key_bufs(d_seg_key_buf_b, d_seg_key_buf_a);
      DoubleBuffer<effective_value_t> val_bufs;
      if constexpr (!keys_only)
      {
        val_bufs = DoubleBuffer<effective_value_t>(d_seg_val_buf_b, d_seg_val_buf_a);
      }

      int pass = 1;
      for (; pass < num_passes; ++pass)
      {
        const bool reset_histogram = pass != num_passes - 1;
        if (const auto error = CubDebug(
              THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
                filter_grid_size, multi_worker_threads_per_block, 0, stream)
                .doit(filter_kernel_ptr,
                      d_key_segments_it,
                      d_key_segments_out_it,
                      effective_d_value_segments_it,
                      effective_d_value_segments_out_it,
                      segment_sizes,
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
        // Per-segment epilogue for this filter pass: one CTA per large segment, runs the
        // trailing-partial-tile work (when the policy enables `full_tiles_only_filter`), then the
        // prefix-sum + bucket-finder + per-mode counter update + optional histogram reset. It
        // re-derives the per-segment mode (early_stop / buffered / unbuffered) from the same counter
        // fields the filter agent saw, so no extra device-side flag is needed.
        //
        // Takes the full filter-agent argument list so it can instantiate the agent and call
        // `agent.process_partial_for_segment(queue_idx, pass)`. When `full_tiles_only_filter ==
        // false` the agent body is `if constexpr`-eliminated; the args are passed but unused.
        if (const auto error = CubDebug(
              THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
                finalize_filter_grid_size, multi_worker_threads_per_block, 0, stream)
                .doit(finalize_filter_kernel_ptr,
                      d_key_segments_it,
                      d_key_segments_out_it,
                      effective_d_value_segments_it,
                      effective_d_value_segments_out_it,
                      segment_sizes,
                      segment_id_provider,
                      static_cast<const large_segment_tile_offset_t*>(d_large_segments_tile_offsets),
                      d_seg_counters,
                      d_seg_histograms,
                      key_bufs.Current(),
                      val_bufs.Current(),
                      key_bufs.Alternate(),
                      val_bufs.Alternate(),
                      large_segments_count_it,
                      candidate_buffer_length,
                      static_cast<OffsetT>(coefficient_for_candidate_buffer),
                      pass,
                      total_bits,
                      decomposer,
                      reset_histogram)))
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
              .doit(last_filter_kernel_ptr,
                    d_key_segments_it,
                    d_key_segments_out_it,
                    effective_d_value_segments_it,
                    effective_d_value_segments_out_it,
                    segment_sizes,
                    k,
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
