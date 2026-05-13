// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private `BlockFilterAccumulating` -- single-stream sister of
//! `BlockFilter`. Buffers items matching a unary `IdentifySelected(key) -> bool`
//! predicate in shared memory across multiple `Partition()` calls and flushes
//! only when the buffer fills (or in the terminal `epilogue()`).
//!
//! Shares the same "safe-both" interface as `BlockFilter`: sinks + identify op
//! captured at ctor; per-call `Partition(scratch, keys, [num_items,] value_sources)`;
//! argless `epilogue()`. Implements the same multi-round overflow algorithm as
//! `BlockPartitionAccumulatingCandidates`, but with only one stream.
//!
//! Algorithm (mirrored on `BlockPartitionAccumulatingCandidates::accumulating_partition_base`):
//!   1. Fused classify + reserve-into-smem-buffer loop. `identify_selected_op`
//!      runs once per item. Rejected and out-of-bounds items get
//!      `positions[j] = -1`. Otherwise the item's smem slot index is
//!      `atomicAdd(&counter, 1)`.
//!   2. Multi-round overflow loop. See the comment in
//!      `block_partition_accumulating.cuh` for the same algorithm; here only the
//!      single selected stream is buffered.
//!   3. `epilogue()`: terminal cooperative flush of any remaining buffered
//!      items. Counter is < BufferCapacity at this point by construction.
//!
//! `LazyValueLoad` follows the same convention as `BlockFilter` (and the
//! accumulating partition variants): when `false`, each call up-front loads
//! each value channel into a per-thread `value_t reg_values[ItemsPerThread]`
//! array; when `true`, the up-front load is skipped and `data_source.gather_one(j)`
//! runs only at scatter sites for non-rejected items.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/topk/block_filter.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
namespace bf_acc_detail
{
// Per-stream counter + flush-broadcast slots in TempStorage. Same layout as
// `bp_acc_detail::stream_counters_t` but lives in the filter namespace so the
// header dependency stays one-way (block_filter_accumulating.cuh ->
// block_filter.cuh) without pulling in the accumulating partition header.
template <typename OffsetT>
struct stream_counters_t
{
  int counter;
  OffsetT base;
  OffsetT granted;
};

// Per-channel value array slot in TempStorage.
template <typename ValueT, int Capacity>
struct value_buf_slot_t
{
  ValueT values[Capacity];
};

// Persistent TempStorage layout for an accumulating filter. One smem buffer
// (key + per-channel values) plus the stream's counter + broadcast slots.
// Wrapped in `cub::Uninitialized<>` at the public layer.
template <typename KeyT, typename OffsetT, typename ValueTypesTuple, int Capacity>
struct accumulating_filter_temp_storage_t
{
  stream_counters_t<OffsetT> cnt;
  KeyT keys[Capacity];
  CUB_NS_QUALIFIER::detail::phase_aggregate<bp_detail::map_tuple_t<value_buf_slot_t, ValueTypesTuple, Capacity>>
    per_channel_values;
};
} // namespace bf_acc_detail

//---------------------------------------------------------------------
// `BlockFilterAccumulating`
//
// Single-stream sister of `BlockFilter` (single output, single buffer,
// single counter). Used by the agent's `early_stop`-mode pass when the
// policy selects `BlockFilterStrategy::AccumulatingFilter`.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          int BufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename SelectedReserveOp,
          typename SelectedKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksTuple = ::cuda::std::tuple<>,
          typename ValueTypesTuple        = ::cuda::std::tuple<>,
          bool LazyValueLoad              = false>
class BlockFilterAccumulating
{
public:
  static constexpr int tile_items         = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  static_assert(BufferCapacity >= 1, "Accumulating filter requires BufferCapacity >= 1.");
  static_assert(num_value_channels <= 1,
                "Accumulating filter supports keys-only or single-value-channel today; multi-channel needs a "
                "heterogeneous register-array tuple analogous to the BlockFilter shared_mem path.");
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");

  // Internal `_TempStorage` is the actual buffer + counters; the publicly-exposed
  // `TempStorage` wraps it in `cub::Uninitialized<>` so the user can declare
  // `__shared__ filter_t::TempStorage` directly.
  using _TempStorage = bf_acc_detail::accumulating_filter_temp_storage_t<KeyT, SelectedOffsetT, ValueTypesTuple, BufferCapacity>;
  struct TempStorage : CUB_NS_QUALIFIER::Uninitialized<_TempStorage>
  {};

  // Empty per-call scratch -- everything the class needs lives in `TempStorage`.
  struct ScratchStorage
  {};

  // COLLECTIVE ctor: all threads in the block must construct together.
  // Unwraps the `Uninitialized<>` wrapper via `.Alias()`, zero-inits the smem
  // counter (thread 0), then `__syncthreads()`.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockFilterAccumulating(
    TempStorage& storage,
    SelectedReserveOp& reserve_selected,
    SelectedKeyOutTransformOp& selected_key_transform,
    SelectedKeyOutIt selected_keys_out,
    ValueChannelSinksTuple& value_channel_sinks,
    IdentifySelectedOp& identify_selected_op)
      : ts_(storage.Alias())
      , reserve_sel_(reserve_selected)
      , sel_xform_(selected_key_transform)
      , sel_iter_(selected_keys_out)
      , sinks_(value_channel_sinks)
      , identify_op_(identify_selected_op)
  {
    if (threadIdx.x == 0)
    {
      ts_.cnt.counter = 0;
    }
    __syncthreads();
  }

  // Full-tile overload.
  template <typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Partition(ScratchStorage& /*scratch*/, const KeyT (&keys)[ItemsPerThread], ValueSourcesTuple& value_sources)
  {
    filter_impl<true>(keys, /*num_items=*/tile_items, value_sources);
  }

  // Partial-tile overload.
  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Partition(
    ScratchStorage& /*scratch*/,
    const KeyT (&keys)[ItemsPerThread],
    NumItemsT num_items,
    ValueSourcesTuple& value_sources)
  {
    filter_impl<false>(keys, static_cast<int>(num_items), value_sources);
  }

  // Terminal flush: drain any remaining buffered items.
  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue()
  {
    __syncthreads();
    const int leftover = ts_.cnt.counter;
    if (leftover > 0)
    {
      cooperative_flush_round(leftover);
      __syncthreads();
      if (threadIdx.x == 0)
      {
        ts_.cnt.counter = 0;
      }
      __syncthreads();
    }
  }

private:
  using channel_value_t = typename bp_detail::value_t_or_default<ValueTypesTuple>::type;

  // Eagerly load the (single) channel's per-thread values from the per-call source.
  // No-op when keys-only or when LazyValueLoad is true.
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void eager_load_value_channel(
    ValueSourcesTuple& value_sources, channel_value_t (&reg_values)[ItemsPerThread], int num_items)
  {
    (void) value_sources;
    (void) reg_values;
    (void) num_items;
    if constexpr (!LazyValueLoad && num_value_channels == 1)
    {
      auto& src      = ::cuda::std::get<0>(value_sources);
      using source_t = ::cuda::std::remove_reference_t<decltype(src)>;
      static_assert(::cuda::std::is_same_v<typename source_t::value_t, channel_value_t>,
                    "Per-call value source's value_t must match the class-level ValueTypesTuple element.");
      typename source_t::ScratchStorage scratch{};
      if constexpr (IsFull)
      {
        auto h = src.submit_load(scratch);
        h.complete_load(reg_values);
      }
      else
      {
        auto h = src.submit_load(scratch, num_items);
        h.complete_load(reg_values);
      }
    }
  }

  // Shared body for both Partition() overloads. Classify-into-positions[]
  // then overflow loop. Single stream -- every kept item goes into the smem
  // buffer.
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  filter_impl(const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the filter pairs them positionally.");

    int num_thread_items;
    if constexpr (IsFull)
    {
      num_thread_items = ItemsPerThread;
      (void) num_items;
    }
    else
    {
      const int tb_offset = static_cast<int>(threadIdx.x) * ItemsPerThread;
      num_thread_items =
        (tb_offset >= num_items) ? 0 : static_cast<int>((::cuda::std::min) (ItemsPerThread, num_items - tb_offset));
    }

    channel_value_t reg_values[ItemsPerThread]{};
    eager_load_value_channel<IsFull>(value_sources, reg_values, num_items);

    auto get_value = [&](int j) -> channel_value_t {
      if constexpr (LazyValueLoad && num_value_channels == 1)
      {
        auto& src = ::cuda::std::get<0>(value_sources);
        return src.gather_one(j);
      }
      else
      {
        return reg_values[j];
      }
    };

    // Step 1: fused classify + reserve. positions[j] encodes the per-item state.
    int positions[ItemsPerThread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const bool is_valid = IsFull ? true : (j < num_thread_items);
      const bool keep     = is_valid ? static_cast<bool>(identify_op_(keys[j])) : false;
      positions[j]        = keep ? atomicAdd(&ts_.cnt.counter, 1) : -1;
    }
    __syncthreads();

    // Step 2: multi-round overflow loop.
    overflow_loop(positions, keys, get_value);
  }

  template <typename GetValueFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  overflow_loop(int (&positions)[ItemsPerThread], const KeyT (&keys)[ItemsPerThread], GetValueFn get_value)
  {
    while (true)
    {
      const int cnt = ts_.cnt.counter;
      if (cnt < BufferCapacity)
      {
        scatter_pending_to_smem(positions, keys, get_value, /*upper_bound=*/cnt);
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < ItemsPerThread; ++j)
        {
          if (positions[j] >= 0)
          {
            positions[j] = -1;
          }
        }
        __syncthreads();
        break;
      }
      else if (cnt == BufferCapacity)
      {
        scatter_pending_to_smem(positions, keys, get_value, /*upper_bound=*/BufferCapacity);
        __syncthreads();
        cooperative_flush_round(BufferCapacity);
        __syncthreads();
        if (threadIdx.x == 0)
        {
          ts_.cnt.counter = 0;
        }
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < ItemsPerThread; ++j)
        {
          if (positions[j] >= 0)
          {
            positions[j] = -1;
          }
        }
        __syncthreads();
        break;
      }
      else
      {
        scatter_pending_to_smem(positions, keys, get_value, /*upper_bound=*/BufferCapacity);
        __syncthreads();
        cooperative_flush_round(BufferCapacity);
        __syncthreads();
        if (threadIdx.x == 0)
        {
          ts_.cnt.counter = cnt - BufferCapacity;
        }
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < ItemsPerThread; ++j)
        {
          if (positions[j] >= 0)
          {
            positions[j] -= BufferCapacity;
          }
        }
        __syncthreads();
      }
    }
  }

  template <typename GetValueFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void scatter_pending_to_smem(
    int (&positions)[ItemsPerThread], const KeyT (&keys)[ItemsPerThread], GetValueFn get_value, int upper_bound)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (positions[j] >= 0 && positions[j] < upper_bound)
      {
        ts_.keys[positions[j]] = keys[j];
        if constexpr (num_value_channels == 1)
        {
          CUB_NS_QUALIFIER::detail::at<0>(ts_.per_channel_values).values[positions[j]] = get_value(j);
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_round(int count)
  {
    if (threadIdx.x == 0)
    {
      const auto r    = reserve_sel_(static_cast<SelectedOffsetT>(count));
      ts_.cnt.base    = r.first;
      ts_.cnt.granted = static_cast<SelectedOffsetT>(r.second);
    }
    __syncthreads();

    const SelectedOffsetT base = ts_.cnt.base;
    const SelectedOffsetT to_write =
      SelectedReserveOp::may_grant_less ? ts_.cnt.granted : static_cast<SelectedOffsetT>(count);

    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
    {
      sel_iter_[base + static_cast<SelectedOffsetT>(i)] = sel_xform_(ts_.keys[i]);
    }
    if constexpr (num_value_channels == 1)
    {
      auto& sink = ::cuda::std::get<0>(sinks_);
      auto& vs   = CUB_NS_QUALIFIER::detail::at<0>(ts_.per_channel_values);
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
      {
        sink.selected_values_out[base + static_cast<SelectedOffsetT>(i)] = sink.selected_value_transform(vs.values[i]);
      }
    }
  }

  // ---------------------------------------------------------------
  // Member state.
  // ---------------------------------------------------------------
  _TempStorage& ts_;
  SelectedReserveOp& reserve_sel_;
  SelectedKeyOutTransformOp& sel_xform_;
  SelectedKeyOutIt sel_iter_;
  ValueChannelSinksTuple& sinks_;
  IdentifySelectedOp& identify_op_;
};

//---------------------------------------------------------------------
// `strategy_to_filter_class<Strategy, ...>` -- compile-time selector mapping a
// `BlockFilterStrategy` value to the corresponding filter class type.
//
// The four `Atomics*` / `Staged` / `SharedMem` values map to `BlockFilter`; the
// `AccumulatingFilter` value maps to `BlockFilterAccumulating`. The agent uses
// this metafunction to derive `filter_t` from a policy enum.
//---------------------------------------------------------------------
template <BlockFilterStrategy Strategy,
          int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename SelectedReserveOp,
          typename SelectedKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksTuple,
          typename ValueTypesTuple,
          typename DataSourceScratchTypesTuple,
          bool LazyValueLoad>
struct strategy_to_filter_class
{
  using type = BlockFilter<
    BlockThreads,
    ItemsPerThread,
    Strategy,
    KeyT,
    SelectedOffsetT,
    SelectedReserveOp,
    SelectedKeyOutTransformOp,
    SelectedKeyOutIt,
    IdentifySelectedOp,
    ValueChannelSinksTuple,
    ValueTypesTuple,
    DataSourceScratchTypesTuple,
    LazyValueLoad>;
};

template <int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename SelectedReserveOp,
          typename SelectedKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksTuple,
          typename ValueTypesTuple,
          typename DataSourceScratchTypesTuple,
          bool LazyValueLoad>
struct strategy_to_filter_class<
  BlockFilterStrategy::AccumulatingFilter,
  BlockThreads,
  ItemsPerThread,
  AccumulatingBufferCapacity,
  KeyT,
  SelectedOffsetT,
  SelectedReserveOp,
  SelectedKeyOutTransformOp,
  SelectedKeyOutIt,
  IdentifySelectedOp,
  ValueChannelSinksTuple,
  ValueTypesTuple,
  DataSourceScratchTypesTuple,
  LazyValueLoad>
{
  // The accumulating filter loads value channels via stack-local
  // `source_t::ScratchStorage` and so doesn't consume the
  // `DataSourceScratchTypesTuple` parameter; it's accepted for parity with the
  // non-accumulating branch (the agent always supplies it).
  using type = BlockFilterAccumulating<
    BlockThreads,
    ItemsPerThread,
    AccumulatingBufferCapacity,
    KeyT,
    SelectedOffsetT,
    SelectedReserveOp,
    SelectedKeyOutTransformOp,
    SelectedKeyOutIt,
    IdentifySelectedOp,
    ValueChannelSinksTuple,
    ValueTypesTuple,
    LazyValueLoad>;
};

template <BlockFilterStrategy Strategy,
          int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename SelectedReserveOp,
          typename SelectedKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksTuple,
          typename ValueTypesTuple,
          typename DataSourceScratchTypesTuple,
          bool LazyValueLoad>
using strategy_to_filter_class_t = typename strategy_to_filter_class<
  Strategy,
  BlockThreads,
  ItemsPerThread,
  AccumulatingBufferCapacity,
  KeyT,
  SelectedOffsetT,
  SelectedReserveOp,
  SelectedKeyOutTransformOp,
  SelectedKeyOutIt,
  IdentifySelectedOp,
  ValueChannelSinksTuple,
  ValueTypesTuple,
  DataSourceScratchTypesTuple,
  LazyValueLoad>::type;
} // namespace detail::topk

CUB_NAMESPACE_END
