// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private `block_filter_accumulating` -- single-stream sister of
//! `BlockFilter`. Buffers items matching a unary `IdentifySelected(key) -> bool`
//! predicate in shared memory across multiple `partition()` calls and flushes
//! only when the buffer fills (or in the terminal `epilogue()`).
//!
//! Shares the same "safe-both" interface as `BlockFilter`: sinks + identify op
//! captured at ctor; per-call `partition(scratch, keys, [num_items,] value_sources)`;
//! argless `epilogue()`. Implements the same multi-round overflow algorithm as
//! `block_partition_accumulating_candidates`, but with only one stream.
//!
//! Algorithm (mirrored on `block_partition_accumulating_candidates::accumulating_partition_base`):
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
#include <cub/detail/topk/block_partition_accumulating.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
// Persistent TempStorage layout for an accumulating filter. One smem buffer
// (key + per-channel values) plus the stream's counter + broadcast slots.
// Wrapped in `cub::Uninitialized<>` at the public layer. Reuses
// `stream_counters_t` and `value_buf_slot_t` from the accumulating partition.
template <typename KeyT, typename OffsetT, typename ValueTypesTuple, int Capacity>
struct accumulating_filter_temp_storage_t
{
  stream_counters_t<OffsetT> cnt;
  KeyT keys[Capacity];
  CUB_NS_QUALIFIER::detail::phase_aggregate<map_tuple_t<value_buf_slot_t, ValueTypesTuple, Capacity>>
    per_channel_values;
};

//---------------------------------------------------------------------
// `block_filter_accumulating`
//
// Single-stream sister of `BlockFilter` (single output, single buffer,
// single counter). Used by the agent's `early_stop`-mode pass when the
// policy selects `block_filter_strategy::accumulating_filter`.
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
class block_filter_accumulating
{
public:
  static constexpr int tile_items         = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  // Compile-time upper bound on `overflow_loop` iterations per `partition()` call.
  // See the sibling class `block_partition_accumulating_candidates::max_flush_iters`
  // for the full derivation. When `BufferCapacity >= tile_items` this evaluates
  // to 2 and NVCC can straight-line the loop.
  static constexpr int max_flush_iters = (tile_items + BufferCapacity - 1) / BufferCapacity + 1;

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
  using _TempStorage = accumulating_filter_temp_storage_t<KeyT, SelectedOffsetT, ValueTypesTuple, BufferCapacity>;
  struct TempStorage : CUB_NS_QUALIFIER::Uninitialized<_TempStorage>
  {};

  // Empty per-call scratch -- everything the class needs lives in `TempStorage`.
  struct ScratchStorage
  {};

  // COLLECTIVE ctor: all threads in the block must construct together.
  // Unwraps the `Uninitialized<>` wrapper via `.Alias()`, zero-inits the smem
  // counter (thread 0), then `__syncthreads()`.
  _CCCL_DEVICE _CCCL_FORCEINLINE block_filter_accumulating(
    TempStorage& storage,
    SelectedReserveOp& reserve_selected,
    SelectedKeyOutTransformOp& selected_key_transform,
    SelectedKeyOutIt selected_keys_out,
    ValueChannelSinksTuple& value_channel_sinks,
    IdentifySelectedOp& identify_selected_op)
      : temp_storage(storage.Alias())
      , reserve_sel(reserve_selected)
      , sel_xform(selected_key_transform)
      , sel_iter(selected_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_selected_op)
  {
    if (threadIdx.x == 0)
    {
      temp_storage.cnt.counter = 0;
    }
    __syncthreads();
  }

  // Full-tile overload.
  template <typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& /*scratch*/, const KeyT (&keys)[ItemsPerThread], ValueSourcesTuple& value_sources)
  {
    filter_impl<true>(keys, /*num_items=*/tile_items, value_sources);
  }

  // Partial-tile overload.
  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
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
    const int leftover = temp_storage.cnt.counter;
    if (leftover > 0)
    {
      cooperative_flush_partial(leftover);
      __syncthreads();
      if (threadIdx.x == 0)
      {
        temp_storage.cnt.counter = 0;
      }
      __syncthreads();
    }
  }

private:
  using channel_value_t = typename value_t_or_default<ValueTypesTuple>::type;

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

  // Shared body for both partition() overloads. Classify-into-positions[]
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
      const bool keep     = is_valid ? static_cast<bool>(identify_op(keys[j])) : false;
      positions[j]        = keep ? atomicAdd(&temp_storage.cnt.counter, 1) : -1;
    }
    __syncthreads();

    // Step 2: multi-round overflow loop.
    overflow_loop(positions, keys, get_value);
  }

  // Multi-round overflow loop. See `block_partition_accumulating_candidates::overflow_loop`
  // for the design notes -- the counted-`for` shape, the merged `cnt >= Capacity`
  // branch, and the `Capacity >= tile_items` fast path apply identically here.
  template <typename GetValueFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  overflow_loop(int (&positions)[ItemsPerThread], const KeyT (&keys)[ItemsPerThread], GetValueFn get_value)
  {
    for (int iter = 0; iter < max_flush_iters; ++iter)
    {
      const int cnt = temp_storage.cnt.counter;
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
        return;
      }
      // cnt >= BufferCapacity: flush the leading Capacity-worth and renumber.
      scatter_pending_to_smem(positions, keys, get_value, /*upper_bound=*/BufferCapacity);
      __syncthreads();
      cooperative_flush_full_buffer();
      __syncthreads();
      if (threadIdx.x == 0)
      {
        temp_storage.cnt.counter = cnt - BufferCapacity;
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

  template <typename GetValueFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void scatter_pending_to_smem(
    int (&positions)[ItemsPerThread], const KeyT (&keys)[ItemsPerThread], GetValueFn get_value, int upper_bound)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (positions[j] >= 0 && positions[j] < upper_bound)
      {
        temp_storage.keys[positions[j]] = keys[j];
        if constexpr (num_value_channels == 1)
        {
          CUB_NS_QUALIFIER::detail::at<0>(temp_storage.per_channel_values).values[positions[j]] = get_value(j);
        }
      }
    }
  }

  // Cooperative-flush primitives. See the matching pair in
  // `block_partition_accumulating_candidates` (`block_partition_accumulating.cuh`)
  // for the full design notes: the hot-path full-buffer overload constant-folds
  // its `count` to `BufferCapacity` and unrolls the strided output loop into
  // `full_flush_waves = Capacity / BlockThreads` register-stride waves plus an
  // optional trailing partial wave; the partial overload preserves the runtime
  // shape for the once-per-kernel terminal flush.
  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_full_buffer()
  {
    static constexpr int full_flush_waves = BufferCapacity / BlockThreads;
    static constexpr int trailing_count   = BufferCapacity % BlockThreads;

    if (threadIdx.x == 0)
    {
      const auto r    = reserve_sel(static_cast<SelectedOffsetT>(BufferCapacity));
      temp_storage.cnt.base    = r.first;
      temp_storage.cnt.granted = static_cast<SelectedOffsetT>(r.second);
    }
    __syncthreads();

    const SelectedOffsetT base = temp_storage.cnt.base;
    const SelectedOffsetT to_write =
      SelectedReserveOp::may_grant_less ? temp_storage.cnt.granted : static_cast<SelectedOffsetT>(BufferCapacity);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int w = 0; w < full_flush_waves; ++w)
    {
      const int i = w * BlockThreads + static_cast<int>(threadIdx.x);
      if (!SelectedReserveOp::may_grant_less || static_cast<SelectedOffsetT>(i) < to_write)
      {
        sel_iter[base + static_cast<SelectedOffsetT>(i)] = sel_xform(temp_storage.keys[i]);
      }
    }
    if constexpr (trailing_count != 0)
    {
      const int i = full_flush_waves * BlockThreads + static_cast<int>(threadIdx.x);
      if (static_cast<int>(threadIdx.x) < trailing_count
          && (!SelectedReserveOp::may_grant_less || static_cast<SelectedOffsetT>(i) < to_write))
      {
        sel_iter[base + static_cast<SelectedOffsetT>(i)] = sel_xform(temp_storage.keys[i]);
      }
    }

    if constexpr (num_value_channels == 1)
    {
      auto& sink = ::cuda::std::get<0>(sinks);
      auto& vs   = CUB_NS_QUALIFIER::detail::at<0>(temp_storage.per_channel_values);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int w = 0; w < full_flush_waves; ++w)
      {
        const int i = w * BlockThreads + static_cast<int>(threadIdx.x);
        if (!SelectedReserveOp::may_grant_less || static_cast<SelectedOffsetT>(i) < to_write)
        {
          sink.selected_values_out[base + static_cast<SelectedOffsetT>(i)] =
            sink.selected_value_transform(vs.values[i]);
        }
      }
      if constexpr (trailing_count != 0)
      {
        const int i = full_flush_waves * BlockThreads + static_cast<int>(threadIdx.x);
        if (static_cast<int>(threadIdx.x) < trailing_count
            && (!SelectedReserveOp::may_grant_less || static_cast<SelectedOffsetT>(i) < to_write))
        {
          sink.selected_values_out[base + static_cast<SelectedOffsetT>(i)] =
            sink.selected_value_transform(vs.values[i]);
        }
      }
    }
  }

  // Partial flush used by `epilogue()`. `count` is in `[1, BufferCapacity)` at runtime.
  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_partial(int count)
  {
    if (threadIdx.x == 0)
    {
      const auto r    = reserve_sel(static_cast<SelectedOffsetT>(count));
      temp_storage.cnt.base    = r.first;
      temp_storage.cnt.granted = static_cast<SelectedOffsetT>(r.second);
    }
    __syncthreads();

    const SelectedOffsetT base = temp_storage.cnt.base;
    const SelectedOffsetT to_write =
      SelectedReserveOp::may_grant_less ? temp_storage.cnt.granted : static_cast<SelectedOffsetT>(count);

    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
    {
      sel_iter[base + static_cast<SelectedOffsetT>(i)] = sel_xform(temp_storage.keys[i]);
    }
    if constexpr (num_value_channels == 1)
    {
      auto& sink = ::cuda::std::get<0>(sinks);
      auto& vs   = CUB_NS_QUALIFIER::detail::at<0>(temp_storage.per_channel_values);
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
      {
        sink.selected_values_out[base + static_cast<SelectedOffsetT>(i)] = sink.selected_value_transform(vs.values[i]);
      }
    }
  }

  // ---------------------------------------------------------------
  // Member state.
  // ---------------------------------------------------------------
  _TempStorage& temp_storage;
  SelectedReserveOp& reserve_sel;
  SelectedKeyOutTransformOp& sel_xform;
  SelectedKeyOutIt sel_iter;
  ValueChannelSinksTuple& sinks;
  IdentifySelectedOp& identify_op;
};

//---------------------------------------------------------------------
// `strategy_to_filter_class<Strategy, ...>` -- compile-time selector mapping a
// `block_filter_strategy` value (and an `InlinedClassify` bool) to the
// corresponding filter class type.
//
// The non-accumulating strategy values map to one of the three classes in
// `block_filter.cuh`:
//   - `Atomics`                -> `block_filter_atomics<..., LazyValueLoad, InlinedClassify>`
//   - `Staged`                 -> `block_filter_staged<..., LazyValueLoad, InlinedClassify>`
//   - `SharedMem`              -> `block_filter_shared_mem<..., LazyValueLoad, InlinedClassify>`
//   - `AccumulatingFilter`     -> `block_filter_accumulating`
//                                 (with `CandidateBufferCapacity` filled in from the
//                                 metafunction's own `AccumulatingBufferCapacity` arg).
//                                 The accumulating variant always classifies inline,
//                                 so the `InlinedClassify` bool has no effect there.
//
// The agent uses this metafunction to derive `filter_t` from a policy enum.
//---------------------------------------------------------------------
template <block_filter_strategy Strategy,
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
          bool LazyValueLoad,
          bool InlinedClassify>
struct strategy_to_filter_class
{
private:
  using atomics_t = block_filter_atomics<
    BlockThreads,
    ItemsPerThread,
    InlinedClassify,
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

  using staged_t = block_filter_staged<
    BlockThreads,
    ItemsPerThread,
    KeyT,
    SelectedOffsetT,
    SelectedReserveOp,
    SelectedKeyOutTransformOp,
    SelectedKeyOutIt,
    IdentifySelectedOp,
    ValueChannelSinksTuple,
    ValueTypesTuple,
    DataSourceScratchTypesTuple,
    LazyValueLoad,
    InlinedClassify>;

  using shared_mem_t = block_filter_shared_mem<
    BlockThreads,
    ItemsPerThread,
    KeyT,
    SelectedOffsetT,
    SelectedReserveOp,
    SelectedKeyOutTransformOp,
    SelectedKeyOutIt,
    IdentifySelectedOp,
    ValueChannelSinksTuple,
    ValueTypesTuple,
    DataSourceScratchTypesTuple,
    LazyValueLoad,
    InlinedClassify>;

public:
  using type = ::cuda::std::conditional_t<
    Strategy == block_filter_strategy::staged, staged_t,
    ::cuda::std::conditional_t<Strategy == block_filter_strategy::shared_mem, shared_mem_t, atomics_t>>;
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
          bool LazyValueLoad,
          bool InlinedClassify>
struct strategy_to_filter_class<
  block_filter_strategy::accumulating_filter,
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
  LazyValueLoad,
  InlinedClassify>
{
  // The accumulating filter always classifies inline (no separate pre-classify
  // step), so it does not consume the `InlinedClassify` parameter. It also loads
  // value channels via stack-local `source_t::ScratchStorage` and so doesn't
  // consume the `DataSourceScratchTypesTuple` parameter; both are accepted for
  // parity with the non-accumulating branch (the agent always supplies them).
  using type = block_filter_accumulating<
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

template <block_filter_strategy Strategy,
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
          bool LazyValueLoad,
          bool InlinedClassify>
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
  LazyValueLoad,
  InlinedClassify>::type;
} // namespace detail::topk

CUB_NAMESPACE_END
