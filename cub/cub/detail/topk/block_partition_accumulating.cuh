// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private accumulating partition primitive `BlockPartitionAccumulatingCandidates`
//! -- sister class to `BlockPartition` that buffers the `candidate` stream (key +
//! per-channel values per slot) in shared memory across multiple `Partition()` calls
//! and flushes only when the buffer fills. Selected items go direct-to-global through
//! `reserve_sel_`. Used by the agent's `buffered`-mode pass.
//!
//! The early_stop / "buffer the selected stream" path lives in the dedicated
//! single-stream `BlockFilterAccumulating` primitive
//! (`block_filter_accumulating.cuh`).
//!
//! Shares `BlockPartition`'s "safe-both" interface: same ctor shape
//! `(TempStorage&, reserve_sel, reserve_cand, sel_xform, cand_xform, sel_it, cand_it,
//! value_channel_sinks, identify_candidates_op, candidate_callback_op)`, same
//! per-call `Partition(scratch, keys, [num_items,] value_sources)`, and an argless
//! `epilogue()`. Sinks + classify hooks are captured at ctor so the consistency
//! invariant for accumulating across calls is enforced by construction.
//!
//! Per-tile algorithm:
//!   1. Fused classify + reserve + act loop. `identify_op` runs once per item.
//!      Rejected and out-of-bounds items get `positions[j] = -1`. Candidate items
//!      buffer into smem with `positions[j] = atomicAdd(&counter, 1)`; selected
//!      items go direct-to-global via `reserve_sel_` and `positions[j] = -1`.
//!      `positions[]` encodes both classification (skip vs. pending) and the
//!      smem slot index.
//!   2. Multi-round overflow loop (cooperative):
//!        - if `counter < BufferCapacity`: scatter pending items to smem; defer
//!          the global flush so subsequent `Partition()` calls can keep
//!          accumulating.
//!        - if `counter == BufferCapacity`: scatter all pending items to smem +
//!          cooperative flush + reset `counter` to 0.
//!        - if `counter > BufferCapacity`: scatter only items with `positions[j] <
//!          BufferCapacity` to smem + cooperative flush + renumber positions
//!          (subtract BufferCapacity from the still-pending ones) + decrement
//!          counter; loop until `counter <= BufferCapacity`.
//!   3. `epilogue()` (called by the agent after all `Partition()` calls): if the
//!      counter is non-zero, run a single round of the cooperative flush. Counter is
//!      < BufferCapacity by construction at this point.
//!
//! `LazyValueLoad` follows the same convention as `BlockPartition`: when `false`
//! (default), each call up-front loads each value channel into a per-thread
//! `value_t reg_values[ItemsPerThread]` array via `data_source.complete_load(...)`.
//! When `true`, the up-front load is skipped and `data_source.gather_one(j)` is
//! called only at scatter sites for non-rejected items.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

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
namespace bp_acc_detail
{
// Per-stream counter + flush-broadcast slots in TempStorage.
//   `counter`  -- per-tile reservation counter (0..BufferCapacity); persisted across
//                 Partition() calls so the buffer can accumulate across tiles.
//   `base`     -- broadcast: written by thread 0 inside `cooperative_flush_round`,
//                 read by every thread before the strided write.
//   `granted`  -- broadcast: same; relevant only for `may_grant_less` reserve ops.
template <typename OffsetT>
struct stream_counters_t
{
  int counter;
  OffsetT base;
  OffsetT granted;
};

// Per-channel value array slot in TempStorage. Indexed by element type (the
// per-channel `value_t` from the agent-supplied `ValueTypesTuple`).
template <typename ValueT, int Capacity>
struct value_buf_slot_t
{
  ValueT values[Capacity];
};

// Persistent TempStorage layout. One smem buffer (key + per-channel values) plus
// the candidate stream's counter + broadcast slots. The accumulating class wraps
// this in `cub::Uninitialized<...>` and exposes the wrapper as the public
// `TempStorage` so users can declare `__shared__ partition_t::TempStorage`
// without tripping CUDA's "dynamic initialization not supported for `__shared__`"
// rule.
template <typename KeyT, typename OffsetT, typename ValueTypesTuple, int Capacity>
struct accumulating_temp_storage_t
{
  stream_counters_t<OffsetT> cnt;
  KeyT keys[Capacity];
  CUB_NS_QUALIFIER::detail::phase_aggregate<bp_detail::map_tuple_t<value_buf_slot_t, ValueTypesTuple, Capacity>>
    per_channel_values;
};
} // namespace bp_acc_detail

//---------------------------------------------------------------------
// `BlockPartitionAccumulatingCandidates`
//
// Buffers items classified `candidate` in shared memory across multiple `Partition()`
// calls; selected items go direct-to-global via `reserve_sel_`. Used by the agent's
// `buffered`-mode pass.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          int CandidateBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          typename SelectedReserveOp,
          typename CandidateReserveOp,
          typename SelectedKeyOutTransformOp,
          typename CandidateKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename IdentifyCandidatesOp,
          typename CandidateCallbackOp,
          typename ValueChannelSinksTuple = ::cuda::std::tuple<>,
          typename ValueTypesTuple        = ::cuda::std::tuple<>,
          bool LazyValueLoad              = false>
class BlockPartitionAccumulatingCandidates
{
public:
  static constexpr int tile_items         = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  static_assert(CandidateBufferCapacity >= 1, "Accumulating partition requires CandidateBufferCapacity >= 1.");
  static_assert(num_value_channels <= 1,
                "Accumulating partition supports keys-only or single-value-channel today; multi-channel needs a "
                "heterogeneous register-array tuple analogous to the BlockPartition shared_mem path.");
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");

  // Internal `_TempStorage` is the actual buffer + counters. The publicly-exposed
  // `TempStorage` wraps it in `cub::Uninitialized<>` so the user can declare
  // `__shared__ partition_t::TempStorage` directly. The agent's
  // `partition_storage_layout_for_t` selector consults
  // `is_empty_v<TempStorage>` to pick the right smem aliasing layout (the wrapper
  // has a non-trivial `DeviceWord storage[N]` member, so `is_empty_v` is false and
  // the persistent + scratch layout is selected).
  using _TempStorage =
    bp_acc_detail::accumulating_temp_storage_t<KeyT, CandidateOffsetT, ValueTypesTuple, CandidateBufferCapacity>;
  struct TempStorage : CUB_NS_QUALIFIER::Uninitialized<_TempStorage>
  {};

  // Empty per-call scratch -- everything the class needs lives in `TempStorage`.
  struct ScratchStorage
  {};

  // The ctor is a COLLECTIVE operation -- all threads in the block must construct
  // the object together. Internally it unwraps the `Uninitialized<>` wrapper via
  // `.Alias()`, zero-initializes the persistent smem counter (thread 0), and then
  // `__syncthreads()` so all threads observe the initialization before they reach
  // any subsequent `atomicAdd(&counter, ...)` inside `Partition()`.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockPartitionAccumulatingCandidates(
    TempStorage& storage,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksTuple& value_channel_sinks,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op)
      : ts_(storage.Alias())
      , reserve_sel_(reserve_selected)
      , reserve_cand_(reserve_candidate)
      , sel_xform_(selected_key_transform)
      , cand_xform_(candidate_key_transform)
      , sel_iter_(selected_keys_out)
      , cand_iter_(candidate_keys_out)
      , sinks_(value_channel_sinks)
      , identify_op_(identify_candidates_op)
      , callback_op_(candidate_callback_op)
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
    partition_impl<true>(keys, /*num_items=*/tile_items, value_sources);
  }

  // Partial-tile overload.
  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Partition(
    ScratchStorage& /*scratch*/,
    const KeyT (&keys)[ItemsPerThread],
    NumItemsT num_items,
    ValueSourcesTuple& value_sources)
  {
    partition_impl<false>(keys, static_cast<int>(num_items), value_sources);
  }

  // Terminal flush: drain any remaining buffered items. No overflow possible because
  // every `Partition()` call leaves the counter < CandidateBufferCapacity by
  // construction.
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
  // First/only channel's value_t (or `int` if keys-only). Used to size the optional
  // per-thread eager-load register array.
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

  // Shared body for both Partition() overloads. Performs:
  //   - num_thread_items computation (full vs. partial),
  //   - eager value-channel load (when LazyValueLoad == false),
  //   - fused classify + reserve + (direct-write for selected stream) loop,
  //   - multi-round overflow loop on the candidate stream's smem buffer.
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition_impl(const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the partition pairs them positionally.");

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

    // Step 1: fused classify + reserve + act. positions[j] encodes the per-item
    // state -- `-1` for skip (rejected, OOB, or selected-and-already-written), `>=
    // 0` for "still pending; smem slot index".
    int positions[ItemsPerThread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const bool is_valid     = IsFull ? true : (j < num_thread_items);
      const candidate_class c = is_valid ? identify_op_(keys[j]) : candidate_class::rejected;

      if (c == candidate_class::rejected)
      {
        positions[j] = -1;
      }
      else if (c == candidate_class::candidate)
      {
        // Fire the candidate callback (architecture §10.2: every `candidate`-
        // classified item, regardless of whether it ends up dropped during a
        // capped flush). Then reserve the smem slot.
        callback_op_(keys[j]);
        positions[j] = atomicAdd(&ts_.cnt.counter, 1);
      }
      else
      {
        // c == candidate_class::selected: direct global atomic via reserve_sel_.
        const auto r = reserve_sel_(SelectedOffsetT{1});
        bool granted = true;
        if constexpr (SelectedReserveOp::may_grant_less)
        {
          granted = (r.second != SelectedOffsetT{0});
        }
        if (granted)
        {
          sel_iter_[r.first] = sel_xform_(keys[j]);
          if constexpr (num_value_channels == 1)
          {
            auto& sink                        = ::cuda::std::get<0>(sinks_);
            sink.selected_values_out[r.first] = sink.selected_value_transform(get_value(j));
          }
        }
        positions[j] = -1;
      }
    }
    __syncthreads();

    // Step 2: multi-round overflow loop. Drains the smem buffer in
    // CandidateBufferCapacity-sized chunks until all items are either written to
    // smem (deferred for next tile) or flushed to global.
    overflow_loop(positions, keys, get_value);
  }

  // Multi-round overflow loop. positions[] is mutated as items are consumed (set to
  // -1 once they're successfully written to smem and either deferred or flushed).
  template <typename GetValueFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  overflow_loop(int (&positions)[ItemsPerThread], const KeyT (&keys)[ItemsPerThread], GetValueFn get_value)
  {
    while (true)
    {
      const int cnt = ts_.cnt.counter;
      if (cnt < CandidateBufferCapacity)
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
      else if (cnt == CandidateBufferCapacity)
      {
        scatter_pending_to_smem(positions, keys, get_value, /*upper_bound=*/CandidateBufferCapacity);
        __syncthreads();
        cooperative_flush_round(CandidateBufferCapacity);
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
        // cnt > CandidateBufferCapacity: write only items with positions[j] in [0,
        // CandidateBufferCapacity), flush, renumber the rest, decrement counter,
        // loop.
        scatter_pending_to_smem(positions, keys, get_value, /*upper_bound=*/CandidateBufferCapacity);
        __syncthreads();
        cooperative_flush_round(CandidateBufferCapacity);
        __syncthreads();
        if (threadIdx.x == 0)
        {
          ts_.cnt.counter = cnt - CandidateBufferCapacity;
        }
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < ItemsPerThread; ++j)
        {
          if (positions[j] >= 0)
          {
            positions[j] -= CandidateBufferCapacity;
          }
        }
        __syncthreads();
      }
    }
  }

  // Scatter all items with `0 <= positions[j] < upper_bound` into the smem buffer.
  // Caller is responsible for the surrounding sync.
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

  // Cooperative flush of `count` items from the candidate smem buffer to the global
  // iterator.
  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_round(int count)
  {
    if (threadIdx.x == 0)
    {
      const auto r    = reserve_cand_(static_cast<CandidateOffsetT>(count));
      ts_.cnt.base    = r.first;
      ts_.cnt.granted = static_cast<CandidateOffsetT>(r.second);
    }
    __syncthreads();

    const CandidateOffsetT base = ts_.cnt.base;
    const CandidateOffsetT to_write =
      CandidateReserveOp::may_grant_less ? ts_.cnt.granted : static_cast<CandidateOffsetT>(count);

    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
    {
      cand_iter_[base + static_cast<CandidateOffsetT>(i)] = cand_xform_(ts_.keys[i]);
    }
    if constexpr (num_value_channels == 1)
    {
      auto& sink = ::cuda::std::get<0>(sinks_);
      auto& vs   = CUB_NS_QUALIFIER::detail::at<0>(ts_.per_channel_values);
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
      {
        sink.candidate_values_out[base + static_cast<CandidateOffsetT>(i)] =
          sink.candidate_value_transform(vs.values[i]);
      }
    }
  }

  // ---------------------------------------------------------------
  // Member state.
  // ---------------------------------------------------------------
  _TempStorage& ts_;
  SelectedReserveOp& reserve_sel_;
  CandidateReserveOp& reserve_cand_;
  SelectedKeyOutTransformOp& sel_xform_;
  CandidateKeyOutTransformOp& cand_xform_;
  SelectedKeyOutIt sel_iter_;
  CandidateKeyOutIt cand_iter_;
  ValueChannelSinksTuple& sinks_;
  IdentifyCandidatesOp& identify_op_;
  CandidateCallbackOp& callback_op_;
};

//---------------------------------------------------------------------
// `strategy_to_partition_class<Strategy, ...>` -- compile-time selector that maps a
// `BlockPartitionStrategy` enum value (and an `InlinedClassify` bool) to the
// corresponding partition class type.
//
// The non-accumulating strategy values map to one of the three classes in
// `block_partition.cuh`:
//   - `Atomics`                -> `BlockPartitionAtomics<..., LazyValueLoad, InlinedClassify>`
//   - `Staged`                 -> `BlockPartitionStaged<..., LazyValueLoad, InlinedClassify>`
//   - `SharedMem`              -> `BlockPartitionSharedMem<..., LazyValueLoad, InlinedClassify>`
//   - `AccumulatingCandidates` -> `BlockPartitionAccumulatingCandidates`
//                                 (with `CandidateBufferCapacity` filled in from the
//                                 metafunction's own `AccumulatingBufferCapacity` arg).
//                                 The accumulating variant always classifies inline
//                                 (its fused classify-and-act loop has no separate
//                                 pre-classify step), so the `InlinedClassify` bool
//                                 has no effect there.
//
// The agent uses this to define `using buffered_partition_t = typename
// strategy_to_partition_class<...>::type` -- a single point that hides the dispatch.
//---------------------------------------------------------------------
template <BlockPartitionStrategy Strategy,
          int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          typename SelectedReserveOp,
          typename CandidateReserveOp,
          typename SelectedKeyOutTransformOp,
          typename CandidateKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename IdentifyCandidatesOp,
          typename CandidateCallbackOp,
          typename ValueChannelSinksTuple,
          typename ValueTypesTuple,
          typename DataSourceScratchTypesTuple,
          bool LazyValueLoad,
          bool InlinedClassify>
struct strategy_to_partition_class
{
private:
  using atomics_t = BlockPartitionAtomics<
    BlockThreads,
    ItemsPerThread,
    InlinedClassify,
    KeyT,
    SelectedOffsetT,
    CandidateOffsetT,
    SelectedReserveOp,
    CandidateReserveOp,
    SelectedKeyOutTransformOp,
    CandidateKeyOutTransformOp,
    SelectedKeyOutIt,
    CandidateKeyOutIt,
    IdentifyCandidatesOp,
    CandidateCallbackOp,
    ValueChannelSinksTuple,
    ValueTypesTuple,
    DataSourceScratchTypesTuple,
    LazyValueLoad>;

  using staged_t = BlockPartitionStaged<
    BlockThreads,
    ItemsPerThread,
    KeyT,
    SelectedOffsetT,
    CandidateOffsetT,
    SelectedReserveOp,
    CandidateReserveOp,
    SelectedKeyOutTransformOp,
    CandidateKeyOutTransformOp,
    SelectedKeyOutIt,
    CandidateKeyOutIt,
    IdentifyCandidatesOp,
    CandidateCallbackOp,
    ValueChannelSinksTuple,
    ValueTypesTuple,
    DataSourceScratchTypesTuple,
    LazyValueLoad,
    InlinedClassify>;

  using shared_mem_t = BlockPartitionSharedMem<
    BlockThreads,
    ItemsPerThread,
    KeyT,
    SelectedOffsetT,
    CandidateOffsetT,
    SelectedReserveOp,
    CandidateReserveOp,
    SelectedKeyOutTransformOp,
    CandidateKeyOutTransformOp,
    SelectedKeyOutIt,
    CandidateKeyOutIt,
    IdentifyCandidatesOp,
    CandidateCallbackOp,
    ValueChannelSinksTuple,
    ValueTypesTuple,
    DataSourceScratchTypesTuple,
    LazyValueLoad,
    InlinedClassify>;

public:
  using type = ::cuda::std::conditional_t<
    Strategy == BlockPartitionStrategy::Staged, staged_t,
    ::cuda::std::conditional_t<Strategy == BlockPartitionStrategy::SharedMem, shared_mem_t, atomics_t>>;
};

template <int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          typename SelectedReserveOp,
          typename CandidateReserveOp,
          typename SelectedKeyOutTransformOp,
          typename CandidateKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename IdentifyCandidatesOp,
          typename CandidateCallbackOp,
          typename ValueChannelSinksTuple,
          typename ValueTypesTuple,
          typename DataSourceScratchTypesTuple,
          bool LazyValueLoad,
          bool InlinedClassify>
struct strategy_to_partition_class<
  BlockPartitionStrategy::AccumulatingCandidates,
  BlockThreads,
  ItemsPerThread,
  AccumulatingBufferCapacity,
  KeyT,
  SelectedOffsetT,
  CandidateOffsetT,
  SelectedReserveOp,
  CandidateReserveOp,
  SelectedKeyOutTransformOp,
  CandidateKeyOutTransformOp,
  SelectedKeyOutIt,
  CandidateKeyOutIt,
  IdentifyCandidatesOp,
  CandidateCallbackOp,
  ValueChannelSinksTuple,
  ValueTypesTuple,
  DataSourceScratchTypesTuple,
  LazyValueLoad,
  InlinedClassify>
{
  // The accumulating prototype always classifies inline (its fused classify-and-act
  // loop has no separate pre-classify step), so it does not consume the
  // `InlinedClassify` parameter. It also loads value channels via stack-local
  // `source_t::ScratchStorage` and so doesn't consume the
  // `DataSourceScratchTypesTuple` parameter; both are accepted for parity with the
  // non-accumulating branch (the agent always supplies them).
  using type = BlockPartitionAccumulatingCandidates<
    BlockThreads,
    ItemsPerThread,
    AccumulatingBufferCapacity,
    KeyT,
    SelectedOffsetT,
    CandidateOffsetT,
    SelectedReserveOp,
    CandidateReserveOp,
    SelectedKeyOutTransformOp,
    CandidateKeyOutTransformOp,
    SelectedKeyOutIt,
    CandidateKeyOutIt,
    IdentifyCandidatesOp,
    CandidateCallbackOp,
    ValueChannelSinksTuple,
    ValueTypesTuple,
    LazyValueLoad>;
};

template <BlockPartitionStrategy Strategy,
          int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          typename SelectedReserveOp,
          typename CandidateReserveOp,
          typename SelectedKeyOutTransformOp,
          typename CandidateKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename IdentifyCandidatesOp,
          typename CandidateCallbackOp,
          typename ValueChannelSinksTuple,
          typename ValueTypesTuple,
          typename DataSourceScratchTypesTuple,
          bool LazyValueLoad,
          bool InlinedClassify>
using strategy_to_partition_class_t = typename strategy_to_partition_class<
  Strategy,
  BlockThreads,
  ItemsPerThread,
  AccumulatingBufferCapacity,
  KeyT,
  SelectedOffsetT,
  CandidateOffsetT,
  SelectedReserveOp,
  CandidateReserveOp,
  SelectedKeyOutTransformOp,
  CandidateKeyOutTransformOp,
  SelectedKeyOutIt,
  CandidateKeyOutIt,
  IdentifyCandidatesOp,
  CandidateCallbackOp,
  ValueChannelSinksTuple,
  ValueTypesTuple,
  DataSourceScratchTypesTuple,
  LazyValueLoad,
  InlinedClassify>::type;
} // namespace detail::topk

CUB_NAMESPACE_END
