// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private accumulating partition primitive `block_partition_accumulating_candidates`
//! -- sister class to `BlockPartition` that buffers the `candidate` stream (key +
//! per-channel values per slot) in shared memory across multiple `partition()` calls
//! and flushes only when the buffer fills. Selected items go direct-to-global through
//! `reserve_sel`. Used by the agent's `buffered`-mode pass.
//!
//! The early_stop / "buffer the selected stream" path lives in the dedicated
//! single-stream `block_filter_accumulating` primitive
//! (`block_filter_accumulating.cuh`).
//!
//! Shares `BlockPartition`'s "safe-both" interface: same ctor shape
//! `(TempStorage&, reserve_sel, reserve_cand, sel_xform, cand_xform, sel_it, cand_it,
//! value_channel_sinks, identify_candidates_op, candidate_callback_op)`, same
//! per-call `partition(scratch, keys, [num_items,] value_sources)`, and an argless
//! `epilogue()`. Sinks + classify hooks are captured at ctor so the consistency
//! invariant for accumulating across calls is enforced by construction.
//!
//! Per-tile algorithm:
//!   1. Fused classify + reserve + act loop. `identify_op` runs once per item.
//!      Rejected and out-of-bounds items get `positions[j] = -1`. Candidate items
//!      buffer into smem with `positions[j] = atomicAdd(&counter, 1)`; selected
//!      items go direct-to-global via `reserve_sel` and `positions[j] = -1`.
//!      `positions[]` encodes both classification (skip vs. pending) and the
//!      smem slot index.
//!   2. Multi-round overflow loop (cooperative):
//!        - if `counter < BufferCapacity`: scatter pending items to smem; defer
//!          the global flush so subsequent `partition()` calls can keep
//!          accumulating.
//!        - if `counter == BufferCapacity`: scatter all pending items to smem +
//!          cooperative flush + reset `counter` to 0.
//!        - if `counter > BufferCapacity`: scatter only items with `positions[j] <
//!          BufferCapacity` to smem + cooperative flush + renumber positions
//!          (subtract BufferCapacity from the still-pending ones) + decrement
//!          counter; loop until `counter <= BufferCapacity`.
//!   3. `epilogue()` (called by the agent after all `partition()` calls): if the
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
//                 partition() calls so the buffer can accumulate across tiles.
//   `base`     -- broadcast: written by thread 0 inside the cooperative-flush
//                 primitives, read by every thread before the strided write.
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
// `block_partition_accumulating_candidates`
//
// Buffers items classified `candidate` in shared memory across multiple `partition()`
// calls; selected items go direct-to-global via `reserve_sel`. Used by the agent's
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
class block_partition_accumulating_candidates
{
public:
  static constexpr int tile_items         = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  // Compile-time upper bound on the number of times `overflow_loop` iterates per
  // `partition()` call. By the cross-call invariant the entrant counter is in
  // `[0, CandidateBufferCapacity - 1]`, and each call adds at most `tile_items`
  // atomicAdd reservations, so the post-add counter is bounded by
  // `CandidateBufferCapacity - 1 + tile_items`. Each `>=`-branch round drains
  // `CandidateBufferCapacity`, so the number of draining rounds is
  // `ceil(tile_items / CandidateBufferCapacity)`. The final non-draining round
  // accounts for the +1. For configurations with
  // `CandidateBufferCapacity >= tile_items` this collapses to 2, letting NVCC
  // straight-line the loop -- the second iteration is provably the
  // `cnt < Capacity` branch because the post-flush counter is in `[0, Capacity - 1]`.
  static constexpr int max_flush_iters =
    (tile_items + CandidateBufferCapacity - 1) / CandidateBufferCapacity + 1;

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
  // any subsequent `atomicAdd(&counter, ...)` inside `partition()`.
  _CCCL_DEVICE _CCCL_FORCEINLINE block_partition_accumulating_candidates(
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
      : temp_storage(storage.Alias())
      , reserve_sel(reserve_selected)
      , reserve_cand(reserve_candidate)
      , sel_xform(selected_key_transform)
      , cand_xform(candidate_key_transform)
      , sel_iter(selected_keys_out)
      , cand_iter(candidate_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_candidates_op)
      , callback_op(candidate_callback_op)
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
    partition_impl<true>(keys, /*num_items=*/tile_items, value_sources);
  }

  // Partial-tile overload.
  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& /*scratch*/,
    const KeyT (&keys)[ItemsPerThread],
    NumItemsT num_items,
    ValueSourcesTuple& value_sources)
  {
    partition_impl<false>(keys, static_cast<int>(num_items), value_sources);
  }

  // Terminal flush: drain any remaining buffered items. No overflow possible because
  // every `partition()` call leaves the counter < CandidateBufferCapacity by
  // construction.
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

  // Shared body for both partition() overloads. Performs:
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
      const candidate_class c = is_valid ? identify_op(keys[j]) : candidate_class::rejected;

      if (c == candidate_class::rejected)
      {
        positions[j] = -1;
      }
      else if (c == candidate_class::candidate)
      {
        // Fire the candidate callback (architecture §10.2: every `candidate`-
        // classified item, regardless of whether it ends up dropped during a
        // capped flush). Then reserve the smem slot.
        callback_op(keys[j]);
        positions[j] = atomicAdd(&temp_storage.cnt.counter, 1);
      }
      else
      {
        // c == candidate_class::selected: direct global atomic via reserve_sel.
        const auto r = reserve_sel(SelectedOffsetT{1});
        bool granted = true;
        if constexpr (SelectedReserveOp::may_grant_less)
        {
          granted = (r.second != SelectedOffsetT{0});
        }
        if (granted)
        {
          sel_iter[r.first] = sel_xform(keys[j]);
          if constexpr (num_value_channels == 1)
          {
            auto& sink                        = ::cuda::std::get<0>(sinks);
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

  // Multi-round overflow loop. positions[] is mutated as items are consumed: a
  // flushed item's slot index becomes negative after the renumber step (`pos -=
  // Capacity`), which doubles as the sentinel that excludes it from subsequent
  // rounds via the `positions[j] >= 0` predicate in `scatter_pending_to_smem`.
  //
  // The loop is bounded by the compile-time `max_flush_iters`, computed from
  // `tile_items` and `CandidateBufferCapacity`. With this bound the loop
  // structure is a counted `for` (giving NVCC a static iteration count for
  // register-lifetime analysis and unroll heuristics) instead of the previous
  // `while (true) { ...; break; }` shape. The `cnt == Capacity` and
  // `cnt > Capacity` cases have been folded into a single `cnt >= Capacity`
  // branch: their bodies differ only in the post-flush counter being set to `0`
  // vs `cnt - Capacity`, and `cnt - Capacity == 0` when `cnt == Capacity`, so a
  // unified `cnt - Capacity` covers both. Eliminating the third branch shortens
  // the basic blocks the allocator has to widen across.
  //
  // When `Capacity >= tile_items` the bound is exactly 2, and the second
  // iteration is provably the `cnt < Capacity` (drain-remainder) branch -- the
  // post-flush counter lands in `[0, Capacity - 1]` by construction. NVCC then
  // straight-lines the body and the `positions[]` mutation only happens once.
  template <typename GetValueFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  overflow_loop(int (&positions)[ItemsPerThread], const KeyT (&keys)[ItemsPerThread], GetValueFn get_value)
  {
    for (int iter = 0; iter < max_flush_iters; ++iter)
    {
      const int cnt = temp_storage.cnt.counter;
      if (cnt < CandidateBufferCapacity)
      {
        // Drain-remainder branch: scatter what's left into smem, mark the
        // pending slots as consumed (the buffer accumulates across the next
        // `partition()` call), and exit.
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
      // Flush branch (`cnt >= Capacity`): emit the leading buffer-worth of items
      // to the global candidate stream, then renumber any still-pending items so
      // that their slot indices fall back into `[0, Capacity)` for the next
      // iteration. Items just flushed go negative and drop out of subsequent
      // rounds.
      scatter_pending_to_smem(positions, keys, get_value, /*upper_bound=*/CandidateBufferCapacity);
      __syncthreads();
      cooperative_flush_full_buffer();
      __syncthreads();
      if (threadIdx.x == 0)
      {
        temp_storage.cnt.counter = cnt - CandidateBufferCapacity;
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
        temp_storage.keys[positions[j]] = keys[j];
        if constexpr (num_value_channels == 1)
        {
          CUB_NS_QUALIFIER::detail::at<0>(temp_storage.per_channel_values).values[positions[j]] = get_value(j);
        }
      }
    }
  }

  // Cooperative-flush primitives. Two overloads split along the lifetime
  // boundary between the overflow loop (always flushes exactly
  // `CandidateBufferCapacity` items) and the terminal `epilogue()` (flushes a
  // runtime `leftover` in `[1, Capacity)`):
  //
  //   - `cooperative_flush_full_buffer()` -- the hot-path overload. The flush
  //     count is the compile-time `CandidateBufferCapacity`, so the strided
  //     output loop splits into `full_flush_waves = Capacity / BlockThreads`
  //     fully-unrolled register-stride writes plus one optional trailing
  //     partial wave bound-checked against `Capacity % BlockThreads`. With
  //     `BlockThreads = 512` and the Hopper+ tuning of
  //     `Capacity = tile_items in {1024, 2048, 4096, 8192}`, the full-waves
  //     count is in `{2, 4, 8, 16}` -- all small enough for ptxas to unroll
  //     cleanly. The reserve op runs once (thread 0) for the entire buffer.
  //
  //   - `cooperative_flush_partial(int count)` -- the terminal overload, used
  //     by `epilogue()`. `count` is `< Capacity` at runtime; the same shape as
  //     before is fine because this runs at most once per kernel invocation.
  //
  // Splitting these out replaces the prior single `cooperative_flush_round(int
  // count)` which forced ptxas to assume a runtime `count` on the hot path
  // even though both call sites passed compile-time constants.
  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_full_buffer()
  {
    static constexpr int full_flush_waves = CandidateBufferCapacity / BlockThreads;
    static constexpr int trailing_count   = CandidateBufferCapacity % BlockThreads;

    if (threadIdx.x == 0)
    {
      const auto r    = reserve_cand(static_cast<CandidateOffsetT>(CandidateBufferCapacity));
      temp_storage.cnt.base    = r.first;
      temp_storage.cnt.granted = static_cast<CandidateOffsetT>(r.second);
    }
    __syncthreads();

    const CandidateOffsetT base = temp_storage.cnt.base;
    const CandidateOffsetT to_write =
      CandidateReserveOp::may_grant_less ? temp_storage.cnt.granted : static_cast<CandidateOffsetT>(CandidateBufferCapacity);

    // Keys stream: full waves first, then an optional trailing partial wave.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int w = 0; w < full_flush_waves; ++w)
    {
      const int i = w * BlockThreads + static_cast<int>(threadIdx.x);
      if (!CandidateReserveOp::may_grant_less || static_cast<CandidateOffsetT>(i) < to_write)
      {
        cand_iter[base + static_cast<CandidateOffsetT>(i)] = cand_xform(temp_storage.keys[i]);
      }
    }
    if constexpr (trailing_count != 0)
    {
      const int i = full_flush_waves * BlockThreads + static_cast<int>(threadIdx.x);
      if (static_cast<int>(threadIdx.x) < trailing_count
          && (!CandidateReserveOp::may_grant_less || static_cast<CandidateOffsetT>(i) < to_write))
      {
        cand_iter[base + static_cast<CandidateOffsetT>(i)] = cand_xform(temp_storage.keys[i]);
      }
    }

    // Values channel (optional): same shape.
    if constexpr (num_value_channels == 1)
    {
      auto& sink = ::cuda::std::get<0>(sinks);
      auto& vs   = CUB_NS_QUALIFIER::detail::at<0>(temp_storage.per_channel_values);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int w = 0; w < full_flush_waves; ++w)
      {
        const int i = w * BlockThreads + static_cast<int>(threadIdx.x);
        if (!CandidateReserveOp::may_grant_less || static_cast<CandidateOffsetT>(i) < to_write)
        {
          sink.candidate_values_out[base + static_cast<CandidateOffsetT>(i)] =
            sink.candidate_value_transform(vs.values[i]);
        }
      }
      if constexpr (trailing_count != 0)
      {
        const int i = full_flush_waves * BlockThreads + static_cast<int>(threadIdx.x);
        if (static_cast<int>(threadIdx.x) < trailing_count
            && (!CandidateReserveOp::may_grant_less || static_cast<CandidateOffsetT>(i) < to_write))
        {
          sink.candidate_values_out[base + static_cast<CandidateOffsetT>(i)] =
            sink.candidate_value_transform(vs.values[i]);
        }
      }
    }
  }

  // Partial flush used by `epilogue()`. `count` is in `[1, CandidateBufferCapacity)`
  // at runtime, so no compile-time wave decomposition is available.
  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_partial(int count)
  {
    if (threadIdx.x == 0)
    {
      const auto r    = reserve_cand(static_cast<CandidateOffsetT>(count));
      temp_storage.cnt.base    = r.first;
      temp_storage.cnt.granted = static_cast<CandidateOffsetT>(r.second);
    }
    __syncthreads();

    const CandidateOffsetT base = temp_storage.cnt.base;
    const CandidateOffsetT to_write =
      CandidateReserveOp::may_grant_less ? temp_storage.cnt.granted : static_cast<CandidateOffsetT>(count);

    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
    {
      cand_iter[base + static_cast<CandidateOffsetT>(i)] = cand_xform(temp_storage.keys[i]);
    }
    if constexpr (num_value_channels == 1)
    {
      auto& sink = ::cuda::std::get<0>(sinks);
      auto& vs   = CUB_NS_QUALIFIER::detail::at<0>(temp_storage.per_channel_values);
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
  _TempStorage& temp_storage;
  SelectedReserveOp& reserve_sel;
  CandidateReserveOp& reserve_cand;
  SelectedKeyOutTransformOp& sel_xform;
  CandidateKeyOutTransformOp& cand_xform;
  SelectedKeyOutIt sel_iter;
  CandidateKeyOutIt cand_iter;
  ValueChannelSinksTuple& sinks;
  IdentifyCandidatesOp& identify_op;
  CandidateCallbackOp& callback_op;
};

//---------------------------------------------------------------------
// `strategy_to_partition_class<Strategy, ...>` -- compile-time selector that maps a
// `block_partition_strategy` enum value (and an `InlinedClassify` bool) to the
// corresponding partition class type.
//
// The non-accumulating strategy values map to one of the three classes in
// `block_partition.cuh`:
//   - `Atomics`                -> `block_partition_atomics<..., LazyValueLoad, InlinedClassify>`
//   - `Staged`                 -> `block_partition_staged<..., LazyValueLoad, InlinedClassify>`
//   - `SharedMem`              -> `block_partition_shared_mem<..., LazyValueLoad, InlinedClassify>`
//   - `AccumulatingCandidates` -> `block_partition_accumulating_candidates`
//                                 (with `CandidateBufferCapacity` filled in from the
//                                 metafunction's own `AccumulatingBufferCapacity` arg).
//                                 The accumulating variant always classifies inline
//                                 (its fused classify-and-act loop has no separate
//                                 pre-classify step), so the `InlinedClassify` bool
//                                 has no effect there.
//   - `SpeculativeBoth`        -> `block_partition_speculative` (in
//                                 `block_partition_speculative.cuh` -- the
//                                 partial specialization lives there to keep the
//                                 include cost optional). Consumes both
//                                 `AccumulatingBufferCapacity` (candidate stream)
//                                 and `SpeculativeSelectedBufferCapacity` (selected
//                                 stream; `0` short-circuits the selected smem
//                                 buffer to pure-Atomics).
//
// `SpeculativeSelectedBufferCapacity` is ignored by every strategy except
// `SpeculativeBoth`; the slot is unconditionally present on the metafunction
// signature so each agent can thread its tuning value through without having
// to inspect the strategy value.
//
// The agent uses this to define `using buffered_partition_t = typename
// strategy_to_partition_class<...>::type` -- a single point that hides the dispatch.
//---------------------------------------------------------------------
template <block_partition_strategy Strategy,
          int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          int SpeculativeSelectedBufferCapacity,
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
  using atomics_t = block_partition_atomics<
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

  using staged_t = block_partition_staged<
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

  using shared_mem_t = block_partition_shared_mem<
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
    Strategy == block_partition_strategy::staged, staged_t,
    ::cuda::std::conditional_t<Strategy == block_partition_strategy::shared_mem, shared_mem_t, atomics_t>>;
};

template <int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          int SpeculativeSelectedBufferCapacity,
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
  block_partition_strategy::accumulating_candidates,
  BlockThreads,
  ItemsPerThread,
  AccumulatingBufferCapacity,
  SpeculativeSelectedBufferCapacity,
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
  // non-accumulating branch (the agent always supplies them). The single-stream
  // accumulating variant only buffers the candidate stream and so does not consume
  // `SpeculativeSelectedBufferCapacity` either.
  using type = block_partition_accumulating_candidates<
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

template <block_partition_strategy Strategy,
          int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          int SpeculativeSelectedBufferCapacity,
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
  SpeculativeSelectedBufferCapacity,
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
