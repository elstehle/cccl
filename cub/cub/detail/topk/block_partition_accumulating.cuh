// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private accumulating partition primitives. Sister classes to `BlockPartition`
//! that buffer one partition stream (key + per-channel values per slot) in shared
//! memory across multiple `Partition()` calls and flush only when the buffer fills.
//!
//! Two public sister classes share an internal `accumulating_partition_base` (a
//! template parameterized on which stream is buffered):
//!
//!   - `BlockPartitionAccumulatingCandidates` -- buffers items classified
//!     `candidate`; selected items go direct-to-global through `reserve_sel_`.
//!     Used when `HasCandidates == true` (the agent's `buffered`-mode pass).
//!
//!   - `BlockPartitionAccumulatingSelected` -- buffers items classified `selected`
//!     (or `candidate` collapsed to `selected`); never writes via `reserve_cand_`.
//!     Used when `HasCandidates == false` (the agent's `early_stop`-mode pass).
//!
//! Both classes share `BlockPartition`'s "safe-both" interface: same ctor shape
//! `(TempStorage&, reserve_sel, reserve_cand, sel_xform, cand_xform, sel_it, cand_it,
//! value_channel_sinks)`, same per-call `Partition()` signature
//! `(scratch, keys, [num_items,] HasCandidates_ic, identify, callback, value_sources)`,
//! and an argless `epilogue()`. Sinks are captured at ctor so the consistency
//! invariant for accumulating across calls is enforced by construction.
//!
//! Per-tile algorithm (mirrored on both classes, parameterized by `BufferedStream`):
//!   1. Fused classify + reserve + act loop. `identify_op` runs once per item.
//!      Rejected and out-of-bounds items get `positions[j] = -1`. For the buffered
//!      stream class: the item's smem slot index is `atomicAdd(&counter, 1)`. For the
//!      non-buffered stream (only reachable from `Candidates` variant): a direct
//!      global atomic via `reserve_sel_` writes the key (and lazily-or-eagerly the
//!      value), and `positions[j] = -1`. There is no `classes[ItemsPerThread]`
//!      register array -- `positions[]` encodes both classification (skip vs.
//!      pending) and the slot index.
//!   2. Multi-round overflow loop (cooperative).
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
//!      < BufferCapacity by construction at this point (the per-tile loop wouldn't
//!      have left it equal to or above the capacity), so no overflow loop is needed.
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
namespace bp_acc_detail
{
// Compile-time tag selecting which classification stream is buffered in smem.
enum class buffered_stream
{
  selected,
  candidate,
};

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

// Per-channel value array slot in TempStorage. Mirrors `bp_detail::values_slot`.
template <typename Sink, int Capacity>
struct value_buf_slot_t
{
  typename Sink::value_t values[Capacity];
};

// Persistent TempStorage layout for an accumulating class. One smem buffer (key +
// per-channel values) plus the stream's counter + broadcast slots. The explicit
// empty ctor/dtor make the type "no-dynamic-init" so it can be declared as a
// `__shared__` variable directly (CUDA rejects dynamic initialization of static
// `__shared__` variables otherwise -- `phase_aggregate` wraps a `cuda::std::tuple`
// whose default ctor would otherwise count as dynamic).
template <typename KeyT, typename OffsetT, typename SinksTuple, int Capacity>
struct accumulating_temp_storage_t
{
  stream_counters_t<OffsetT> cnt;
  KeyT keys[Capacity];
  CUB_NS_QUALIFIER::detail::phase_aggregate<bp_detail::map_tuple_t<value_buf_slot_t, SinksTuple, Capacity>>
    per_channel_values;

  _CCCL_HOST_DEVICE accumulating_temp_storage_t() {}
  _CCCL_HOST_DEVICE ~accumulating_temp_storage_t() {}
};

//---------------------------------------------------------------------
// Internal accumulating partition implementation, parameterized by `BufferedStream`.
//
// The two public sister classes pin `BufferedStream` to either `candidate` or
// `selected` and otherwise inherit the implementation as-is.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          int BufferCapacity,
          buffered_stream BufferedStream,
          typename KeyT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          typename SelectedReserveOp,
          typename CandidateReserveOp,
          typename SelectedKeyOutTransformOp,
          typename CandidateKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename ValueChannelSinksTuple = ::cuda::std::tuple<>,
          bool LazyValueLoad              = false>
class accumulating_partition_base
{
public:
  static constexpr int tile_items              = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels      = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);
  static constexpr bool needs_persistent_state = true;

  static_assert(BufferCapacity >= 1, "Accumulating partition requires BufferCapacity >= 1.");
  static_assert(num_value_channels <= 1,
                "Accumulating partition supports keys-only or single-value-channel today; multi-channel needs a "
                "heterogeneous register-array tuple analogous to the BlockPartition shared_mem path.");

  // Pick the offset type used for the buffered stream's reservation/flush counter.
  using buffered_offset_t =
    ::cuda::std::conditional_t<BufferedStream == buffered_stream::candidate, CandidateOffsetT, SelectedOffsetT>;
  using buffered_reserve_op_t =
    ::cuda::std::conditional_t<BufferedStream == buffered_stream::candidate, CandidateReserveOp, SelectedReserveOp>;

  using TempStorage = accumulating_temp_storage_t<KeyT, buffered_offset_t, ValueChannelSinksTuple, BufferCapacity>;

  // Empty per-call scratch -- everything the class needs lives in `TempStorage`.
  struct ScratchStorage
  {};

  // The ctor is a COLLECTIVE operation -- all threads in the block must construct
  // the object together. Internally it zero-initializes the persistent smem counter
  // (thread 0) and then `__syncthreads()` so all threads observe the initialization
  // before they reach any subsequent `atomicAdd(&counter, ...)` inside `Partition()`.
  _CCCL_DEVICE _CCCL_FORCEINLINE accumulating_partition_base(
    TempStorage& storage,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksTuple& value_channel_sinks)
      : ts_(storage)
      , reserve_sel_(reserve_selected)
      , reserve_cand_(reserve_candidate)
      , sel_xform_(selected_key_transform)
      , cand_xform_(candidate_key_transform)
      , sel_iter_(selected_keys_out)
      , cand_iter_(candidate_keys_out)
      , sinks_(value_channel_sinks)
  {
    if (threadIdx.x == 0)
    {
      ts_.cnt.counter = 0;
    }
    __syncthreads();
  }

  // Full-tile overload.
  template <bool HasCandidates,
            typename IdentifyCandidatesOp,
            typename CandidateCallbackOp,
            typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Partition(
    ScratchStorage& /*scratch*/,
    const KeyT (&keys)[ItemsPerThread],
    ::cuda::std::integral_constant<bool, HasCandidates>,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op,
    ValueSourcesTuple& value_sources)
  {
    static_assert_has_candidates_for_variant<HasCandidates>();
    partition_impl<true, HasCandidates>(
      keys, /*num_items=*/tile_items, identify_candidates_op, candidate_callback_op, value_sources);
  }

  // Partial-tile overload.
  template <bool HasCandidates,
            typename NumItemsT,
            typename IdentifyCandidatesOp,
            typename CandidateCallbackOp,
            typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Partition(
    ScratchStorage& /*scratch*/,
    const KeyT (&keys)[ItemsPerThread],
    NumItemsT num_items,
    ::cuda::std::integral_constant<bool, HasCandidates>,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op,
    ValueSourcesTuple& value_sources)
  {
    static_assert_has_candidates_for_variant<HasCandidates>();
    partition_impl<false, HasCandidates>(
      keys,
      static_cast<int>(num_items),
      identify_candidates_op,
      candidate_callback_op,
      value_sources);
  }

  // Terminal flush: drain any remaining buffered items. No overflow possible because
  // every `Partition()` call leaves the counter < BufferCapacity by construction.
  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue()
  {
    // Sync to publish all in-flight Partition() writes to the smem buffer.
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
  template <bool HasCandidates>
  _CCCL_DEVICE _CCCL_FORCEINLINE static constexpr void static_assert_has_candidates_for_variant()
  {
    if constexpr (BufferedStream == buffered_stream::candidate)
    {
      static_assert(HasCandidates,
                    "BlockPartitionAccumulatingCandidates requires HasCandidates == true; for the early_stop / "
                    "HasCandidates == false case use BlockPartitionAccumulatingSelected.");
    }
    else
    {
      static_assert(!HasCandidates,
                    "BlockPartitionAccumulatingSelected requires HasCandidates == false; for the buffered / "
                    "HasCandidates == true case use BlockPartitionAccumulatingCandidates.");
    }
  }

  // First/only channel's value_t (or `int` if keys-only). Used to size the optional
  // per-thread eager-load register array.
  using channel_value_t = typename bp_detail::first_channel_value<ValueChannelSinksTuple>::type;

  // Eagerly load the (single) channel's per-thread values from the per-call source.
  // No-op when keys-only or when LazyValueLoad is true.
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  eager_load_value_channel(ValueSourcesTuple& value_sources, channel_value_t (&reg_values)[ItemsPerThread], int num_items)
  {
    (void) value_sources;
    (void) reg_values;
    (void) num_items;
    if constexpr (!LazyValueLoad && num_value_channels == 1)
    {
      auto& src      = ::cuda::std::get<0>(value_sources);
      using source_t = ::cuda::std::remove_reference_t<decltype(src)>;
      using sink_t   = ::cuda::std::tuple_element_t<0, ValueChannelSinksTuple>;
      static_assert(::cuda::std::is_same_v<typename source_t::value_t, typename sink_t::value_t>,
                    "Per-call value source's value_t must match the class-level sink's value_t.");
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
  //   - fused classify + reserve + (direct-write for non-buffered stream) loop,
  //   - multi-round overflow loop on the buffered stream's smem buffer.
  template <bool IsFull,
            bool HasCandidates,
            typename IdentifyCandidatesOp,
            typename CandidateCallbackOp,
            typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_impl(
    const KeyT (&keys)[ItemsPerThread],
    int num_items,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op,
    ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the partition pairs them positionally.");

    // Per-thread number of valid items (full vs. partial path).
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

    // Optional per-thread reg array for eager value loads. Sized to `int` for the
    // keys-only path (unused; ptxas should DCE it).
    channel_value_t reg_values[ItemsPerThread]{};
    eager_load_value_channel<IsFull>(value_sources, reg_values, num_items);

    // Per-tile lambda that fetches the channel value at item j either from the
    // pre-loaded reg array (eager) or via gather_one (lazy). Statically guarded for
    // the keys-only path.
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
      const candidate_class c = is_valid ? identify_candidates_op(keys[j]) : candidate_class::rejected;

      if (c == candidate_class::rejected)
      {
        positions[j] = -1;
      }
      else if constexpr (BufferedStream == buffered_stream::candidate)
      {
        // HasCandidates == true here (asserted at Partition() call site).
        if (c == candidate_class::candidate)
        {
          // Fire the candidate callback (architecture §10.2: every `candidate`
          // classified item, regardless of whether it ends up dropped during a
          // capped flush). Then reserve the smem slot.
          candidate_callback_op(keys[j]);
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
      else
      {
        // BufferedStream == selected, HasCandidates == false. The classifier
        // collapses candidate -> selected, so any non-rejected item is buffered as
        // selected.
        positions[j] = atomicAdd(&ts_.cnt.counter, 1);
      }
    }
    __syncthreads();

    // Step 2: multi-round overflow loop. Drains the smem buffer in BufferCapacity-
    // sized chunks until all items are either written to smem (deferred for next
    // tile) or flushed to global.
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
      if (cnt < BufferCapacity)
      {
        // Defer flush: write all pending items to smem and stop. Subsequent
        // Partition() calls keep accumulating into the same buffer.
        scatter_pending_to_smem(positions, keys, get_value, /*upper_bound=*/cnt);
        // Mark all pending items as consumed so the next round's writes (if any)
        // don't double-write -- actually unnecessary because we `break`, but keeps
        // positions[] in a consistent state for the test driver.
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
        // Perfect fit: write all pending items, flush the full buffer, reset counter.
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
        // cnt > BufferCapacity: write only items with positions[j] in [0,
        // BufferCapacity), flush, renumber the rest, decrement counter, loop.
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
        // loop again; new counter value is `cnt - BufferCapacity`.
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

  // Cooperative flush of `count` items from the smem buffer to the global iterator.
  // Reads sinks (reserve op, iterator, transform, per-channel sinks) from class
  // members, picking the right side based on `BufferedStream`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_round(int count)
  {
    // thread 0 reserves the global write range and broadcasts (base, granted).
    if (threadIdx.x == 0)
    {
      if constexpr (BufferedStream == buffered_stream::candidate)
      {
        const auto r    = reserve_cand_(static_cast<CandidateOffsetT>(count));
        ts_.cnt.base    = r.first;
        ts_.cnt.granted = static_cast<CandidateOffsetT>(r.second);
      }
      else
      {
        const auto r    = reserve_sel_(static_cast<SelectedOffsetT>(count));
        ts_.cnt.base    = r.first;
        ts_.cnt.granted = static_cast<SelectedOffsetT>(r.second);
      }
    }
    __syncthreads();

    const buffered_offset_t base    = ts_.cnt.base;
    const buffered_offset_t to_write =
      buffered_reserve_op_t::may_grant_less ? ts_.cnt.granted : static_cast<buffered_offset_t>(count);

    // Strided coalesced write of keys + per-channel values, picking the right sink
    // side based on `BufferedStream`.
    if constexpr (BufferedStream == buffered_stream::candidate)
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
      {
        cand_iter_[base + static_cast<buffered_offset_t>(i)] = cand_xform_(ts_.keys[i]);
      }
      if constexpr (num_value_channels == 1)
      {
        auto& sink = ::cuda::std::get<0>(sinks_);
        auto& vs   = CUB_NS_QUALIFIER::detail::at<0>(ts_.per_channel_values);
        for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
        {
          sink.candidate_values_out[base + static_cast<buffered_offset_t>(i)] =
            sink.candidate_value_transform(vs.values[i]);
        }
      }
    }
    else
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
      {
        sel_iter_[base + static_cast<buffered_offset_t>(i)] = sel_xform_(ts_.keys[i]);
      }
      if constexpr (num_value_channels == 1)
      {
        auto& sink = ::cuda::std::get<0>(sinks_);
        auto& vs   = CUB_NS_QUALIFIER::detail::at<0>(ts_.per_channel_values);
        for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
        {
          sink.selected_values_out[base + static_cast<buffered_offset_t>(i)] =
            sink.selected_value_transform(vs.values[i]);
        }
      }
    }
  }

  // ---------------------------------------------------------------
  // Member state: TempStorage reference + sinks captured by the ctor.
  // ---------------------------------------------------------------
  TempStorage& ts_;
  SelectedReserveOp& reserve_sel_;
  CandidateReserveOp& reserve_cand_;
  SelectedKeyOutTransformOp& sel_xform_;
  CandidateKeyOutTransformOp& cand_xform_;
  SelectedKeyOutIt sel_iter_;
  CandidateKeyOutIt cand_iter_;
  ValueChannelSinksTuple& sinks_;
};
} // namespace bp_acc_detail

//---------------------------------------------------------------------
// `BlockPartitionAccumulatingCandidates`
//
// Buffers items classified `candidate` in shared memory across multiple `Partition()`
// calls; selected items go direct-to-global via `reserve_sel_`. Used by the agent's
// `buffered`-mode pass (`HasCandidates == true`).
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
          typename ValueChannelSinksTuple = ::cuda::std::tuple<>,
          bool LazyValueLoad              = false>
class BlockPartitionAccumulatingCandidates
    : public bp_acc_detail::accumulating_partition_base<
        BlockThreads,
        ItemsPerThread,
        CandidateBufferCapacity,
        bp_acc_detail::buffered_stream::candidate,
        KeyT,
        SelectedOffsetT,
        CandidateOffsetT,
        SelectedReserveOp,
        CandidateReserveOp,
        SelectedKeyOutTransformOp,
        CandidateKeyOutTransformOp,
        SelectedKeyOutIt,
        CandidateKeyOutIt,
        ValueChannelSinksTuple,
        LazyValueLoad>
{
  using base_t = bp_acc_detail::accumulating_partition_base<
    BlockThreads,
    ItemsPerThread,
    CandidateBufferCapacity,
    bp_acc_detail::buffered_stream::candidate,
    KeyT,
    SelectedOffsetT,
    CandidateOffsetT,
    SelectedReserveOp,
    CandidateReserveOp,
    SelectedKeyOutTransformOp,
    CandidateKeyOutTransformOp,
    SelectedKeyOutIt,
    CandidateKeyOutIt,
    ValueChannelSinksTuple,
    LazyValueLoad>;

public:
  using base_t::base_t; // inherit ctor
};

//---------------------------------------------------------------------
// `BlockPartitionAccumulatingSelected`
//
// Buffers items classified `selected` (or `candidate` collapsed to `selected` when
// HasCandidates == false). Never writes via `reserve_cand_`. Used by the agent's
// `early_stop`-mode pass.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          int SelectedBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          typename SelectedReserveOp,
          typename CandidateReserveOp,
          typename SelectedKeyOutTransformOp,
          typename CandidateKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename ValueChannelSinksTuple = ::cuda::std::tuple<>,
          bool LazyValueLoad              = false>
class BlockPartitionAccumulatingSelected
    : public bp_acc_detail::accumulating_partition_base<
        BlockThreads,
        ItemsPerThread,
        SelectedBufferCapacity,
        bp_acc_detail::buffered_stream::selected,
        KeyT,
        SelectedOffsetT,
        CandidateOffsetT,
        SelectedReserveOp,
        CandidateReserveOp,
        SelectedKeyOutTransformOp,
        CandidateKeyOutTransformOp,
        SelectedKeyOutIt,
        CandidateKeyOutIt,
        ValueChannelSinksTuple,
        LazyValueLoad>
{
  using base_t = bp_acc_detail::accumulating_partition_base<
    BlockThreads,
    ItemsPerThread,
    SelectedBufferCapacity,
    bp_acc_detail::buffered_stream::selected,
    KeyT,
    SelectedOffsetT,
    CandidateOffsetT,
    SelectedReserveOp,
    CandidateReserveOp,
    SelectedKeyOutTransformOp,
    CandidateKeyOutTransformOp,
    SelectedKeyOutIt,
    CandidateKeyOutIt,
    ValueChannelSinksTuple,
    LazyValueLoad>;

public:
  using base_t::base_t; // inherit ctor
};

//---------------------------------------------------------------------
// `strategy_to_partition_class<Strategy, ...>` -- compile-time selector that maps a
// `BlockPartitionStrategy` enum value to the corresponding partition class type.
//
// The four `Atomics*` / `Staged` / `SharedMem` strategy values map to
// `BlockPartition<Strategy, ...>`. The two `Accumulating*` values map to the matching
// accumulating sister class (with `BufferCapacity` filled in from the metafunction's
// own template arg).
//
// The agent uses this to define `using partition_t = typename
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
          typename ValueChannelSinksTuple,
          bool LazyValueLoad>
struct strategy_to_partition_class
{
  // Default: non-accumulating strategies map to BlockPartition. The static_assert
  // inside BlockPartition catches any value not in {AtomicsPreClassify,
  // AtomicsInlinedClassify, Staged, SharedMem}.
  using type = BlockPartition<
    BlockThreads,
    ItemsPerThread,
    Strategy,
    KeyT,
    SelectedOffsetT,
    CandidateOffsetT,
    SelectedReserveOp,
    CandidateReserveOp,
    SelectedKeyOutTransformOp,
    CandidateKeyOutTransformOp,
    SelectedKeyOutIt,
    CandidateKeyOutIt,
    ValueChannelSinksTuple,
    LazyValueLoad>;
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
          typename ValueChannelSinksTuple,
          bool LazyValueLoad>
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
  ValueChannelSinksTuple,
  LazyValueLoad>
{
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
    ValueChannelSinksTuple,
    LazyValueLoad>;
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
          typename ValueChannelSinksTuple,
          bool LazyValueLoad>
struct strategy_to_partition_class<
  BlockPartitionStrategy::AccumulatingSelected,
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
  ValueChannelSinksTuple,
  LazyValueLoad>
{
  using type = BlockPartitionAccumulatingSelected<
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
    ValueChannelSinksTuple,
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
          typename ValueChannelSinksTuple,
          bool LazyValueLoad>
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
  ValueChannelSinksTuple,
  LazyValueLoad>::type;
} // namespace detail::topk

CUB_NAMESPACE_END
