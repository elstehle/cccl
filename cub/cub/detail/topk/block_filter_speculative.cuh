// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private `block_filter_speculative` -- single-stream sister of
//! `BlockFilter` / `block_filter_accumulating`. Like `block_filter_accumulating`
//! it buffers items matching a unary `IdentifySelected(key) -> bool` predicate
//! in a fixed-size smem buffer across multiple `partition()` calls and flushes
//! cooperatively when the buffer fills. Unlike the accumulating variant the
//! per-item slot reservation is *speculative + branchless*: each kept item
//! does `pos = atomicAdd(&counter, 1)`, ORs `(pos >= Cap) << j` into a
//! per-thread `overflow_bits` register, and writes the key to
//! `temp_storage.keys[min(pos, Cap)]` *unconditionally*. The trailing sentinel slot
//! at index `Cap` (the smem buffer is sized `Cap + 1`) absorbs racy
//! overflow writes; the cooperative flush only reads `[0, Cap)`. After the
//! classify loop a separate drain loop walks `overflow_bits`, emitting each
//! flagged item via per-item `reserve_sel(1)` (Atomics-equivalent
//! behaviour for the overflow tail). When the post-`partition()` counter
//! reaches or exceeds `BufferCapacity`, a single cooperative full-buffer
//! flush emits the leading `BufferCapacity` items and resets the counter
//! to 0.
//!
//! The design point versus `block_filter_accumulating`: the latter keeps a
//! per-thread `positions[ItemsPerThread]` array live across the multi-round
//! `overflow_loop`, raising the register high-water mark by `ItemsPerThread`
//! ints. `block_filter_speculative` replaces that with a single `uint32_t
//! overflow_bits` (one bit per item per thread) -- the per-thread `keys[]`
//! and optional `reg_values[]` arrays die at the end of `partition()`
//! because the cooperative flush only reads from smem. The intent is
//! **register parity with `block_filter_atomics`**, with the cross-tile
//! batching benefit (one cooperative store per `BufferCapacity` items)
//! preserved on sparse streams. Dense streams degrade gracefully to
//! Atomics-equivalent per-item global stores.
//!
//! The branchless `min(pos, Cap)` write keeps the classify-loop body as
//! close to `block_filter_atomics`'s straight-line scatter as possible (one
//! atomicAdd + one write per kept item), avoiding the per-item predicated
//! arm that an explicit `if (pos < Cap)` would force the compiler to plumb
//! through.
//!
//! Shares the same "safe-both" ctor + per-call interface as `BlockFilter` and
//! `block_filter_accumulating`:
//!   - Sinks + classify hook captured at ctor.
//!   - Per-call `partition(scratch, keys, [num_items,] value_sources)`.
//!   - Argless `epilogue()` for the terminal partial flush.
//!
//! `LazyValueLoad` semantics match the sibling accumulating class: when
//! `false`, per-thread `reg_values[ItemsPerThread]` is loaded upfront; when
//! `true`, values are gathered at scatter sites via `gather_one(j)`. Lazy
//! loading is the natural default for the Speculative variant because it
//! removes the per-thread `ItemsPerThread`-sized values array, which would
//! otherwise stay alive across the inline-drain branch.

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
#include <cub/detail/topk/block_filter_accumulating.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{

//---------------------------------------------------------------------
// `block_filter_speculative`
//
// Speculative single-stream sister of `BlockFilter`. Reuses the
// `accumulating_filter_temp_storage_t` layout (one counter
// + flush-broadcast slots, one keys buffer, optional per-channel values
// buffer) since the storage requirements are identical to
// `block_filter_accumulating`. Distinguished from `block_filter_accumulating`
// by the per-call control flow:
//
//   1. Fused classify + speculative reserve + (smem | bit-flag) loop.
//      `keep` items do `pos = atomicAdd(&counter, 1)`. The bit-flag is
//      written unconditionally for every `keep` item via a bool->uint32
//      cast (`overflow_bits |= (uint32(pos >= Capacity) << j)`); the smem
//      write happens iff `pos < Capacity`.
//   2. Overflow drain: per-bit-set item does a per-item
//      `reserve_sel(1)` and writes direct-to-global. The drain uses only
//      register state (`keys[]`, `overflow_bits`, optional `reg_values[]`),
//      so the cooperative flush below can fully reuse those registers.
//   3. `__syncthreads()` -- finalizes counter visibility + smem-write
//      visibility for the cooperative flush.
//   4. If `counter >= BufferCapacity`: one full-buffer cooperative flush;
//      reset counter to 0; sync. The counter is bounded above by
//      `BufferCapacity + tile_items - 1` (entrant counter is in
//      `[0, Capacity)`, max `tile_items` atomicAdds per call), so the
//      cooperative flush is single-round by construction.
//   5. `epilogue()`: terminal partial flush of any remaining items.
//
// `assert(ItemsPerThread <= 32)` because the bit-flag is a `uint32_t`. With
// `BlockThreads = 512` and `tile_items` up to 8192 on B200/B300, `ItemsPerThread`
// is at most 16 -- comfortably within the limit.
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
          bool LazyValueLoad              = false,
          bool InlinedClassify            = true>
class block_filter_speculative
{
public:
  static constexpr int tile_items         = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  static_assert(BufferCapacity >= 1, "Speculative filter requires BufferCapacity >= 1.");
  static_assert(ItemsPerThread <= 32, "Speculative filter overflow bit-mask is a uint32_t; ItemsPerThread must be <= 32.");
  static_assert(num_value_channels <= 1,
                "Speculative filter supports keys-only or single-value-channel today; multi-channel needs a "
                "heterogeneous register-array tuple analogous to the BlockFilter shared_mem path.");
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");

  // Reuse the accumulating filter's TempStorage layout -- the storage
  // requirements (one counter + one keys buffer + optional per-channel values
  // buffer) are identical. Wrapped in `cub::Uninitialized<>` at the public
  // layer.
  // The speculative variant allocates `BufferCapacity + 1` smem slots: the
  // trailing "sentinel" slot lets the classify loop write *unconditionally*
  // with `idx = min(pos, BufferCapacity)`, eliminating the per-item branch
  // on `pos < BufferCapacity`. The cooperative flush still emits only the
  // leading `BufferCapacity` slots, so the sentinel content is never read
  // -- it just absorbs racy overflow writes. Hypothesis: the unconditional
  // write keeps `pos` short-lived and lets ptxas avoid keeping it across a
  // predicated arm.
  static constexpr int buffer_storage_slots = BufferCapacity + 1;
  using _TempStorage =
    accumulating_filter_temp_storage_t<KeyT, SelectedOffsetT, ValueTypesTuple, buffer_storage_slots>;
  struct TempStorage : CUB_NS_QUALIFIER::Uninitialized<_TempStorage>
  {};

  struct ScratchStorage
  {};

  // COLLECTIVE ctor.
  _CCCL_DEVICE _CCCL_FORCEINLINE block_filter_speculative(
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

  // Storage type for the optional eager-loaded per-thread values array. The
  // size-1 dummy specialization (used when LazyValueLoad is true or the agent
  // is keys-only) mirrors the `int unused_values[1]{}` trick from
  // `block_partition_atomics::partition_atomics_fused` -- ptxas was observed to
  // keep `ItemsPerThread` zero-init writes alive even though the array is dead
  // under Lazy mode (the consuming `else`-branch in the per-item lambda is
  // discarded by `if constexpr`, but the local declaration's value-init still
  // survives liveness analysis). Collapsing to a 1-element placeholder costs 1
  // register instead of `ItemsPerThread`.
  static constexpr bool kEagerLoadValues = (!LazyValueLoad && num_value_channels == 1);
  static constexpr int kRegValuesSize    = kEagerLoadValues ? ItemsPerThread : 1;

  // Dispatches eager value-loading when applicable. No-op under Lazy or keys-only.
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void eager_load_value_channel(
    ValueSourcesTuple& value_sources, channel_value_t (&reg_values)[kRegValuesSize], int num_items)
  {
    (void) value_sources;
    (void) reg_values;
    (void) num_items;
    if constexpr (kEagerLoadValues)
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

  // Shared body for both partition() overloads. Dispatches the classify
  // path on `InlinedClassify` (mirrors `block_filter_atomics::filter_impl`):
  // when true, the inlined classifier re-evaluates `identify_op` inside
  // the scatter loop's `classifier(keys[j], j)` calls; when false, the
  // `precomputed_filter_classifier`'s ctor materializes
  // `bool kept[ItemsPerThread]` up front and the scatter loop just reads
  // `kept[j]`. The shared fused scatter helper takes the classifier as a
  // template parameter so the same SASS body services both choices.
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  filter_impl(const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the filter pairs them positionally.");

    const int num_thread_items = compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    if constexpr (InlinedClassify)
    {
      auto classifier = make_inlined_filter_classifier<IsFull>(identify_op, num_thread_items);
      filter_speculative_fused<IsFull>(keys, num_thread_items, classifier, value_sources, num_items);
    }
    else
    {
      precomputed_filter_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op};
      filter_speculative_fused<IsFull>(keys, num_thread_items, classifier, value_sources, num_items);
    }
  }

  // -----------------------------------------------------------------
  // Fused classify-and-scatter for the speculative filter: same
  // structural shape as `block_filter_atomics::filter_atomics_fused`
  // (single-arm per-item loop driven by an indexed `(key,j) -> bool`
  // classifier), but the kept-item write goes to the smem buffer via the
  // branchless `min(pos, Cap)` pattern with overflow tracked in a
  // per-thread bit-mask. The post-classify drain emits overflowed items
  // via per-item global atomics (Atomics-equivalent hot path).
  // -----------------------------------------------------------------
  template <bool IsFull, typename Classifier, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_speculative_fused(
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    ValueSourcesTuple& value_sources,
    int num_items)
  {
    (void) num_thread_items;

    // Size-1 placeholder when not eagerly loading (Lazy or keys-only); full-size
    // register array only when needed. See `kRegValuesSize` comment above.
    channel_value_t reg_values[kRegValuesSize]{};
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

    // Step 1: classify + smem-scatter loop, *branchless on the
    // capacity bound*. Each kept item does `pos = atomicAdd(&counter, 1)`,
    // ORs `(pos >= Cap) << j` into `overflow_bits`, and writes
    // `temp_storage.keys[min(pos, Cap)] = keys[j]` *unconditionally*. The trailing
    // sentinel slot at `temp_storage.keys[Cap]` absorbs racy overflow writes; the
    // cooperative flush only reads `[0, Cap)` so the sentinel content is
    // discarded. The structural intent: keep this loop as close to
    // `block_filter_atomics`'s scatter shape as possible (one straight-line
    // atomicAdd + write per kept item, no per-item branch), so ptxas
    // doesn't have to keep `pos` alive across a predicated arm and the
    // register footprint matches the Atomics baseline.
    ::cuda::std::uint32_t overflow_bits = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const bool keep = classifier(keys[j], j);
      if (keep)
      {
        const int pos = atomicAdd(&temp_storage.cnt.counter, 1);
        overflow_bits |= (static_cast<::cuda::std::uint32_t>(pos >= BufferCapacity) << j);
        const int idx           = (pos < BufferCapacity) ? pos : BufferCapacity;
        temp_storage.keys[idx]           = keys[j];
        if constexpr (num_value_channels == 1)
        {
          CUB_NS_QUALIFIER::detail::at<0>(temp_storage.per_channel_values).values[idx] = get_value(j);
        }
      }
    }

    // Step 2: deferred drain of overflowed items. Identical to
    // `block_filter_atomics`'s per-item scatter loop (per-item `reserve_sel(1)`
    // + direct-to-global writes); reads only register state
    // (`keys[]`, `overflow_bits`, optional eager `reg_values[]`).
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if ((overflow_bits >> j) & 1u)
      {
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
      }
    }

    // Step 3: finalize counter + smem-write visibility before the
    // (potentially-fired) cooperative flush. Note: the per-thread `keys[]` and
    // `reg_values[]` arrays die at function return; the flush below only
    // touches `temp_storage.keys[]` / `temp_storage.per_channel_values`, both in smem.
    __syncthreads();

    // Step 4: full-buffer cooperative flush iff the buffer reached capacity.
    // Counter <= Capacity + tile_items - 1 by construction, so a single round
    // suffices: after flushing the leading `Capacity` items and resetting the
    // counter to 0, the remaining `cnt - Capacity` items (if any) are already
    // accounted for -- they went through Step 2's deferred drain.
    const int cnt_after = temp_storage.cnt.counter;
    if (cnt_after >= BufferCapacity)
    {
      cooperative_flush_full_buffer();
      __syncthreads();
      if (threadIdx.x == 0)
      {
        temp_storage.cnt.counter = 0;
      }
      __syncthreads();
    }
  }

  // Cooperative-flush primitives -- byte-identical in shape to the
  // accumulating sibling's versions. See
  // `block_partition_accumulating.cuh` for the wave-decomposition design
  // notes (full-waves count = Capacity / BlockThreads + optional trailing
  // partial wave).
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

  // Partial flush used by `epilogue()`. `count` is in `[1, BufferCapacity)`
  // at runtime, so no compile-time wave decomposition is available.
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
// `strategy_to_filter_class` specialization for
// `block_filter_strategy::speculative_filter`. Lives in this header so the
// agent (or any other consumer) only pays the include cost when actually
// using the Speculative strategy.
//---------------------------------------------------------------------
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
  block_filter_strategy::speculative_filter,
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
  // `InlinedClassify` is threaded through to `block_filter_speculative`'s
  // classify path (`filter_impl` dispatches between `inlined_filter_classifier`
  // and `precomputed_filter_classifier`). `DataSourceScratchTypesTuple` is
  // not consumed -- the Speculative filter gathers values via stack-local
  // `source_t::ScratchStorage`.
  using type = block_filter_speculative<
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
    LazyValueLoad,
    InlinedClassify>;
};

} // namespace detail::topk

CUB_NAMESPACE_END
