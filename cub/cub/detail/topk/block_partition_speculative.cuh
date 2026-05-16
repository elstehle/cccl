// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private speculative-accumulating partition primitive
//! `block_partition_speculative` -- dual-stream sister of
//! `block_partition_accumulating_candidates` and `BlockPartition`. Like the
//! accumulating sibling, both the candidate and the selected streams accumulate
//! across multiple `partition()` calls into fixed-size shared-memory buffers,
//! and the cooperative full-buffer flush amortizes the per-item global-atomic
//! cost. Unlike the accumulating sibling, each per-item slot reservation is
//! *speculative + branchless*: an item does `pos = atomicAdd(&counter, 1)`,
//! ORs `(pos >= Cap) << j` into a per-thread `*_overflow_bits` register, and
//! writes the key to `temp_storage.{cand,sel}.keys[min(pos, Cap)]` *unconditionally*.
//! The trailing sentinel slot at index `Cap` (each enabled buffer is sized
//! `Cap + 1`) absorbs racy overflow writes; the cooperative flush only
//! reads `[0, Cap)`. A pair of post-classify drain loops walks the
//! overflow bit-masks and emits each flagged item via per-item
//! `reserve_*_(1)` -- the same hot path as `block_partition_atomics`.
//!
//! The motivating design point is **register parity with `block_partition_atomics`**.
//! `block_partition_accumulating_candidates` keeps a per-thread
//! `positions[ItemsPerThread]` array live across the multi-round
//! `overflow_loop`, which raises the kernel's register high-water mark by
//! `ItemsPerThread` ints. The speculative variant replaces that with two
//! `uint32_t` bit-masks (one bit per item per thread per stream) -- the
//! per-thread `keys[]` and optional `reg_values[]` arrays die at function
//! return and the cooperative flush only touches smem. The branchless
//! `min(pos, Cap)` write keeps the classify-loop body identical in shape
//! to `block_partition_atomics`'s scatter (atomicAdd + write, no per-item
//! predicated arm). On sparse streams the cross-tile batching benefit (one
//! cooperative store per `Capacity` items) is preserved; on dense streams
//! the overflow drain degrades to Atomics-equivalent per-item stores.
//!
//! `SelectedBufferCapacity == 0` short-circuits the selected stream to pure
//! `block_partition_atomics`-style behaviour (per-item `reserve_sel(1)` direct
//! to global, no smem buffer, no cooperative flush). This is the natural
//! tuning when the selected stream is dense (e.g., the agent's `last_filter`
//! pass).
//!
//! Shares the same "safe-both" ctor + per-call interface as `BlockPartition`
//! and `block_partition_accumulating_candidates`:
//!   - Sinks + classify hook + candidate callback captured at ctor.
//!   - Per-call `partition(scratch, keys, [num_items,] value_sources)`.
//!   - Argless `epilogue()` for the terminal partial flush.
//!
//! `LazyValueLoad = true` is the natural default for the Speculative variant:
//! with eager value-loading the `reg_values[ItemsPerThread]` array lives
//! across the classify-and-act loop, which materially raises the register
//! high-water mark. Lazy loading restricts the per-thread state during the
//! loop to just `keys[]`.

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
#include <cub/detail/topk/block_partition_accumulating.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{

//---------------------------------------------------------------------
// `block_partition_speculative`
//
// Algorithm (per-tile):
//   1. Optionally eager-load values into `reg_values[ItemsPerThread]`
//      (skipped when `LazyValueLoad`).
//   2. Fused classify + speculative reserve + (smem | bit-flag) loop:
//        - `rejected`  items: no-op.
//        - `candidate` items: `callback_op` fires (architecture: every
//          candidate-classified item, regardless of where the post-add index
//          lands). Then `pos = atomicAdd(&cand_cnt.counter, 1)`, the bit-mask
//          gets `(pos >= CandCap) << j` via an unbranched bool->uint32 cast,
//          and the smem write is predicated on `pos < CandCap`.
//        - `selected`  items (when `SelectedBufferCapacity > 0`): symmetric
//          to candidate but against the selected arm.
//        - `selected`  items (when `SelectedBufferCapacity == 0`): the bit
//          is unconditionally set so the post-loop drain emits the item via
//          per-item `reserve_sel(1)` (pure-Atomics behaviour).
//   3. Drain selected-overflow items first (priority: selected items are
//      unconditionally in the final output; if `reserve_*_` is back-grow-capped
//      we want them to claim slots before candidates).
//   4. Drain candidate-overflow items.
//   5. `__syncthreads()` -- finalizes counter + smem-write visibility for
//      the cooperative flush(es).
//   6. Conditionally flush each buffer that reached capacity, resetting its
//      counter to 0.
//
// `assert(ItemsPerThread <= 32)` because each overflow bit-mask is a
// `uint32_t`. With `BlockThreads = 512` and `tile_items` up to 8192 on
// B200/B300, `ItemsPerThread <= 16` -- comfortably within the limit.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          int CandidateBufferCapacity,
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
          typename IdentifyCandidatesOp,
          typename CandidateCallbackOp,
          typename ValueChannelSinksT = CUB_NS_QUALIFIER::NullType,
          typename ValueT             = CUB_NS_QUALIFIER::NullType,
          bool LazyValueLoad          = false,
          bool InlinedClassify        = true>
class block_partition_speculative
{
public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;

  static_assert(CandidateBufferCapacity >= 1,
                "Speculative partition requires CandidateBufferCapacity >= 1. To disable candidate-stream buffering "
                "altogether, use block_partition_atomics.");
  static_assert(SelectedBufferCapacity >= 0,
                "SelectedBufferCapacity must be non-negative. Set it to 0 to bypass the selected smem buffer.");
  static_assert(ItemsPerThread <= 32,
                "Speculative partition overflow bit-mask is a uint32_t; ItemsPerThread must be <= 32.");

  static constexpr bool selected_buffered = (SelectedBufferCapacity > 0);

  // Sentinel-slot storage. Each enabled smem buffer is sized `Capacity + 1`
  // so the classify loop can use `min(pos, Capacity)` as an unconditional
  // index instead of an `if (pos < Capacity)` branch; the trailing slot
  // absorbs racy overflow writes which are then drained per-item to global
  // via the post-classify `*_overflow_bits` walk. The cooperative flush
  // reads only `[0, Capacity)`. The `0`-capacity selected stream stays at
  // size 0 so the empty-base packing path still applies.
  static constexpr int cand_buffer_storage_slots = CandidateBufferCapacity + 1;
  static constexpr int sel_buffer_storage_slots  = selected_buffered ? SelectedBufferCapacity + 1 : 0;

  // Per-stream counter + flush-broadcast slots in TempStorage. Same layout as
  // `block_partition_accumulating_candidates::stream_counters_t`; nested here
  // so the speculative partition is self-contained.
  template <typename OffsetT>
  struct stream_counters_t
  {
    int counter;
    OffsetT base;
    OffsetT granted;
  };

  // Per-stream persistent storage. Keeps the dual-stream nature explicit by
  // instantiating two parallel `speculative_stream_storage_t` arms in the
  // outer `_TempStorage`. The `Capacity == 0` case (selected stream bypassing
  // its smem buffer in the pure-Atomics fallback) lands on
  // `speculative_stream_storage_empty`, which has `is_empty_v<> == true` so
  // the surrounding TempStorage benefits from empty-base-style packing.
  //
  // The `values[]` array lives directly in the per-stream storage in
  // single-value mode; in keys-only mode the `_keys_only` partial
  // specialization drops it entirely.
  template <typename OffsetT, int Capacity, bool KeysOnly>
  struct speculative_stream_storage_full
  {
    stream_counters_t<OffsetT> cnt;
    KeyT keys[Capacity];
    ValueT values[Capacity];
  };
  template <typename OffsetT, int Capacity>
  struct speculative_stream_storage_full<OffsetT, Capacity, /*KeysOnly=*/true>
  {
    stream_counters_t<OffsetT> cnt;
    KeyT keys[Capacity];
  };
  struct speculative_stream_storage_empty
  {};
  template <typename OffsetT, int Capacity>
  using speculative_stream_storage_t = ::cuda::std::conditional_t<
    Capacity == 0,
    speculative_stream_storage_empty,
    speculative_stream_storage_full<OffsetT, Capacity, keys_only>>;

  // Persistent TempStorage layout. Two per-stream arms; the selected arm is
  // empty when `SelectedCapacity == 0`. Wrapped in `cub::Uninitialized<>` at
  // the public layer.
  struct speculative_partition_temp_storage_t
  {
    speculative_stream_storage_t<CandidateOffsetT, cand_buffer_storage_slots> cand;
    speculative_stream_storage_t<SelectedOffsetT, sel_buffer_storage_slots> sel;
  };

  using _TempStorage = speculative_partition_temp_storage_t;
  struct TempStorage : CUB_NS_QUALIFIER::Uninitialized<_TempStorage>
  {};

  struct ScratchStorage
  {};

  // COLLECTIVE ctor.
  _CCCL_DEVICE _CCCL_FORCEINLINE block_partition_speculative(
    TempStorage& storage,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksT& value_channel_sinks,
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
      temp_storage.cand.cnt.counter = 0;
      if constexpr (selected_buffered)
      {
        temp_storage.sel.cnt.counter = 0;
      }
    }
    __syncthreads();
  }

  // Full-tile overload.
  template <typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& /*scratch*/, const KeyT (&keys)[ItemsPerThread], ValueSourceT& value_source)
  {
    partition_impl<true>(keys, /*num_items=*/tile_items, value_source);
  }

  // Partial-tile overload.
  template <typename NumItemsT, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& /*scratch*/,
    const KeyT (&keys)[ItemsPerThread],
    NumItemsT num_items,
    ValueSourceT& value_source)
  {
    partition_impl<false>(keys, static_cast<int>(num_items), value_source);
  }

  // Terminal flush: drain any remaining buffered items in each arm.
  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue()
  {
    __syncthreads();
    const int cand_leftover = temp_storage.cand.cnt.counter;
    if (cand_leftover > 0)
    {
      cooperative_flush_partial_cand(cand_leftover);
      __syncthreads();
      if (threadIdx.x == 0)
      {
        temp_storage.cand.cnt.counter = 0;
      }
      __syncthreads();
    }
    if constexpr (selected_buffered)
    {
      const int sel_leftover = temp_storage.sel.cnt.counter;
      if (sel_leftover > 0)
      {
        cooperative_flush_partial_sel(sel_leftover);
        __syncthreads();
        if (threadIdx.x == 0)
        {
          temp_storage.sel.cnt.counter = 0;
        }
        __syncthreads();
      }
    }
  }

private:
  using channel_value_t = ::cuda::std::conditional_t<keys_only, int, ValueT>;

  // Storage type for the optional eager-loaded per-thread values array. The
  // size-1 dummy specialization (used when LazyValueLoad is true or the agent
  // is keys-only) mirrors the `int unused_values[1]{}` trick from
  // `block_partition_atomics::partition_atomics_fused` -- ptxas was observed to
  // keep `ItemsPerThread` zero-init writes alive even though the array is dead
  // under Lazy mode (the consuming `else`-branch in the per-item lambda is
  // discarded by `if constexpr`, but the local declaration's value-init still
  // survives liveness analysis). Collapsing to a 1-element placeholder costs 1
  // register instead of `ItemsPerThread`.
  static constexpr bool kEagerLoadValues = (!LazyValueLoad && !keys_only);
  static constexpr int kRegValuesSize    = kEagerLoadValues ? ItemsPerThread : 1;

  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void eager_load_value_channel(
    ValueSourceT& value_source, channel_value_t (&reg_values)[kRegValuesSize], int num_items)
  {
    (void) value_source;
    (void) reg_values;
    (void) num_items;
    if constexpr (kEagerLoadValues)
    {
      static_assert(::cuda::std::is_same_v<typename ValueSourceT::value_t, ValueT>,
                    "Per-call value source's value_t must match the class-level ValueT template parameter.");
      typename ValueSourceT::ScratchStorage scratch{};
      if constexpr (IsFull)
      {
        auto h = value_source.submit_load(scratch);
        h.complete_load(reg_values);
      }
      else
      {
        auto h = value_source.submit_load(scratch, num_items);
        h.complete_load(reg_values);
      }
    }
  }

  // Shared body for both partition() overloads. Dispatches the classify
  // path on `InlinedClassify` (mirrors `block_partition_atomics::partition_impl`):
  // when true, the inlined classifier re-evaluates `identify_op` inside the
  // scatter loop's `classifier(keys[j], j)` calls; when false, the
  // `precomputed_classifier`'s ctor materializes `candidate_class
  // classes[ItemsPerThread]` up front (firing the candidate callback for
  // every candidate item) and the scatter loop just reads `classes[j]`. The
  // shared fused scatter helper takes the classifier + per-mode callback as
  // template parameters so the same SASS body services both choices.
  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition_impl(const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourceT& value_source)
  {
    const int num_thread_items = compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    if constexpr (InlinedClassify)
    {
      auto classifier = make_inlined_classifier<IsFull>(identify_op, num_thread_items);
      partition_speculative_fused<IsFull>(keys, num_thread_items, classifier, callback_op, value_source, num_items);
    }
    else
    {
      // `precomputed_classifier`'s ctor fires `callback_op` for every
      // candidate item; the scatter loop then uses `noop_callback_op` to
      // avoid double-firing.
      precomputed_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op, callback_op};
      noop_callback_op noop_cb{};
      partition_speculative_fused<IsFull>(keys, num_thread_items, classifier, noop_cb, value_source, num_items);
    }
  }

  // -----------------------------------------------------------------
  // Fused classify-and-scatter for the speculative variant: same
  // structural shape as `block_partition_atomics::partition_atomics_fused`
  // (a unified per-item loop driven by an indexed classifier), but each
  // arm writes to its smem buffer using the branchless `min(pos, Cap)`
  // pattern and tracks overflow in a per-thread bit-mask. The two
  // post-classify drain loops emit overflowed items via per-item
  // global atomics (Atomics-equivalent hot path).
  // -----------------------------------------------------------------
  template <bool IsFull, typename Classifier, typename CandidateCallbackOpT, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_speculative_fused(
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    CandidateCallbackOpT& candidate_callback_op,
    ValueSourceT& value_source,
    int num_items)
  {
    (void) num_thread_items;
    (void) value_source;

    // Size-1 placeholder when not eagerly loading (Lazy or keys-only); full-size
    // register array only when needed. See `kRegValuesSize` comment above.
    channel_value_t reg_values[kRegValuesSize]{};
    eager_load_value_channel<IsFull>(value_source, reg_values, num_items);

    auto get_value = [&](int j) -> channel_value_t {
      if constexpr (LazyValueLoad && !keys_only)
      {
        return value_source.gather_one(j);
      }
      else
      {
        return reg_values[j];
      }
    };

    // Step 1: classify + smem-scatter loop, *branchless on the capacity
    // bound*. Two independent arms per unrolled item -- mirroring
    // `block_partition_atomics`'s deliberate `if (selected) {...} if
    // (candidate) {...}` pattern, which avoids the per-item indirect-branch
    // table ptxas emits for an `if/else if` cascade. Each arm does
    // `pos = atomicAdd(&counter, 1)`, ORs `(pos >= Cap) << j` into its
    // bit-mask, and writes `temp_storage.{cand,sel}.keys[min(pos, Cap)] = keys[j]`
    // *unconditionally*. The trailing sentinel slot at index `Cap` absorbs
    // racy overflow writes; the cooperative flush only reads `[0, Cap)`.
    // The structural intent: keep this loop's hot path identical to the
    // Atomics scatter (one atomicAdd + one write per item), so ptxas
    // doesn't have to keep `pos` alive across a predicated arm.
    //
    // `SelectedBufferCapacity == 0` short-circuits to pure-Atomics behaviour
    // for selected: the bit is set unconditionally and the post-loop drain
    // emits the item via per-item `reserve_sel(1)`.
    ::cuda::std::uint32_t cand_overflow_bits = 0;
    ::cuda::std::uint32_t sel_overflow_bits  = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const candidate_class c = classifier(keys[j], j);

      // Architecture §10.2: callback fires for every `candidate`-classified
      // item. With `InlinedClassify=false` this is a `noop_callback_op`
      // (the real callback already fired from the classifier's ctor); with
      // `InlinedClassify=true` it's the real `callback_op`.
      if (c == candidate_class::candidate)
      {
        candidate_callback_op(keys[j]);
      }

      if (c == candidate_class::selected)
      {
        if constexpr (selected_buffered)
        {
          const int pos = atomicAdd(&temp_storage.sel.cnt.counter, 1);
          sel_overflow_bits |= (static_cast<::cuda::std::uint32_t>(pos >= SelectedBufferCapacity) << j);
          const int idx             = (pos < SelectedBufferCapacity) ? pos : SelectedBufferCapacity;
          temp_storage.sel.keys[idx] = keys[j];
          if constexpr (!keys_only)
          {
            temp_storage.sel.values[idx] = get_value(j);
          }
        }
        else
        {
          // SelectedBufferCapacity == 0: drain in the post-loop step.
          sel_overflow_bits |= (::cuda::std::uint32_t{1} << j);
        }
      }

      if (c == candidate_class::candidate)
      {
        const int pos = atomicAdd(&temp_storage.cand.cnt.counter, 1);
        cand_overflow_bits |= (static_cast<::cuda::std::uint32_t>(pos >= CandidateBufferCapacity) << j);
        const int idx              = (pos < CandidateBufferCapacity) ? pos : CandidateBufferCapacity;
        temp_storage.cand.keys[idx] = keys[j];
        if constexpr (!keys_only)
        {
          temp_storage.cand.values[idx] = get_value(j);
        }
      }
    }

    // Step 2: drain selected-overflow items first. Architecture: selected
    // items are unconditionally in the final output; if `reserve_*_` is
    // back-grow-capped, draining selected before candidates lets selected
    // claim global slots first.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if ((sel_overflow_bits >> j) & 1u)
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
          if constexpr (!keys_only)
          {
            sinks.selected_values_out[r.first] = sinks.selected_value_transform(get_value(j));
          }
        }
      }
    }

    // Step 3: drain candidate-overflow items.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if ((cand_overflow_bits >> j) & 1u)
      {
        const auto r = reserve_cand(CandidateOffsetT{1});
        bool granted = true;
        if constexpr (CandidateReserveOp::may_grant_less)
        {
          granted = (r.second != CandidateOffsetT{0});
        }
        if (granted)
        {
          cand_iter[r.first] = cand_xform(keys[j]);
          if constexpr (!keys_only)
          {
            sinks.candidate_values_out[r.first] = sinks.candidate_value_transform(get_value(j));
          }
        }
      }
    }

    // Step 4: finalize counter + smem-write visibility before the cooperative
    // flushes. Note: the per-thread `keys[]` and `reg_values[]` arrays die at
    // function return; the flushes below only touch smem.
    __syncthreads();

    const int cand_cnt_after = temp_storage.cand.cnt.counter;
    if (cand_cnt_after >= CandidateBufferCapacity)
    {
      cooperative_flush_full_buffer_cand();
      __syncthreads();
      if (threadIdx.x == 0)
      {
        temp_storage.cand.cnt.counter = 0;
      }
      __syncthreads();
    }

    if constexpr (selected_buffered)
    {
      const int sel_cnt_after = temp_storage.sel.cnt.counter;
      if (sel_cnt_after >= SelectedBufferCapacity)
      {
        cooperative_flush_full_buffer_sel();
        __syncthreads();
        if (threadIdx.x == 0)
        {
          temp_storage.sel.cnt.counter = 0;
        }
        __syncthreads();
      }
    }
  }

  // ---------------------------------------------------------------
  // Cooperative-flush primitives. Two pairs (candidate / selected), one
  // full-buffer and one partial per pair. See
  // `block_partition_accumulating.cuh` for the wave-decomposition design
  // notes -- full-waves count is `Capacity / BlockThreads` (compile-time
  // unrolled) plus an optional trailing partial wave.
  // ---------------------------------------------------------------
  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_full_buffer_cand()
  {
    static constexpr int full_flush_waves = CandidateBufferCapacity / BlockThreads;
    static constexpr int trailing_count   = CandidateBufferCapacity % BlockThreads;

    if (threadIdx.x == 0)
    {
      const auto r         = reserve_cand(static_cast<CandidateOffsetT>(CandidateBufferCapacity));
      temp_storage.cand.cnt.base    = r.first;
      temp_storage.cand.cnt.granted = static_cast<CandidateOffsetT>(r.second);
    }
    __syncthreads();

    const CandidateOffsetT base = temp_storage.cand.cnt.base;
    const CandidateOffsetT to_write =
      CandidateReserveOp::may_grant_less ? temp_storage.cand.cnt.granted : static_cast<CandidateOffsetT>(CandidateBufferCapacity);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int w = 0; w < full_flush_waves; ++w)
    {
      const int i = w * BlockThreads + static_cast<int>(threadIdx.x);
      if (!CandidateReserveOp::may_grant_less || static_cast<CandidateOffsetT>(i) < to_write)
      {
        cand_iter[base + static_cast<CandidateOffsetT>(i)] = cand_xform(temp_storage.cand.keys[i]);
      }
    }
    if constexpr (trailing_count != 0)
    {
      const int i = full_flush_waves * BlockThreads + static_cast<int>(threadIdx.x);
      if (static_cast<int>(threadIdx.x) < trailing_count
          && (!CandidateReserveOp::may_grant_less || static_cast<CandidateOffsetT>(i) < to_write))
      {
        cand_iter[base + static_cast<CandidateOffsetT>(i)] = cand_xform(temp_storage.cand.keys[i]);
      }
    }

    if constexpr (!keys_only)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int w = 0; w < full_flush_waves; ++w)
      {
        const int i = w * BlockThreads + static_cast<int>(threadIdx.x);
        if (!CandidateReserveOp::may_grant_less || static_cast<CandidateOffsetT>(i) < to_write)
        {
          sinks.candidate_values_out[base + static_cast<CandidateOffsetT>(i)] =
            sinks.candidate_value_transform(temp_storage.cand.values[i]);
        }
      }
      if constexpr (trailing_count != 0)
      {
        const int i = full_flush_waves * BlockThreads + static_cast<int>(threadIdx.x);
        if (static_cast<int>(threadIdx.x) < trailing_count
            && (!CandidateReserveOp::may_grant_less || static_cast<CandidateOffsetT>(i) < to_write))
        {
          sinks.candidate_values_out[base + static_cast<CandidateOffsetT>(i)] =
            sinks.candidate_value_transform(temp_storage.cand.values[i]);
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_partial_cand(int count)
  {
    if (threadIdx.x == 0)
    {
      const auto r         = reserve_cand(static_cast<CandidateOffsetT>(count));
      temp_storage.cand.cnt.base    = r.first;
      temp_storage.cand.cnt.granted = static_cast<CandidateOffsetT>(r.second);
    }
    __syncthreads();

    const CandidateOffsetT base = temp_storage.cand.cnt.base;
    const CandidateOffsetT to_write =
      CandidateReserveOp::may_grant_less ? temp_storage.cand.cnt.granted : static_cast<CandidateOffsetT>(count);

    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
    {
      cand_iter[base + static_cast<CandidateOffsetT>(i)] = cand_xform(temp_storage.cand.keys[i]);
    }
    if constexpr (!keys_only)
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
      {
        sinks.candidate_values_out[base + static_cast<CandidateOffsetT>(i)] =
          sinks.candidate_value_transform(temp_storage.cand.values[i]);
      }
    }
  }

  // Selected-stream flush primitives. Only instantiated when
  // `selected_buffered == true`.
  template <bool Enabled = selected_buffered, ::cuda::std::enable_if_t<Enabled, int> = 0>
  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_full_buffer_sel()
  {
    static constexpr int full_flush_waves = SelectedBufferCapacity / BlockThreads;
    static constexpr int trailing_count   = SelectedBufferCapacity % BlockThreads;

    if (threadIdx.x == 0)
    {
      const auto r        = reserve_sel(static_cast<SelectedOffsetT>(SelectedBufferCapacity));
      temp_storage.sel.cnt.base    = r.first;
      temp_storage.sel.cnt.granted = static_cast<SelectedOffsetT>(r.second);
    }
    __syncthreads();

    const SelectedOffsetT base = temp_storage.sel.cnt.base;
    const SelectedOffsetT to_write =
      SelectedReserveOp::may_grant_less ? temp_storage.sel.cnt.granted : static_cast<SelectedOffsetT>(SelectedBufferCapacity);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int w = 0; w < full_flush_waves; ++w)
    {
      const int i = w * BlockThreads + static_cast<int>(threadIdx.x);
      if (!SelectedReserveOp::may_grant_less || static_cast<SelectedOffsetT>(i) < to_write)
      {
        sel_iter[base + static_cast<SelectedOffsetT>(i)] = sel_xform(temp_storage.sel.keys[i]);
      }
    }
    if constexpr (trailing_count != 0)
    {
      const int i = full_flush_waves * BlockThreads + static_cast<int>(threadIdx.x);
      if (static_cast<int>(threadIdx.x) < trailing_count
          && (!SelectedReserveOp::may_grant_less || static_cast<SelectedOffsetT>(i) < to_write))
      {
        sel_iter[base + static_cast<SelectedOffsetT>(i)] = sel_xform(temp_storage.sel.keys[i]);
      }
    }

    if constexpr (!keys_only)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int w = 0; w < full_flush_waves; ++w)
      {
        const int i = w * BlockThreads + static_cast<int>(threadIdx.x);
        if (!SelectedReserveOp::may_grant_less || static_cast<SelectedOffsetT>(i) < to_write)
        {
          sinks.selected_values_out[base + static_cast<SelectedOffsetT>(i)] =
            sinks.selected_value_transform(temp_storage.sel.values[i]);
        }
      }
      if constexpr (trailing_count != 0)
      {
        const int i = full_flush_waves * BlockThreads + static_cast<int>(threadIdx.x);
        if (static_cast<int>(threadIdx.x) < trailing_count
            && (!SelectedReserveOp::may_grant_less || static_cast<SelectedOffsetT>(i) < to_write))
        {
          sinks.selected_values_out[base + static_cast<SelectedOffsetT>(i)] =
            sinks.selected_value_transform(temp_storage.sel.values[i]);
        }
      }
    }
  }

  template <bool Enabled = selected_buffered, ::cuda::std::enable_if_t<Enabled, int> = 0>
  _CCCL_DEVICE _CCCL_FORCEINLINE void cooperative_flush_partial_sel(int count)
  {
    if (threadIdx.x == 0)
    {
      const auto r        = reserve_sel(static_cast<SelectedOffsetT>(count));
      temp_storage.sel.cnt.base    = r.first;
      temp_storage.sel.cnt.granted = static_cast<SelectedOffsetT>(r.second);
    }
    __syncthreads();

    const SelectedOffsetT base = temp_storage.sel.cnt.base;
    const SelectedOffsetT to_write =
      SelectedReserveOp::may_grant_less ? temp_storage.sel.cnt.granted : static_cast<SelectedOffsetT>(count);

    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
    {
      sel_iter[base + static_cast<SelectedOffsetT>(i)] = sel_xform(temp_storage.sel.keys[i]);
    }
    if constexpr (!keys_only)
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(to_write); i += BlockThreads)
      {
        sinks.selected_values_out[base + static_cast<SelectedOffsetT>(i)] =
          sinks.selected_value_transform(temp_storage.sel.values[i]);
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
  ValueChannelSinksT& sinks;
  IdentifyCandidatesOp& identify_op;
  CandidateCallbackOp& callback_op;
};

//---------------------------------------------------------------------
// `strategy_to_partition_class` specialization for
// `block_partition_strategy::speculative_both`. Consumes both the per-stream
// capacity slots from the metafunction's template parameter list -- the
// candidate-stream buffer capacity is the shared `AccumulatingBufferCapacity`
// slot (reused with the accumulating partition), and the selected-stream
// buffer capacity is the dedicated `SpeculativeSelectedBufferCapacity` slot
// (`0` short-circuits the selected smem buffer to pure-Atomics).
//---------------------------------------------------------------------
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
          typename ValueChannelSinksT,
          typename ValueT,
          typename ValueDataSourceScratchT,
          bool LazyValueLoad,
          bool InlinedClassify>
struct strategy_to_partition_class<
  block_partition_strategy::speculative_both,
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
  ValueChannelSinksT,
  ValueT,
  ValueDataSourceScratchT,
  LazyValueLoad,
  InlinedClassify>
{
  using type = block_partition_speculative<
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
    ValueChannelSinksT,
    ValueT,
    LazyValueLoad,
    InlinedClassify>;
};

} // namespace detail::topk

CUB_NAMESPACE_END
