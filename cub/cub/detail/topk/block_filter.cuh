// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private non-accumulating filter primitives (single-stream siblings of the
//! `BlockPartition*` primitives in `block_partition.cuh`). Three self-contained
//! class templates -- one per filter strategy:
//!
//!   - `BlockFilterAtomics<..., InlinedClassify>` -- no smem; per-kept-item global
//!     atomic + scatter. `InlinedClassify == false` precomputes a `kept[ItemsPerThread]`
//!     register array up front; `InlinedClassify == true` recomputes the predicate
//!     at each scatter use-site.
//!   - `BlockFilterStaged` -- smem scatter into a keys arena + cooperative coalesced
//!     store; per-channel values run sequentially after the keys phase.
//!   - `BlockFilterSharedMem` -- typed `keys[]` + per-channel `values[]` packed
//!     into the same arena; a single coalesced store.
//!
//! Interface ("safe-both") contract shared with the accumulating sister class
//! `BlockFilterAccumulating` (`block_filter_accumulating.cuh`):
//!   - All sinks (reserve op, output iterator, transform, value-channel sink
//!     tuple) AND the `identify_selected_op` predicate are captured by ctor and
//!     stored as members. Per-call args reduce to per-tile data plus a bare
//!     `cuda::std::tuple<TileDataSource...>` for value sources.
//!   - `epilogue()` is argless on every variant. The three non-accumulating
//!     primitives' `epilogue()` is a no-op. The accumulating variant's `epilogue()`
//!     performs a terminal flush of any remaining buffered items.
//!
//! Strategy selection is done by `strategy_to_filter_class_t<Strategy, ...>` in
//! `block_filter_accumulating.cuh`, which maps a `BlockFilterStrategy` enum value
//! to one of the three classes here (or to the accumulating sister class).

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
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
//---------------------------------------------------------------------
// Strategy selector for the filter primitives. Mirrors `BlockPartitionStrategy`:
// picks the *filtering* shape; the orthogonal `InlinedClassify` axis is a
// separate template / policy bool that every non-accumulating primitive accepts.
// The mapping from a strategy enum value to a class is performed by
// `strategy_to_filter_class_t<...>` in `block_filter_accumulating.cuh`.
//---------------------------------------------------------------------
enum class BlockFilterStrategy
{
  Atomics,
  Staged,
  SharedMem,
  AccumulatingFilter,
  // SpeculativeFilter accumulates the selected stream in a fixed-size smem
  // buffer, but uses a *speculative* slot reservation: items whose atomicAdd
  // index lands within the buffer go to smem, items beyond capacity fall back
  // to per-item global atomics (Atomics-equivalent). The trade is one extra
  // per-thread uint32 bitmask and one extra `__syncthreads()` per partition()
  // call in exchange for keeping `positions[]` cross-iteration-dead, which
  // restores register parity with `Atomics` while preserving the cooperative
  // batched flush on sparse streams. See `block_filter_speculative.cuh`.
  SpeculativeFilter,
};

//---------------------------------------------------------------------
// Per-channel value-sink bundle for the single-stream filter primitives.
// Sibling of `value_channel_sinks_t` in `block_partition.cuh`, but holding only
// the selected-stream iter + transform (the filter has no candidate stream).
//---------------------------------------------------------------------
template <typename SelectedValuesOutIt, typename SelectedValueTransform>
struct value_channel_sinks_filter_t
{
  SelectedValuesOutIt selected_values_out;
  SelectedValueTransform selected_value_transform;
};

//---------------------------------------------------------------------
// Shared scratch-storage building blocks. Per-strategy assembled `ScratchStorage`
// structs live as nested types of the three filter classes below; this
// `bf_detail` namespace just holds the pieces (single-stream counters,
// classifiers) that they compose from. The shared multi-channel slots
// (`bp_detail::staged_channel_phase`, `bp_detail::delegate_load_slot`,
// `bp_detail::values_slot`, `bp_detail::map_tuple_t`, ...) live in
// `block_partition.cuh` and are reused here.
//---------------------------------------------------------------------

namespace bf_detail
{
// Single-stream counter + broadcast slots for `Staged` / `SharedMem` filter
// strategies. Phase 1 uses a 32-bit smem atomic (`counter`); Phase 2 reuses the
// same word as the global base via a union (separated by `__syncthreads()`).
// `granted_*` lives outside the union (broadcast across all threads).
template <typename SelectedOffsetT>
struct filter_counters
{
  union
  {
    int counter;
    SelectedOffsetT global_base;
  };
  SelectedOffsetT granted_selected;
};

// Adapter so the atomics fused scatter can be a single function template over an
// "indexed classifier" with signature `(KeyT, int j) -> bool`. The inlined wrapper
// encapsulates the `is_valid` partial-tile check.
template <bool IsFull, typename Op>
struct inlined_filter_classifier
{
  Op& op;
  int num_thread_items;

  template <typename KeyT>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(const KeyT& key, int j) const
  {
    if constexpr (IsFull)
    {
      (void) j;
      return op(key);
    }
    else
    {
      return (j < num_thread_items) ? static_cast<bool>(op(key)) : false;
    }
  }
};

template <bool IsFull, typename Op>
_CCCL_DEVICE _CCCL_FORCEINLINE inlined_filter_classifier<IsFull, Op>
make_inlined_filter_classifier(Op& op, int num_thread_items)
{
  return inlined_filter_classifier<IsFull, Op>{op, num_thread_items};
}

// Precomputed-classes adapter: at construction runs the predicate over the
// per-thread keys array; `operator()(KeyT, int j)` returns the cached bool.
// Items past `num_thread_items` (partial path) are forced to `false`. Reusable
// across `BlockFilterAtomics<InlinedClassify=false>`, `BlockFilterStaged`,
// and `BlockFilterSharedMem` -- the latter two consume `.kept` directly.
template <typename KeyT, int ItemsPerThread, bool IsFull>
struct precomputed_filter_classifier
{
  bool kept[ItemsPerThread];

  template <typename Op>
  _CCCL_DEVICE _CCCL_FORCEINLINE
  precomputed_filter_classifier(const KeyT (&keys)[ItemsPerThread], int num_thread_items, Op& op)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const bool is_valid = IsFull ? true : (j < num_thread_items);
      kept[j]             = is_valid ? static_cast<bool>(op(keys[j])) : false;
    }
  }

  template <typename KeyT_>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(const KeyT_& /*key*/, int j) const
  {
    return kept[j];
  }
};

} // namespace bf_detail

//---------------------------------------------------------------------
// `BlockFilterAtomics` -- per-kept-item global atomic + scatter, no smem.
// `InlinedClassify` selects between the precomputed-classes form (materializes a
// `kept[]` register array up front) and the inlined-classify form (recomputes the
// predicate at each scatter use-site, frees the registers that would hold
// `kept[]`). Mapped from `BlockFilterStrategy::Atomics`. The `InlinedClassify`
// axis is independent and is also accepted (with the same semantics) by
// `BlockFilterStaged` and `BlockFilterSharedMem`.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          bool InlinedClassify,
          typename KeyT,
          typename SelectedOffsetT,
          typename SelectedReserveOp,
          typename SelectedKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksTuple      = ::cuda::std::tuple<>,
          typename ValueTypesTuple             = ::cuda::std::tuple<>,
          typename DataSourceScratchTypesTuple = ::cuda::std::tuple<>,
          bool LazyValueLoad                   = false>
class BlockFilterAtomics
{
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueTypesTuple>::value
                  == ::cuda::std::tuple_size<DataSourceScratchTypesTuple>::value,
                "ValueTypesTuple and DataSourceScratchTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value <= 1,
                "BlockFilterAtomics supports keys-only or single-value-channel today; "
                "multi-channel needs a per-channel value array.");

public:
  static constexpr int tile_items         = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  // Class-lifetime persistent state. Empty (no carried state across partition() calls).
  struct TempStorage
  {};

  // Per-tile scratch. Empty: the atomics strategies hold no smem state -- per-item
  // scatter goes direct to the user's iterator via the captured reserve op.
  struct ScratchStorage
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockFilterAtomics(
    TempStorage& /*storage*/,
    SelectedReserveOp& reserve_selected,
    SelectedKeyOutTransformOp& selected_key_transform,
    SelectedKeyOutIt selected_keys_out,
    ValueChannelSinksTuple& value_channel_sinks,
    IdentifySelectedOp& identify_selected_op)
      : reserve_sel_(reserve_selected)
      , sel_xform_(selected_key_transform)
      , sel_iter_(selected_keys_out)
      , sinks_(value_channel_sinks)
      , identify_op_(identify_selected_op)
  {}

  // Full-tile overload: no per-item bound check inside the classify loop.
  template <typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourcesTuple& value_sources)
  {
    filter_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_sources);
  }

  // Partial-tile overload: classify loop bound-checks against num_items.
  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourcesTuple& value_sources)
  {
    filter_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_sources);
  }

  // No-op terminal flush. Present for parity with `BlockFilterAccumulating`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the filter pairs them positionally.");

    const int num_thread_items = bp_detail::compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    if constexpr (InlinedClassify)
    {
      auto classifier = bf_detail::make_inlined_filter_classifier<IsFull>(identify_op_, num_thread_items);
      filter_atomics_fused<IsFull>(buffer, keys, num_thread_items, classifier, value_sources);
    }
    else
    {
      bf_detail::precomputed_filter_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op_};
      filter_atomics_fused<IsFull>(buffer, keys, num_thread_items, classifier, value_sources);
    }
  }

  // -----------------------------------------------------------------
  // Fused scatter: drives a unified per-item loop via a user-supplied
  // `Classifier` with signature `(KeyT, int j) -> bool`. The classifier
  // abstracts the precomputed-vs-inlined decision.
  // -----------------------------------------------------------------
  template <bool IsFull, typename Classifier, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_atomics_fused(
    ScratchStorage& /*buffer*/,
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    ValueSourcesTuple& value_sources)
  {
    if constexpr (num_value_channels == 1)
    {
      auto& src      = ::cuda::std::get<0>(value_sources);
      using source_t = ::cuda::std::remove_reference_t<decltype(src)>;
      using value_t  = ::cuda::std::tuple_element_t<0, ValueTypesTuple>;
      static_assert(::cuda::std::is_same_v<typename source_t::value_t, value_t>,
                    "Per-call value source's value_t must match the class-level ValueTypesTuple element.");

      if constexpr (LazyValueLoad)
      {
        int unused_values[1]{};
        filter_atomics_fused_scatter<IsFull, /*KeysOnly=*/false>(
          keys, num_thread_items, classifier, value_sources, unused_values);
      }
      else
      {
        typename source_t::ScratchStorage chan_scratch{};
        auto h = src.submit_load(chan_scratch);
        value_t values[ItemsPerThread]{};
        h.complete_load(values);

        filter_atomics_fused_scatter<IsFull, /*KeysOnly=*/false>(
          keys, num_thread_items, classifier, value_sources, values);
      }
    }
    else
    {
      int unused_dummy[1]{};
      (void) unused_dummy;
      filter_atomics_fused_scatter<IsFull, /*KeysOnly=*/true>(
        keys, num_thread_items, classifier, value_sources, unused_dummy);
    }
  }

  template <bool IsFull, bool KeysOnly, typename Classifier, typename ValueSourcesTuple, typename ValuesArr>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_atomics_fused_scatter(
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    ValueSourcesTuple& value_sources,
    ValuesArr& values)
  {
    (void) num_thread_items;
    (void) value_sources;

    auto get_value = [&](int j) {
      if constexpr (LazyValueLoad)
      {
        auto& src = ::cuda::std::get<0>(value_sources);
        return src.gather_one(j);
      }
      else
      {
        return values[j];
      }
    };

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const bool keep = classifier(keys[j], j);
      if (keep)
      {
        const auto r = reserve_sel_(SelectedOffsetT{1});
        bool granted = true;
        if constexpr (SelectedReserveOp::may_grant_less)
        {
          granted = (r.second != SelectedOffsetT{0});
        }
        if (granted)
        {
          sel_iter_[r.first] = sel_xform_(keys[j]);
          if constexpr (!KeysOnly)
          {
            auto& sink                        = ::cuda::std::get<0>(sinks_);
            sink.selected_values_out[r.first] = sink.selected_value_transform(get_value(j));
          }
        }
      }
    }
  }

  // Captured at ctor; used by every partition() call.
  SelectedReserveOp& reserve_sel_;
  SelectedKeyOutTransformOp& sel_xform_;
  SelectedKeyOutIt sel_iter_;
  ValueChannelSinksTuple& sinks_;
  IdentifySelectedOp& identify_op_;
};

//---------------------------------------------------------------------
// `BlockFilterStaged` -- smem scatter into a keys arena + cooperative coalesced
// store. Per-channel value path runs sequentially after the keys phase: each
// channel loads (sub-brokered scratch), scatters into the channel's `values[]`
// slot, then cooperatively stores. Mapped from `BlockFilterStrategy::Staged`.
//
// `InlinedClassify` selects between materializing a `kept[ItemsPerThread]`
// register array up front and recomputing the predicate at the per-item
// smem-scatter site. The filter has no candidate callback so the choice is
// purely about register pressure vs recomputation.
//
// `LazyValueLoad` selects between eagerly loading a tile of values into a
// register array up front (default) and gathering only the surviving values
// directly into smem via the source's `gather_one(j)` operation. Only honored
// when the value source supports `gather_one`.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          typename KeyT,
          typename SelectedOffsetT,
          typename SelectedReserveOp,
          typename SelectedKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksTuple      = ::cuda::std::tuple<>,
          typename ValueTypesTuple             = ::cuda::std::tuple<>,
          typename DataSourceScratchTypesTuple = ::cuda::std::tuple<>,
          bool LazyValueLoad                   = false,
          bool InlinedClassify                 = false>
class BlockFilterStaged
{
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueTypesTuple>::value
                  == ::cuda::std::tuple_size<DataSourceScratchTypesTuple>::value,
                "ValueTypesTuple and DataSourceScratchTypesTuple must have the same length.");

public:
  static constexpr int tile_items         = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  struct TempStorage
  {};

  using value_channel_meta_tuple_t =
    bp_detail::zip_value_channel_metas_t<ValueTypesTuple, DataSourceScratchTypesTuple>;

  // Per-tile scratch. `phase` is a phase_union: phase 1 (key scatter) uses the
  // `keys[]` arena; phase 2 (per-channel value scatter) reuses the same smem
  // through the `per_channel` view. `cnt` lives outside the union because it
  // carries the per-tile count and the broadcast `granted_*` slot across the
  // whole `partition()` call.
  struct ScratchStorage
  {
    union phase_t
    {
      KeyT keys[tile_items];
      CUB_NS_QUALIFIER::detail::phase_union<
        bp_detail::map_tuple_t<bp_detail::staged_channel_phase, value_channel_meta_tuple_t, tile_items>>
        per_channel;

      _CCCL_HOST_DEVICE phase_t() {}
      _CCCL_HOST_DEVICE ~phase_t() {}
    } phase;

    bf_detail::filter_counters<SelectedOffsetT> cnt;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockFilterStaged(
    TempStorage& /*storage*/,
    SelectedReserveOp& reserve_selected,
    SelectedKeyOutTransformOp& selected_key_transform,
    SelectedKeyOutIt selected_keys_out,
    ValueChannelSinksTuple& value_channel_sinks,
    IdentifySelectedOp& identify_selected_op)
      : reserve_sel_(reserve_selected)
      , sel_xform_(selected_key_transform)
      , sel_iter_(selected_keys_out)
      , sinks_(value_channel_sinks)
      , identify_op_(identify_selected_op)
  {}

  template <typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourcesTuple& value_sources)
  {
    filter_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_sources);
  }

  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourcesTuple& value_sources)
  {
    filter_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_sources);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the filter pairs them positionally.");

    const int num_thread_items = bp_detail::compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    int positions[ItemsPerThread];

    if (threadIdx.x == 0)
    {
      buffer.cnt.counter = 0;
    }
    __syncthreads();

    if constexpr (InlinedClassify)
    {
      auto classifier = bf_detail::make_inlined_filter_classifier<IsFull>(identify_op_, num_thread_items);
      classify_and_scatter_keys(buffer, keys, classifier, positions);
    }
    else
    {
      bf_detail::precomputed_filter_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op_};
      classify_and_scatter_keys(buffer, keys, classifier, positions);
    }
    __syncthreads();

    const int selected_cnt = buffer.cnt.counter;

    if (threadIdx.x == 0)
    {
      const auto sel              = reserve_sel_(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_base      = sel.first;
      buffer.cnt.granted_selected = static_cast<SelectedOffsetT>(sel.second);
    }
    __syncthreads();

    const SelectedOffsetT sel_base = buffer.cnt.global_base;
    const SelectedOffsetT sel_to_write =
      SelectedReserveOp::may_grant_less ? buffer.cnt.granted_selected : static_cast<SelectedOffsetT>(selected_cnt);

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter_[sel_base + static_cast<SelectedOffsetT>(i)] = sel_xform_(buffer.phase.keys[i]);
    }

    if constexpr (num_value_channels > 0)
    {
      __syncthreads();
      bp_detail::tuple_for_each(value_sources, [&](auto& src, auto I_ic) {
        constexpr int I = static_cast<int>(decltype(I_ic)::value);
        using source_t  = ::cuda::std::remove_reference_t<decltype(src)>;
        using value_t   = ::cuda::std::tuple_element_t<I, ValueTypesTuple>;
        static_assert(::cuda::std::is_same_v<typename source_t::value_t, value_t>,
                      "Per-call value source's value_t must match the class-level ValueTypesTuple element.");

        auto& sink       = ::cuda::std::get<I>(sinks_);
        auto& chan_phase = CUB_NS_QUALIFIER::detail::at<I>(buffer.phase.per_channel);
        if constexpr (LazyValueLoad)
        {
          // Lazy: skip the eager tile-wide load. Each thread gathers only the
          // values of its surviving items via `src.gather_one(j)` and writes
          // them straight into the smem arena. The data-source's `load` slot
          // (aliased with `chan_phase.values` via the phase union) goes
          // unused on this path.
          (void) num_items;
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int j = 0; j < ItemsPerThread; ++j)
          {
            if (positions[j] >= 0)
            {
              chan_phase.values[positions[j]] = src.gather_one(j);
            }
          }
        }
        else
        {
          // Eager: load the full tile into a register array, then scatter the
          // surviving values into the smem arena.
          value_t reg_values[ItemsPerThread]{};
          if constexpr (IsFull)
          {
            auto h = src.submit_load(chan_phase.load);
            h.complete_load(reg_values);
          }
          else
          {
            auto h = src.submit_load(chan_phase.load, num_items);
            h.complete_load(reg_values);
          }
          __syncthreads();

          _CCCL_PRAGMA_UNROLL_FULL()
          for (int j = 0; j < ItemsPerThread; ++j)
          {
            if (positions[j] >= 0)
            {
              chan_phase.values[positions[j]] = reg_values[j];
            }
          }
        }
        __syncthreads();

        for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
        {
          sink.selected_values_out[sel_base + static_cast<SelectedOffsetT>(i)] =
            sink.selected_value_transform(chan_phase.values[i]);
        }
        __syncthreads();
      });
    }
    __syncthreads();
  }

  // Mode-agnostic Phase 1 body: classify every per-thread item and scatter the
  // kept ones into the smem arena. `positions[j]` records the smem slot for use
  // by the value channels; `-1` marks dropped items. `Classifier` exposes
  // `operator()(KeyT, int j) -> bool` for both inlined and precomputed modes.
  template <typename Classifier>
  _CCCL_DEVICE _CCCL_FORCEINLINE void classify_and_scatter_keys(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], Classifier& classifier, int (&positions)[ItemsPerThread])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (classifier(keys[j], j))
      {
        const int pos          = atomicAdd(&buffer.cnt.counter, 1);
        buffer.phase.keys[pos] = keys[j];
        positions[j]           = pos;
      }
      else
      {
        positions[j] = -1;
      }
    }
  }

  SelectedReserveOp& reserve_sel_;
  SelectedKeyOutTransformOp& sel_xform_;
  SelectedKeyOutIt sel_iter_;
  ValueChannelSinksTuple& sinks_;
  IdentifySelectedOp& identify_op_;
};

//---------------------------------------------------------------------
// `BlockFilterSharedMem` -- keys + per-channel values coexist in smem (within
// `phase.kv`), then a single coalesced flush. Pre-Phase-1 delegate loads alias
// with the kv arena via the top-level phase union. Mapped from
// `BlockFilterStrategy::SharedMem`.
//
// Single-value-channel only today; multi-channel needs a heterogeneous
// register-array tuple.
//
// `InlinedClassify` selects between materializing a `kept[ItemsPerThread]`
// register array up front and recomputing the predicate at the per-item
// kv-scatter site.
//
// `LazyValueLoad` selects between the eager pre-Phase-1 delegate load (default)
// and gathering only the surviving values directly into the kv arena via the
// source's `gather_one(j)` operation. Only honored when the value source
// supports `gather_one`.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          typename KeyT,
          typename SelectedOffsetT,
          typename SelectedReserveOp,
          typename SelectedKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksTuple      = ::cuda::std::tuple<>,
          typename ValueTypesTuple             = ::cuda::std::tuple<>,
          typename DataSourceScratchTypesTuple = ::cuda::std::tuple<>,
          bool LazyValueLoad                   = false,
          bool InlinedClassify                 = false>
class BlockFilterSharedMem
{
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueTypesTuple>::value
                  == ::cuda::std::tuple_size<DataSourceScratchTypesTuple>::value,
                "ValueTypesTuple and DataSourceScratchTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value <= 1,
                "BlockFilterSharedMem supports keys-only or single-value-channel today; "
                "multi-channel needs a heterogeneous register-array tuple.");

public:
  static constexpr int tile_items         = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  struct TempStorage
  {};

  using value_channel_meta_tuple_t =
    bp_detail::zip_value_channel_metas_t<ValueTypesTuple, DataSourceScratchTypesTuple>;

  // Per-tile scratch. The `phase` union has two views: `delegate_loads` (pre-Phase-1
  // delegate-load staging area for value channels) and `kv` (the keys + per-channel
  // values arena used for scatter and cooperative flush). `cnt` lives outside the
  // union, same role as in BlockFilterStaged.
  struct ScratchStorage
  {
    struct keys_and_values_t
    {
      KeyT keys[tile_items];
      CUB_NS_QUALIFIER::detail::phase_aggregate<
        bp_detail::map_tuple_t<bp_detail::values_slot, value_channel_meta_tuple_t, tile_items>>
        per_channel_values;
    };

    union phase_t
    {
      CUB_NS_QUALIFIER::detail::phase_union<bp_detail::map_tuple1_t<bp_detail::delegate_load_slot, value_channel_meta_tuple_t>>
        delegate_loads;
      keys_and_values_t kv;

      _CCCL_HOST_DEVICE phase_t() {}
      _CCCL_HOST_DEVICE ~phase_t() {}
    } phase;

    bf_detail::filter_counters<SelectedOffsetT> cnt;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockFilterSharedMem(
    TempStorage& /*storage*/,
    SelectedReserveOp& reserve_selected,
    SelectedKeyOutTransformOp& selected_key_transform,
    SelectedKeyOutIt selected_keys_out,
    ValueChannelSinksTuple& value_channel_sinks,
    IdentifySelectedOp& identify_selected_op)
      : reserve_sel_(reserve_selected)
      , sel_xform_(selected_key_transform)
      , sel_iter_(selected_keys_out)
      , sinks_(value_channel_sinks)
      , identify_op_(identify_selected_op)
  {}

  template <typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourcesTuple& value_sources)
  {
    filter_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_sources);
  }

  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourcesTuple& value_sources)
  {
    filter_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_sources);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the filter pairs them positionally.");

    const int num_thread_items = bp_detail::compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    // Pre-Phase-1: when eagerly loading, fetch the channel's values into a register
    // array via the delegate-load slot. After this, the delegate_loads view of the
    // phase union is dead and we transition to the kv view at the next
    // __syncthreads. Skipped under `LazyValueLoad` -- the kv-scatter loop below
    // pulls values via `gather_one(j)` only for surviving items.
    using channel_value_t = typename bp_detail::value_t_or_default<ValueTypesTuple>::type;
    channel_value_t reg_values[ItemsPerThread]{};
    if constexpr (num_value_channels == 1 && !LazyValueLoad)
    {
      auto& src       = ::cuda::std::get<0>(value_sources);
      auto& load_slot = CUB_NS_QUALIFIER::detail::at<0>(buffer.phase.delegate_loads);
      if constexpr (IsFull)
      {
        auto h = src.submit_load(load_slot.load);
        h.complete_load(reg_values);
      }
      else
      {
        auto h = src.submit_load(load_slot.load, num_items);
        h.complete_load(reg_values);
      }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
      buffer.cnt.counter = 0;
    }
    __syncthreads();

    if constexpr (InlinedClassify)
    {
      auto classifier = bf_detail::make_inlined_filter_classifier<IsFull>(identify_op_, num_thread_items);
      classify_and_scatter_kv(buffer, keys, classifier, value_sources, reg_values);
    }
    else
    {
      bf_detail::precomputed_filter_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op_};
      classify_and_scatter_kv(buffer, keys, classifier, value_sources, reg_values);
    }
    __syncthreads();

    const int selected_cnt = buffer.cnt.counter;

    if (threadIdx.x == 0)
    {
      const auto sel              = reserve_sel_(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_base      = sel.first;
      buffer.cnt.granted_selected = static_cast<SelectedOffsetT>(sel.second);
    }
    __syncthreads();

    const SelectedOffsetT sel_base = buffer.cnt.global_base;
    const SelectedOffsetT sel_to_write =
      SelectedReserveOp::may_grant_less ? buffer.cnt.granted_selected : static_cast<SelectedOffsetT>(selected_cnt);

    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter_[sel_base + static_cast<SelectedOffsetT>(i)] = sel_xform_(buffer.phase.kv.keys[i]);
    }

    if constexpr (num_value_channels > 0)
    {
      bp_detail::tuple_for_each(sinks_, [&](auto& sink, auto I_ic) {
        constexpr int I = static_cast<int>(decltype(I_ic)::value);
        auto& vs        = CUB_NS_QUALIFIER::detail::at<I>(buffer.phase.kv.per_channel_values);
        for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
        {
          sink.selected_values_out[sel_base + static_cast<SelectedOffsetT>(i)] =
            sink.selected_value_transform(vs.values[i]);
        }
      });
    }
    __syncthreads();
  }

  // Mode-agnostic Phase 1 body: classify every per-thread item and scatter the
  // surviving (key, value) pairs into the kv arena. `Classifier` exposes
  // `operator()(KeyT, int j) -> bool` for both modes; the value source is
  // consulted via `gather_one` only when `LazyValueLoad == true`.
  template <typename Classifier, typename ValueSourcesTuple, typename RegValuesArr>
  _CCCL_DEVICE _CCCL_FORCEINLINE void classify_and_scatter_kv(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    Classifier& classifier,
    ValueSourcesTuple& value_sources,
    const RegValuesArr& reg_values)
  {
    (void) value_sources;
    (void) reg_values;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (!classifier(keys[j], j))
      {
        continue;
      }
      const int idx             = atomicAdd(&buffer.cnt.counter, 1);
      buffer.phase.kv.keys[idx] = keys[j];
      if constexpr (num_value_channels == 1)
      {
        if constexpr (LazyValueLoad)
        {
          auto& src = ::cuda::std::get<0>(value_sources);
          CUB_NS_QUALIFIER::detail::at<0>(buffer.phase.kv.per_channel_values).values[idx] = src.gather_one(j);
        }
        else
        {
          CUB_NS_QUALIFIER::detail::at<0>(buffer.phase.kv.per_channel_values).values[idx] = reg_values[j];
        }
      }
    }
  }

  SelectedReserveOp& reserve_sel_;
  SelectedKeyOutTransformOp& sel_xform_;
  SelectedKeyOutIt sel_iter_;
  ValueChannelSinksTuple& sinks_;
  IdentifySelectedOp& identify_op_;
};
} // namespace detail::topk

CUB_NAMESPACE_END
