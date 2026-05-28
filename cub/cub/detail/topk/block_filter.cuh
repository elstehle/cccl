// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private non-accumulating filter primitives (single-stream siblings of the
//! `BlockPartition*` primitives in `block_partition.cuh`). Three self-contained
//! class templates -- one per filter strategy:
//!
//!   - `block_filter_atomics<..., InlinedClassify>` -- no smem; per-kept-item global
//!     atomic + scatter. `InlinedClassify == false` precomputes a `kept[ItemsPerThread]`
//!     register array up front; `InlinedClassify == true` recomputes the predicate
//!     at each scatter use-site.
//!   - `block_filter_staged` -- smem scatter into a keys arena + cooperative coalesced
//!     store; per-channel values run sequentially after the keys phase.
//!   - `block_filter_shared_mem` -- typed `keys[]` + per-channel `values[]` packed
//!     into the same arena; a single coalesced store.
//!
//! Interface ("safe-both") contract shared with the accumulating sister class
//! `block_filter_accumulating` (`block_filter_accumulating.cuh`):
//!   - All sinks (reserve op, output iterator, transform, value-channel sink
//!     tuple) AND the `identify_selected_op` predicate are captured by ctor and
//!     stored as members. Per-call args reduce to per-tile data plus a bare
//!     `cuda::std::tuple<TileDataSource...>` for value sources.
//!   - `epilogue()` is argless on every variant. The three non-accumulating
//!     primitives' `epilogue()` is a no-op. The accumulating variant's `epilogue()`
//!     performs a terminal flush of any remaining buffered items.
//!
//! Strategy selection is done by `strategy_to_filter_class_t<Strategy, ...>` in
//! `block_filter_accumulating.cuh`, which maps a `block_filter_strategy` enum value
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
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
//---------------------------------------------------------------------
// Strategy selector for the filter primitives. Mirrors `block_partition_strategy`:
// picks the *filtering* shape; the orthogonal `InlinedClassify` axis is a
// separate template / policy bool that every non-accumulating primitive accepts.
// The mapping from a strategy enum value to a class is performed by
// `strategy_to_filter_class_t<...>` in `block_filter_accumulating.cuh`.
//---------------------------------------------------------------------
enum class block_filter_strategy
{
  atomics,
  staged,
  shared_mem,
  accumulating_filter,
  // SpeculativeFilter accumulates the selected stream in a fixed-size smem
  // buffer, but uses a *speculative* slot reservation: items whose atomicAdd
  // index lands within the buffer go to smem, items beyond capacity fall back
  // to per-item global atomics (Atomics-equivalent). The trade is one extra
  // per-thread uint32 bitmask and one extra `__syncthreads()` per partition()
  // call in exchange for keeping `positions[]` cross-iteration-dead, which
  // restores register parity with `Atomics` while preserving the cooperative
  // batched flush on sparse streams. See `block_filter_speculative.cuh`.
  speculative_filter,
};

//---------------------------------------------------------------------
// Per-channel value-sink bundle for the single-stream filter primitives.
// Sibling of `value_channel_sinks_t` in `block_partition.cuh`, but holding
// only the selected-stream output iterator. See the note on the partition
// sibling for the rationale behind dropping the previous identity-only
// `selected_value_transform`.
//---------------------------------------------------------------------
template <typename SelectedValuesOutIt>
struct value_channel_sinks_filter_t
{
  SelectedValuesOutIt selected_values_out;
};

//---------------------------------------------------------------------
// Shared scratch-storage building blocks. Per-strategy assembled `ScratchStorage`
// structs live as nested types of the three filter classes below; the helpers
// here (single-stream counters, classifiers) are what they compose from.
//---------------------------------------------------------------------

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
// across `block_filter_atomics<InlinedClassify=false>`, `block_filter_staged`,
// and `block_filter_shared_mem` -- the latter two consume `.kept` directly.
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


//---------------------------------------------------------------------
// `block_filter_atomics` -- per-kept-item global atomic + scatter, no smem.
// `InlinedClassify` selects between the precomputed-classes form (materializes a
// `kept[]` register array up front) and the inlined-classify form (recomputes the
// predicate at each scatter use-site, frees the registers that would hold
// `kept[]`). Mapped from `block_filter_strategy::atomics`. The `InlinedClassify`
// axis is independent and is also accepted (with the same semantics) by
// `block_filter_staged` and `block_filter_shared_mem`.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          bool InlinedClassify,
          typename KeyT,
          typename SelectedOffsetT,
          typename SelectedReserveOp,
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksT      = CUB_NS_QUALIFIER::NullType,
          typename ValueT                  = CUB_NS_QUALIFIER::NullType,
          typename ValueDataSourceScratchT = CUB_NS_QUALIFIER::NullType,
          bool LazyValueLoad               = false>
class block_filter_atomics
{
public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;

  // Class-lifetime persistent state. Empty (no carried state across partition() calls).
  struct TempStorage
  {};

  // Per-tile scratch. The atomics strategies hold no scatter-side smem (per-item
  // scatter goes direct to the user's iterator via the captured reserve op), but
  // they DO own the per-tile load scratch for the value channel: see the matching
  // doc on `block_partition_atomics::ScratchStorage` for the full rationale. The
  // struct is wrapped in `cub::Uninitialized<>` so it can sit in `__shared__` even
  // when `ValueDataSourceScratchT` carries a non-trivial union ctor / dtor.
private:
  struct _ScratchStorage
  {
    ValueDataSourceScratchT value_load;
  };

public:
  struct ScratchStorage : CUB_NS_QUALIFIER::Uninitialized<_ScratchStorage>
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE block_filter_atomics(
    TempStorage& /*storage*/,
    SelectedReserveOp reserve_selected,
    SelectedKeyOutIt selected_keys_out,
    ValueChannelSinksT value_channel_sinks,
    IdentifySelectedOp& identify_selected_op)
      : reserve_sel(reserve_selected)
      
      , sel_iter(selected_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_selected_op)
  {}

  // Full-tile overload: no per-item bound check inside the classify loop.
  template <typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourceT& value_source)
  {
    filter_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_source);
  }

  // Partial-tile overload: classify loop bound-checks against num_items.
  template <typename NumItemsT, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourceT& value_source)
  {
    filter_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_source);
  }

  // No-op terminal flush. Present for parity with `block_filter_accumulating`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourceT& value_source)
  {
    const int num_thread_items = compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    if constexpr (InlinedClassify)
    {
      auto classifier = make_inlined_filter_classifier<IsFull>(identify_op, num_thread_items);
      filter_atomics_fused<IsFull>(buffer, keys, num_thread_items, classifier, value_source);
    }
    else
    {
      precomputed_filter_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op};
      filter_atomics_fused<IsFull>(buffer, keys, num_thread_items, classifier, value_source);
    }
  }

  // -----------------------------------------------------------------
  // Fused scatter: drives a unified per-item loop via a user-supplied
  // `Classifier` with signature `(KeyT, int j) -> bool`. The classifier
  // abstracts the precomputed-vs-inlined decision.
  // -----------------------------------------------------------------
  template <bool IsFull, typename Classifier, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_atomics_fused(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    ValueSourceT& value_source)
  {
    if constexpr (!keys_only)
    {
      static_assert(::cuda::std::is_same_v<typename ValueSourceT::value_t, ValueT>,
                    "Per-call value source's value_t must match the class-level ValueT template parameter.");
      static_assert(::cuda::std::is_same_v<typename ValueSourceT::ScratchStorage, ValueDataSourceScratchT>,
                    "Per-call value source's ScratchStorage must match the class-level "
                    "ValueDataSourceScratchT template parameter (the agent is responsible for picking "
                    "the smem-backed scratch type up front so it can flow through ScratchStorage).");

      if constexpr (LazyValueLoad)
      {
        int unused_values[1]{};
        filter_atomics_fused_scatter<IsFull>(
          keys, num_thread_items, classifier, value_source, unused_values);
      }
      else
      {
        // Smem-backed scratch for the value-channel load. See the matching comment
        // in `block_partition_atomics::partition_atomics_fused`.
        auto& chan_scratch = buffer.Alias().value_load;
        auto h             = value_source.submit_load(chan_scratch);
        ValueT values[ItemsPerThread]{};
        h.complete_load(values);

        filter_atomics_fused_scatter<IsFull>(
          keys, num_thread_items, classifier, value_source, values);
      }
    }
    else
    {
      int unused_dummy[1]{};
      filter_atomics_fused_scatter<IsFull>(keys, num_thread_items, classifier, value_source, unused_dummy);
    }
  }

  template <bool IsFull, typename Classifier, typename ValueSourceT, typename ValuesArr>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_atomics_fused_scatter(
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    ValueSourceT& value_source,
    ValuesArr& values)
  {
    (void) num_thread_items;
    (void) value_source;
    (void) values;

    auto get_value = [&](int j) {
      if constexpr (LazyValueLoad)
      {
        return value_source.gather_one(j);
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
        const auto r = reserve_sel(SelectedOffsetT{1});
        bool granted = true;
        if constexpr (SelectedReserveOp::may_grant_less)
        {
          granted = (r.second != SelectedOffsetT{0});
        }
        if (granted)
        {
          sel_iter[r.first] = keys[j];
          if constexpr (!keys_only)
          {
            sinks.selected_values_out[r.first] = get_value(j);
          }
        }
      }
    }
  }

  // Captured at ctor; used by every partition() call.
  SelectedReserveOp reserve_sel;
  SelectedKeyOutIt sel_iter;
  ValueChannelSinksT sinks;
  IdentifySelectedOp& identify_op;
};

//---------------------------------------------------------------------
// `block_filter_staged` -- smem scatter into a keys arena + cooperative coalesced
// store. Per-channel value path runs sequentially after the keys phase: each
// channel loads (sub-brokered scratch), scatters into the channel's `values[]`
// slot, then cooperatively stores. Mapped from `block_filter_strategy::staged`.
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
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksT      = CUB_NS_QUALIFIER::NullType,
          typename ValueT                  = CUB_NS_QUALIFIER::NullType,
          typename ValueDataSourceScratchT = CUB_NS_QUALIFIER::NullType,
          bool LazyValueLoad               = false,
          bool InlinedClassify             = false>
class block_filter_staged
{
public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;

  struct TempStorage
  {};

  // Per-tile scratch. `phase` is a phase union: phase 1 (key scatter) uses the
  // `keys[]` arena; phase 2 (value scatter) reuses the same smem through the
  // `value_phase` view, which itself is an internal union over the data source's
  // `load` scratch and the per-tile `values[]` array. `cnt` lives outside the
  // union because it carries the per-tile count and the broadcast `granted_*`
  // slot across the whole `partition()` call. Mirrors the sister layout in
  // `block_partition_staged`. Both phase unions are wrapped in
  // `cub::Uninitialized<>` to keep the public ScratchStorage free of explicit
  // ctor / dtor declarations.
  struct value_phase_full
  {
    union _payload
    {
      ValueDataSourceScratchT load;
      ValueT values[tile_items];
    };
    CUB_NS_QUALIFIER::Uninitialized<_payload> storage;
  };
  struct value_phase_empty
  {};
  using value_phase_t = ::cuda::std::conditional_t<keys_only, value_phase_empty, value_phase_full>;

  struct ScratchStorage
  {
    union _phase_payload
    {
      KeyT keys[tile_items];
      value_phase_t value_phase;
    };
    CUB_NS_QUALIFIER::Uninitialized<_phase_payload> phase;
    filter_counters<SelectedOffsetT> cnt;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE block_filter_staged(
    TempStorage& /*storage*/,
    SelectedReserveOp reserve_selected,
    SelectedKeyOutIt selected_keys_out,
    ValueChannelSinksT value_channel_sinks,
    IdentifySelectedOp& identify_selected_op)
      : reserve_sel(reserve_selected)
      
      , sel_iter(selected_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_selected_op)
  {}

  template <typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourceT& value_source)
  {
    filter_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_source);
  }

  template <typename NumItemsT, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourceT& value_source)
  {
    filter_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_source);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourceT& value_source)
  {
    (void) value_source;
    (void) num_items;

    const int num_thread_items = compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    int positions[ItemsPerThread];

    if (threadIdx.x == 0)
    {
      buffer.cnt.counter = 0;
    }
    __syncthreads();

    if constexpr (InlinedClassify)
    {
      auto classifier = make_inlined_filter_classifier<IsFull>(identify_op, num_thread_items);
      classify_and_scatter_keys(buffer, keys, classifier, positions);
    }
    else
    {
      precomputed_filter_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op};
      classify_and_scatter_keys(buffer, keys, classifier, positions);
    }
    __syncthreads();

    const int selected_cnt = buffer.cnt.counter;

    if (threadIdx.x == 0)
    {
      const auto sel              = reserve_sel(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_base      = sel.first;
      buffer.cnt.granted_selected = static_cast<SelectedOffsetT>(sel.second);
    }
    __syncthreads();

    const SelectedOffsetT sel_base = buffer.cnt.global_base;
    const SelectedOffsetT sel_to_write =
      SelectedReserveOp::may_grant_less ? buffer.cnt.granted_selected : static_cast<SelectedOffsetT>(selected_cnt);

    auto& phase = buffer.phase.Alias();

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter[sel_base + static_cast<SelectedOffsetT>(i)] = phase.keys[i];
    }

    if constexpr (!keys_only)
    {
      static_assert(::cuda::std::is_same_v<typename ValueSourceT::value_t, ValueT>,
                    "Per-call value source's value_t must match the class-level ValueT template parameter.");

      __syncthreads();
      auto& vphase = phase.value_phase.storage.Alias();
      if constexpr (LazyValueLoad)
      {
        // Lazy: skip the eager tile-wide load. Each thread gathers only the
        // values of its surviving items via `value_source.gather_one(j)` and
        // writes them straight into the smem arena. The data-source's `load`
        // slot (aliased with `vphase.values` via the value-phase union) goes
        // unused on this path.
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < ItemsPerThread; ++j)
        {
          if (positions[j] >= 0)
          {
            vphase.values[positions[j]] = value_source.gather_one(j);
          }
        }
      }
      else
      {
        // Eager: load the full tile into a register array, then scatter the
        // surviving values into the smem arena.
        ValueT reg_values[ItemsPerThread]{};
        if constexpr (IsFull)
        {
          auto h = value_source.submit_load(vphase.load);
          h.complete_load(reg_values);
        }
        else
        {
          auto h = value_source.submit_load(vphase.load, num_items);
          h.complete_load(reg_values);
        }
        __syncthreads();

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < ItemsPerThread; ++j)
        {
          if (positions[j] >= 0)
          {
            vphase.values[positions[j]] = reg_values[j];
          }
        }
      }
      __syncthreads();

      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
      {
        sinks.selected_values_out[sel_base + static_cast<SelectedOffsetT>(i)] =
          vphase.values[i];
      }
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
        const int pos                  = atomicAdd(&buffer.cnt.counter, 1);
        buffer.phase.Alias().keys[pos] = keys[j];
        positions[j]                   = pos;
      }
      else
      {
        positions[j] = -1;
      }
    }
  }

  SelectedReserveOp reserve_sel;
  SelectedKeyOutIt sel_iter;
  ValueChannelSinksT sinks;
  IdentifySelectedOp& identify_op;
};

//---------------------------------------------------------------------
// `block_filter_shared_mem` -- keys + value array coexist in smem (within
// `phase.kv`), then a single coalesced flush. Pre-Phase-1 delegate loads alias
// with the kv arena via the top-level phase union. Mapped from
// `block_filter_strategy::shared_mem`.
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
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksT      = CUB_NS_QUALIFIER::NullType,
          typename ValueT                  = CUB_NS_QUALIFIER::NullType,
          typename ValueDataSourceScratchT = CUB_NS_QUALIFIER::NullType,
          bool LazyValueLoad               = false,
          bool InlinedClassify             = false>
class block_filter_shared_mem
{
public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;

  struct TempStorage
  {};

  // Per-tile scratch. The `phase` union has two views: `delegate_loads` (pre-Phase-1
  // delegate-load staging area for the value channel) and `kv` (the keys + values
  // arena used for scatter and cooperative flush). `cnt` lives outside the union,
  // same role as in `block_filter_staged`. The `delegate_loads` slot and the
  // `values[]` array both collapse to empty placeholders in the keys-only build.
  struct keys_and_values_full
  {
    KeyT keys[tile_items];
    ValueT values[tile_items];
  };
  struct keys_and_values_keys_only
  {
    KeyT keys[tile_items];
  };
  using keys_and_values_t = ::cuda::std::conditional_t<keys_only, keys_and_values_keys_only, keys_and_values_full>;

  struct delegate_load_full
  {
    ValueDataSourceScratchT load;
  };
  struct delegate_load_empty
  {};
  using delegate_load_t = ::cuda::std::conditional_t<keys_only, delegate_load_empty, delegate_load_full>;

  // The phase union carries non-trivial alternatives (the data-source's `load`
  // scratch may be `multi_source_data_source::ScratchStorage`); wrapping it in
  // `cub::Uninitialized<>` keeps the public ScratchStorage free of explicit
  // ctor / dtor declarations.
  struct ScratchStorage
  {
    union _phase_payload
    {
      delegate_load_t delegate_loads;
      keys_and_values_t kv;
    };
    CUB_NS_QUALIFIER::Uninitialized<_phase_payload> phase;
    filter_counters<SelectedOffsetT> cnt;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE block_filter_shared_mem(
    TempStorage& /*storage*/,
    SelectedReserveOp reserve_selected,
    SelectedKeyOutIt selected_keys_out,
    ValueChannelSinksT value_channel_sinks,
    IdentifySelectedOp& identify_selected_op)
      : reserve_sel(reserve_selected)
      
      , sel_iter(selected_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_selected_op)
  {}

  template <typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourceT& value_source)
  {
    filter_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_source);
  }

  template <typename NumItemsT, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourceT& value_source)
  {
    filter_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_source);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourceT& value_source)
  {
    (void) value_source;
    (void) num_items;

    const int num_thread_items = compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    // Pre-Phase-1: when eagerly loading, fetch the channel's values into a register
    // array via the delegate-load slot. After this, the delegate_loads view of the
    // phase union is dead and we transition to the kv view at the next
    // __syncthreads. Skipped under `LazyValueLoad` -- the kv-scatter loop below
    // pulls values via `gather_one(j)` only for surviving items.
    using reg_values_t = ::cuda::std::conditional_t<keys_only, int, ValueT>;
    reg_values_t reg_values[ItemsPerThread]{};
    if constexpr (!keys_only && !LazyValueLoad)
    {
      static_assert(::cuda::std::is_same_v<typename ValueSourceT::value_t, ValueT>,
                    "Per-call value source's value_t must match the class-level ValueT template parameter.");
      auto& load_slot = buffer.phase.Alias().delegate_loads;
      if constexpr (IsFull)
      {
        auto h = value_source.submit_load(load_slot.load);
        h.complete_load(reg_values);
      }
      else
      {
        auto h = value_source.submit_load(load_slot.load, num_items);
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
      auto classifier = make_inlined_filter_classifier<IsFull>(identify_op, num_thread_items);
      classify_and_scatter_kv(buffer, keys, classifier, value_source, reg_values);
    }
    else
    {
      precomputed_filter_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op};
      classify_and_scatter_kv(buffer, keys, classifier, value_source, reg_values);
    }
    __syncthreads();

    const int selected_cnt = buffer.cnt.counter;

    if (threadIdx.x == 0)
    {
      const auto sel              = reserve_sel(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_base      = sel.first;
      buffer.cnt.granted_selected = static_cast<SelectedOffsetT>(sel.second);
    }
    __syncthreads();

    const SelectedOffsetT sel_base = buffer.cnt.global_base;
    const SelectedOffsetT sel_to_write =
      SelectedReserveOp::may_grant_less ? buffer.cnt.granted_selected : static_cast<SelectedOffsetT>(selected_cnt);

    auto& kv = buffer.phase.Alias().kv;
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter[sel_base + static_cast<SelectedOffsetT>(i)] = kv.keys[i];
    }

    if constexpr (!keys_only)
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
      {
        sinks.selected_values_out[sel_base + static_cast<SelectedOffsetT>(i)] =
          kv.values[i];
      }
    }
    __syncthreads();
  }

  // Mode-agnostic Phase 1 body: classify every per-thread item and scatter the
  // surviving (key, value) pairs into the kv arena. `Classifier` exposes
  // `operator()(KeyT, int j) -> bool` for both modes; the value source is
  // consulted via `gather_one` only when `LazyValueLoad == true`.
  template <typename Classifier, typename ValueSourceT, typename RegValuesArr>
  _CCCL_DEVICE _CCCL_FORCEINLINE void classify_and_scatter_kv(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    Classifier& classifier,
    ValueSourceT& value_source,
    const RegValuesArr& reg_values)
  {
    (void) value_source;
    (void) reg_values;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (!classifier(keys[j], j))
      {
        continue;
      }
      const int idx = atomicAdd(&buffer.cnt.counter, 1);
      auto& kv      = buffer.phase.Alias().kv;
      kv.keys[idx]  = keys[j];
      if constexpr (!keys_only)
      {
        if constexpr (LazyValueLoad)
        {
          kv.values[idx] = value_source.gather_one(j);
        }
        else
        {
          kv.values[idx] = reg_values[j];
        }
      }
    }
  }

  SelectedReserveOp reserve_sel;
  SelectedKeyOutIt sel_iter;
  ValueChannelSinksT sinks;
  IdentifySelectedOp& identify_op;
};
} // namespace detail::topk

CUB_NAMESPACE_END
