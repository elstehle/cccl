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
//! Strategy selection is done by `strategy_to_filter_class_t<Strategy, ...>` below. The
//! default tuning uses only `block_filter_atomics`.

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
#include <cub/detail/topk/empty_storage.cuh>
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
// Strategy selector for the filter primitives. Mirrors `block_partition_strategy`: the default
// tuning uses only `atomics` (`block_filter_atomics`). The enum retains the other (currently
// unimplemented) values for policy/ABI compatibility; `strategy_to_filter_class_t` maps every
// strategy to `block_filter_atomics` in this build.
//---------------------------------------------------------------------
enum class block_filter_strategy
{
  atomics,
  staged,
  shared_mem,
  accumulating_filter,
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
  using TempStorage = empty_storage_t;

  // Per-tile scratch. The atomics strategies hold no scatter-side smem (per-item
  // scatter goes direct to the user's iterator via the captured reserve op), but
  // they DO own the per-tile load scratch for the value channel: see the matching
  // doc on `block_partition_atomics::ScratchStorage` for the full rationale. When
  // the value-channel scratch is itself empty, we publish `ScratchStorage` as the
  // canonical empty marker so consumers can elide barriers / setup transitively.

private:
  static constexpr bool _scratch_storage_is_empty = is_empty_storage_v<ValueDataSourceScratchT>;

  struct _ScratchStorage_full
  {
    ValueDataSourceScratchT value_load;
  };

  struct _ScratchStorage_wrapped : CUB_NS_QUALIFIER::Uninitialized<_ScratchStorage_full>
  {};

public:
  using ScratchStorage =
    ::cuda::std::conditional_t<_scratch_storage_is_empty, empty_storage_t, _ScratchStorage_wrapped>;

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
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourceT& value_source)
  {
    filter_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_source);
  }

  // No-op terminal flush. Present for parity with `block_filter_accumulating`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  filter_impl(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourceT& value_source)
  {
    const int num_thread_items = compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    if constexpr (InlinedClassify)
    {
      auto classifier = make_inlined_filter_classifier<IsFull>(identify_op, num_thread_items);
      filter_atomics_fused<IsFull>(buffer, keys, num_thread_items, classifier, value_source);
    }
    else
    {
      precomputed_filter_classifier<KeyT, ItemsPerThread, IsFull> classifier{keys, num_thread_items, identify_op};
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
        filter_atomics_fused_scatter<IsFull>(keys, num_thread_items, classifier, value_source, unused_values);
      }
      else
      {
        // Smem-backed scratch for the value-channel load. See the matching comment
        // in `block_partition_atomics::partition_atomics_fused`.
        ValueT values[ItemsPerThread]{};
        if constexpr (_scratch_storage_is_empty)
        {
          ValueDataSourceScratchT chan_scratch_dummy{};
          auto h = value_source.submit_load(chan_scratch_dummy);
          h.complete_load(values);
        }
        else
        {
          auto& chan_scratch = buffer.Alias().value_load;
          auto h             = value_source.submit_load(chan_scratch);
          h.complete_load(values);
        }

        filter_atomics_fused_scatter<IsFull>(keys, num_thread_items, classifier, value_source, values);
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

// Maps a `block_filter_strategy` to its filter class. The default tuning only uses `atomics`;
// the strategy / buffer-capacity template parameters are retained for call-site compatibility
// but ignored here.
template <block_filter_strategy Strategy,
          int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename SelectedReserveOp,
          typename SelectedKeyOutIt,
          typename IdentifySelectedOp,
          typename ValueChannelSinksT,
          typename ValueT,
          typename ValueDataSourceScratchT,
          bool LazyValueLoad,
          bool InlinedClassify>
using strategy_to_filter_class_t = block_filter_atomics<
  BlockThreads,
  ItemsPerThread,
  InlinedClassify,
  KeyT,
  SelectedOffsetT,
  SelectedReserveOp,
  SelectedKeyOutIt,
  IdentifySelectedOp,
  ValueChannelSinksT,
  ValueT,
  ValueDataSourceScratchT,
  LazyValueLoad>;
} // namespace detail::topk

CUB_NAMESPACE_END
