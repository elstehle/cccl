// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private `BlockFilter` (single-stream sibling of `BlockPartition` from
//! `block_partition.cuh`). Per-tile 1-way filter over a unary
//! `IdentifySelected(key) -> bool` predicate, with reserve-driven global writes.
//!
//! Interface ("safe-both") contract shared with the accumulating sister class
//! `BlockFilterAccumulating` (`block_filter_accumulating.cuh`):
//!   - All sinks (reserve op, output iterator, transform, value-channel sink
//!     tuple) AND the `identify_selected_op` predicate are captured by ctor and
//!     stored as members. Per-call args reduce to per-tile data plus a bare
//!     `cuda::std::tuple<TileDataSource...>` for value sources.
//!   - `epilogue()` is argless on every variant. `BlockFilter::epilogue()` is a
//!     no-op (the strategy is non-accumulating). The accumulating variant's
//!     `epilogue()` performs a terminal flush of any remaining buffered items.
//!   - `Strategy` carries the classify-mode encoding directly: `AtomicsPreClassify`
//!     materializes a `kept[ItemsPerThread]` register array up front;
//!     `AtomicsInlinedClassify` recomputes the predicate at each scatter site.
//!     `Staged` and `SharedMem` always behave as "pre-classify".

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
// the four `Atomics*` / `Staged` / `SharedMem` values map to `BlockFilter`, and
// `AccumulatingFilter` maps to `BlockFilterAccumulating`.
//---------------------------------------------------------------------
enum class BlockFilterStrategy
{
  AtomicsPreClassify,
  AtomicsInlinedClassify,
  Staged,
  SharedMem,
  AccumulatingFilter,
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

// Per-strategy scratch declarations. The Atomics* strategies hold no scratch;
// Staged / SharedMem hold a strategy-specific phase arena plus the counter.

struct atomics_filter_scratch
{};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT>
struct staged_filter_scratch
{
  union phase_t
  {
    KeyT keys[TileItems];
    CUB_NS_QUALIFIER::detail::phase_union<
      bp_detail::map_tuple_t<bp_detail::staged_channel_phase, ValueChannelMetaTuple, TileItems>>
      per_channel;

    _CCCL_HOST_DEVICE phase_t() {}
    _CCCL_HOST_DEVICE ~phase_t() {}
  } phase;

  filter_counters<SelectedOffsetT> cnt;
};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT>
struct shared_mem_filter_scratch
{
  struct keys_and_values_t
  {
    KeyT keys[TileItems];
    CUB_NS_QUALIFIER::detail::phase_aggregate<bp_detail::map_tuple_t<bp_detail::values_slot, ValueChannelMetaTuple, TileItems>>
      per_channel_values;
  };

  union phase_t
  {
    CUB_NS_QUALIFIER::detail::phase_union<bp_detail::map_tuple1_t<bp_detail::delegate_load_slot, ValueChannelMetaTuple>>
      delegate_loads;
    keys_and_values_t kv;

    _CCCL_HOST_DEVICE phase_t() {}
    _CCCL_HOST_DEVICE ~phase_t() {}
  } phase;

  filter_counters<SelectedOffsetT> cnt;
};

template <BlockFilterStrategy Strategy,
          typename KeyT,
          typename ValueChannelMetaTuple,
          int TileItems,
          typename SelectedOffsetT>
struct strategy_filter_scratch_selector;

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT>
struct strategy_filter_scratch_selector<BlockFilterStrategy::AtomicsPreClassify,
                                        KeyT,
                                        ValueChannelMetaTuple,
                                        TileItems,
                                        SelectedOffsetT>
{
  using type = atomics_filter_scratch;
};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT>
struct strategy_filter_scratch_selector<BlockFilterStrategy::AtomicsInlinedClassify,
                                        KeyT,
                                        ValueChannelMetaTuple,
                                        TileItems,
                                        SelectedOffsetT>
{
  using type = atomics_filter_scratch;
};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT>
struct strategy_filter_scratch_selector<BlockFilterStrategy::Staged, KeyT, ValueChannelMetaTuple, TileItems, SelectedOffsetT>
{
  using type = staged_filter_scratch<KeyT, ValueChannelMetaTuple, TileItems, SelectedOffsetT>;
};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT>
struct strategy_filter_scratch_selector<BlockFilterStrategy::SharedMem,
                                        KeyT,
                                        ValueChannelMetaTuple,
                                        TileItems,
                                        SelectedOffsetT>
{
  using type = shared_mem_filter_scratch<KeyT, ValueChannelMetaTuple, TileItems, SelectedOffsetT>;
};

// Adapter so `filter_atomics_fused_scatter` can be a single function template
// over an "indexed classifier" with signature `(KeyT, int j) -> bool`. The
// inlined wrapper encapsulates the `is_valid` partial-tile check.
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
// Items past `num_thread_items` (partial path) are forced to `false`.
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

// Compile-time predicate: which `BlockFilterStrategy` values map to the
// non-accumulating `BlockFilter` class.
template <BlockFilterStrategy S>
inline constexpr bool is_block_filter_strategy_v =
  (S == BlockFilterStrategy::AtomicsPreClassify) //
  || (S == BlockFilterStrategy::AtomicsInlinedClassify) //
  || (S == BlockFilterStrategy::Staged) //
  || (S == BlockFilterStrategy::SharedMem);

template <BlockFilterStrategy S>
inline constexpr bool is_atomics_filter_strategy_v =
  (S == BlockFilterStrategy::AtomicsPreClassify) || (S == BlockFilterStrategy::AtomicsInlinedClassify);

template <BlockFilterStrategy S>
inline constexpr bool is_inlined_filter_classify_v = (S == BlockFilterStrategy::AtomicsInlinedClassify);

} // namespace bf_detail

//---------------------------------------------------------------------
// `BlockFilter` -- the non-accumulating filter primitive.
//
// Sinks AND `identify_selected_op` are bound at ctor; per-call `Partition()`
// only takes per-tile data and a bare `cuda::std::tuple<TileDataSource...>`
// of value sources. `epilogue()` is a `_CCCL_FORCEINLINE` no-op for parity
// with the accumulating sister class.
//---------------------------------------------------------------------

template <int BlockThreads,
          int ItemsPerThread,
          BlockFilterStrategy Strategy,
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
class BlockFilter
{
  static_assert(bf_detail::is_block_filter_strategy_v<Strategy>,
                "BlockFilter only handles the non-accumulating strategies; the AccumulatingFilter strategy maps to "
                "BlockFilterAccumulating (see block_filter_accumulating.cuh).");
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueTypesTuple>::value
                  == ::cuda::std::tuple_size<DataSourceScratchTypesTuple>::value,
                "ValueTypesTuple and DataSourceScratchTypesTuple must have the same length.");

public:
  static constexpr int tile_items            = BlockThreads * ItemsPerThread;
  static constexpr BlockFilterStrategy strat = Strategy;
  static constexpr int num_value_channels    = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  // Class-lifetime persistent state. Empty (no carried state across Partition() calls).
  struct TempStorage
  {};

  using value_channel_meta_tuple_t =
    bp_detail::zip_value_channel_metas_t<ValueTypesTuple, DataSourceScratchTypesTuple>;

  using ScratchStorage =
    typename bf_detail::strategy_filter_scratch_selector<Strategy,
                                                         KeyT,
                                                         value_channel_meta_tuple_t,
                                                         tile_items,
                                                         SelectedOffsetT>::type;

  // Ctor (safe-both shape): captures sinks + identify op. The TempStorage
  // parameter is unused (BlockFilter has no persistent state) but is taken
  // for parity with the accumulating sister class.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockFilter(
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
  Partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourcesTuple& value_sources)
  {
    filter_impl<true>(buffer, keys, /*num_items=*/tile_items, value_sources);
  }

  // Partial-tile overload: classify loop bound-checks against num_items.
  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourcesTuple& value_sources)
  {
    filter_impl<false>(buffer, keys, static_cast<int>(num_items), value_sources);
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

    if constexpr (bf_detail::is_atomics_filter_strategy_v<Strategy>)
    {
      if constexpr (bf_detail::is_inlined_filter_classify_v<Strategy>)
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
    else
    {
      // Staged / SharedMem strategies still need a precomputed `kept[]`
      // because their cooperative smem-scatter reads it more than once.
      bool kept[ItemsPerThread];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        const bool is_valid = IsFull ? true : (j < num_thread_items);
        kept[j]             = is_valid ? static_cast<bool>(identify_op_(keys[j])) : false;
      }

      if constexpr (Strategy == BlockFilterStrategy::Staged)
      {
        filter_staged<IsFull>(buffer, keys, kept, value_sources, num_items);
      }
      else
      {
        filter_shared_mem<IsFull>(buffer, keys, kept, value_sources, num_items);
      }
    }
  }

  // -----------------------------------------------------------------
  // Atomics strategies (fused): scatter via a unified per-item loop driven by a
  // user-supplied `Classifier` with signature `(KeyT, int j) -> bool`.
  // -----------------------------------------------------------------
  template <bool IsFull, typename Classifier, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_atomics_fused(
    ScratchStorage& /*buffer*/,
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    ValueSourcesTuple& value_sources)
  {
    static_assert(num_value_channels <= 1,
                  "atomics filter supports keys-only or single-value-channel "
                  "today; multi-channel needs a per-channel value array.");

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

  // -----------------------------------------------------------------
  // Staged strategy: smem scatter into `buffer.phase.keys` then cooperative
  // coalesced store via reserve op. Per-channel value path runs sequentially
  // after the keys phase.
  // -----------------------------------------------------------------
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_staged(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const bool (&kept)[ItemsPerThread],
    ValueSourcesTuple& value_sources,
    int num_items)
  {
    int positions[ItemsPerThread];

    if (threadIdx.x == 0)
    {
      buffer.cnt.counter = 0;
    }
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (kept[j])
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
    __syncthreads();

    const int selected_cnt = buffer.cnt.counter;

    if (threadIdx.x == 0)
    {
      const auto sel                = reserve_sel_(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_base        = sel.first;
      buffer.cnt.granted_selected   = static_cast<SelectedOffsetT>(sel.second);
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

  // -----------------------------------------------------------------
  // SharedMem strategy: keys + per-channel values coexist in smem (within
  // `phase.kv`), then a single coalesced flush.
  // -----------------------------------------------------------------
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_shared_mem(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const bool (&kept)[ItemsPerThread],
    ValueSourcesTuple& value_sources,
    int num_items)
  {
    static_assert(num_value_channels <= 1,
                  "shared_mem filter supports keys-only or single-value-channel "
                  "today; multi-channel needs a heterogeneous register-array tuple.");

    [[maybe_unused]] auto load_channel_values = [&](auto& reg_values_out) {
      if constexpr (num_value_channels == 1)
      {
        auto& src       = ::cuda::std::get<0>(value_sources);
        auto& load_slot = CUB_NS_QUALIFIER::detail::at<0>(buffer.phase.delegate_loads);
        if constexpr (IsFull)
        {
          auto h = src.submit_load(load_slot.load);
          h.complete_load(reg_values_out);
        }
        else
        {
          auto h = src.submit_load(load_slot.load, num_items);
          h.complete_load(reg_values_out);
        }
      }
      else
      {
        (void) reg_values_out;
      }
    };

    using channel_value_t = typename bp_detail::value_t_or_default<ValueTypesTuple>::type;
    channel_value_t reg_values[ItemsPerThread]{};
    load_channel_values(reg_values);

    __syncthreads();
    if (threadIdx.x == 0)
    {
      buffer.cnt.counter = 0;
    }
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (!kept[j])
      {
        continue;
      }
      const int idx             = atomicAdd(&buffer.cnt.counter, 1);
      buffer.phase.kv.keys[idx] = keys[j];
      if constexpr (num_value_channels == 1)
      {
        CUB_NS_QUALIFIER::detail::at<0>(buffer.phase.kv.per_channel_values).values[idx] = reg_values[j];
      }
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

  // ---------------------------------------------------------------
  // Member state: sinks + identify op captured by the ctor. Stored by reference
  // for stateful op types (matches the convention used by `BlockPartition`).
  // ---------------------------------------------------------------
  SelectedReserveOp& reserve_sel_;
  SelectedKeyOutTransformOp& sel_xform_;
  SelectedKeyOutIt sel_iter_;
  ValueChannelSinksTuple& sinks_;
  IdentifySelectedOp& identify_op_;
};
} // namespace detail::topk

CUB_NAMESPACE_END
