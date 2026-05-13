// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private `BlockPartition` (architecture §9). Per-tile 2-way partition of
//! classified items into "selected" and "candidate" streams, with reserve-driven
//! global writes.
//!
//! Interface ("safe-both") contract shared with the accumulating sister classes
//! `BlockPartitionAccumulatingCandidates` / `BlockPartitionAccumulatingSelected`
//! (`block_partition_accumulating.cuh`):
//!   - All sinks (reserve ops, output iterators, transforms, value-channel sink
//!     tuple) are captured by ctor and stored as members. Per-call args reduce to
//!     per-tile data plus a bare `cuda::std::tuple<TileDataSource...>` for value
//!     sources.
//!   - `epilogue()` is argless on every variant. `BlockPartition::epilogue()` is a
//!     no-op (the strategy is non-accumulating). The accumulating variants'
//!     `epilogue()` performs a terminal flush of any remaining buffered items.
//!   - `Strategy` carries the classify-mode encoding directly: `AtomicsPreClassify`
//!     materializes a `classes[ItemsPerThread]` register array up front;
//!     `AtomicsInlinedClassify` recomputes the classification at each scatter
//!     site. `Staged` and `SharedMem` always behave as "pre-classify" (their
//!     cooperative scatter reads the array more than once).
//!   - The per-channel value bundle splits along the lifetime boundary:
//!     `value_channel_sinks_t` (captured at ctor) carries the iters + transforms
//!     + per-channel `value_t` and `data_source_scratch_t` typedefs;
//!     a per-call `value_sources_tuple_t` carries the live `TileDataSource`
//!     instances that the agent has called `set_tile_base()` on for the current
//!     tile.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
//---------------------------------------------------------------------
// Shared types
//---------------------------------------------------------------------

// The three classes emitted by top-k's classifier. `rejected` is also the
// out-of-bounds marker; the partial-tile path forces OOB items to `rejected` inside
// `BlockPartition::Partition`.
enum class candidate_class
{
  selected,
  candidate,
  rejected,
};

// Strategy selector for the partition primitives. Folds in what used to be a
// separate `BlockPartitionClassifyMode` knob -- the four `Atomics*` / `Staged` /
// `SharedMem` values map to `BlockPartition`, the two `Accumulating*` values map
// to the two accumulating sister classes.
//
//   AtomicsPreClassify     -- no smem; per-non-rejected-item global atomic + scatter,
//                             per-item class precomputed into a `classes[]` register
//                             array up front.
//   AtomicsInlinedClassify -- same as above, but the classification is recomputed
//                             inline at each scatter use-site (frees the registers
//                             that would hold `classes[]`).
//   Staged                 -- smem scatter + cooperative coalesced store; phase_union
//                             arena. Always uses pre-classified `classes[]`.
//   SharedMem              -- typed `keys[]` + per-channel `values[]` arrays + per-
//                             channel delegate-load union. Always uses pre-classified
//                             `classes[]`.
//   AccumulatingCandidates -- BlockPartitionAccumulatingCandidates: candidate stream
//                             buffered in smem and accumulated across multiple tiles;
//                             selected stream goes direct-to-global. Used when
//                             HasCandidates == true.
//   AccumulatingSelected   -- BlockPartitionAccumulatingSelected: selected stream
//                             buffered (HasCandidates == false; classifier collapses
//                             candidate -> selected so the selected stream is the
//                             only one).
enum class BlockPartitionStrategy
{
  AtomicsPreClassify,
  AtomicsInlinedClassify,
  Staged,
  SharedMem,
  AccumulatingCandidates,
  AccumulatingSelected,
};

//---------------------------------------------------------------------
// Per-channel value-sink bundle (captured at ctor; lifetime = class instance).
//
// The single `value_channel_t` struct used previously bundled the per-tile
// `data_source` with the persistent sink-side state (iterators + transforms).
// The "safe-both" interface splits that bundle along the lifetime boundary:
// this struct holds only the four sink-side members. Per-channel `value_t` and
// `data_source_scratch_t` (needed for sizing smem in the Staged / SharedMem
// strategies) come from the agent-supplied `ValueTypesTuple` and
// `DataSourceScratchTypesTuple` template parameters of the partition class --
// not from any iterator's `value_type`, which can be `void` for output
// iterators.
//---------------------------------------------------------------------
template <typename SelectedValuesOutIt,
          typename CandidateValuesOutIt,
          typename SelectedValueTransform,
          typename CandidateValueTransform>
struct value_channel_sinks_t
{
  SelectedValuesOutIt selected_values_out;
  CandidateValuesOutIt candidate_values_out;
  SelectedValueTransform selected_value_transform;
  CandidateValueTransform candidate_value_transform;
};

//---------------------------------------------------------------------
// Per-strategy ScratchStorage layouts (architecture §9.4).
//---------------------------------------------------------------------

namespace bp_detail
{
// Counters for the staged / shared_mem strategies. Phase 1 uses 32-bit smem atomics
// (`int counters[2]`); Phase 2 uses the actual offset types for the global bases. The
// two phases are separated by `__syncthreads()` so the union reuse is safe. `cnt`
// lives outside the per-phase union (architecture O8): it stays alive across the
// entire `Partition()` call.
//
// `granted_*` are broadcast-only fields written by thread 0 after the reserve op
// returns and read by every thread before the cooperative flush. They live outside
// the phase union to avoid corrupting `global_bases.*`.
template <typename SelectedOffsetT, typename CandidateOffsetT>
struct partition_counters
{
  union
  {
    int counters[2];
    struct
    {
      SelectedOffsetT selected;
      CandidateOffsetT candidate;
    } global_bases;
  };
  SelectedOffsetT granted_selected;
  CandidateOffsetT granted_candidate;
};

// Per-channel "meta" record carrying the per-channel `value_t` and the data
// source's `ScratchStorage` type. Built by zipping the agent-supplied
// `ValueTypesTuple` and `DataSourceScratchTypesTuple` template arguments. The
// per-strategy scratch helpers below consume this meta record (one per channel)
// rather than reading typedefs off the sinks struct -- the sinks struct has
// only the runtime sink members and no typedefs.
template <typename ValueT, typename DataSourceScratchT>
struct value_channel_meta_t
{
  using value_t               = ValueT;
  using data_source_scratch_t = DataSourceScratchT;
};

// Zip a `ValueTypesTuple` and a `DataSourceScratchTypesTuple` (parallel tuples
// of element type) into a single `tuple<value_channel_meta_t<...>, ...>` that
// the per-strategy scratch helpers consume.
template <typename ValueTypesTuple, typename DataSourceScratchTypesTuple>
struct zip_value_channel_metas;

template <typename... Vs, typename... Ss>
struct zip_value_channel_metas<::cuda::std::tuple<Vs...>, ::cuda::std::tuple<Ss...>>
{
  static_assert(sizeof...(Vs) == sizeof...(Ss),
                "ValueTypesTuple and DataSourceScratchTypesTuple must have the same length.");
  using type = ::cuda::std::tuple<value_channel_meta_t<Vs, Ss>...>;
};

template <typename ValueTypesTuple, typename DataSourceScratchTypesTuple>
using zip_value_channel_metas_t =
  typename zip_value_channel_metas<ValueTypesTuple, DataSourceScratchTypesTuple>::type;

// Per-channel "loading or scattering" slot for the staged strategy. Inside one
// channel's processing window the data-source scratch and the per-channel value
// array are mutually exclusive in time -> internal phase_union.
template <typename Meta, int N>
struct staged_channel_phase
{
  union
  {
    typename Meta::data_source_scratch_t load;
    typename Meta::value_t values[N];
  };
};

// Per-channel "loading" slot for the shared_mem strategy.
template <typename Meta>
struct delegate_load_slot
{
  typename Meta::data_source_scratch_t load;
};

// Per-channel "values" slot for the shared_mem strategy.
template <typename Meta, int N>
struct values_slot
{
  typename Meta::value_t values[N];
};

// Map a tuple type-parameter through a (Sink, int) -> Out template. Equivalent to
// the architecture's `map_tuple_t<F, Tuple, N>`.
template <template <typename, int> class F, typename Tuple, int N>
struct map_tuple;

template <template <typename, int> class F, int N, typename... Cs>
struct map_tuple<F, ::cuda::std::tuple<Cs...>, N>
{
  using type = ::cuda::std::tuple<F<Cs, N>...>;
};

template <template <typename, int> class F, typename Tuple, int N>
using map_tuple_t = typename map_tuple<F, Tuple, N>::type;

template <template <typename> class F, typename Tuple>
struct map_tuple1;

template <template <typename> class F, typename... Cs>
struct map_tuple1<F, ::cuda::std::tuple<Cs...>>
{
  using type = ::cuda::std::tuple<F<Cs>...>;
};

template <template <typename> class F, typename Tuple>
using map_tuple1_t = typename map_tuple1<F, Tuple>::type;

// First element of a tuple (or a stand-in `int` when the tuple is empty). Used by
// `BlockPartition::partition_shared_mem` and the accumulating classes to size a
// per-channel register array conditionally on the first channel's `value_t`.
// The `Tuple` is expected to be either a `ValueTypesTuple` (whose elements are
// already value types) or a `ValueChannelMetaTuple` (whose elements expose a
// `value_t` typedef); the `value_t_or_default` and `meta_value_t_or_default`
// variants below select the right idiom.
template <typename Tuple>
struct value_t_or_default
{
  using type = int;
};

template <typename Head, typename... Rest>
struct value_t_or_default<::cuda::std::tuple<Head, Rest...>>
{
  using type = Head;
};

template <typename Tuple>
struct meta_value_t_or_default
{
  using type = int;
};

template <typename Head, typename... Rest>
struct meta_value_t_or_default<::cuda::std::tuple<Head, Rest...>>
{
  using type = typename Head::value_t;
};

// Strategy-specific scratch declarations.

struct atomics_scratch
{};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct staged_scratch
{
  union phase_t
  {
    KeyT keys[TileItems];
    CUB_NS_QUALIFIER::detail::phase_union<map_tuple_t<staged_channel_phase, ValueChannelMetaTuple, TileItems>>
      per_channel;

    _CCCL_HOST_DEVICE phase_t() {}
    _CCCL_HOST_DEVICE ~phase_t() {}
  } phase;

  partition_counters<SelectedOffsetT, CandidateOffsetT> cnt;
};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct shared_mem_scratch
{
  struct keys_and_values_t
  {
    KeyT keys[TileItems];
    CUB_NS_QUALIFIER::detail::phase_aggregate<map_tuple_t<values_slot, ValueChannelMetaTuple, TileItems>>
      per_channel_values;
  };

  union phase_t
  {
    CUB_NS_QUALIFIER::detail::phase_union<map_tuple1_t<delegate_load_slot, ValueChannelMetaTuple>> delegate_loads;
    keys_and_values_t kv;

    _CCCL_HOST_DEVICE phase_t() {}
    _CCCL_HOST_DEVICE ~phase_t() {}
  } phase;

  partition_counters<SelectedOffsetT, CandidateOffsetT> cnt;
};

template <BlockPartitionStrategy Strategy,
          typename KeyT,
          typename ValueChannelMetaTuple,
          int TileItems,
          typename SelectedOffsetT,
          typename CandidateOffsetT>
struct strategy_scratch_selector;

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct strategy_scratch_selector<BlockPartitionStrategy::AtomicsPreClassify,
                                 KeyT,
                                 ValueChannelMetaTuple,
                                 TileItems,
                                 SelectedOffsetT,
                                 CandidateOffsetT>
{
  using type = atomics_scratch;
};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct strategy_scratch_selector<BlockPartitionStrategy::AtomicsInlinedClassify,
                                 KeyT,
                                 ValueChannelMetaTuple,
                                 TileItems,
                                 SelectedOffsetT,
                                 CandidateOffsetT>
{
  using type = atomics_scratch;
};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct strategy_scratch_selector<BlockPartitionStrategy::Staged,
                                 KeyT,
                                 ValueChannelMetaTuple,
                                 TileItems,
                                 SelectedOffsetT,
                                 CandidateOffsetT>
{
  using type = staged_scratch<KeyT, ValueChannelMetaTuple, TileItems, SelectedOffsetT, CandidateOffsetT>;
};

template <typename KeyT, typename ValueChannelMetaTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct strategy_scratch_selector<BlockPartitionStrategy::SharedMem,
                                 KeyT,
                                 ValueChannelMetaTuple,
                                 TileItems,
                                 SelectedOffsetT,
                                 CandidateOffsetT>
{
  using type = shared_mem_scratch<KeyT, ValueChannelMetaTuple, TileItems, SelectedOffsetT, CandidateOffsetT>;
};

// Compile-time tuple iteration helper. Calls `f(at<I>(tuple), integral_constant<int,I>)`
// for every element of the tuple.
template <typename Tuple, typename Fn, ::cuda::std::size_t... Is>
_CCCL_DEVICE _CCCL_FORCEINLINE void tuple_for_each_impl(Tuple&& t, Fn&& f, ::cuda::std::index_sequence<Is...>)
{
  (f(::cuda::std::get<Is>(t), ::cuda::std::integral_constant<::cuda::std::size_t, Is>{}), ...);
}

template <typename Tuple, typename Fn>
_CCCL_DEVICE _CCCL_FORCEINLINE void tuple_for_each(Tuple&& t, Fn&& f)
{
  constexpr auto sz = ::cuda::std::tuple_size<::cuda::std::remove_reference_t<Tuple>>::value;
  tuple_for_each_impl(
    ::cuda::std::forward<Tuple>(t), ::cuda::std::forward<Fn>(f), ::cuda::std::make_index_sequence<sz>{});
}

// Adapter that lets `partition_atomics_fused_scatter` be a single function template
// over an "indexed classifier" with signature `(KeyT, int j) -> candidate_class`. The
// inlined wrapper holds a reference to the original `IdentifyCandidatesOp` and the
// per-tile valid count, encapsulating the `is_valid` check so the outer scatter loop
// can call the classifier unconditionally. Marked `_CCCL_FORCEINLINE` so ptxas sees
// through the indirection and emits identical SASS to a direct call.
template <bool IsFull, typename Op>
struct inlined_classifier
{
  Op& op;
  int num_thread_items;

  template <typename KeyT>
  _CCCL_DEVICE _CCCL_FORCEINLINE candidate_class operator()(KeyT key, int j) const
  {
    if constexpr (IsFull)
    {
      (void) j;
      return op(key);
    }
    else
    {
      return (j < num_thread_items) ? op(key) : candidate_class::rejected;
    }
  }
};

template <bool IsFull, typename Op>
_CCCL_DEVICE _CCCL_FORCEINLINE inlined_classifier<IsFull, Op> make_inlined_classifier(Op& op, int num_thread_items)
{
  return inlined_classifier<IsFull, Op>{op, num_thread_items};
}

// Precomputed-classes adapter for the unified `partition_atomics_fused_scatter`. The
// constructor runs the classify loop up front (mirroring partition_impl's existing
// Step 1): it computes `classes[j] = op(keys[j])` for valid items, fires
// `candidate_callback_op` for `candidate`-classified items when `HasCandidates`, and
// collapses `candidate -> selected` otherwise. Items past `num_thread_items` (partial
// path) are forced to `rejected` so the outer scatter loop's redundant `is_valid`
// check never observes a different class.
//
// `operator()(KeyT, int j)` returns `classes[j]`, ignoring the key. Forceinlined so
// the outer scatter's classifier call collapses to a direct register read of
// `classes[j]` and any redundant `is_valid` selection becomes statically `classes[j]`.
template <typename KeyT, int ItemsPerThread, bool IsFull, bool HasCandidates>
struct precomputed_classifier
{
  candidate_class classes[ItemsPerThread];

  template <typename Op, typename CallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE
  precomputed_classifier(const KeyT (&keys)[ItemsPerThread], int num_thread_items, Op& op, CallbackOp& callback)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const bool is_valid = IsFull ? true : (j < num_thread_items);
      classes[j]          = is_valid ? op(keys[j]) : candidate_class::rejected;

      if constexpr (HasCandidates)
      {
        // Architecture §10.2: callback fires for every `candidate`-classified item,
        // including ones the candidate reserve op subsequently drops (cap clamp).
        if (is_valid && classes[j] == candidate_class::candidate)
        {
          callback(keys[j]);
        }
      }
      else
      {
        // !HasCandidates: collapse `candidate` onto `selected`. The callback is
        // statically guaranteed never to fire.
        if (is_valid && classes[j] == candidate_class::candidate)
        {
          classes[j] = candidate_class::selected;
        }
      }
    }
  }

  template <typename KeyT_>
  _CCCL_DEVICE _CCCL_FORCEINLINE candidate_class operator()(KeyT_ /*key*/, int j) const
  {
    return classes[j];
  }
};

// No-op callback for cases where the candidate callback has already been fired up
// front (e.g., by `precomputed_classifier`'s constructor). The empty body lets ptxas
// DCE the callback firing in the unified scatter loop.
struct noop_callback_op
{
  template <typename KeyT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(KeyT /*key*/) const
  {}
};

// Compile-time predicate: which `BlockPartitionStrategy` values map to the
// non-accumulating `BlockPartition` class.
template <BlockPartitionStrategy S>
inline constexpr bool is_block_partition_strategy_v =
  (S == BlockPartitionStrategy::AtomicsPreClassify) //
  || (S == BlockPartitionStrategy::AtomicsInlinedClassify) //
  || (S == BlockPartitionStrategy::Staged) //
  || (S == BlockPartitionStrategy::SharedMem);

template <BlockPartitionStrategy S>
inline constexpr bool is_atomics_strategy_v =
  (S == BlockPartitionStrategy::AtomicsPreClassify) || (S == BlockPartitionStrategy::AtomicsInlinedClassify);

template <BlockPartitionStrategy S>
inline constexpr bool is_inlined_classify_v = (S == BlockPartitionStrategy::AtomicsInlinedClassify);

//---------------------------------------------------------------------
// `partition_storage_layout` -- agent-side smem layout helper.
//
// Picks the right storage shape for the union (`partition_t::TempStorage`, prefix-sum
// state, load scratch, partition scratch) based on whether the partition class
// carries persistent state across calls. Both shapes are temporally safe -- prefix
// sum runs strictly after the partition's terminal `epilogue()` and only on the
// last block, and load+partition are sequential within a tile.
//
//   NeedsPersistent == false (BlockPartition; `TempStorage` is empty):
//     `partition_state` is empty; the union spans
//     {keys_source_scratch | prefix_sum | partition_scratch}, matching today's
//     three-way union and giving byte-equivalent smem footprint.
//
//   NeedsPersistent == true  (Accumulating variants; `TempStorage` is non-empty):
//     `partition_state` carries the per-stream slot buffers + counters across all
//     tiles; it aliases with `prefix_sum` (only safe because prefix_sum runs after
//     epilogue on the last block). The per-tile scratch union spans
//     {keys_source_scratch | partition_scratch}; for the accumulating variants
//     `partition_scratch` is empty so the union collapses to just keys_source_scratch.
//
// All four pieces of storage are accessed via the four `get_*()` member functions so
// the agent's call sites are layout-agnostic.
//
// Agents should generally consume `partition_storage_layout_for_t<...>` (below) which
// derives `NeedsPersistent` from `cuda::std::is_empty_v<typename PartitionT::TempStorage>`
// automatically.
//---------------------------------------------------------------------
template <bool NeedsPersistent, typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
struct partition_storage_layout;

template <typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
struct partition_storage_layout</*NeedsPersistent=*/false, PartitionT, KeysSourceScratchT, PrefixSumT>
{
  union scratch_t
  {
    KeysSourceScratchT keys_source_scratch;
    PrefixSumT prefix_sum;
    typename PartitionT::ScratchStorage partition_scratch;

    _CCCL_HOST_DEVICE scratch_t() {}
    _CCCL_HOST_DEVICE ~scratch_t() {}
  } scratch;

  // Empty TempStorage -- accessed only for parity with the Accumulating layout. Sized
  // to zero by the compiler (PartitionT::TempStorage is `struct{}`).
  typename PartitionT::TempStorage partition_state;

  _CCCL_DEVICE _CCCL_FORCEINLINE typename PartitionT::TempStorage& get_partition_state()
  {
    return partition_state;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE PrefixSumT& get_prefix_sum()
  {
    return scratch.prefix_sum;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE KeysSourceScratchT& get_keys_source_scratch()
  {
    return scratch.keys_source_scratch;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE typename PartitionT::ScratchStorage& get_partition_scratch()
  {
    return scratch.partition_scratch;
  }
};

template <typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
struct partition_storage_layout</*NeedsPersistent=*/true, PartitionT, KeysSourceScratchT, PrefixSumT>
{
  union persistent_t
  {
    typename PartitionT::TempStorage partition_state;
    PrefixSumT prefix_sum;

    _CCCL_HOST_DEVICE persistent_t() {}
    _CCCL_HOST_DEVICE ~persistent_t() {}
  } persistent;

  union scratch_t
  {
    KeysSourceScratchT keys_source_scratch;
    typename PartitionT::ScratchStorage partition_scratch;

    _CCCL_HOST_DEVICE scratch_t() {}
    _CCCL_HOST_DEVICE ~scratch_t() {}
  } scratch;

  _CCCL_DEVICE _CCCL_FORCEINLINE typename PartitionT::TempStorage& get_partition_state()
  {
    return persistent.partition_state;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE PrefixSumT& get_prefix_sum()
  {
    return persistent.prefix_sum;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE KeysSourceScratchT& get_keys_source_scratch()
  {
    return scratch.keys_source_scratch;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE typename PartitionT::ScratchStorage& get_partition_scratch()
  {
    return scratch.partition_scratch;
  }
};

// Agent-friendly alias that auto-derives `NeedsPersistent` from
// `cuda::std::is_empty_v<typename PartitionT::TempStorage>`. The partition class
// itself doesn't have to expose any explicit "needs-persistent-state" trait --
// `BlockPartition`'s `TempStorage = struct{}` is empty (so we get the 3-way
// union), and the accumulating variants' `TempStorage` is non-empty (so we get
// the persistent + scratch layout). Empty-struct vs. wrapped-data is a clean
// signal because `cub::Uninitialized<T>` carries a `DeviceWord storage[N]`
// member, which makes `is_empty_v` correctly return `false` on the wrapper.
template <typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
using partition_storage_layout_for_t =
  partition_storage_layout<!::cuda::std::is_empty_v<typename PartitionT::TempStorage>,
                           PartitionT,
                           KeysSourceScratchT,
                           PrefixSumT>;

} // namespace bp_detail

//---------------------------------------------------------------------
// `BlockPartition` -- the non-accumulating partition primitive (architecture §9.2).
//
// Sinks are bound at ctor; per-call `Partition()` only takes per-tile data and a
// bare `cuda::std::tuple<TileDataSource...>` of value sources. `epilogue()` is a
// `_CCCL_FORCEINLINE` no-op for parity with the accumulating sister classes -- the
// compiler DCEs the call.
//---------------------------------------------------------------------

template <int BlockThreads,
          int ItemsPerThread,
          BlockPartitionStrategy Strategy,
          typename KeyT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          typename SelectedReserveOp,
          typename CandidateReserveOp,
          typename SelectedKeyOutTransformOp,
          typename CandidateKeyOutTransformOp,
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename ValueChannelSinksTuple        = ::cuda::std::tuple<>,
          typename ValueTypesTuple               = ::cuda::std::tuple<>,
          typename DataSourceScratchTypesTuple   = ::cuda::std::tuple<>,
          bool LazyValueLoad                     = false>
class BlockPartition
{
  static_assert(bp_detail::is_block_partition_strategy_v<Strategy>,
                "BlockPartition only handles the non-accumulating strategies; the AccumulatingCandidates / "
                "AccumulatingSelected strategies map to BlockPartitionAccumulatingCandidates / "
                "BlockPartitionAccumulatingSelected (see block_partition_accumulating.cuh).");
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueTypesTuple>::value
                  == ::cuda::std::tuple_size<DataSourceScratchTypesTuple>::value,
                "ValueTypesTuple and DataSourceScratchTypesTuple must have the same length.");

public:
  static constexpr int tile_items               = BlockThreads * ItemsPerThread;
  static constexpr BlockPartitionStrategy strat = Strategy;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  // Class-lifetime persistent state. Empty (no carried state across Partition() calls).
  // The agent dispatches on `is_empty_v<TempStorage>` to pick the optimal smem
  // aliasing layout (3-way union for empty, otherwise persistent + scratch).
  struct TempStorage
  {};

  // Per-channel meta tuple: zips `ValueTypesTuple` and `DataSourceScratchTypesTuple`
  // into one tuple of `bp_detail::value_channel_meta_t<value_t, scratch_t>` records
  // that the per-strategy scratch helpers consume.
  using value_channel_meta_tuple_t =
    bp_detail::zip_value_channel_metas_t<ValueTypesTuple, DataSourceScratchTypesTuple>;

  // Method-call typed scratch; strategy-specific. Architecture §9.4.
  using ScratchStorage =
    typename bp_detail::strategy_scratch_selector<Strategy,
                                                  KeyT,
                                                  value_channel_meta_tuple_t,
                                                  tile_items,
                                                  SelectedOffsetT,
                                                  CandidateOffsetT>::type;

  // Ctor (safe-both shape): captures sinks. The TempStorage parameter is unused
  // (BlockPartition has no persistent state) but is taken for parity with the
  // accumulating sister classes so the agent can construct any of the three
  // partition variants with the same call.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockPartition(
    TempStorage& /*storage*/,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksTuple& value_channel_sinks)
      : reserve_sel_(reserve_selected)
      , reserve_cand_(reserve_candidate)
      , sel_xform_(selected_key_transform)
      , cand_xform_(candidate_key_transform)
      , sel_iter_(selected_keys_out)
      , cand_iter_(candidate_keys_out)
      , sinks_(value_channel_sinks)
  {}

  // Full-tile overload: no per-item bound check inside the classify loop.
  //
  // The class-level `LazyValueLoad` template arg (default `false`) controls whether
  // the per-thread `values[ItemsPerThread]` register array is populated up front via
  // the value channel's `data_source.complete_load(...)` (eager) or skipped, with the
  // scatter loop calling `data_source.gather_one(j)` only for non-rejected items
  // (lazy). Only honored on the Atomics* strategies with at most one value channel.
  template <bool HasCandidates, typename IdentifyCandidatesOp, typename CandidateCallbackOp, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Partition(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    ::cuda::std::integral_constant<bool, HasCandidates>,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op,
    ValueSourcesTuple& value_sources)
  {
    partition_impl<true, HasCandidates>(
      buffer, keys, /*num_items=*/tile_items, identify_candidates_op, candidate_callback_op, value_sources);
  }

  // Partial-tile overload: classify loop bound-checks against num_items.
  template <bool HasCandidates,
            typename NumItemsT,
            typename IdentifyCandidatesOp,
            typename CandidateCallbackOp,
            typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Partition(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    NumItemsT num_items,
    ::cuda::std::integral_constant<bool, HasCandidates>,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op,
    ValueSourcesTuple& value_sources)
  {
    partition_impl<false, HasCandidates>(
      buffer,
      keys,
      static_cast<int>(num_items),
      identify_candidates_op,
      candidate_callback_op,
      value_sources);
  }

  // No-op terminal flush. Present for parity with the accumulating sister classes; the
  // call site collapses to nothing.
  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  // Shared body for both overloads. `IsFull` is the compile-time switch that elides
  // the per-item classify-loop bound check on the hot full-tile path. `num_items` is
  // the runtime per-tile valid count (only used when `IsFull == false`).
  // The class-level `LazyValueLoad` propagates into `partition_atomics_fused`;
  // ignored by the Staged / SharedMem strategies.
  template <bool IsFull,
            bool HasCandidates,
            typename IdentifyCandidatesOp,
            typename CandidateCallbackOp,
            typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_impl(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    int num_items,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op,
    ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the partition pairs them positionally.");

    // Per-thread number of valid items. Full path: every slot is valid; partial path:
    // derive from the global per-tile num_items.
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

    // Atomics* strategies: route through the unified `partition_atomics_fused` for
    // both classify modes. The classifier abstraction picks whether the per-item
    // class is recomputed at each use site (`AtomicsInlinedClassify`) or read from
    // a pre-filled register array (`AtomicsPreClassify`). For the precomputed case
    // the classifier's ctor also fires the candidate callback up front, so the
    // scatter receives `noop_callback_op` to statically elide the second firing.
    if constexpr (bp_detail::is_atomics_strategy_v<Strategy>)
    {
      if constexpr (bp_detail::is_inlined_classify_v<Strategy>)
      {
        auto classifier = bp_detail::make_inlined_classifier<IsFull>(identify_candidates_op, num_thread_items);
        partition_atomics_fused<IsFull, HasCandidates>(
          buffer, keys, num_thread_items, classifier, candidate_callback_op, value_sources);
      }
      else
      {
        bp_detail::precomputed_classifier<KeyT, ItemsPerThread, IsFull, HasCandidates> classifier{
          keys, num_thread_items, identify_candidates_op, candidate_callback_op};
        bp_detail::noop_callback_op noop_cb{};
        partition_atomics_fused<IsFull, HasCandidates>(
          buffer, keys, num_thread_items, classifier, noop_cb, value_sources);
      }
    }
    else
    {
      // Staged / SharedMem strategies still need a precomputed `classes[]` because
      // their cooperative smem-scatter reads it more than once.
      candidate_class classes[ItemsPerThread];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        const bool is_valid = IsFull ? true : (j < num_thread_items);
        classes[j]          = is_valid ? identify_candidates_op(keys[j]) : candidate_class::rejected;

        if constexpr (HasCandidates)
        {
          // Architecture §10.2: callback fires for every `candidate`-classified item,
          // including ones the candidate reserve op subsequently drops (cap clamp).
          if (is_valid && classes[j] == candidate_class::candidate)
          {
            candidate_callback_op(keys[j]);
          }
        }
        else
        {
          // HasCandidates == false: collapse `candidate` onto `selected`. The callback
          // is statically guaranteed never to fire.
          if (is_valid && classes[j] == candidate_class::candidate)
          {
            classes[j] = candidate_class::selected;
          }
        }
      }

      if constexpr (Strategy == BlockPartitionStrategy::Staged)
      {
        partition_staged<IsFull, HasCandidates>(buffer, keys, classes, value_sources, num_items);
      }
      else
      {
        partition_shared_mem<IsFull, HasCandidates>(buffer, keys, classes, value_sources, num_items);
      }
    }
  }

  // -----------------------------------------------------------------
  // Atomics strategies (fused): scatter via a unified per-item loop driven by a
  // user-supplied `Classifier` with signature `(KeyT, int j) -> candidate_class`.
  // The classifier abstracts the precomputed-vs-inlined decision.
  // -----------------------------------------------------------------
  template <bool IsFull,
            bool HasCandidates,
            typename Classifier,
            typename CandidateCallbackOp,
            typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_atomics_fused(
    ScratchStorage& /*buffer*/,
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    CandidateCallbackOp& candidate_callback_op,
    ValueSourcesTuple& value_sources)
  {
    // Atomics* strategies don't load value channels into smem -- they scatter each
    // value as soon as the item claims a global slot. Eager (LazyValueLoad==false)
    // pre-loads per-channel values into a register array; lazy fetches via
    // `gather_one(j)` at the scatter site for non-rejected items only.
    static_assert(num_value_channels <= 1,
                  "atomics partition supports keys-only or single-value-channel "
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
        partition_atomics_fused_scatter<IsFull, HasCandidates, /*KeysOnly=*/false>(
          keys, num_thread_items, classifier, candidate_callback_op, value_sources, unused_values);
      }
      else
      {
        typename source_t::ScratchStorage chan_scratch{};
        auto h = src.submit_load(chan_scratch);
        value_t values[ItemsPerThread]{};
        h.complete_load(values);

        partition_atomics_fused_scatter<IsFull, HasCandidates, /*KeysOnly=*/false>(
          keys, num_thread_items, classifier, candidate_callback_op, value_sources, values);
      }
    }
    else
    {
      int unused_dummy[1]{};
      (void) unused_dummy;
      partition_atomics_fused_scatter<IsFull, HasCandidates, /*KeysOnly=*/true>(
        keys, num_thread_items, classifier, candidate_callback_op, value_sources, unused_dummy);
    }
  }

  // Unified scatter loop for the Atomics* strategies. Two independent 2-way (do/skip)
  // branches per unrolled item -- avoids the per-item indirect-branch table in
  // `c[0x2]` that ptxas would emit for a 3-way `rejected` / `selected` / `candidate`
  // cascade.
  template <bool IsFull,
            bool HasCandidates,
            bool KeysOnly,
            typename Classifier,
            typename CandidateCallbackOp,
            typename ValueSourcesTuple,
            typename ValuesArr>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_atomics_fused_scatter(
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    CandidateCallbackOp& candidate_callback_op,
    ValueSourcesTuple& value_sources,
    ValuesArr& values)
  {
    (void) num_thread_items;

    // Helper: fetch the per-item value either from the pre-loaded register array
    // (`LazyValueLoad == false`) or via on-demand single-item gather through the
    // value channel's data source (`LazyValueLoad == true`). For keys-only paths
    // the helper is never called (statically guarded).
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

    if constexpr (HasCandidates)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        // The classifier encapsulates the `is_valid` check internally, returning
        // `rejected` for items past `num_thread_items` on partial tiles.
        const candidate_class c = classifier(keys[j], j);

        // Architecture §10.2: callback fires for every `candidate`-classified item,
        // including ones the candidate reserve op subsequently drops (cap clamp).
        if (c == candidate_class::candidate)
        {
          candidate_callback_op(keys[j]);
        }

        if (c == candidate_class::selected)
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
              auto& sink                       = ::cuda::std::get<0>(sinks_);
              sink.selected_values_out[r.first] = sink.selected_value_transform(get_value(j));
            }
          }
        }
        if (c == candidate_class::candidate)
        {
          const auto r = reserve_cand_(CandidateOffsetT{1});
          bool granted = true;
          if constexpr (CandidateReserveOp::may_grant_less)
          {
            granted = (r.second != CandidateOffsetT{0});
          }
          if (granted)
          {
            cand_iter_[r.first] = cand_xform_(keys[j]);
            if constexpr (!KeysOnly)
            {
              auto& sink                        = ::cuda::std::get<0>(sinks_);
              sink.candidate_values_out[r.first] = sink.candidate_value_transform(get_value(j));
            }
          }
        }
      }
    }
    else
    {
      // `!HasCandidates`: `candidate`-classified items collapse onto `selected`. A
      // single `!= rejected` guard suffices; the callback is statically a no-op and
      // doesn't fire.
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        const candidate_class c = classifier(keys[j], j);

        if (c != candidate_class::rejected)
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
              auto& sink                       = ::cuda::std::get<0>(sinks_);
              sink.selected_values_out[r.first] = sink.selected_value_transform(get_value(j));
            }
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------
  // Staged strategy: smem scatter into `buffer.phase.keys` then cooperative
  // coalesced store via reserve op. Per-channel value path runs sequentially after
  // the keys phase: each channel loads (sub-brokered scratch), scatters into the
  // channel's `values[]` slot, then cooperatively stores.
  // -----------------------------------------------------------------
  template <bool IsFull, bool HasCandidates, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_staged(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    ValueSourcesTuple& value_sources,
    int num_items)
  {
    int positions[ItemsPerThread];

    // Phase 1: scatter keys + remember per-thread positions.
    if (threadIdx.x == 0)
    {
      buffer.cnt.counters[0] = 0;
      if constexpr (HasCandidates)
      {
        buffer.cnt.counters[1] = 0;
      }
    }
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (classes[j] == candidate_class::rejected)
      {
        positions[j] = -1;
      }
      else if constexpr (!HasCandidates)
      {
        const int pos          = atomicAdd(&buffer.cnt.counters[0], 1);
        buffer.phase.keys[pos] = keys[j];
        positions[j]           = pos;
      }
      else
      {
        if (classes[j] == candidate_class::selected)
        {
          const int pos          = atomicAdd(&buffer.cnt.counters[0], 1);
          buffer.phase.keys[pos] = keys[j];
          positions[j]           = pos;
        }
        else // candidate
        {
          const int pos          = atomicAdd(&buffer.cnt.counters[1], 1);
          const int idx          = tile_items - 1 - pos;
          buffer.phase.keys[idx] = keys[j];
          positions[j]           = idx;
        }
      }
    }
    __syncthreads();

    // Phase 2: snapshot counts; thread 0 claims global bases via reserve ops.
    const int selected_cnt  = buffer.cnt.counters[0];
    const int candidate_cnt = HasCandidates ? buffer.cnt.counters[1] : 0;

    if (threadIdx.x == 0)
    {
      const auto sel                   = reserve_sel_(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_bases.selected = sel.first;
      buffer.cnt.granted_selected      = static_cast<SelectedOffsetT>(sel.second);
      if constexpr (HasCandidates)
      {
        const auto cand                   = reserve_cand_(static_cast<CandidateOffsetT>(candidate_cnt));
        buffer.cnt.global_bases.candidate = cand.first;
        buffer.cnt.granted_candidate      = static_cast<CandidateOffsetT>(cand.second);
      }
    }
    __syncthreads();

    const SelectedOffsetT sel_base   = buffer.cnt.global_bases.selected;
    const CandidateOffsetT cand_base = HasCandidates ? buffer.cnt.global_bases.candidate : CandidateOffsetT{};
    const SelectedOffsetT sel_to_write =
      SelectedReserveOp::may_grant_less ? buffer.cnt.granted_selected : static_cast<SelectedOffsetT>(selected_cnt);
    const CandidateOffsetT cand_to_write =
      HasCandidates
        ? (CandidateReserveOp::may_grant_less
             ? buffer.cnt.granted_candidate
             : static_cast<CandidateOffsetT>(candidate_cnt))
        : CandidateOffsetT{};

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter_[sel_base + static_cast<SelectedOffsetT>(i)] = sel_xform_(buffer.phase.keys[i]);
    }
    if constexpr (HasCandidates)
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
      {
        cand_iter_[cand_base + static_cast<CandidateOffsetT>(i)] =
          cand_xform_(buffer.phase.keys[tile_items - candidate_cnt + i]);
      }
    }

    // Per-channel values phases. Each channel's load + scatter is sequential in time
    // (sub-brokered through `buffer.phase.per_channel`). After the keys cooperative
    // store, the keys[] arena is no longer needed -- we sync and reuse the union slot.
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
        if constexpr (HasCandidates)
        {
          for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
          {
            sink.candidate_values_out[cand_base + static_cast<CandidateOffsetT>(i)] =
              sink.candidate_value_transform(chan_phase.values[tile_items - candidate_cnt + i]);
          }
        }
        __syncthreads();
      });
    }
    __syncthreads();
  }

  // -----------------------------------------------------------------
  // SharedMem strategy: keys + per-channel values coexist in smem (within `phase.kv`),
  // then a single coalesced flush per stream. Pre-Phase-1 delegate loads alias with
  // the kv arena via the top-level union.
  // -----------------------------------------------------------------
  template <bool IsFull, bool HasCandidates, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_shared_mem(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    ValueSourcesTuple& value_sources,
    int num_items)
  {
    static_assert(num_value_channels <= 1,
                  "shared_mem partition supports keys-only or single-value-channel "
                  "today; multi-channel needs a heterogeneous register-array tuple.");

    // Pre-Phase-1: load the (single) channel's values into registers via the
    // delegate-load slot. After this, the delegate_loads view of the phase union is
    // dead and we transition to the kv view at the next __syncthreads.
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

    // Phase 1: scatter keys + values into the kv arena. Both coexist within `kv`.
    __syncthreads();
    if (threadIdx.x == 0)
    {
      buffer.cnt.counters[0] = 0;
      if constexpr (HasCandidates)
      {
        buffer.cnt.counters[1] = 0;
      }
    }
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (classes[j] == candidate_class::rejected)
      {
        continue;
      }
      int idx;
      if constexpr (!HasCandidates)
      {
        idx = atomicAdd(&buffer.cnt.counters[0], 1);
      }
      else
      {
        if (classes[j] == candidate_class::selected)
        {
          idx = atomicAdd(&buffer.cnt.counters[0], 1);
        }
        else
        {
          const int pos = atomicAdd(&buffer.cnt.counters[1], 1);
          idx           = tile_items - 1 - pos;
        }
      }
      buffer.phase.kv.keys[idx] = keys[j];
      if constexpr (num_value_channels == 1)
      {
        CUB_NS_QUALIFIER::detail::at<0>(buffer.phase.kv.per_channel_values).values[idx] = reg_values[j];
      }
    }
    __syncthreads();

    const int selected_cnt  = buffer.cnt.counters[0];
    const int candidate_cnt = HasCandidates ? buffer.cnt.counters[1] : 0;

    if (threadIdx.x == 0)
    {
      const auto sel                   = reserve_sel_(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_bases.selected = sel.first;
      buffer.cnt.granted_selected      = static_cast<SelectedOffsetT>(sel.second);
      if constexpr (HasCandidates)
      {
        const auto cand                   = reserve_cand_(static_cast<CandidateOffsetT>(candidate_cnt));
        buffer.cnt.global_bases.candidate = cand.first;
        buffer.cnt.granted_candidate      = static_cast<CandidateOffsetT>(cand.second);
      }
    }
    __syncthreads();

    const SelectedOffsetT sel_base   = buffer.cnt.global_bases.selected;
    const CandidateOffsetT cand_base = HasCandidates ? buffer.cnt.global_bases.candidate : CandidateOffsetT{};
    const SelectedOffsetT sel_to_write =
      SelectedReserveOp::may_grant_less ? buffer.cnt.granted_selected : static_cast<SelectedOffsetT>(selected_cnt);
    const CandidateOffsetT cand_to_write =
      HasCandidates
        ? (CandidateReserveOp::may_grant_less
             ? buffer.cnt.granted_candidate
             : static_cast<CandidateOffsetT>(candidate_cnt))
        : CandidateOffsetT{};

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter_[sel_base + static_cast<SelectedOffsetT>(i)] = sel_xform_(buffer.phase.kv.keys[i]);
    }
    if constexpr (HasCandidates)
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
      {
        cand_iter_[cand_base + static_cast<CandidateOffsetT>(i)] =
          cand_xform_(buffer.phase.kv.keys[tile_items - candidate_cnt + i]);
      }
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
        if constexpr (HasCandidates)
        {
          for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
          {
            sink.candidate_values_out[cand_base + static_cast<CandidateOffsetT>(i)] =
              sink.candidate_value_transform(vs.values[tile_items - candidate_cnt + i]);
          }
        }
      });
    }
    __syncthreads();
  }

  // ---------------------------------------------------------------
  // Member state: sinks captured by the ctor, used by every Partition() call and by
  // the (no-op) epilogue(). Reserve ops + transforms + sinks tuple are stored by
  // reference (matching the previous per-call by-reference convention -- the agent's
  // factory-created instances outlive the partition object); iterators are stored by
  // value (CUB convention).
  // ---------------------------------------------------------------
  SelectedReserveOp& reserve_sel_;
  CandidateReserveOp& reserve_cand_;
  SelectedKeyOutTransformOp& sel_xform_;
  CandidateKeyOutTransformOp& cand_xform_;
  SelectedKeyOutIt sel_iter_;
  CandidateKeyOutIt cand_iter_;
  ValueChannelSinksTuple& sinks_;
};
} // namespace detail::topk

CUB_NAMESPACE_END
