// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private non-accumulating partition primitives (architecture §9). Three
//! self-contained class templates -- one per partitioning strategy:
//!
//!   - `block_partition_atomics<..., InlinedClassify>` -- no smem; per-non-rejected-item
//!     global atomic + scatter. `InlinedClassify == false` precomputes a
//!     `classes[ItemsPerThread]` register array up front; `InlinedClassify == true`
//!     recomputes the classification at each scatter use-site (frees the registers
//!     that would hold `classes[]`).
//!   - `block_partition_staged` -- smem scatter into a keys arena + cooperative
//!     coalesced store; per-channel values run sequentially after the keys phase,
//!     sub-brokering one slot of the phase union.
//!   - `block_partition_shared_mem` -- typed `keys[]` + per-channel `values[]` packed
//!     into the same arena; a single coalesced store per stream.
//!
//! Interface ("safe-both") contract shared with the accumulating sister class
//! `block_partition_accumulating_candidates` (`block_partition_accumulating.cuh`):
//!   - All sinks (reserve ops, output iterators, transforms, value-channel sink
//!     tuple) AND the classify hooks (`identify_candidates_op`,
//!     `candidate_callback_op`) are captured by ctor and stored as members.
//!     Per-call args reduce to per-tile data plus a bare
//!     `cuda::std::tuple<TileDataSource...>` for value sources.
//!   - `epilogue()` is argless on every variant. The three non-accumulating
//!     primitives' `epilogue()` is a no-op. The accumulating sister's `epilogue()`
//!     performs a terminal flush of any remaining buffered items.
//!   - The per-channel value bundle splits along the lifetime boundary:
//!     `value_channel_sinks_t` (captured at ctor) carries the iters + transforms
//!     + per-channel `value_t` and `data_source_scratch_t` typedefs;
//!     a per-call `value_sources_tuple_t` carries the live `TileDataSource`
//!     instances that the agent has called `set_tile_base()` on for the current
//!     tile.
//!
//! These primitives always operate as a true 2-way partition. The single-stream
//! "filter" path (where the classifier collapsed `candidate -> selected`) lives in
//! the dedicated `BlockFilter*` primitives in `block_filter.cuh`.
//!
//! Strategy selection is done by `strategy_to_partition_class_t<Strategy, ...>` in
//! `block_partition_accumulating.cuh`, which maps a `block_partition_strategy` enum
//! value to one of the three classes here (or to the accumulating sister class).

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
// the primitives' `partition()` calls.
enum class candidate_class
{
  selected,
  candidate,
  rejected,
};

// Strategy selector for the partition primitives. Picks the *partitioning* shape;
// the orthogonal `InlinedClassify` axis is a separate template / policy bool that
// every non-accumulating primitive accepts. The mapping from a strategy enum value
// to a class is performed by `strategy_to_partition_class_t<...>` in
// `block_partition_accumulating.cuh`.
//
//   Atomics                -- block_partition_atomics. No smem; per-non-rejected-item
//                             global atomic + scatter.
//   Staged                 -- block_partition_staged. Smem scatter into a keys arena +
//                             cooperative coalesced store; per-channel values run
//                             sequentially after the keys phase.
//   SharedMem              -- block_partition_shared_mem. Typed `keys[]` + per-channel
//                             `values[]` packed into the same arena; a single
//                             coalesced store per stream.
//   AccumulatingCandidates -- block_partition_accumulating_candidates: candidate stream
//                             buffered in smem and accumulated across multiple tiles;
//                             selected stream goes direct-to-global. Used by the
//                             agent's `buffered`-mode pass.
//   SpeculativeBoth        -- block_partition_speculative: both candidate and selected
//                             streams accumulate in smem, but with speculative slot
//                             reservation -- items overflowing the buffer fall back
//                             to per-item global atomics. Trades one bit-mask uint32
//                             per stream + one sync for register parity with
//                             `Atomics`. Setting the selected-stream capacity to 0
//                             via the agent policy degrades that stream to pure
//                             atomics (useful when the selected stream is dense).
enum class block_partition_strategy
{
  atomics,
  staged,
  shared_mem,
  accumulating_candidates,
  speculative_both,
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
// Shared scratch-storage building blocks (architecture §9.4). Per-strategy
// assembled `ScratchStorage` structs live as nested types of the three
// partition classes below; this `bp_detail` namespace just holds the pieces
// (counters, per-channel slots, tuple-mapping helpers, classifiers) that they
// compose from.
//---------------------------------------------------------------------

namespace bp_detail
{
// Counters for the staged / shared_mem strategies. Phase 1 uses 32-bit smem atomics
// (`int counters[2]`); Phase 2 uses the actual offset types for the global bases. The
// two phases are separated by `__syncthreads()` so the union reuse is safe. `cnt`
// lives outside the per-phase union (architecture O8): it stays alive across the
// entire `partition()` call.
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
// the SharedMem primitive (sizing a per-channel register array conditionally on the
// first channel's `value_t`) and the accumulating sister class. The `Tuple` is
// expected to be either a `ValueTypesTuple` (whose elements are already value
// types) or a `ValueChannelMetaTuple` (whose elements expose a `value_t` typedef);
// the `value_t_or_default` and `meta_value_t_or_default` variants below select
// the right idiom.
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
// can call the classifier unconditionally.
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

// Precomputed-classes adapter: builds a `classes[ItemsPerThread]` register array
// up front and fires the candidate callback once per `candidate`-classified item.
// Reusable across `block_partition_atomics<InlinedClassify=false>`, `block_partition_staged`,
// and `block_partition_shared_mem` -- the latter two consume the `classes` array
// directly from `.classes`.
//
// `operator()(KeyT, int j)` returns `classes[j]`, ignoring the key.
template <typename KeyT, int ItemsPerThread, bool IsFull>
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

      // Architecture §10.2: callback fires for every `candidate`-classified item,
      // including ones the candidate reserve op subsequently drops (cap clamp).
      if (is_valid && classes[j] == candidate_class::candidate)
      {
        callback(keys[j]);
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

// Computes the per-thread valid-item count for a tile, given the per-tile valid
// count. On the full-tile path returns `ItemsPerThread` unconditionally.
template <bool IsFull, int ItemsPerThread>
_CCCL_DEVICE _CCCL_FORCEINLINE int compute_num_thread_items(int num_items)
{
  if constexpr (IsFull)
  {
    (void) num_items;
    return ItemsPerThread;
  }
  else
  {
    const int tb_offset = static_cast<int>(threadIdx.x) * ItemsPerThread;
    return (tb_offset >= num_items) ? 0 : static_cast<int>((::cuda::std::min) (ItemsPerThread, num_items - tb_offset));
  }
}

//---------------------------------------------------------------------
// `partition_storage_layout` -- agent-side smem layout helper.
//
// Picks the right storage shape for the union (`partition_t::TempStorage`, prefix-sum
// state, load scratch, partition scratch) based on whether the partition class
// carries persistent state across calls. Both shapes are temporally safe -- prefix
// sum runs strictly after the partition's terminal `epilogue()` and only on the
// last block, and load+partition are sequential within a tile.
//
//   NeedsPersistent == false (BlockPartition{Atomics,Staged,SharedMem} / BlockFilter*;
//     `TempStorage` is empty): `partition_state` is empty; the union spans
//     {keys_source_scratch | prefix_sum | partition_scratch}, giving byte-equivalent
//     smem footprint to the original three-way union.
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
// the non-accumulating primitives' `TempStorage = struct{}` is empty (so we get
// the 3-way union), and the accumulating variants' `TempStorage` is non-empty (so
// we get the persistent + scratch layout). Empty-struct vs. wrapped-data is a
// clean signal because `cub::Uninitialized<T>` carries a `DeviceWord storage[N]`
// member, which makes `is_empty_v` correctly return `false` on the wrapper.
template <typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
using partition_storage_layout_for_t =
  partition_storage_layout<!::cuda::std::is_empty_v<typename PartitionT::TempStorage>,
                           PartitionT,
                           KeysSourceScratchT,
                           PrefixSumT>;

} // namespace bp_detail

//---------------------------------------------------------------------
// `block_partition_atomics` -- per-non-rejected-item global atomic + scatter,
// no smem. `InlinedClassify` selects between the precomputed-classes form
// (smaller scatter loop, larger live register set for the `classes[]` array)
// and the inlined-classify form (recomputes classification at each scatter
// use-site, frees those registers). Mapped from
// `block_partition_strategy::atomics`. The `InlinedClassify` axis is independent
// and is also accepted (with the same semantics) by `block_partition_staged` and
// `block_partition_shared_mem`.
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
          bool InlinedClassify,
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
          typename ValueChannelSinksTuple      = ::cuda::std::tuple<>,
          typename ValueTypesTuple             = ::cuda::std::tuple<>,
          typename DataSourceScratchTypesTuple = ::cuda::std::tuple<>,
          bool LazyValueLoad                   = false>
class block_partition_atomics
{
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueTypesTuple>::value
                  == ::cuda::std::tuple_size<DataSourceScratchTypesTuple>::value,
                "ValueTypesTuple and DataSourceScratchTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value <= 1,
                "block_partition_atomics supports keys-only or single-value-channel today; "
                "multi-channel needs a per-channel value array.");

public:
  static constexpr int tile_items         = BlockThreads * ItemsPerThread;
  static constexpr int num_value_channels = static_cast<int>(::cuda::std::tuple_size<ValueChannelSinksTuple>::value);

  // Class-lifetime persistent state. Empty (no carried state across partition() calls).
  struct TempStorage
  {};

  // Per-tile scratch. Empty: the atomics strategies hold no smem state -- per-item
  // scatter goes direct to the user's iterators via the captured reserve ops.
  struct ScratchStorage
  {};

  // Ctor (safe-both shape): captures sinks + classify hooks. The TempStorage
  // parameter is unused (Atomics has no persistent state) but is taken for
  // parity with the accumulating sister class.
  _CCCL_DEVICE _CCCL_FORCEINLINE block_partition_atomics(
    TempStorage& /*storage*/,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksTuple& value_channel_sinks,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op)
      : reserve_sel(reserve_selected)
      , reserve_cand(reserve_candidate)
      , sel_xform(selected_key_transform)
      , cand_xform(candidate_key_transform)
      , sel_iter(selected_keys_out)
      , cand_iter(candidate_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_candidates_op)
      , callback_op(candidate_callback_op)
  {}

  // Full-tile overload: no per-item bound check inside the classify loop.
  template <typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourcesTuple& value_sources)
  {
    partition_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_sources);
  }

  // Partial-tile overload: classify loop bound-checks against num_items.
  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourcesTuple& value_sources)
  {
    partition_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_sources);
  }

  // No-op terminal flush. Present for parity with the accumulating sister class; the
  // call site collapses to nothing.
  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  // Top-level dispatch: peels off `cand_reserve_open` at the tile boundary
  // (only when the candidate reserve op can grant less than requested) and
  // selects the `HasCandidateStream` specialization. The per-thread flag is
  // mutated inside the `HasCandidateStream=true` scatter when a thread
  // observes a 0-grant; subsequent tiles then take the cheaper specialization
  // for that thread.
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the partition pairs them positionally.");

    const int num_thread_items = bp_detail::compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    if constexpr (CandidateReserveOp::may_grant_less)
    {
      if (cand_reserve_open)
      {
        partition_dispatch_classify<IsFull, /*HasCandidateStream=*/true>(buffer, keys, num_thread_items, value_sources);
      }
      else
      {
        partition_dispatch_classify<IsFull, /*HasCandidateStream=*/false>(buffer, keys, num_thread_items, value_sources);
      }
    }
    else
    {
      // `may_grant_less=false`: the reserve op never grants 0, so the
      // candidate stream stays open for the lifetime of this thread. Skip
      // the runtime branch and the `HasCandidateStream=false` template
      // instantiation entirely.
      partition_dispatch_classify<IsFull, /*HasCandidateStream=*/true>(buffer, keys, num_thread_items, value_sources);
    }
  }

  // Inner classifier-dispatch: same shape as the old `partition_impl` body,
  // factored out so the cand-stream dispatch above doesn't need to duplicate
  // it. Both `HasCandidateStream` paths flow through here.
  template <bool IsFull, bool HasCandidateStream, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_dispatch_classify(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_thread_items, ValueSourcesTuple& value_sources)
  {
    if constexpr (InlinedClassify)
    {
      auto classifier = bp_detail::make_inlined_classifier<IsFull>(identify_op, num_thread_items);
      partition_atomics_fused<IsFull, HasCandidateStream>(
        buffer, keys, num_thread_items, classifier, callback_op, value_sources);
    }
    else
    {
      bp_detail::precomputed_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op, callback_op};
      bp_detail::noop_callback_op noop_cb{};
      partition_atomics_fused<IsFull, HasCandidateStream>(
        buffer, keys, num_thread_items, classifier, noop_cb, value_sources);
    }
  }

  // -----------------------------------------------------------------
  // Fused scatter: drives a unified per-item loop via a user-supplied
  // `Classifier` with signature `(KeyT, int j) -> candidate_class`. The
  // classifier abstracts the precomputed-vs-inlined decision; the callback is
  // either the original `candidate_callback_op` (inlined-classify path) or a
  // `noop_callback_op` (precomputed-classify path, where the classifier
  // already fired callbacks at construction).
  //
  // `HasCandidateStream` selects between the full path (per-candidate
  // `reserve_cand` + write) and the closed-stream specialization (no
  // candidate-side reserve/write at all; the candidate callback is still
  // fired, so the threshold-update protocol is preserved). The dispatch is
  // peeled by `partition_impl` from the per-thread `cand_reserve_open`
  // flag.
  // -----------------------------------------------------------------
  template <bool IsFull, bool HasCandidateStream, typename Classifier, typename CandidateCallbackOpT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_atomics_fused(
    ScratchStorage& /*buffer*/,
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    CandidateCallbackOpT& candidate_callback_op,
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
        partition_atomics_fused_scatter<IsFull, HasCandidateStream, /*KeysOnly=*/false>(
          keys, num_thread_items, classifier, candidate_callback_op, value_sources, unused_values);
      }
      else
      {
        typename source_t::ScratchStorage chan_scratch{};
        auto h = src.submit_load(chan_scratch);
        value_t values[ItemsPerThread]{};
        h.complete_load(values);

        partition_atomics_fused_scatter<IsFull, HasCandidateStream, /*KeysOnly=*/false>(
          keys, num_thread_items, classifier, candidate_callback_op, value_sources, values);
      }
    }
    else
    {
      int unused_dummy[1]{};
      (void) unused_dummy;
      partition_atomics_fused_scatter<IsFull, HasCandidateStream, /*KeysOnly=*/true>(
        keys, num_thread_items, classifier, candidate_callback_op, value_sources, unused_dummy);
    }
  }

  // Unified scatter loop. Two independent 2-way (do/skip) branches per unrolled
  // item -- avoids the per-item indirect-branch table in `c[0x2]` that ptxas would
  // emit for a 3-way `rejected` / `selected` / `candidate` cascade.
  //
  // With `HasCandidateStream=false`, the per-item `reserve_cand` + candidate
  // write block is elided entirely (the candidate callback still fires for
  // every candidate-classified item). The `any_cand_granted_zero` flag is
  // also elided -- it only exists in the `HasCandidateStream=true` path,
  // where it records that the per-thread candidate stream just closed so
  // the outer dispatcher can take the cheaper specialization on the next
  // tile.
  template <bool IsFull,
            bool HasCandidateStream,
            bool KeysOnly,
            typename Classifier,
            typename CandidateCallbackOpT,
            typename ValueSourcesTuple,
            typename ValuesArr>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_atomics_fused_scatter(
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    CandidateCallbackOpT& candidate_callback_op,
    ValueSourcesTuple& value_sources,
    ValuesArr& values)
  {
    (void) num_thread_items;

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

    // Tracks per-thread "saw 0 from `reserve_cand`" within this tile. Folded
    // into `cand_reserve_open` after the loop so the next tile dispatches to
    // the `HasCandidateStream=false` specialization for this thread. Lives in
    // the `HasCandidateStream=true` && `may_grant_less=true` path only;
    // everywhere else the gating `if constexpr`s leave it dead. The
    // `[[maybe_unused]]` attribute suppresses the "unused variable" warning
    // on the dead-code paths without taking the variable's address (which a
    // `(void)` cast would).
    [[maybe_unused]] bool any_cand_granted_zero = false;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const candidate_class c = classifier(keys[j], j);

      // Architecture §10.2: callback fires for every `candidate`-classified item,
      // including ones the candidate reserve op subsequently drops (cap clamp)
      // and the ones we drop here when the candidate stream has closed.
      if (c == candidate_class::candidate)
      {
        candidate_callback_op(keys[j]);
      }

      if (c == candidate_class::selected)
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
          if constexpr (!KeysOnly)
          {
            auto& sink                        = ::cuda::std::get<0>(sinks);
            sink.selected_values_out[r.first] = sink.selected_value_transform(get_value(j));
          }
        }
      }
      if constexpr (HasCandidateStream)
      {
        if (c == candidate_class::candidate)
        {
          const auto r = reserve_cand(CandidateOffsetT{1});
          bool granted = true;
          if constexpr (CandidateReserveOp::may_grant_less)
          {
            granted               = (r.second != CandidateOffsetT{0});
            any_cand_granted_zero = any_cand_granted_zero || !granted;
          }
          if (granted)
          {
            cand_iter[r.first] = cand_xform(keys[j]);
            if constexpr (!KeysOnly)
            {
              auto& sink                         = ::cuda::std::get<0>(sinks);
              sink.candidate_values_out[r.first] = sink.candidate_value_transform(get_value(j));
            }
          }
        }
      }
    }

    if constexpr (HasCandidateStream && CandidateReserveOp::may_grant_less)
    {
      if (any_cand_granted_zero)
      {
        cand_reserve_open = false;
      }
    }
  }

  // Captured at ctor; used by every partition() call.
  SelectedReserveOp& reserve_sel;
  CandidateReserveOp& reserve_cand;
  SelectedKeyOutTransformOp& sel_xform;
  CandidateKeyOutTransformOp& cand_xform;
  SelectedKeyOutIt sel_iter;
  CandidateKeyOutIt cand_iter;
  ValueChannelSinksTuple& sinks;
  IdentifyCandidatesOp& identify_op;
  CandidateCallbackOp& callback_op;

  // Per-thread monotonic flag for the candidate stream. Only consulted when
  // `CandidateReserveOp::may_grant_less` is true (otherwise the reserve op
  // never grants less than requested and the flag is dead code).
  //
  // Top-k guarantees: once `reserve_cand` grants 0 for any thread, the
  // device-global candidate counter is past the back-grow cap, so every
  // *subsequent* call from *any* thread also grants 0. We exploit that by
  // tracking the per-thread observation inside the scatter loop and, once
  // set, dispatching the next tile's classify+scatter to a
  // `HasCandidateStream=false`-specialized instantiation that drops the
  // per-item candidate-reserve atomic + candidate-write entirely. Items
  // classified as `candidate` still fire the candidate callback (architecture
  // §10.2) and skip the selected stream, equivalent to the granted-0 drop
  // path in the full-stream specialization.
  //
  // Convergence is bounded: any thread that called `reserve_cand(1)` and
  // observed 0 has flag=false by tile end; any thread that didn't classify
  // a candidate in that tile retains flag=true, but its next candidate
  // observation will set the flag, so all threads converge to flag=false
  // within one extra tile of tail-divergence. The flag is `may_grant_less`-
  // gated at compile time, so the `may_grant_less=false` path pays nothing
  // (no runtime branch, no extra template instantiation).
  bool cand_reserve_open = true;
};

//---------------------------------------------------------------------
// `block_partition_staged` -- smem scatter into a keys arena + cooperative coalesced
// store. Per-channel value path runs sequentially after the keys phase: each
// channel loads (sub-brokered scratch), scatters into the channel's `values[]`
// slot, then cooperatively stores. Mapped from `block_partition_strategy::staged`.
//
// `InlinedClassify` selects between materializing a `classes[ItemsPerThread]`
// register array up front (the candidate callback then fires from
// `precomputed_classifier`'s ctor) and recomputing the classification at the
// per-item smem-scatter site (the candidate callback then fires inline in the
// scatter loop). The choice is independent of the Atomics version's
// `InlinedClassify`.
//
// `LazyValueLoad` selects between eagerly loading a tile of values into a
// register array up front (default) and gathering only the surviving values
// directly into smem via the source's `gather_one(j)` operation. Only honored
// when the value source supports `gather_one` (the agent enforces this by
// composing `multi_source_data_source<direct_data_source, direct_data_source>`).
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
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
          typename ValueChannelSinksTuple      = ::cuda::std::tuple<>,
          typename ValueTypesTuple             = ::cuda::std::tuple<>,
          typename DataSourceScratchTypesTuple = ::cuda::std::tuple<>,
          bool LazyValueLoad                   = false,
          bool InlinedClassify                 = false>
class block_partition_staged
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
  // carries the per-tile counts and the broadcast `granted_*` slots across the
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

    bp_detail::partition_counters<SelectedOffsetT, CandidateOffsetT> cnt;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE block_partition_staged(
    TempStorage& /*storage*/,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksTuple& value_channel_sinks,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op)
      : reserve_sel(reserve_selected)
      , reserve_cand(reserve_candidate)
      , sel_xform(selected_key_transform)
      , cand_xform(candidate_key_transform)
      , sel_iter(selected_keys_out)
      , cand_iter(candidate_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_candidates_op)
      , callback_op(candidate_callback_op)
  {}

  template <typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourcesTuple& value_sources)
  {
    partition_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_sources);
  }

  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourcesTuple& value_sources)
  {
    partition_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_sources);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the partition pairs them positionally.");

    const int num_thread_items = bp_detail::compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    int positions[ItemsPerThread];

    // Phase 1: scatter keys + remember per-thread positions.
    if (threadIdx.x == 0)
    {
      buffer.cnt.counters[0] = 0;
      buffer.cnt.counters[1] = 0;
    }
    __syncthreads();

    if constexpr (InlinedClassify)
    {
      auto classifier = bp_detail::make_inlined_classifier<IsFull>(identify_op, num_thread_items);
      classify_and_scatter_keys</*FireCallbackInline=*/true>(buffer, keys, classifier, positions);
    }
    else
    {
      // `precomputed_classifier`'s ctor fires the candidate callback for every
      // `candidate`-classified item, so the scatter loop below only needs to
      // route into the smem arena.
      bp_detail::precomputed_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op, callback_op};
      classify_and_scatter_keys</*FireCallbackInline=*/false>(buffer, keys, classifier, positions);
    }
    __syncthreads();

    // Phase 2: snapshot counts; thread 0 claims global bases via reserve ops.
    const int selected_cnt  = buffer.cnt.counters[0];
    const int candidate_cnt = buffer.cnt.counters[1];

    if (threadIdx.x == 0)
    {
      const auto sel                   = reserve_sel(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_bases.selected = sel.first;
      buffer.cnt.granted_selected      = static_cast<SelectedOffsetT>(sel.second);

      const auto cand                   = reserve_cand(static_cast<CandidateOffsetT>(candidate_cnt));
      buffer.cnt.global_bases.candidate = cand.first;
      buffer.cnt.granted_candidate      = static_cast<CandidateOffsetT>(cand.second);
    }
    __syncthreads();

    const SelectedOffsetT sel_base   = buffer.cnt.global_bases.selected;
    const CandidateOffsetT cand_base = buffer.cnt.global_bases.candidate;
    const SelectedOffsetT sel_to_write =
      SelectedReserveOp::may_grant_less ? buffer.cnt.granted_selected : static_cast<SelectedOffsetT>(selected_cnt);
    const CandidateOffsetT cand_to_write = CandidateReserveOp::may_grant_less
                                           ? buffer.cnt.granted_candidate
                                           : static_cast<CandidateOffsetT>(candidate_cnt);

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter[sel_base + static_cast<SelectedOffsetT>(i)] = sel_xform(buffer.phase.keys[i]);
    }
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
    {
      cand_iter[cand_base + static_cast<CandidateOffsetT>(i)] =
        cand_xform(buffer.phase.keys[tile_items - candidate_cnt + i]);
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

        auto& sink       = ::cuda::std::get<I>(sinks);
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
          // Eager: load the full tile of values into a register array, then
          // scatter the surviving ones into the smem arena.
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
        for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
        {
          sink.candidate_values_out[cand_base + static_cast<CandidateOffsetT>(i)] =
            sink.candidate_value_transform(chan_phase.values[tile_items - candidate_cnt + i]);
        }
        __syncthreads();
      });
    }
    __syncthreads();
  }

  // Mode-agnostic Phase 1 body: classify every per-thread item, fire the candidate
  // callback inline if the precomputed-classifier ctor hasn't already done so, and
  // scatter the key into the smem arena (selected at the front, candidate at the
  // back). `positions[j]` records the smem slot for use by the value channels;
  // `-1` marks rejected items. `Classifier` exposes `operator()(KeyT, int j) ->
  // candidate_class` for both inlined and precomputed modes.
  template <bool FireCallbackInline, typename Classifier>
  _CCCL_DEVICE _CCCL_FORCEINLINE void classify_and_scatter_keys(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], Classifier& classifier, int (&positions)[ItemsPerThread])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const candidate_class c = classifier(keys[j], j);
      if constexpr (FireCallbackInline)
      {
        // Architecture §10.2: callback fires for every `candidate`-classified item.
        if (c == candidate_class::candidate)
        {
          callback_op(keys[j]);
        }
      }
      if (c == candidate_class::rejected)
      {
        positions[j] = -1;
      }
      else if (c == candidate_class::selected)
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
// `block_partition_shared_mem` -- keys + per-channel values coexist in smem (within
// `phase.kv`), then a single coalesced flush per stream. Pre-Phase-1 delegate
// loads alias with the kv arena via the top-level phase union. Mapped from
// `block_partition_strategy::shared_mem`.
//
// Single-value-channel only today; multi-channel needs a heterogeneous
// register-array tuple.
//
// `InlinedClassify` selects between materializing a `classes[ItemsPerThread]`
// register array up front (the candidate callback then fires from
// `precomputed_classifier`'s ctor) and recomputing the classification at the
// per-item smem-scatter site (the candidate callback then fires inline in the
// scatter loop). Independent of the Atomics version's `InlinedClassify`.
//
// `LazyValueLoad` selects between the eager pre-Phase-1 delegate load (default)
// and gathering only the surviving values directly into the kv arena via the
// source's `gather_one(j)` operation. Only honored when the value source
// supports `gather_one` (the agent enforces this by composing
// `multi_source_data_source<direct_data_source, direct_data_source>`).
//---------------------------------------------------------------------
template <int BlockThreads,
          int ItemsPerThread,
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
          typename ValueChannelSinksTuple      = ::cuda::std::tuple<>,
          typename ValueTypesTuple             = ::cuda::std::tuple<>,
          typename DataSourceScratchTypesTuple = ::cuda::std::tuple<>,
          bool LazyValueLoad                   = false,
          bool InlinedClassify                 = false>
class block_partition_shared_mem
{
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value
                  == ::cuda::std::tuple_size<ValueTypesTuple>::value,
                "ValueChannelSinksTuple and ValueTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueTypesTuple>::value
                  == ::cuda::std::tuple_size<DataSourceScratchTypesTuple>::value,
                "ValueTypesTuple and DataSourceScratchTypesTuple must have the same length.");
  static_assert(::cuda::std::tuple_size<ValueChannelSinksTuple>::value <= 1,
                "block_partition_shared_mem supports keys-only or single-value-channel today; "
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
  // union, same role as in block_partition_staged.
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

    bp_detail::partition_counters<SelectedOffsetT, CandidateOffsetT> cnt;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE block_partition_shared_mem(
    TempStorage& /*storage*/,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksTuple& value_channel_sinks,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op)
      : reserve_sel(reserve_selected)
      , reserve_cand(reserve_candidate)
      , sel_xform(selected_key_transform)
      , cand_xform(candidate_key_transform)
      , sel_iter(selected_keys_out)
      , cand_iter(candidate_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_candidates_op)
      , callback_op(candidate_callback_op)
  {}

  template <typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourcesTuple& value_sources)
  {
    partition_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_sources);
  }

  template <typename NumItemsT, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourcesTuple& value_sources)
  {
    partition_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_sources);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourcesTuple>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourcesTuple& value_sources)
  {
    static_assert(::cuda::std::tuple_size<ValueSourcesTuple>::value == num_value_channels,
                  "Per-call value sources tuple must have the same length as the class-level value channel sinks "
                  "tuple; the partition pairs them positionally.");

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

    // Phase 1: scatter keys + values into the kv arena. Both coexist within `kv`.
    __syncthreads();
    if (threadIdx.x == 0)
    {
      buffer.cnt.counters[0] = 0;
      buffer.cnt.counters[1] = 0;
    }
    __syncthreads();

    if constexpr (InlinedClassify)
    {
      auto classifier = bp_detail::make_inlined_classifier<IsFull>(identify_op, num_thread_items);
      classify_and_scatter_kv</*FireCallbackInline=*/true>(buffer, keys, classifier, value_sources, reg_values);
    }
    else
    {
      // `precomputed_classifier`'s ctor fires the candidate callback for every
      // `candidate`-classified item, so the scatter loop only needs to route into
      // the kv arena.
      bp_detail::precomputed_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op, callback_op};
      classify_and_scatter_kv</*FireCallbackInline=*/false>(buffer, keys, classifier, value_sources, reg_values);
    }
    __syncthreads();

    const int selected_cnt  = buffer.cnt.counters[0];
    const int candidate_cnt = buffer.cnt.counters[1];

    if (threadIdx.x == 0)
    {
      const auto sel                   = reserve_sel(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_bases.selected = sel.first;
      buffer.cnt.granted_selected      = static_cast<SelectedOffsetT>(sel.second);

      const auto cand                   = reserve_cand(static_cast<CandidateOffsetT>(candidate_cnt));
      buffer.cnt.global_bases.candidate = cand.first;
      buffer.cnt.granted_candidate      = static_cast<CandidateOffsetT>(cand.second);
    }
    __syncthreads();

    const SelectedOffsetT sel_base   = buffer.cnt.global_bases.selected;
    const CandidateOffsetT cand_base = buffer.cnt.global_bases.candidate;
    const SelectedOffsetT sel_to_write =
      SelectedReserveOp::may_grant_less ? buffer.cnt.granted_selected : static_cast<SelectedOffsetT>(selected_cnt);
    const CandidateOffsetT cand_to_write = CandidateReserveOp::may_grant_less
                                           ? buffer.cnt.granted_candidate
                                           : static_cast<CandidateOffsetT>(candidate_cnt);

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter[sel_base + static_cast<SelectedOffsetT>(i)] = sel_xform(buffer.phase.kv.keys[i]);
    }
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
    {
      cand_iter[cand_base + static_cast<CandidateOffsetT>(i)] =
        cand_xform(buffer.phase.kv.keys[tile_items - candidate_cnt + i]);
    }

    if constexpr (num_value_channels > 0)
    {
      bp_detail::tuple_for_each(sinks, [&](auto& sink, auto I_ic) {
        constexpr int I = static_cast<int>(decltype(I_ic)::value);
        auto& vs        = CUB_NS_QUALIFIER::detail::at<I>(buffer.phase.kv.per_channel_values);
        for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
        {
          sink.selected_values_out[sel_base + static_cast<SelectedOffsetT>(i)] =
            sink.selected_value_transform(vs.values[i]);
        }
        for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
        {
          sink.candidate_values_out[cand_base + static_cast<CandidateOffsetT>(i)] =
            sink.candidate_value_transform(vs.values[tile_items - candidate_cnt + i]);
        }
      });
    }
    __syncthreads();
  }

  // Mode-agnostic Phase 1 body: classify every per-thread item, fire the candidate
  // callback inline if the precomputed-classifier ctor hasn't already done so, and
  // scatter the surviving (key, value) pairs into the kv arena. `Classifier`
  // exposes `operator()(KeyT, int j) -> candidate_class` for both modes; the
  // value source is consulted via `gather_one` only when `LazyValueLoad == true`.
  template <bool FireCallbackInline, typename Classifier, typename ValueSourcesTuple, typename RegValuesArr>
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
      const candidate_class c = classifier(keys[j], j);
      if constexpr (FireCallbackInline)
      {
        if (c == candidate_class::candidate)
        {
          callback_op(keys[j]);
        }
      }
      if (c == candidate_class::rejected)
      {
        continue;
      }
      int idx;
      if (c == candidate_class::selected)
      {
        idx = atomicAdd(&buffer.cnt.counters[0], 1);
      }
      else
      {
        const int pos = atomicAdd(&buffer.cnt.counters[1], 1);
        idx           = tile_items - 1 - pos;
      }
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
} // namespace detail::topk

CUB_NAMESPACE_END
