// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private `BlockPartition` (architecture §9). Per-tile 2-way partition of
//! classified items into "selected" and "candidate" streams, with reserve-driven
//! global writes. Replaces the legacy `cub::detail::topk::BlockPartition` template
//! that previously lived at `cub/agent/topk/block_partition.cuh`.
//!
//! Differences vs the legacy primitive (architecture §2 gap analysis):
//!   - Two `Partition()` overloads: full and partial. Full elides the OOB classify
//!     check; partial bound-checks against `num_items`.
//!   - Reserve callbacks (`atomic_reserve_range_op`, `back_grow_capped_reserve_op`)
//!     live outside the primitive. Cap and back-write logic move into
//!     `back_grow_capped_reserve_op`'s `(base, granted)` math; the primitive itself is
//!     cap- and back-write-agnostic.
//!   - Per-call `IdentifyCandidatesOp` + compile-time `HasCandidates` integral_constant
//!     replaces a pre-classified `classes[]` array. The primitive does the
//!     classification itself and statically elides candidate-side machinery when
//!     `HasCandidates == false`.
//!   - Per-call `CandidateCallbackOp` fires for every item classified `candidate`
//!     (architecture §10) -- the histogram update lives in the agent today.
//!   - `ValueChannelsTuple` carries arbitrarily many value channels, each bundling a
//!     `TileDataSource` with selected / candidate output iterators and key-output
//!     transform ops (per-stream identity for now).
//!   - Sub-brokers value-channel `TileDataSource::ScratchStorage` out of the
//!     `ScratchStorage` arena via the `phase_union` / `phase_aggregate` typed
//!     accessors (`cub::detail::at<I>`).

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

// Strategy selector for `BlockPartition`. Spelled in the legacy CamelCase form so it
// can be a drop-in replacement for the legacy enum that the tuning policy already
// exposes (`topk_policy::partition_strategy`).
//
//   Atomics   -- no smem; per-non-rejected-item global atomic + scatter.
//   Staged    -- smem scatter + cooperative coalesced store; phase_union arena.
//   SharedMem -- typed `keys[]` + per-channel `values[]` arrays + per-channel
//                delegate-load union (architecture §9.4).
enum class BlockPartitionStrategy
{
  Atomics,
  Staged,
  SharedMem,
};

// Compile-time selector for how `Partition()` materializes the per-item classification
// of input keys.
//
//   precomputed -- run `identify_candidates_op` once per item up front, store the
//                  results in a register-resident `candidate_class classes[ItemsPerThread]`
//                  array, then have the scatter loop read from it. Required by the
//                  `Staged` and `SharedMem` strategies (their cooperative smem scatter
//                  reads the array more than once).
//
//   inlined     -- skip the `classes[]` array entirely; recompute the classification
//                  inline at each scatter-loop use-site via `identify_candidates_op`.
//                  Frees the registers that would otherwise hold `classes[]` and lets
//                  ptxas kill `keys[j]` earlier when `identify_candidates_op` is
//                  cheap. Pays a small re-evaluation cost when the operator is
//                  expensive (e.g., types whose radix path involves a non-trivial
//                  decomposer or bit-twiddle). Only the `Atomics` strategy supports
//                  this mode (the smem-coordinating strategies need the array).
enum class BlockPartitionClassifyMode
{
  precomputed,
  inlined,
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

// Per-channel "loading or scattering" slot for the staged strategy. Inside one
// channel's processing window the data-source scratch and the per-channel value
// array are mutually exclusive in time -> internal phase_union.
template <typename Channel, int N>
struct staged_channel_phase
{
  union
  {
    typename Channel::data_source_t::ScratchStorage load;
    typename Channel::value_t values[N];
  };
};

// Per-channel "loading" slot for the shared_mem strategy.
template <typename Channel>
struct delegate_load_slot
{
  typename Channel::data_source_t::ScratchStorage load;
};

// Per-channel "values" slot for the shared_mem strategy.
template <typename Channel, int N>
struct values_slot
{
  typename Channel::value_t values[N];
};

// Map a tuple type-parameter through a (Channel, int) -> Out template. Equivalent to
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

// First channel's value_t (or a stand-in `int` when the tuple is empty). Used by the
// shared_mem strategy to size a per-channel register array conditionally.
template <typename Tuple>
struct first_channel_value
{
  using type = int;
};

template <typename Head, typename... Rest>
struct first_channel_value<::cuda::std::tuple<Head, Rest...>>
{
  using type = typename Head::value_t;
};

// Strategy-specific scratch declarations.

struct atomics_scratch
{};

template <typename KeyT, typename ValueChannelsTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct staged_scratch
{
  union phase_t
  {
    KeyT keys[TileItems];
    CUB_NS_QUALIFIER::detail::phase_union<map_tuple_t<staged_channel_phase, ValueChannelsTuple, TileItems>>
      per_channel;

    _CCCL_HOST_DEVICE phase_t() {}
    _CCCL_HOST_DEVICE ~phase_t() {}
  } phase;

  partition_counters<SelectedOffsetT, CandidateOffsetT> cnt;
};

template <typename KeyT, typename ValueChannelsTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct shared_mem_scratch
{
  struct keys_and_values_t
  {
    KeyT keys[TileItems];
    CUB_NS_QUALIFIER::detail::phase_aggregate<map_tuple_t<values_slot, ValueChannelsTuple, TileItems>>
      per_channel_values;
  };

  union phase_t
  {
    CUB_NS_QUALIFIER::detail::phase_union<map_tuple1_t<delegate_load_slot, ValueChannelsTuple>> delegate_loads;
    keys_and_values_t kv;

    _CCCL_HOST_DEVICE phase_t() {}
    _CCCL_HOST_DEVICE ~phase_t() {}
  } phase;

  partition_counters<SelectedOffsetT, CandidateOffsetT> cnt;
};

template <BlockPartitionStrategy Strategy,
          typename KeyT,
          typename ValueChannelsTuple,
          int TileItems,
          typename SelectedOffsetT,
          typename CandidateOffsetT>
struct strategy_scratch_selector;

template <typename KeyT, typename ValueChannelsTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct strategy_scratch_selector<BlockPartitionStrategy::Atomics,
                                 KeyT,
                                 ValueChannelsTuple,
                                 TileItems,
                                 SelectedOffsetT,
                                 CandidateOffsetT>
{
  using type = atomics_scratch;
};

template <typename KeyT, typename ValueChannelsTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct strategy_scratch_selector<BlockPartitionStrategy::Staged,
                                 KeyT,
                                 ValueChannelsTuple,
                                 TileItems,
                                 SelectedOffsetT,
                                 CandidateOffsetT>
{
  using type = staged_scratch<KeyT, ValueChannelsTuple, TileItems, SelectedOffsetT, CandidateOffsetT>;
};

template <typename KeyT, typename ValueChannelsTuple, int TileItems, typename SelectedOffsetT, typename CandidateOffsetT>
struct strategy_scratch_selector<BlockPartitionStrategy::SharedMem,
                                 KeyT,
                                 ValueChannelsTuple,
                                 TileItems,
                                 SelectedOffsetT,
                                 CandidateOffsetT>
{
  using type = shared_mem_scratch<KeyT, ValueChannelsTuple, TileItems, SelectedOffsetT, CandidateOffsetT>;
};

// Compile-time tuple iteration helper. Calls `f(at<I>(tuple), integral_constant<int,I>)`
// for every element of the tuple. Used by BlockPartition for sub-brokering value-channel
// scratch and propagating per-channel calls.
template <typename Tuple, typename Fn, ::cuda::std::size_t... Is>
_CCCL_DEVICE _CCCL_FORCEINLINE void
tuple_for_each_impl(Tuple&& t, Fn&& f, ::cuda::std::index_sequence<Is...>)
{
  (f(::cuda::std::get<Is>(t), ::cuda::std::integral_constant<::cuda::std::size_t, Is>{}), ...);
}

template <typename Tuple, typename Fn>
_CCCL_DEVICE _CCCL_FORCEINLINE void tuple_for_each(Tuple&& t, Fn&& f)
{
  constexpr auto sz = ::cuda::std::tuple_size<::cuda::std::remove_reference_t<Tuple>>::value;
  tuple_for_each_impl(::cuda::std::forward<Tuple>(t), ::cuda::std::forward<Fn>(f), ::cuda::std::make_index_sequence<sz>{});
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
_CCCL_DEVICE _CCCL_FORCEINLINE inlined_classifier<IsFull, Op>
make_inlined_classifier(Op& op, int num_thread_items)
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

} // namespace bp_detail

//---------------------------------------------------------------------
// `BlockPartition` -- the new BlockPartition (architecture §9.2).
//---------------------------------------------------------------------

template <int BlockThreads,
          int ItemsPerThread,
          BlockPartitionStrategy Strategy,
          typename KeyT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          typename ValueChannelsTuple                = ::cuda::std::tuple<>,
          BlockPartitionClassifyMode ClassifyMode    = BlockPartitionClassifyMode::precomputed>
class BlockPartition
{
  static_assert(ClassifyMode == BlockPartitionClassifyMode::precomputed
                  || Strategy == BlockPartitionStrategy::Atomics,
                "BlockPartitionClassifyMode::inlined is only supported with the Atomics "
                "strategy; staged / shared-mem scatter requires a precomputed classes[] array.");

public:
  static constexpr int tile_items                       = BlockThreads * ItemsPerThread;
  static constexpr BlockPartitionStrategy strat         = Strategy;
  static constexpr BlockPartitionClassifyMode classify  = ClassifyMode;
  static constexpr int num_value_channels =
    static_cast<int>(::cuda::std::tuple_size<ValueChannelsTuple>::value);

  // Class-lifetime persistent state. Empty (no carried state across Partition() calls).
  struct TempStorage
  {};

  // Method-call typed scratch; strategy-specific. Architecture §9.4.
  using ScratchStorage = typename bp_detail::strategy_scratch_selector<Strategy,
                                                                      KeyT,
                                                                      ValueChannelsTuple,
                                                                      tile_items,
                                                                      SelectedOffsetT,
                                                                      CandidateOffsetT>::type;

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockPartition() = default;

  // Full-tile overload: no per-item bound check inside the classify loop; each
  // value-channel data source is invoked with the no-arg submit_load.
  template <bool HasCandidates,
            typename IdentifyCandidatesOp,
            typename CandidateCallbackOp,
            typename SelectedReserveOp,
            typename CandidateReserveOp,
            typename SelectedKeyOutTransformOp,
            typename CandidateKeyOutTransformOp,
            typename SelectedKeyOutIt,
            typename CandidateKeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Partition(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    ::cuda::std::integral_constant<bool, HasCandidates>,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelsTuple& value_channels)
  {
    partition_impl<true, HasCandidates>(
      buffer,
      keys,
      /*num_items=*/tile_items,
      identify_candidates_op,
      candidate_callback_op,
      reserve_selected,
      reserve_candidate,
      selected_key_transform,
      candidate_key_transform,
      selected_keys_out,
      candidate_keys_out,
      value_channels);
  }

  // Partial-tile overload: classify loop bound-checks against num_items; each
  // value-channel data source is invoked with the (scratch, num_items) submit_load.
  template <bool HasCandidates,
            typename NumItemsT,
            typename IdentifyCandidatesOp,
            typename CandidateCallbackOp,
            typename SelectedReserveOp,
            typename CandidateReserveOp,
            typename SelectedKeyOutTransformOp,
            typename CandidateKeyOutTransformOp,
            typename SelectedKeyOutIt,
            typename CandidateKeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Partition(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    NumItemsT num_items,
    ::cuda::std::integral_constant<bool, HasCandidates>,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelsTuple& value_channels)
  {
    partition_impl<false, HasCandidates>(
      buffer,
      keys,
      static_cast<int>(num_items),
      identify_candidates_op,
      candidate_callback_op,
      reserve_selected,
      reserve_candidate,
      selected_key_transform,
      candidate_key_transform,
      selected_keys_out,
      candidate_keys_out,
      value_channels);
  }

private:
  // Shared body for both overloads. `IsFull` is the compile-time switch that elides
  // the per-item classify-loop bound check on the hot full-tile path. `num_items` is
  // the runtime per-tile valid count (only used when `IsFull == false`).
  template <bool IsFull,
            bool HasCandidates,
            typename IdentifyCandidatesOp,
            typename CandidateCallbackOp,
            typename SelectedReserveOp,
            typename CandidateReserveOp,
            typename SelectedKeyOutTransformOp,
            typename CandidateKeyOutTransformOp,
            typename SelectedKeyOutIt,
            typename CandidateKeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_impl(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    int num_items,
    IdentifyCandidatesOp& identify_candidates_op,
    CandidateCallbackOp& candidate_callback_op,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelsTuple& value_channels)
  {
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
        (tb_offset >= num_items)
          ? 0
          : static_cast<int>((::cuda::std::min) (ItemsPerThread, num_items - tb_offset));
    }

    // Atomics strategy: route through the unified `partition_atomics_fused` for both
    // classify modes. The classifier abstraction picks whether the per-item class is
    // recomputed at each use site (`inlined`) or read from a pre-filled register array
    // (`precomputed`). For the precomputed case the classifier's ctor also fires the
    // candidate callback up front, so the scatter receives `noop_callback_op` to
    // statically elide the second firing.
    if constexpr (Strategy == BlockPartitionStrategy::Atomics)
    {
      if constexpr (ClassifyMode == BlockPartitionClassifyMode::inlined)
      {
        auto classifier = bp_detail::make_inlined_classifier<IsFull>(identify_candidates_op, num_thread_items);
        partition_atomics_fused<IsFull, HasCandidates>(
          buffer,
          keys,
          num_thread_items,
          classifier,
          candidate_callback_op,
          reserve_selected,
          reserve_candidate,
          selected_key_transform,
          candidate_key_transform,
          selected_keys_out,
          candidate_keys_out,
          value_channels);
      }
      else
      {
        bp_detail::precomputed_classifier<KeyT, ItemsPerThread, IsFull, HasCandidates> classifier{
          keys, num_thread_items, identify_candidates_op, candidate_callback_op};
        bp_detail::noop_callback_op noop_cb{};
        partition_atomics_fused<IsFull, HasCandidates>(
          buffer,
          keys,
          num_thread_items,
          classifier,
          noop_cb,
          reserve_selected,
          reserve_candidate,
          selected_key_transform,
          candidate_key_transform,
          selected_keys_out,
          candidate_keys_out,
          value_channels);
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
        const bool is_valid =
          IsFull ? true : (j < num_thread_items);
        classes[j] = is_valid ? identify_candidates_op(keys[j]) : candidate_class::rejected;

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
        partition_staged<IsFull, HasCandidates>(
          buffer,
          keys,
          classes,
          reserve_selected,
          reserve_candidate,
          selected_key_transform,
          candidate_key_transform,
          selected_keys_out,
          candidate_keys_out,
          value_channels,
          num_items);
      }
      else
      {
        partition_shared_mem<IsFull, HasCandidates>(
          buffer,
          keys,
          classes,
          reserve_selected,
          reserve_candidate,
          selected_key_transform,
          candidate_key_transform,
          selected_keys_out,
          candidate_keys_out,
          value_channels,
          num_items);
      }
    }
  }

  // -----------------------------------------------------------------
  // Atomics strategy (fused): scatter via a unified per-item loop driven by a
  // user-supplied `Classifier` with signature `(KeyT, int j) -> candidate_class`.
  // The classifier abstracts the precomputed-vs-inlined decision: callers pass either
  // a `bp_detail::inlined_classifier` (recompute via `identify_candidates_op` at each
  // use site) or a `bp_detail::precomputed_classifier` (look up a pre-filled
  // `classes[]` array). The `candidate_callback_op` fires inside the scatter loop;
  // callers who already fired it (e.g., as part of `precomputed_classifier`'s ctor)
  // pass `bp_detail::noop_callback_op` to statically elide the second firing.
  // -----------------------------------------------------------------
  template <bool IsFull,
            bool HasCandidates,
            typename Classifier,
            typename CandidateCallbackOp,
            typename SelectedReserveOp,
            typename CandidateReserveOp,
            typename SelectedKeyOutTransformOp,
            typename CandidateKeyOutTransformOp,
            typename SelectedKeyOutIt,
            typename CandidateKeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_atomics_fused(
    ScratchStorage& /*buffer*/,
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    CandidateCallbackOp& candidate_callback_op,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelsTuple& value_channels)
  {
    // Atomics strategy doesn't load value channels into smem -- it scatters each value
    // as soon as it claims a global slot. Materialize the channel's values here via
    // its data source; for `direct_data_source` the sub-brokered scratch is empty.
    static_assert(num_value_channels <= 1,
                  "atomics partition supports keys-only or single-value-channel "
                  "today; multi-channel needs a per-channel value array.");

    if constexpr (num_value_channels == 1)
    {
      auto& ch        = ::cuda::std::get<0>(value_channels);
      using channel_t = ::cuda::std::remove_reference_t<decltype(ch)>;
      using value_t   = typename channel_t::value_t;
      typename channel_t::data_source_t::ScratchStorage chan_scratch{};
      auto h = ch.data_source.submit_load(chan_scratch);
      value_t values[ItemsPerThread]{};
      h.complete_load(values);

      partition_atomics_fused_scatter<IsFull, HasCandidates, /*KeysOnly=*/false>(
        keys,
        num_thread_items,
        classifier,
        candidate_callback_op,
        reserve_selected,
        reserve_candidate,
        selected_key_transform,
        candidate_key_transform,
        selected_keys_out,
        candidate_keys_out,
        value_channels,
        values);
    }
    else
    {
      int unused_dummy[1]{};
      (void) unused_dummy;
      partition_atomics_fused_scatter<IsFull, HasCandidates, /*KeysOnly=*/true>(
        keys,
        num_thread_items,
        classifier,
        candidate_callback_op,
        reserve_selected,
        reserve_candidate,
        selected_key_transform,
        candidate_key_transform,
        selected_keys_out,
        candidate_keys_out,
        value_channels,
        unused_dummy);
    }
  }

  // Unified scatter loop for the Atomics strategy. Two independent 2-way (do/skip)
  // branches per unrolled item -- avoids the per-item indirect-branch table in
  // `c[0x2]` that ptxas would emit for a 3-way `rejected` / `selected` / `candidate`
  // cascade. The per-item `candidate_class` is provided by an indexed classifier with
  // signature `(key, j) -> candidate_class`, which encapsulates both the inlined-vs-
  // precomputed policy and the partial-tile `is_valid` handling:
  //   - `inlined_classifier` recomputes via `identify_candidates_op(key)` and checks
  //     `j < num_thread_items` internally for partial tiles.
  //   - `precomputed_classifier` runs the classify loop in its constructor (also
  //     firing the `candidate_callback_op` for `candidate`-classified items), stores
  //     the result in a register-resident `classes[]`, and indexes by `j`.
  // The per-item runtime work is the same as the legacy precomputed scatter (each
  // non-rejected item still does exactly one atomic reserve + one scatter); when
  // `precomputed_classifier` is supplied, this generates byte-identical SASS to the
  // earlier dedicated `partition_atomics_scatter` implementation.
  template <bool IsFull,
            bool HasCandidates,
            bool KeysOnly,
            typename Classifier,
            typename CandidateCallbackOp,
            typename SelectedReserveOp,
            typename CandidateReserveOp,
            typename SelectedKeyOutTransformOp,
            typename CandidateKeyOutTransformOp,
            typename SelectedKeyOutIt,
            typename CandidateKeyOutIt,
            typename ValuesArr>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_atomics_fused_scatter(
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    CandidateCallbackOp& candidate_callback_op,
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelsTuple& value_channels,
    ValuesArr& values)
  {
    if constexpr (HasCandidates)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        // The classifier encapsulates the `is_valid` check internally, returning
        // `rejected` for items past `num_thread_items` on partial tiles. This lets
        // the outer scatter loop be a single unconditional call site.
        const candidate_class c = classifier(keys[j], j);

        // Architecture §10.2: callback fires for every `candidate`-classified item,
        // including ones the candidate reserve op subsequently drops (cap clamp).
        if (c == candidate_class::candidate)
        {
          candidate_callback_op(keys[j]);
        }

        if (c == candidate_class::selected)
        {
          const auto r = reserve_selected(SelectedOffsetT{1});
          bool granted = true;
          if constexpr (SelectedReserveOp::may_grant_less)
          {
            granted = (r.second != SelectedOffsetT{0});
          }
          if (granted)
          {
            selected_keys_out[r.first] = selected_key_transform(keys[j]);
            if constexpr (!KeysOnly)
            {
              auto& ch                        = ::cuda::std::get<0>(value_channels);
              ch.selected_values_out[r.first] = ch.selected_value_transform(values[j]);
            }
          }
        }
        if (c == candidate_class::candidate)
        {
          const auto r = reserve_candidate(CandidateOffsetT{1});
          bool granted = true;
          if constexpr (CandidateReserveOp::may_grant_less)
          {
            granted = (r.second != CandidateOffsetT{0});
          }
          if (granted)
          {
            candidate_keys_out[r.first] = candidate_key_transform(keys[j]);
            if constexpr (!KeysOnly)
            {
              auto& ch                         = ::cuda::std::get<0>(value_channels);
              ch.candidate_values_out[r.first] = ch.candidate_value_transform(values[j]);
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
          const auto r = reserve_selected(SelectedOffsetT{1});
          bool granted = true;
          if constexpr (SelectedReserveOp::may_grant_less)
          {
            granted = (r.second != SelectedOffsetT{0});
          }
          if (granted)
          {
            selected_keys_out[r.first] = selected_key_transform(keys[j]);
            if constexpr (!KeysOnly)
            {
              auto& ch                        = ::cuda::std::get<0>(value_channels);
              ch.selected_values_out[r.first] = ch.selected_value_transform(values[j]);
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
  template <bool IsFull,
            bool HasCandidates,
            typename SelectedReserveOp,
            typename CandidateReserveOp,
            typename SelectedKeyOutTransformOp,
            typename CandidateKeyOutTransformOp,
            typename SelectedKeyOutIt,
            typename CandidateKeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_staged(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelsTuple& value_channels,
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
        const int pos              = atomicAdd(&buffer.cnt.counters[0], 1);
        buffer.phase.keys[pos]     = keys[j];
        positions[j]               = pos;
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
      const auto sel                   = reserve_selected(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_bases.selected = sel.first;
      buffer.cnt.granted_selected      = static_cast<SelectedOffsetT>(sel.second);
      if constexpr (HasCandidates)
      {
        const auto cand                   = reserve_candidate(static_cast<CandidateOffsetT>(candidate_cnt));
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
        ? (CandidateReserveOp::may_grant_less ? buffer.cnt.granted_candidate
                                              : static_cast<CandidateOffsetT>(candidate_cnt))
        : CandidateOffsetT{};

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      selected_keys_out[sel_base + static_cast<SelectedOffsetT>(i)] =
        selected_key_transform(buffer.phase.keys[i]);
    }
    if constexpr (HasCandidates)
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
      {
        candidate_keys_out[cand_base + static_cast<CandidateOffsetT>(i)] =
          candidate_key_transform(buffer.phase.keys[tile_items - candidate_cnt + i]);
      }
    }

    // Per-channel values phases. Each channel's load + scatter is sequential in time
    // (sub-brokered through `buffer.phase.per_channel`). After the keys cooperative
    // store, the keys[] arena is no longer needed -- we sync and reuse the union slot.
    if constexpr (num_value_channels > 0)
    {
      __syncthreads();
      bp_detail::tuple_for_each(value_channels, [&](auto& ch, auto I_ic) {
        constexpr int I = static_cast<int>(decltype(I_ic)::value);
        using channel_t = ::cuda::std::remove_reference_t<decltype(ch)>;
        using value_t   = typename channel_t::value_t;

        auto& chan_phase = CUB_NS_QUALIFIER::detail::at<I>(buffer.phase.per_channel);
        value_t reg_values[ItemsPerThread]{};
        if constexpr (IsFull)
        {
          auto h = ch.data_source.submit_load(chan_phase.load);
          h.complete_load(reg_values);
        }
        else
        {
          auto h = ch.data_source.submit_load(chan_phase.load, num_items);
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
          ch.selected_values_out[sel_base + static_cast<SelectedOffsetT>(i)] =
            ch.selected_value_transform(chan_phase.values[i]);
        }
        if constexpr (HasCandidates)
        {
          for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
          {
            ch.candidate_values_out[cand_base + static_cast<CandidateOffsetT>(i)] =
              ch.candidate_value_transform(chan_phase.values[tile_items - candidate_cnt + i]);
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
  template <bool IsFull,
            bool HasCandidates,
            typename SelectedReserveOp,
            typename CandidateReserveOp,
            typename SelectedKeyOutTransformOp,
            typename CandidateKeyOutTransformOp,
            typename SelectedKeyOutIt,
            typename CandidateKeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_shared_mem(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    SelectedReserveOp& reserve_selected,
    CandidateReserveOp& reserve_candidate,
    SelectedKeyOutTransformOp& selected_key_transform,
    CandidateKeyOutTransformOp& candidate_key_transform,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelsTuple& value_channels,
    int num_items)
  {
    // shared_mem keeps the design simple by supporting at most one value channel for
    // now; multi-channel needs a heterogeneous tuple of register arrays for the
    // pre-Phase-1 channel loads, which we punt on until the agent actually needs it.
    static_assert(num_value_channels <= 1,
                  "shared_mem partition supports keys-only or single-value-channel "
                  "today; multi-channel needs a heterogeneous register-array tuple.");

    // Pre-Phase-1: load the (single) channel's values into registers via the
    // delegate-load slot. After this, the delegate_loads view of the phase union is
    // dead and we transition to the kv view at the next __syncthreads.
    [[maybe_unused]] auto load_channel_values = [&](auto& reg_values_out) {
      if constexpr (num_value_channels == 1)
      {
        auto& ch        = ::cuda::std::get<0>(value_channels);
        auto& load_slot = CUB_NS_QUALIFIER::detail::at<0>(buffer.phase.delegate_loads);
        if constexpr (IsFull)
        {
          auto h = ch.data_source.submit_load(load_slot.load);
          h.complete_load(reg_values_out);
        }
        else
        {
          auto h = ch.data_source.submit_load(load_slot.load, num_items);
          h.complete_load(reg_values_out);
        }
      }
      else
      {
        (void) reg_values_out;
      }
    };

    using channel_value_t = typename bp_detail::first_channel_value<ValueChannelsTuple>::type;
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
      const auto sel                   = reserve_selected(static_cast<SelectedOffsetT>(selected_cnt));
      buffer.cnt.global_bases.selected = sel.first;
      buffer.cnt.granted_selected      = static_cast<SelectedOffsetT>(sel.second);
      if constexpr (HasCandidates)
      {
        const auto cand                   = reserve_candidate(static_cast<CandidateOffsetT>(candidate_cnt));
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
        ? (CandidateReserveOp::may_grant_less ? buffer.cnt.granted_candidate
                                              : static_cast<CandidateOffsetT>(candidate_cnt))
        : CandidateOffsetT{};

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      selected_keys_out[sel_base + static_cast<SelectedOffsetT>(i)] =
        selected_key_transform(buffer.phase.kv.keys[i]);
    }
    if constexpr (HasCandidates)
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
      {
        candidate_keys_out[cand_base + static_cast<CandidateOffsetT>(i)] =
          candidate_key_transform(buffer.phase.kv.keys[tile_items - candidate_cnt + i]);
      }
    }

    if constexpr (num_value_channels > 0)
    {
      bp_detail::tuple_for_each(value_channels, [&](auto& ch, auto I_ic) {
        constexpr int I = static_cast<int>(decltype(I_ic)::value);
        auto& vs        = CUB_NS_QUALIFIER::detail::at<I>(buffer.phase.kv.per_channel_values);
        for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
        {
          ch.selected_values_out[sel_base + static_cast<SelectedOffsetT>(i)] =
            ch.selected_value_transform(vs.values[i]);
        }
        if constexpr (HasCandidates)
        {
          for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
          {
            ch.candidate_values_out[cand_base + static_cast<CandidateOffsetT>(i)] =
              ch.candidate_value_transform(vs.values[tile_items - candidate_cnt + i]);
          }
        }
      });
    }
    __syncthreads();
  }

};

} // namespace detail::topk

CUB_NAMESPACE_END
