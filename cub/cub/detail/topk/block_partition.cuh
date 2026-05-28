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
//!   - All sinks (reserve ops, output iterators, transforms, value-channel sinks)
//!     AND the classify hooks (`identify_candidates_op`, `candidate_callback_op`)
//!     are captured by ctor and stored as members. Per-call args reduce to per-tile
//!     data plus the live value `TileDataSource` for the current tile.
//!   - `epilogue()` is argless on every variant. The three non-accumulating
//!     primitives' `epilogue()` is a no-op. The accumulating sister's `epilogue()`
//!     performs a terminal flush of any remaining buffered items.
//!   - The value-channel bundle splits along the lifetime boundary:
//!     `value_channel_sinks_t` (captured at ctor) carries the iters + transforms;
//!     the `ValueT` and per-channel `data_source_scratch_t` are forwarded as
//!     dedicated template parameters; and the per-call `ValueSourceT` carries the
//!     live `TileDataSource` instance that the agent has called `set_tile_base()`
//!     on for the current tile. Keys-only callers pass `cub::NullType` for all
//!     value-related parameters and primitives gate their value-paths on an
//!     internal `keys_only` constexpr derived from `ValueT`.
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

#include <cub/detail/topk/empty_storage.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
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
// Holds the two sink-side output iterators. The class previously also carried
// per-channel value transforms (selected_value_transform /
// candidate_value_transform), but every instantiation in the topk pipeline
// used `cuda::std::identity` for those, so the transforms were dropped: writes
// now go directly into the output iterators. If a non-identity transform is
// ever wanted again the right place to put it is inside the iterator itself
// (`cuda::transform_output_iterator`), which is already how the indexed-value
// gather path threads a non-trivial transform end-to-end.
//
// The matching per-channel `ValueT` and `ValueDataSourceScratchT` (needed for
// sizing smem in the Staged / SharedMem strategies) come from the
// agent-supplied template parameters of the partition class -- not from any
// iterator's `value_type`, which can be `void` for output iterators.
//---------------------------------------------------------------------
template <typename SelectedValuesOutIt, typename CandidateValuesOutIt>
struct value_channel_sinks_t
{
  SelectedValuesOutIt selected_values_out;
  CandidateValuesOutIt candidate_values_out;
};

//---------------------------------------------------------------------
// Shared scratch-storage building blocks (architecture §9.4). Per-strategy
// assembled `ScratchStorage` structs live as nested types of the three
// partition classes below; the helpers here (counters, per-channel slots,
// tuple-mapping helpers, classifiers) are what they compose from.
//---------------------------------------------------------------------

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
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename IdentifyCandidatesOp,
          typename CandidateCallbackOp,
          typename ValueChannelSinksT      = CUB_NS_QUALIFIER::NullType,
          typename ValueT                  = CUB_NS_QUALIFIER::NullType,
          typename ValueDataSourceScratchT = CUB_NS_QUALIFIER::NullType,
          bool LazyValueLoad               = false>
class block_partition_atomics
{
public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  // Compile-time keys-only vs keys+values selector. Mirrors the
  // `KEYS_ONLY = is_same_v<ValueT, NullType>` convention used by the radix-sort
  // and merge-sort agents (see `agent_radix_sort_downsweep.cuh` and
  // `agent_merge_sort.cuh`).
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;

  // Class-lifetime persistent state. Empty (no carried state across partition() calls).
  using TempStorage = empty_storage_t;

  // Per-tile scratch. The atomics strategies hold no scatter-side smem (per-item
  // scatter goes direct to the user's iterators via the captured reserve ops),
  // but they DO own the per-tile load scratch for the value channel: the value
  // `TileDataSource` may need its own staging buffer (e.g. `BlockLoad::TempStorage`
  // for the sync block-load variants, or the TMA staging buffer for the async
  // variant). Routing it through `ScratchStorage` keeps it in shared memory
  // instead of letting it land on the per-thread stack.
  //
  // When the value-channel scratch type is itself empty (the typical
  // `multi_source<direct, direct>` configuration), we publish `ScratchStorage`
  // as the canonical empty marker so consumers can detect that and elide
  // setup work / barriers (see `empty_storage.cuh`). Otherwise we wrap the
  // (non-trivial) inner in `cub::Uninitialized<>` for safe `__shared__`
  // placement.
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

  // Ctor (safe-both shape): captures sinks + classify hooks. The TempStorage
  // parameter is unused (Atomics has no persistent state) but is taken for
  // parity with the accumulating sister class.
  _CCCL_DEVICE _CCCL_FORCEINLINE block_partition_atomics(
    TempStorage& /*storage*/,
    SelectedReserveOp reserve_selected,
    CandidateReserveOp reserve_candidate,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksT value_channel_sinks,
    IdentifyCandidatesOp identify_candidates_op,
    CandidateCallbackOp candidate_callback_op)
      : reserve_sel(reserve_selected)
      , reserve_cand(reserve_candidate)
      
      
      , sel_iter(selected_keys_out)
      , cand_iter(candidate_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_candidates_op)
      , callback_op(candidate_callback_op)
  {}

  // Full-tile overload: no per-item bound check inside the classify loop.
  template <typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourceT& value_source)
  {
    partition_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_source);
  }

  // Partial-tile overload: classify loop bound-checks against num_items.
  template <typename NumItemsT, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourceT& value_source)
  {
    partition_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_source);
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
  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourceT& value_source)
  {
    const int num_thread_items = compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

    if constexpr (CandidateReserveOp::may_grant_less)
    {
      if (cand_reserve_open)
      {
        partition_dispatch_classify<IsFull, /*HasCandidateStream=*/true>(buffer, keys, num_thread_items, value_source);
      }
      else
      {
        partition_dispatch_classify<IsFull, /*HasCandidateStream=*/false>(buffer, keys, num_thread_items, value_source);
      }
    }
    else
    {
      // `may_grant_less=false`: the reserve op never grants 0, so the
      // candidate stream stays open for the lifetime of this thread. Skip
      // the runtime branch and the `HasCandidateStream=false` template
      // instantiation entirely.
      partition_dispatch_classify<IsFull, /*HasCandidateStream=*/true>(buffer, keys, num_thread_items, value_source);
    }
  }

  // Inner classifier-dispatch: same shape as the old `partition_impl` body,
  // factored out so the cand-stream dispatch above doesn't need to duplicate
  // it. Both `HasCandidateStream` paths flow through here.
  template <bool IsFull, bool HasCandidateStream, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_dispatch_classify(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_thread_items, ValueSourceT& value_source)
  {
    if constexpr (InlinedClassify)
    {
      auto classifier = make_inlined_classifier<IsFull>(identify_op, num_thread_items);
      partition_atomics_fused<IsFull, HasCandidateStream>(
        buffer, keys, num_thread_items, classifier, callback_op, value_source);
    }
    else
    {
      precomputed_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op, callback_op};
      noop_callback_op noop_cb{};
      partition_atomics_fused<IsFull, HasCandidateStream>(
        buffer, keys, num_thread_items, classifier, noop_cb, value_source);
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
  // `reserve_cand` + write + callback) and the closed-stream specialization
  // (no candidate-side reserve / write / callback at all). The dispatch is
  // peeled by `partition_impl` from the per-thread `cand_reserve_open` flag.
  // The candidate callback fires only for items that actually get written --
  // see the contract doc on `partition_atomics_fused_scatter` below.
  // -----------------------------------------------------------------
  template <bool IsFull, bool HasCandidateStream, typename Classifier, typename CandidateCallbackOpT, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_atomics_fused(
    ScratchStorage& buffer,
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    CandidateCallbackOpT& candidate_callback_op,
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
        partition_atomics_fused_scatter<IsFull, HasCandidateStream>(
          keys, num_thread_items, classifier, candidate_callback_op, value_source, unused_values);
      }
      else
      {
        // Smem-backed scratch for the value-channel load. Lives inside `buffer` (the
        // agent-allocated `__shared__` ScratchStorage) when non-empty, so neither it
        // nor any large staging buffer it embeds (e.g. `BlockLoad::TempStorage`, the
        // TMA buffer) ends up on the per-thread stack. When the value-channel scratch
        // type is empty (typical `multi_source<direct, direct>` config), we hand the
        // value source an on-stack stub the compiler folds away; `buffer` is itself
        // `empty_storage_t` in that case and carries no usable member to read.
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

        partition_atomics_fused_scatter<IsFull, HasCandidateStream>(
          keys, num_thread_items, classifier, candidate_callback_op, value_source, values);
      }
    }
    else
    {
      int unused_dummy[1]{};
      partition_atomics_fused_scatter<IsFull, HasCandidateStream>(
        keys, num_thread_items, classifier, candidate_callback_op, value_source, unused_dummy);
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
            typename Classifier,
            typename CandidateCallbackOpT,
            typename ValueSourceT,
            typename ValuesArr>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_atomics_fused_scatter(
    const KeyT (&keys)[ItemsPerThread],
    int num_thread_items,
    Classifier& classifier,
    CandidateCallbackOpT& candidate_callback_op,
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
          sel_iter[r.first] = keys[j];
          if constexpr (!keys_only)
          {
            sinks.selected_values_out[r.first] = get_value(j);
          }
        }
      }
      // Candidate stream: the callback fires for every `candidate`-classified item
      // *that ends up written* to the candidate output. Items dropped by the cap
      // clamp (reserve_cand returns granted=0) or by the closed-stream specialization
      // (`HasCandidateStream == false`) are silently skipped, **including the
      // callback**. This is safe because:
      //   - The only non-no-op callback in the pipeline is filter-buffered's
      //     `histogram_callback_op_t`. Filter-buffered uses `atomic_reserve_range_op`
      //     (`may_grant_less == false`), so `granted` is always true and
      //     `cand_reserve_open` never turns off -- the histogram observes every
      //     candidate item exactly the same as before this refactor.
      //   - The only `may_grant_less == true` consumer is last_filter, which pairs
      //     `back_grow_capped_reserve_op` with `topk_noop_candidate_callback_op`. Its
      //     "missed" callbacks are noops, so the observation is moot.
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
            candidate_callback_op(keys[j]);
            cand_iter[r.first] = keys[j];
            if constexpr (!keys_only)
            {
              sinks.candidate_values_out[r.first] = get_value(j);
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
  SelectedReserveOp reserve_sel;
  CandidateReserveOp reserve_cand;
  SelectedKeyOutIt sel_iter;
  CandidateKeyOutIt cand_iter;
  ValueChannelSinksT sinks;
  IdentifyCandidatesOp identify_op;
  CandidateCallbackOp callback_op;

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
  // per-item candidate-reserve atomic, candidate-write, **and candidate
  // callback** entirely. Items classified as `candidate` are silently dropped
  // on this path -- the callback is dropped too because the only consumer of
  // this `may_grant_less=true` configuration is last_filter, which pairs the
  // back-grow-capped reserve with `topk_noop_candidate_callback_op`. See the
  // contract on `partition_atomics_fused_scatter`.
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
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename IdentifyCandidatesOp,
          typename CandidateCallbackOp,
          typename ValueChannelSinksT      = CUB_NS_QUALIFIER::NullType,
          typename ValueT                  = CUB_NS_QUALIFIER::NullType,
          typename ValueDataSourceScratchT = CUB_NS_QUALIFIER::NullType,
          bool LazyValueLoad               = false,
          bool InlinedClassify             = false>
class block_partition_staged
{
public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;

  using TempStorage = empty_storage_t;

  // Per-tile scratch. `phase` is a phase union: phase 1 (key scatter) uses the
  // `keys[]` arena; phase 2 (value scatter) reuses the same smem through the
  // `value_phase` view, which itself is an internal union over the data
  // source's `load` scratch and the per-tile `values[]` array (the load happens
  // before the scatter, sub-brokered). `cnt` lives outside the union because it
  // carries the per-tile counts and the broadcast `granted_*` slots across the
  // whole `partition()` call.
  //
  // The `value_phase_t` member collapses to an empty placeholder via the
  // `value_phase_empty` specialization when `keys_only=true`, so the keys-only
  // configuration pays no smem cost beyond the `keys[]` arena.
  //
  // The phase / value-phase unions are wrapped in `cub::Uninitialized<>` so the
  // public `ScratchStorage` stays free of explicit ctor / dtor declarations even
  // when the underlying data-source scratch carries non-trivial members.
  struct value_phase_full
  {
    union _payload
    {
      ValueDataSourceScratchT load;
      ValueT values[tile_items];
    };
    CUB_NS_QUALIFIER::Uninitialized<_payload> storage;
  };
  using value_phase_t = ::cuda::std::conditional_t<keys_only, empty_storage_t, value_phase_full>;

  struct ScratchStorage
  {
    union _phase_payload
    {
      KeyT keys[tile_items];
      value_phase_t value_phase;
    };
    CUB_NS_QUALIFIER::Uninitialized<_phase_payload> phase;
    partition_counters<SelectedOffsetT, CandidateOffsetT> cnt;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE block_partition_staged(
    TempStorage& /*storage*/,
    SelectedReserveOp reserve_selected,
    CandidateReserveOp reserve_candidate,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksT value_channel_sinks,
    IdentifyCandidatesOp identify_candidates_op,
    CandidateCallbackOp candidate_callback_op)
      : reserve_sel(reserve_selected)
      , reserve_cand(reserve_candidate)
      
      
      , sel_iter(selected_keys_out)
      , cand_iter(candidate_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_candidates_op)
      , callback_op(candidate_callback_op)
  {}

  template <typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourceT& value_source)
  {
    partition_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_source);
  }

  template <typename NumItemsT, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourceT& value_source)
  {
    partition_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_source);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_impl(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourceT& value_source)
  {
    (void) value_source;
    (void) num_items;

    const int num_thread_items = compute_num_thread_items<IsFull, ItemsPerThread>(num_items);

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
      auto classifier = make_inlined_classifier<IsFull>(identify_op, num_thread_items);
      classify_and_scatter_keys</*FireCallbackInline=*/true>(buffer, keys, classifier, positions);
    }
    else
    {
      // `precomputed_classifier`'s ctor fires the candidate callback for every
      // `candidate`-classified item, so the scatter loop below only needs to
      // route into the smem arena.
      precomputed_classifier<KeyT, ItemsPerThread, IsFull> classifier{
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

    auto& phase = buffer.phase.Alias();

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter[sel_base + static_cast<SelectedOffsetT>(i)] = phase.keys[i];
    }
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
    {
      cand_iter[cand_base + static_cast<CandidateOffsetT>(i)] =
        phase.keys[tile_items - candidate_cnt + i];
    }

    // Value phase. The load + scatter is sequential in time (sub-brokered through
    // `buffer.phase.value_phase`). After the keys cooperative store, the keys[]
    // arena is no longer needed -- we sync and reuse the union slot.
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
        // Eager: load the full tile of values into a register array, then
        // scatter the surviving ones into the smem arena.
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
        // Fence the just-completed load's smem writes (to `vphase.load`) before the
        // alias swap in the scatter loop below starts writing to `vphase.values`.
        // When `ValueDataSourceScratchT` is empty the load wrote nothing to smem,
        // so the alias swap is safe without a barrier (see `empty_storage.cuh`).
        if constexpr (!is_empty_storage_v<ValueDataSourceScratchT>)
        {
          __syncthreads();
        }

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
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
      {
        sinks.candidate_values_out[cand_base + static_cast<CandidateOffsetT>(i)] =
          vphase.values[tile_items - candidate_cnt + i];
      }
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
        const int pos                  = atomicAdd(&buffer.cnt.counters[0], 1);
        buffer.phase.Alias().keys[pos] = keys[j];
        positions[j]                   = pos;
      }
      else // candidate
      {
        const int pos                  = atomicAdd(&buffer.cnt.counters[1], 1);
        const int idx                  = tile_items - 1 - pos;
        buffer.phase.Alias().keys[idx] = keys[j];
        positions[j]                   = idx;
      }
    }
  }

  SelectedReserveOp reserve_sel;
  CandidateReserveOp reserve_cand;
  SelectedKeyOutIt sel_iter;
  CandidateKeyOutIt cand_iter;
  ValueChannelSinksT sinks;
  IdentifyCandidatesOp identify_op;
  CandidateCallbackOp callback_op;
};

//---------------------------------------------------------------------
// `block_partition_shared_mem` -- keys + value array coexist in smem (within
// `phase.kv`), then a single coalesced flush per stream. Pre-Phase-1 delegate
// loads alias with the kv arena via the top-level phase union. Mapped from
// `block_partition_strategy::shared_mem`.
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
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename IdentifyCandidatesOp,
          typename CandidateCallbackOp,
          typename ValueChannelSinksT      = CUB_NS_QUALIFIER::NullType,
          typename ValueT                  = CUB_NS_QUALIFIER::NullType,
          typename ValueDataSourceScratchT = CUB_NS_QUALIFIER::NullType,
          bool LazyValueLoad               = false,
          bool InlinedClassify             = false>
class block_partition_shared_mem
{
public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;

  using TempStorage = empty_storage_t;

  // Per-tile scratch. The `phase` union has two views: `delegate_loads` (pre-Phase-1
  // delegate-load staging area for the value channel) and `kv` (the keys + values
  // arena used for scatter and cooperative flush). `cnt` lives outside the union,
  // same role as in `block_partition_staged`. The `delegate_loads` slot and the
  // `values[]` array both collapse to empty placeholders in the keys-only build,
  // so the keys-only configuration pays no smem cost beyond the `keys[]` arena.
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
  using delegate_load_t = ::cuda::std::conditional_t<keys_only, empty_storage_t, delegate_load_full>;

  // The phase union carries non-trivial alternatives (the data-source's `load`
  // scratch may be `multi_source_data_source::ScratchStorage`, which itself
  // wraps a non-trivial union); wrapping it in `cub::Uninitialized<>` keeps
  // the public ScratchStorage free of explicit ctor / dtor declarations.
  struct ScratchStorage
  {
    union _phase_payload
    {
      delegate_load_t delegate_loads;
      keys_and_values_t kv;
    };
    CUB_NS_QUALIFIER::Uninitialized<_phase_payload> phase;
    partition_counters<SelectedOffsetT, CandidateOffsetT> cnt;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE block_partition_shared_mem(
    TempStorage& /*storage*/,
    SelectedReserveOp reserve_selected,
    CandidateReserveOp reserve_candidate,
    SelectedKeyOutIt selected_keys_out,
    CandidateKeyOutIt candidate_keys_out,
    ValueChannelSinksT value_channel_sinks,
    IdentifyCandidatesOp identify_candidates_op,
    CandidateCallbackOp candidate_callback_op)
      : reserve_sel(reserve_selected)
      , reserve_cand(reserve_candidate)
      
      
      , sel_iter(selected_keys_out)
      , cand_iter(candidate_keys_out)
      , sinks(value_channel_sinks)
      , identify_op(identify_candidates_op)
      , callback_op(candidate_callback_op)
  {}

  template <typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], ValueSourceT& value_source)
  {
    partition_impl</*IsFull=*/true>(buffer, keys, /*num_items=*/tile_items, value_source);
  }

  template <typename NumItemsT, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition(
    ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourceT& value_source)
  {
    partition_impl</*IsFull=*/false>(buffer, keys, static_cast<int>(num_items), value_source);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_impl(
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

    // Phase 1: scatter keys + values into the kv arena. Both views coexist within
    // the phase union; switching from `delegate_loads` to `kv` needs a barrier
    // *only* if the eager pre-load above actually wrote to smem -- under
    // `keys_only` / `LazyValueLoad` it didn't run at all, and when
    // `ValueDataSourceScratchT` is empty (typical `multi_source<direct, direct>`
    // configuration) it ran but wrote nothing (see `empty_storage.cuh`).
    if constexpr (!keys_only && !LazyValueLoad && !is_empty_storage_v<ValueDataSourceScratchT>)
    {
      __syncthreads();
    }
    if (threadIdx.x == 0)
    {
      buffer.cnt.counters[0] = 0;
      buffer.cnt.counters[1] = 0;
    }
    __syncthreads();

    if constexpr (InlinedClassify)
    {
      auto classifier = make_inlined_classifier<IsFull>(identify_op, num_thread_items);
      classify_and_scatter_kv</*FireCallbackInline=*/true>(buffer, keys, classifier, value_source, reg_values);
    }
    else
    {
      // `precomputed_classifier`'s ctor fires the candidate callback for every
      // `candidate`-classified item, so the scatter loop only needs to route into
      // the kv arena.
      precomputed_classifier<KeyT, ItemsPerThread, IsFull> classifier{
        keys, num_thread_items, identify_op, callback_op};
      classify_and_scatter_kv</*FireCallbackInline=*/false>(buffer, keys, classifier, value_source, reg_values);
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

    auto& kv = buffer.phase.Alias().kv;

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
    {
      sel_iter[sel_base + static_cast<SelectedOffsetT>(i)] = kv.keys[i];
    }
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
    {
      cand_iter[cand_base + static_cast<CandidateOffsetT>(i)] =
        kv.keys[tile_items - candidate_cnt + i];
    }

    if constexpr (!keys_only)
    {
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(sel_to_write); i += BlockThreads)
      {
        sinks.selected_values_out[sel_base + static_cast<SelectedOffsetT>(i)] =
          kv.values[i];
      }
      for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(cand_to_write); i += BlockThreads)
      {
        sinks.candidate_values_out[cand_base + static_cast<CandidateOffsetT>(i)] =
          kv.values[tile_items - candidate_cnt + i];
      }
    }
    __syncthreads();
  }

  // Mode-agnostic Phase 1 body: classify every per-thread item, fire the candidate
  // callback inline if the precomputed-classifier ctor hasn't already done so, and
  // scatter the surviving (key, value) pairs into the kv arena. `Classifier`
  // exposes `operator()(KeyT, int j) -> candidate_class` for both modes; the
  // value source is consulted via `gather_one` only when `LazyValueLoad == true`.
  template <bool FireCallbackInline, typename Classifier, typename ValueSourceT, typename RegValuesArr>
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
      auto& kv     = buffer.phase.Alias().kv;
      kv.keys[idx] = keys[j];
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
  CandidateReserveOp reserve_cand;
  SelectedKeyOutIt sel_iter;
  CandidateKeyOutIt cand_iter;
  ValueChannelSinksT sinks;
  IdentifyCandidatesOp identify_op;
  CandidateCallbackOp callback_op;
};
} // namespace detail::topk

CUB_NAMESPACE_END
