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
//! Strategy selection is done by `strategy_to_partition_class_t<Strategy, ...>` below. The
//! default tuning uses only `block_partition_atomics`.

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

// Strategy selector for the partition primitives. The default tuning uses only `atomics`
// (`block_partition_atomics`: no smem; per-non-rejected-item global atomic + scatter). The
// orthogonal `InlinedClassify` axis is a separate template / policy bool. The enum retains the
// other (currently unimplemented) values for policy/ABI compatibility; in this build
// `strategy_to_partition_class_t` maps every strategy to `block_partition_atomics`.
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
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], NumItemsT num_items, ValueSourceT& value_source)
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
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  partition_impl(ScratchStorage& buffer, const KeyT (&keys)[ItemsPerThread], int num_items, ValueSourceT& value_source)
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
      precomputed_classifier<KeyT, ItemsPerThread, IsFull> classifier{keys, num_thread_items, identify_op, callback_op};
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

// Maps a `block_partition_strategy` to its partition class. The default tuning only uses
// `atomics`; the strategy / buffer-capacity template parameters are retained for call-site
// compatibility but ignored here.
template <block_partition_strategy Strategy,
          int BlockThreads,
          int ItemsPerThread,
          int AccumulatingBufferCapacity,
          int SpeculativeSelectedBufferCapacity,
          typename KeyT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          typename SelectedReserveOp,
          typename CandidateReserveOp,
          typename SelectedKeyOutIt,
          typename CandidateKeyOutIt,
          typename IdentifyCandidatesOp,
          typename CandidateCallbackOp,
          typename ValueChannelSinksT,
          typename ValueT,
          typename ValueDataSourceScratchT,
          bool LazyValueLoad,
          bool InlinedClassify>
using strategy_to_partition_class_t = block_partition_atomics<
  BlockThreads,
  ItemsPerThread,
  InlinedClassify,
  KeyT,
  SelectedOffsetT,
  CandidateOffsetT,
  SelectedReserveOp,
  CandidateReserveOp,
  SelectedKeyOutIt,
  CandidateKeyOutIt,
  IdentifyCandidatesOp,
  CandidateCallbackOp,
  ValueChannelSinksT,
  ValueT,
  ValueDataSourceScratchT,
  LazyValueLoad>;
} // namespace detail::topk

CUB_NAMESPACE_END
