// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private partition primitive `block_partition_atomics<..., InlinedClassify>`: a true
//! 2-way partition (selected / candidate streams) using no smem -- a per-non-rejected-item
//! global atomic + scatter. `InlinedClassify == false` precomputes a `classes[ItemsPerThread]`
//! register array up front; `InlinedClassify == true` recomputes the classification at each
//! scatter use-site, freeing those registers.
//!
//! Interface contract:
//!   - All sinks (reserve ops, output iterators, value-channel sinks) AND the classify hooks
//!     (`identify_candidates_op`, `candidate_callback_op`) are captured by ctor and stored as
//!     members. Per-call args reduce to per-tile data plus the live value `TileDataSource`.
//!   - The per-channel `ValueT` and `data_source_scratch_t` are forwarded as dedicated template
//!     parameters. Keys-only callers pass `cub::NullType` for all value-related parameters and
//!     gate their value-paths on an internal `keys_only` constexpr derived from `ValueT`.
//!
//! The single-stream "filter" path lives in `block_filter.cuh`.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/topk/candidate_class.cuh>
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

//---------------------------------------------------------------------
// Per-channel value-sink bundle (captured at ctor; lifetime = class instance).
// Holds the two sink-side output iterators.
//
// The matching per-channel `ValueT` and `ValueDataSourceScratchT` come from the
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
// Shared scratch-storage building blocks: counters and classifiers that the
// partition class's nested `ScratchStorage` / classify paths compose from.
//---------------------------------------------------------------------

// Adapter so `partition_atomics_fused_scatter` can be a single function template over an
// "indexed classifier" `(KeyT, int j) -> candidate_class`. Holds a reference to the
// `IdentifyCandidatesOp` and encapsulates the partial-tile `is_valid` check.
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

// Precomputed-classes adapter: builds a `classes[ItemsPerThread]` register array up front
// and fires the candidate callback once per `candidate`-classified item.
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

      // Callback fires for every `candidate`-classified item, including ones the candidate
      // reserve op subsequently drops (cap clamp).
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

// No-op callback for when the candidate callback has already been fired up front (e.g. by
// `precomputed_classifier`'s ctor), so the unified scatter loop's callback call folds away.
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
// `block_partition_atomics` -- per-non-rejected-item global atomic + scatter, no smem.
// `InlinedClassify` selects between the precomputed-classes form (larger live register set
// for `classes[]`) and the inlined-classify form (recomputes classification at each scatter
// use-site, freeing those registers).
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
  // `KEYS_ONLY = is_same_v<ValueT, NullType>` convention used by the radix-sort / merge-sort agents.
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, CUB_NS_QUALIFIER::NullType>;

  // Class-lifetime persistent state. Empty (no carried state across partition() calls).
  using TempStorage = empty_storage_t;

  // Per-tile scratch. The atomics strategy holds no scatter-side smem (per-item scatter goes
  // direct to the user's iterators via the captured reserve ops), but it DOES own the per-tile
  // load scratch for the value channel: the value `TileDataSource` may need its own staging
  // buffer (e.g. `BlockLoad::TempStorage`). Routing it through `ScratchStorage` keeps it in
  // shared memory instead of on the per-thread stack.
  //
  // When the value-channel scratch type is itself empty (the typical
  // `multi_source<direct, direct>` configuration), we publish `ScratchStorage` as the canonical
  // empty marker so consumers can elide setup work / barriers (see `empty_storage.cuh`).
  // Otherwise we wrap the inner in `cub::Uninitialized<>` for safe `__shared__` placement.

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

  // Ctor: captures sinks + classify hooks. The TempStorage parameter is unused
  // (no persistent state).
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

  // No-op terminal flush; the call site collapses to nothing.
  _CCCL_DEVICE _CCCL_FORCEINLINE void epilogue() {}

private:
  // Top-level dispatch: peels off `cand_reserve_open` at the tile boundary (only when the
  // candidate reserve op can grant less than requested) to select the `HasCandidateStream`
  // specialization. The per-thread flag is set inside the `HasCandidateStream=true` scatter
  // on a 0-grant; subsequent tiles then take the cheaper specialization for that thread.
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
      // `may_grant_less=false`: the reserve op never grants 0, so the candidate stream
      // stays open for this thread's lifetime. Skip the runtime branch and the
      // `HasCandidateStream=false` instantiation entirely.
      partition_dispatch_classify<IsFull, /*HasCandidateStream=*/true>(buffer, keys, num_thread_items, value_source);
    }
  }

  // Inner classifier-dispatch: both `HasCandidateStream` paths flow through here.
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
  // Fused scatter: drives a unified per-item loop via a user-supplied `Classifier`
  // `(KeyT, int j) -> candidate_class`, abstracting the precomputed-vs-inlined decision.
  // The callback is either the real `candidate_callback_op` (inlined-classify path) or a
  // `noop_callback_op` (precomputed path, where the classifier already fired callbacks).
  //
  // `HasCandidateStream` selects between the full path (per-candidate `reserve_cand` + write +
  // callback) and the closed-stream specialization (no candidate-side work at all), peeled by
  // `partition_impl` from the per-thread `cand_reserve_open` flag. The candidate callback fires
  // only for items that actually get written -- see `partition_atomics_fused_scatter`.
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
        // Smem-backed scratch for the value-channel load. When non-empty it lives inside
        // `buffer` (the agent-allocated `__shared__` ScratchStorage), so neither it nor any
        // staging buffer it embeds (e.g. `BlockLoad::TempStorage`) lands on the per-thread
        // stack. When empty (typical `multi_source<direct, direct>` config) we pass an
        // on-stack stub the compiler folds away; `buffer` is itself `empty_storage_t`.
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

  // Unified scatter loop. Two independent 2-way (do/skip) branches per unrolled item -- avoids
  // the per-item indirect branch a 3-way `rejected`/`selected`/`candidate` cascade would
  // compile to.
  //
  // With `HasCandidateStream=false` the per-item `reserve_cand` + candidate write is elided
  // (the candidate callback still fires for every candidate-classified item), as is the
  // `any_cand_granted_zero` flag, which exists only on the `HasCandidateStream=true` path to
  // record that the candidate stream just closed for the next tile's dispatch.
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

    // Tracks per-thread "saw 0 from `reserve_cand`" within this tile; folded into
    // `cand_reserve_open` after the loop so the next tile dispatches to the
    // `HasCandidateStream=false` specialization. Live only on the `HasCandidateStream=true`
    // && `may_grant_less=true` path; elsewhere the gating `if constexpr`s leave it dead, and
    // `[[maybe_unused]]` suppresses the unused-variable warning without taking its address
    // (which `(void)` would).
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
      // Candidate stream: the callback fires only for `candidate`-classified items *that end
      // up written*. Items dropped by the cap clamp (`reserve_cand` granted=0) or the
      // closed-stream specialization (`HasCandidateStream == false`) skip both the write and
      // the callback. Safe because the only non-noop callback consumer (filter-buffered's
      // `histogram_callback_op_t`) uses a `may_grant_less == false` reserve op, so it never
      // drops; and the only `may_grant_less == true` consumer (last_filter) pairs with a noop
      // callback, so its missed callbacks are moot.
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
  // `CandidateReserveOp::may_grant_less` is true (otherwise the reserve op never grants less
  // and the flag is dead code).
  //
  // Top-k invariant: once `reserve_cand` grants 0 for any thread, the device-global candidate
  // counter is past the back-grow cap, so every subsequent call from any thread also grants 0.
  // We exploit that by tracking the per-thread observation in the scatter loop and, once set,
  // dispatching the next tile to the `HasCandidateStream=false` specialization (which drops the
  // candidate-reserve atomic, write, and callback). Dropping the callback is safe: the only
  // `may_grant_less=true` consumer is last_filter, paired with `topk_noop_candidate_callback_op`.
  // Convergence is bounded to one extra tile of tail-divergence; the flag is compile-time gated,
  // so the `may_grant_less=false` path pays nothing.
  bool cand_reserve_open = true;
};
} // namespace detail::topk

CUB_NAMESPACE_END
