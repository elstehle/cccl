// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private foundation building blocks: reserve callbacks, generative-iterator
//! trait, and `TileDataSource` specializations.
//!
//! Architecture overview:
//! `[topk-building-blocks-architecture_2c7af1d3.plan.md]`. This header co-locates the
//! pieces of phase P1 (foundation) and the async TMA `TileDataSource` (architecture §7);
//! once the foundation stabilizes a future split into smaller headers is cheap.
//!
//! Layout in this header (mirrors the dependency order):
//!   1. Generative-iterator trait (`is_generative_iterator`, `is_generative_iterator_v`)
//!      -- architecture §7.5; in `cub::detail`.
//!   2. Reserve callbacks (`atomic_reserve_range_op`, `back_grow_capped_reserve_op`)
//!      -- architecture §8; in `cub::detail::topk`.
//!   3. `tile_load_kind` enum -- the unified policy knob spanning sync `BlockLoad`
//!      variants and async TMA -- architecture §2.4; in `cub::detail::topk`.
//!   4. The four `TileDataSource` specializations (in `cub::detail::topk`):
//!        - `direct_data_source`              gmem -> registers, no smem
//!        - `sync_block_load_data_source`     wraps `cub::BlockLoad`
//!        - `async_to_shared_data_source`     wraps `cub::detail::BlockLoadToShared`
//!        - `multi_source_data_source`        runtime-switched two-source adapter
//!   5. The `make_tile_data_source` factory which applies §7.5 to redirect
//!      `cuda::counting_iterator` to `direct_data_source` regardless of the configured
//!      `tile_load_kind`.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load.cuh>
#include <cub/block/block_load_to_shared.cuh>
#include <cub/detail/topk/empty_storage.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__fwd/iterator.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>

CUB_NAMESPACE_BEGIN

namespace detail
{
//---------------------------------------------------------------------
// 1. Generative-iterator trait (architecture §7.5).
//
// Phase 1 is intentionally limited to `cuda::counting_iterator`. Recursion through
// adaptors (e.g., transform_iterator over a counting_iterator) is out of scope until
// the trait moves to libcudacxx with an opt-in inline tag.
//---------------------------------------------------------------------

template <typename It>
struct is_generative_iterator : ::cuda::std::false_type
{};

template <typename Start>
struct is_generative_iterator<::cuda::counting_iterator<Start>> : ::cuda::std::true_type
{};

template <typename It>
struct is_generative_iterator<const It> : is_generative_iterator<It>
{};

template <typename It>
struct is_generative_iterator<volatile It> : is_generative_iterator<It>
{};

template <typename It>
inline constexpr bool is_generative_iterator_v = is_generative_iterator<It>::value;

namespace topk
{
//---------------------------------------------------------------------
// 2. Reserve callbacks (architecture §8).
//
// Both follow the `(base, granted) operator()(n)` contract with the static
// `may_grant_less` trait. Stateless function objects: empty TempStorage / ScratchStorage
// (omitted; treated as empty by the brokering protocol).
//---------------------------------------------------------------------
template <typename OffsetT>
struct atomic_reserve_range_op
{
  static constexpr bool may_grant_less = false;

  OffsetT* counter;

  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::pair<OffsetT, OffsetT> operator()(OffsetT n) const
  {
    const OffsetT base = atomicAdd(counter, n);
    return {base, n};
  }
};

template <typename OffsetT>
struct back_grow_capped_reserve_op
{
  static constexpr bool may_grant_less = true;

  OffsetT* counter;
  
  // The start offset of the reservable region
  OffsetT region_start;

  // The number of items in that can be reserved in total
  OffsetT cap;

  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::pair<OffsetT, OffsetT> operator()(OffsetT n) const
  {
    // The `(cap > prev) ? cap - prev : 0` guard exists because `cap - prev` would
    // underflow (wrap) when `prev >= cap` for unsigned `OffsetT`. 
    const OffsetT prev    = atomicAdd(counter, n);
    const OffsetT avail   = (cap > prev) ? (cap - prev) : OffsetT{0};
    const OffsetT granted = (::cuda::std::min) (n, avail);
    const OffsetT base    = region_start + prev;
    return {base, granted};
  }
};

//---------------------------------------------------------------------
// 3. `tile_load_kind` -- the unified policy knob (architecture §2.4).
//
// Spans the sync `BlockLoadAlgorithm` choices (covering everything the legacy
// `BlockLoadAlgorithm`-based policy entry could express) plus the async TMA path.
// The factory below picks the concrete TileDataSource specialization from this enum.
//---------------------------------------------------------------------

enum class tile_load_kind
{
  direct,
  block_load_direct,
  block_load_striped,
  block_load_vectorize,
  block_load_transpose,
  block_load_warp_transpose,
  block_load_warp_transpose_timesliced,
  block_load_to_shared_async,
};

// Mapping from `tile_load_kind` to `cub::BlockLoadAlgorithm`. Used by the sync data
// source factory; the async kind is handled by a different specialization so we do not
// need a mapping for it here.
template <tile_load_kind Kind>
struct sync_block_load_algo;

template <>
struct sync_block_load_algo<tile_load_kind::block_load_direct>
{
  static constexpr CUB_NS_QUALIFIER::BlockLoadAlgorithm value = CUB_NS_QUALIFIER::BLOCK_LOAD_DIRECT;
};
template <>
struct sync_block_load_algo<tile_load_kind::block_load_striped>
{
  static constexpr CUB_NS_QUALIFIER::BlockLoadAlgorithm value = CUB_NS_QUALIFIER::BLOCK_LOAD_STRIPED;
};
template <>
struct sync_block_load_algo<tile_load_kind::block_load_vectorize>
{
  static constexpr CUB_NS_QUALIFIER::BlockLoadAlgorithm value = CUB_NS_QUALIFIER::BLOCK_LOAD_VECTORIZE;
};
template <>
struct sync_block_load_algo<tile_load_kind::block_load_transpose>
{
  static constexpr CUB_NS_QUALIFIER::BlockLoadAlgorithm value = CUB_NS_QUALIFIER::BLOCK_LOAD_TRANSPOSE;
};
template <>
struct sync_block_load_algo<tile_load_kind::block_load_warp_transpose>
{
  static constexpr CUB_NS_QUALIFIER::BlockLoadAlgorithm value = CUB_NS_QUALIFIER::BLOCK_LOAD_WARP_TRANSPOSE;
};
template <>
struct sync_block_load_algo<tile_load_kind::block_load_warp_transpose_timesliced>
{
  static constexpr CUB_NS_QUALIFIER::BlockLoadAlgorithm value = CUB_NS_QUALIFIER::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED;
};

//---------------------------------------------------------------------
// 4. `TileDataSource` specializations (architecture §7.4).
//
// Contract per architecture §7.2:
//   - Construct with `(InputIt it, TempStorage& state)`.
//   - `set_tile_base(OffsetT)` advances the global offset of the next load.
//   - `submit_load(ScratchStorage&)`            -> full_load_handle.
//   - `submit_load(ScratchStorage&, OffsetT n)` -> partial_load_handle.
//   - Each handle has `complete_load(value_t (&out)[ItemsPerThread])`.
// Default arrangement is BLOCKED: thread t gets items [t*IPT, (t+1)*IPT) of the window.
//---------------------------------------------------------------------

// 4.1 direct_data_source -- no smem; per-thread `it[base + t*IPT + j]`. Hot path.
template <typename InputIt, int BlockThreads, int ItemsPerThread, typename OffsetT = ::cuda::std::int64_t>
class direct_data_source
{
public:
  using value_t = CUB_NS_QUALIFIER::detail::it_value_t<InputIt>;

  // No persistent state and no per-tile scratch -- direct loads go straight from
  // gmem to registers via per-thread `it[base + ...]` accesses. Publishing the
  // canonical empty marker lets transitive empty-storage detection work without
  // any user-defined union ctor / dtor declarations downstream.
  using TempStorage    = empty_storage_t;
  using ScratchStorage = empty_storage_t;

  struct full_load_handle
  {
    InputIt it;

    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[ItemsPerThread])
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        out[j] = it[static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + j];
      }
    }
  };

  struct partial_load_handle
  {
    InputIt it;
    OffsetT num_items;

    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[ItemsPerThread])
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        const OffsetT idx = static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + j;
        out[j]            = (idx < num_items) ? it[idx] : value_t{};
      }
    }
  };

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE direct_data_source(InputIt input_it, TempStorage& /*state*/)
      : it(input_it)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT base)
  {
    tile_base = base;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& /*scratch*/)
  {
    return full_load_handle{it + tile_base};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& /*scratch*/, OffsetT num_items)
  {
    return partial_load_handle{it + tile_base, num_items};
  }

  // On-demand single-item gather. Used by the lazy value-load path in
  // `BlockPartition::partition_atomics_fused_scatter`: instead of loading the
  // full per-thread `values[ItemsPerThread]` array up front, the scatter loop
  // calls this for each non-rejected item, fetching only the values that will
  // actually be written. Mirrors the access pattern of `full_load_handle`
  // (BLOCKED layout): thread `t` owns items `[t*IPT, (t+1)*IPT)` of the tile.
  // The caller is responsible for not gathering past `num_thread_items` on
  // partial tiles -- the partition primitive enforces this by classifying
  // out-of-range items as `rejected`.
  _CCCL_DEVICE _CCCL_FORCEINLINE value_t gather_one(int item_idx) const
  {
    const OffsetT idx = tile_base + static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + item_idx;
    return it[idx];
  }

private:
  InputIt it;
  OffsetT tile_base{};
};

// 4.2 sync_block_load_data_source -- wraps `cub::BlockLoad`. ScratchStorage holds the
// underlying BlockLoad's TempStorage (which is method-call in our taxonomy).
template <typename InputIt,
          int BlockThreads,
          int ItemsPerThread,
          CUB_NS_QUALIFIER::BlockLoadAlgorithm Algo,
          typename OffsetT = ::cuda::std::int64_t>
class sync_block_load_data_source
{
public:
  using value_t      = CUB_NS_QUALIFIER::detail::it_value_t<InputIt>;
  using block_load_t = CUB_NS_QUALIFIER::BlockLoad<value_t, BlockThreads, ItemsPerThread, Algo>;

  // No cross-tile state to carry; the per-tile `BlockLoad::TempStorage` lives
  // in `ScratchStorage` below.
  using TempStorage = empty_storage_t;

  struct ScratchStorage
  {
    typename block_load_t::TempStorage block_load;
  };

  struct full_load_handle
  {
    InputIt it;
    ScratchStorage* scratch;

    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[ItemsPerThread])
    {
      block_load_t(scratch->block_load).Load(it, out);
    }
  };

  struct partial_load_handle
  {
    InputIt it;
    ScratchStorage* scratch;
    OffsetT num_items;

    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[ItemsPerThread])
    {
      block_load_t(scratch->block_load).Load(it, out, num_items);
    }
  };

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE sync_block_load_data_source(InputIt input_it, TempStorage& /*state*/)
      : it(input_it)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT base)
  {
    tile_base = base;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& scratch)
  {
    return full_load_handle{it + tile_base, &scratch};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& scratch, OffsetT num_items)
  {
    return partial_load_handle{it + tile_base, &scratch, num_items};
  }

private:
  InputIt it;
  OffsetT tile_base{};
};

// 4.3 async_to_shared_data_source -- wraps `cub::detail::BlockLoadToShared` (TMA on
// SM90+, cp.async on SM80+, scalar fallback on older arches). The mbarrier handle is
// persistent (`TempStorage`); the staging buffer is method-call (`ScratchStorage`) and
// must remain valid across the entire submit -> complete window (architecture §2.3).
template <typename InputIt,
          int BlockThreads,
          int ItemsPerThread,
          ::cuda::std::size_t GmemAlign = alignof(CUB_NS_QUALIFIER::detail::it_value_t<InputIt>),
          typename OffsetT              = ::cuda::std::int64_t>
class async_to_shared_data_source
{
public:
  using value_t  = CUB_NS_QUALIFIER::detail::it_value_t<InputIt>;
  using loader_t = CUB_NS_QUALIFIER::detail::BlockLoadToShared<BlockThreads>;

  static constexpr int tile_items = BlockThreads * ItemsPerThread;

  struct TempStorage
  {
    typename loader_t::TempStorage barrier;
  };

  // The staging buffer for the in-flight transfer. Sized and aligned per
  // `cub::detail::LoadToSharedBuffer{Size,Align}Bytes<value_t, GmemAlign>(tile_items)`.
  struct alignas(CUB_NS_QUALIFIER::detail::LoadToSharedBufferAlignBytes<value_t>()) ScratchStorage
  {
    char buf[CUB_NS_QUALIFIER::detail::LoadToSharedBufferSizeBytes<value_t, GmemAlign>(
      static_cast<::cuda::std::size_t>(tile_items))];
  };

  struct full_load_handle
  {
    ::cuda::std::span<value_t> span;
    typename loader_t::CommitToken token;
    loader_t* loader;

    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[ItemsPerThread])
    {
      loader->Wait(::cuda::std::move(token));
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        out[j] = span[static_cast<int>(threadIdx.x) * ItemsPerThread + j];
      }
    }
  };

  struct partial_load_handle
  {
    ::cuda::std::span<value_t> span;
    typename loader_t::CommitToken token;
    OffsetT num_items;
    loader_t* loader;

    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[ItemsPerThread])
    {
      loader->Wait(::cuda::std::move(token));
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        const OffsetT idx = static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + j;
        out[j]            = (idx < num_items) ? span[static_cast<int>(idx)] : value_t{};
      }
    }
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE async_to_shared_data_source(InputIt input_it, TempStorage& state)
      : it(input_it)
      , loader(state.barrier)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT base)
  {
    tile_base = base;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& scratch)
  {
    ::cuda::std::span<char> dst{scratch.buf, sizeof(scratch.buf)};
    ::cuda::std::span<const value_t> src{it + tile_base, static_cast<::cuda::std::size_t>(tile_items)};
    auto span  = loader.template CopyAsync<value_t, GmemAlign>(dst, src);
    auto token = loader.Commit();
    return full_load_handle{span, ::cuda::std::move(token), &loader};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& scratch, OffsetT num_items)
  {
    ::cuda::std::span<char> dst{scratch.buf, sizeof(scratch.buf)};
    ::cuda::std::span<const value_t> src{it + tile_base, static_cast<::cuda::std::size_t>(num_items)};
    auto span  = loader.template CopyAsync<value_t, GmemAlign>(dst, src);
    auto token = loader.Commit();
    return partial_load_handle{span, ::cuda::std::move(token), num_items, &loader};
  }

  // Make the persistent mbarrier reusable. Caller (the agent) pairs this with its own
  // lifetime end; the primitive itself does the necessary __syncthreads internally.
  _CCCL_DEVICE _CCCL_FORCEINLINE void invalidate()
  {
    loader.Invalidate();
  }

private:
  InputIt it;
  loader_t loader;
  OffsetT tile_base{};
};

// 4.4 multi_source_data_source -- runtime-switched two-source adapter.
//
// Both underlying sources are alive and the multi-source delegates every
// operation -- `set_tile_base` to both (cheap and lets ptxas hoist constants
// into uniform registers without a branch) and `submit_load` / `gather_one`
// to whichever arm `pick_source_b` selects (the per-tile data only ever
// comes from one arm).
//
// Children ownership: the **agent** owns both child sources and their
// per-source `TempStorage` slots. Each child is constructed by the agent
// against its own agent-owned `TempStorage` instance; the multi-source then
// borrows references to the two constructed children. The multi-source
// itself does not publish a `TempStorage` -- it has no persistent per-tile
// state of its own beyond the two references and `pick_source_b`. This
// keeps the agent / multi-source contract clean (no agent-side introspection
// of an opaque aggregate type) and matches the symmetric story for
// `ScratchStorage`, which the multi-source *does* publish because it owns
// the per-tile alias decision between the two arms' scratch slots.
//
// The shape has two practical wins:
//
//   1. Keeps `<direct, direct>` / `<sync_block_load, direct>` codegen
//      byte-identical to the OLD `(SourceA, SourceB, bool)` value ctor --
//      ptxas still sees both arms as straight-line members and the LDCU
//      hoist / uniform-register propagation around `pick_source_b` keeps
//      firing.
//   2. Composes with future non-copyable / non-movable children
//      (`async_to_shared_data_source`'s embedded `BlockLoadToShared` has
//      `= delete` copy and no implicit move): the multi-source ctor takes
//      references rather than values, so the deleted-copy chain never gets
//      reached. The multi-source itself is non-copyable / non-movable below
//      to keep the lifetime contract symmetric.
//
// Lifetime contract: the agent guarantees both child references outlive the
// multi-source. Existing call sites already satisfy this (children + multi-
// source declared back-to-back in the same enclosing block; the one
// segment-boundary refresh in `agent_batched_topk_last_filter::run` uses
// destroy-then-construct via placement-new on the entire trio).
template <typename SourceA, typename SourceB, typename OffsetT = ::cuda::std::int64_t>
class multi_source_data_source
{
public:
  using value_t = typename SourceA::value_t;
  static_assert(::cuda::std::is_same_v<value_t, typename SourceB::value_t>,
                "multi_source_data_source requires both sources to share value_t");

  // Note: this class intentionally does NOT publish a `TempStorage` member
  // type. Per-source persistent state is owned by the agent as two separate
  // `SourceA::TempStorage` / `SourceB::TempStorage` allocations -- the
  // multi-source has no persistent state to host on top of them. Agents
  // construct the children against their own slots, then build the
  // multi-source with references to the constructed children.

  // Only one of the two sources is active per submit/complete window (`pick_source_b`
  // is set once at construction), so the two scratch slots alias via a union. The
  // union is wrapped in `cub::Uninitialized<>` so callers can place `ScratchStorage`
  // directly in `__shared__` without tripping CUDA's "no dynamic init in shared
  // memory" rule -- the alternatives can carry their own non-trivial ctors / dtors
  // (e.g. another `multi_source_data_source` nested below) and the wrapper sidesteps
  // those by carrying raw byte storage.
  //
  // When *both* children publish an empty ScratchStorage, the aggregate is empty too;
  // collapse to `empty_storage_t` rather than wrap-in-Uninitialized so consumers see
  // the empty signal across class boundaries.
private:
  static constexpr bool _scratch_storage_is_empty =
       is_empty_storage_v<typename SourceA::ScratchStorage>
    && is_empty_storage_v<typename SourceB::ScratchStorage>;

  union _ScratchStorageInner
  {
    typename SourceA::ScratchStorage a;
    typename SourceB::ScratchStorage b;
  };
  struct _ScratchStorageWrapped : CUB_NS_QUALIFIER::Uninitialized<_ScratchStorageInner>
  {};

public:
  using ScratchStorage =
    ::cuda::std::conditional_t<_scratch_storage_is_empty, empty_storage_t, _ScratchStorageWrapped>;

  // Tagged-union load handles. Both alternatives are alive in the small POD; only the
  // one matching `pick_source_b` is initialized via the underlying source's submit, and only
  // it is read in `complete_load`. The runtime branch is constant within a kernel
  // launch (set once by the agent), so the compiler eliminates the dead branch.
  struct full_load_handle
  {
    typename SourceA::full_load_handle a{};
    typename SourceB::full_load_handle b{};
    bool pick_b{};

    template <int IPT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[IPT])
    {
      if (pick_b)
      {
        b.complete_load(out);
      }
      else
      {
        a.complete_load(out);
      }
    }
  };

  struct partial_load_handle
  {
    typename SourceA::partial_load_handle a{};
    typename SourceB::partial_load_handle b{};
    bool pick_b{};

    template <int IPT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[IPT])
    {
      if (pick_b)
      {
        b.complete_load(out);
      }
      else
      {
        a.complete_load(out);
      }
    }
  };

  // Take both child sources by reference. The agent owns the underlying
  // objects -- the multi-source just borrows for delegation. This composes
  // with non-copyable / non-movable children (the proposal's headline
  // future-async support) without forcing the by-value ctor path that would
  // hit a deleted copy ctor for `async_to_shared_data_source`.
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE multi_source_data_source(SourceA& a, SourceB& b, bool pick_b)
      : source_a(a)
      , source_b(b)
      , pick_source_b(pick_b)
  {}

  // Non-copyable / non-movable for two reasons:
  //   1. The class holds references; copy/move-assigning them is brittle.
  //   2. Future children with deleted copy/move (`async_to_shared_data_source`
  //      via `BlockLoadToShared`) would propagate deletion into the
  //      multi-source anyway. Making it explicit here documents the contract
  //      and ensures a caller that accidentally routes through copy/move
  //      fails at the declaration site, not deep inside a template error.
  // Existing call sites are all direct-init in the enclosing scope; the one
  // assignment site at the segment boundary in `agent_batched_topk_last_filter
  // ::run` uses destroy-then-construct via placement-new.
  multi_source_data_source(const multi_source_data_source&)            = delete;
  multi_source_data_source(multi_source_data_source&&)                 = delete;
  multi_source_data_source& operator=(const multi_source_data_source&) = delete;
  multi_source_data_source& operator=(multi_source_data_source&&)      = delete;

  // Both sources alive -- propagate `set_tile_base` to both. Per-tile cost
  // is one extra register store (cheap) and matches the OLD codegen shape
  // that ptxas optimises well (uniform-register hoisting, no branch).
  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT tile_base)
  {
    source_a.set_tile_base(tile_base);
    source_b.set_tile_base(tile_base);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& s)
  {
    // When the aggregate `ScratchStorage` is `empty_storage_t` (both children
    // empty) the union doesn't exist; the children's `submit_load` doesn't
    // actually touch the scratch in that case anyway, so we pass them empty
    // stack-local stubs that the compiler folds away.
    if constexpr (_scratch_storage_is_empty)
    {
      typename SourceA::ScratchStorage a_dummy{};
      typename SourceB::ScratchStorage b_dummy{};
      if (pick_source_b)
      {
        return full_load_handle{{}, source_b.submit_load(b_dummy), true};
      }
      return full_load_handle{source_a.submit_load(a_dummy), {}, false};
    }
    else
    {
      auto& inner = s.Alias();
      if (pick_source_b)
      {
        return full_load_handle{{}, source_b.submit_load(inner.b), true};
      }
      return full_load_handle{source_a.submit_load(inner.a), {}, false};
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& s, OffsetT num_items)
  {
    if constexpr (_scratch_storage_is_empty)
    {
      typename SourceA::ScratchStorage a_dummy{};
      typename SourceB::ScratchStorage b_dummy{};
      if (pick_source_b)
      {
        return partial_load_handle{{}, source_b.submit_load(b_dummy, num_items), true};
      }
      return partial_load_handle{source_a.submit_load(a_dummy, num_items), {}, false};
    }
    else
    {
      auto& inner = s.Alias();
      if (pick_source_b)
      {
        return partial_load_handle{{}, source_b.submit_load(inner.b, num_items), true};
      }
      return partial_load_handle{source_a.submit_load(inner.a, num_items), {}, false};
    }
  }

  // On-demand single-item gather. Dispatches to whichever underlying source is
  // active (`pick_source_b` is set once at construction, so the branch is constant
  // within a kernel launch and ptxas eliminates the dead arm).
  _CCCL_DEVICE _CCCL_FORCEINLINE value_t gather_one(int item_idx) const
  {
    return pick_source_b ? source_b.gather_one(item_idx) : source_a.gather_one(item_idx);
  }

private:
  SourceA& source_a;
  SourceB& source_b;
  bool pick_source_b;
};

//---------------------------------------------------------------------
// 5. `make_tile_data_source` factory (architecture §7.5).
//
// Picks the concrete TileDataSource for the given `tile_load_kind`. Generative iterators
// (today: `cuda::counting_iterator`) are statically downgraded to `direct_data_source`
// regardless of the configured kind -- coalescing/TMA buy nothing for an iterator that
// doesn't live in memory.
//---------------------------------------------------------------------

// Tag-dispatch helper: factory_impl<Kind, IsGen> picks the data source type.
template <tile_load_kind Kind, bool IsGenerative>
struct factory_impl;

template <tile_load_kind Kind>
struct factory_impl<Kind, /*IsGenerative=*/true>
{
  template <typename It, int BlockThreads, int ItemsPerThread, typename OffsetT, ::cuda::std::size_t /*GmemAlign*/>
  using data_source_t = direct_data_source<It, BlockThreads, ItemsPerThread, OffsetT>;
};

template <>
struct factory_impl<tile_load_kind::direct, /*IsGenerative=*/false>
{
  template <typename It, int BlockThreads, int ItemsPerThread, typename OffsetT, ::cuda::std::size_t /*GmemAlign*/>
  using data_source_t = direct_data_source<It, BlockThreads, ItemsPerThread, OffsetT>;
};

template <>
struct factory_impl<tile_load_kind::block_load_to_shared_async, /*IsGenerative=*/false>
{
  template <typename It, int BlockThreads, int ItemsPerThread, typename OffsetT, ::cuda::std::size_t GmemAlign>
  using data_source_t = async_to_shared_data_source<It, BlockThreads, ItemsPerThread, GmemAlign, OffsetT>;
};

#define _CUB_DETAIL_TOPK_SYNC_BL_FACTORY(Kind)                                                                        \
  template <>                                                                                                         \
  struct factory_impl<Kind, /*IsGenerative=*/false>                                                                   \
  {                                                                                                                   \
    template <typename It, int BlockThreads, int ItemsPerThread, typename OffsetT, ::cuda::std::size_t /*GmemAlign*/> \
    using data_source_t =                                                                                             \
      sync_block_load_data_source<It, BlockThreads, ItemsPerThread, sync_block_load_algo<Kind>::value, OffsetT>;      \
  }

_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_direct);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_striped);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_vectorize);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_transpose);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_warp_transpose);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_warp_transpose_timesliced);

#undef _CUB_DETAIL_TOPK_SYNC_BL_FACTORY

// Type alias selecting the concrete TileDataSource type for the given configuration.
// `OffsetT` defaults to `int64_t` to handle large problems; per-call overflow can be
// avoided by passing `int32_t` for known-small inputs.
template <typename It,
          tile_load_kind ConfiguredKind,
          int BlockThreads,
          int ItemsPerThread,
          typename OffsetT              = ::cuda::std::int64_t,
          ::cuda::std::size_t GmemAlign = alignof(CUB_NS_QUALIFIER::detail::it_value_t<It>)>
using tile_data_source_t =
  typename factory_impl<ConfiguredKind,
                                    CUB_NS_QUALIFIER::detail::is_generative_iterator_v<::cuda::std::remove_cv_t<It>>>::
    template data_source_t<It, BlockThreads, ItemsPerThread, OffsetT, GmemAlign>;

template <typename It,
          tile_load_kind ConfiguredKind,
          int BlockThreads,
          int ItemsPerThread,
          typename OffsetT              = ::cuda::std::int64_t,
          ::cuda::std::size_t GmemAlign = alignof(CUB_NS_QUALIFIER::detail::it_value_t<It>)>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto make_tile_data_source(
  It it,
  typename tile_data_source_t<It, ConfiguredKind, BlockThreads, ItemsPerThread, OffsetT, GmemAlign>::TempStorage& state)
{
  using data_source_t = tile_data_source_t<It, ConfiguredKind, BlockThreads, ItemsPerThread, OffsetT, GmemAlign>;
  return data_source_t{it, state};
}
} // namespace topk
} // namespace detail

CUB_NAMESPACE_END
