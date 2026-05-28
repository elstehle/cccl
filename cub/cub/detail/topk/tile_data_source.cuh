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

// 4.4 multi_source_data_source -- runtime-switched two-source adapter folded
// around the *single active source* invariant.
//
// `pick_source_b` is set once at construction and never changes; from that
// point on the two children alias one another at every level. Concretely:
//
//   - `TempStorage` is a union over the two children's TempStorages (wrapped in
//     `Uninitialized<>` so it can sit in `__shared__` even when an arm carries a
//     non-trivial ctor like `async_to_shared`'s mbarrier). When both children
//     publish empty TempStorage, the aggregate collapses to `empty_storage_t`
//     so the empty signal survives across class boundaries (see
//     `empty_storage.cuh`).
//
//   - `ScratchStorage` likewise (already the shape before the refactor).
//
//   - Only the *active* source object exists. The ctor takes two factory
//     callbacks; only the one matching `pick_b` is invoked and its result is
//     placement-new'd into the active arm of `_ActiveSourceStorage`. The
//     inactive arm is raw uninitialized bytes and is never read.
//
//   - `set_tile_base`, `submit_load`, `gather_one` dispatch on the runtime
//     `pick_source_b` and call only the active arm.
//
//   - Load handles are tagged unions over the two children's handle types --
//     only the active arm is initialized in `submit_load`, only the active
//     arm is read in `complete_load`.
//
// Non-copyable / non-movable: the refactor is designed to compose with
// children that have deleted copy / move (notably `async_to_shared_data_source`,
// whose embedded `BlockLoadToShared` has `= delete` copy and no implicit
// move). Existing factory call sites (e.g. `make_keys_source_for_segment`,
// `make_value_channel_sources`) keep working via C++17 mandatory prvalue
// copy elision. Segment-boundary reassignment sites use explicit
// destroy-then-construct via placement-new. See the design proposal in
// `topk_perf_tracking/reports/proposal_multi_source_active_source_refactor.md`
// for the full migration discussion.
template <typename SourceA, typename SourceB, typename OffsetT = ::cuda::std::int64_t>
class multi_source_data_source
{
public:
  using value_t = typename SourceA::value_t;
  static_assert(::cuda::std::is_same_v<value_t, typename SourceB::value_t>,
                "multi_source_data_source requires both sources to share value_t");

private:
  // ---------- TempStorage (the per-source persistent state slot) ----------
  static constexpr bool _temp_storage_is_empty =
       is_empty_storage_v<typename SourceA::TempStorage>
    && is_empty_storage_v<typename SourceB::TempStorage>;

  union _TempStorageInner
  {
    typename SourceA::TempStorage a;
    typename SourceB::TempStorage b;
  };
  struct _TempStorageWrapped : CUB_NS_QUALIFIER::Uninitialized<_TempStorageInner>
  {};

public:
  using TempStorage =
    ::cuda::std::conditional_t<_temp_storage_is_empty, empty_storage_t, _TempStorageWrapped>;

  // ---------- ScratchStorage (the per-tile scratch slot) ----------
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

  // ---------- Tagged-union load handles ----------
  // Only the active arm is initialized in `submit_load`, only the active arm
  // is read in `complete_load`. The runtime `pick_b` tag is constant within a
  // kernel launch (set once by the agent at construction), so ptxas folds the
  // dead arm at codegen.
  struct full_load_handle
  {
    union _H
    {
      typename SourceA::full_load_handle a;
      typename SourceB::full_load_handle b;
      _CCCL_DEVICE _CCCL_FORCEINLINE _H() {}
      _CCCL_DEVICE _CCCL_FORCEINLINE ~_H() {}
    } h;
    bool pick_b;

    template <int IPT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[IPT])
    {
      if (pick_b)
      {
        h.b.complete_load(out);
      }
      else
      {
        h.a.complete_load(out);
      }
    }
  };

  struct partial_load_handle
  {
    union _H
    {
      typename SourceA::partial_load_handle a;
      typename SourceB::partial_load_handle b;
      _CCCL_DEVICE _CCCL_FORCEINLINE _H() {}
      _CCCL_DEVICE _CCCL_FORCEINLINE ~_H() {}
    } h;
    bool pick_b;

    template <int IPT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[IPT])
    {
      if (pick_b)
      {
        h.b.complete_load(out);
      }
      else
      {
        h.a.complete_load(out);
      }
    }
  };

  // ---------- Active-source storage ----------
  //
  // Raw-byte union over the two source types. The ctor placement-news the
  // active arm; the dtor placement-deletes it. The inactive arm is never
  // constructed -- its bytes are uninitialized and never read. The union has
  // a trivial user-declared default ctor / dtor so the outer class controls
  // the lifetime explicitly.
private:
  union _ActiveSourceStorage
  {
    SourceA a;
    SourceB b;
    _CCCL_DEVICE _CCCL_FORCEINLINE _ActiveSourceStorage() {}
    _CCCL_DEVICE _CCCL_FORCEINLINE ~_ActiveSourceStorage() {}
  };

public:
  // Factory-callback ctor:
  //   - `ts`:      aggregate per-source TempStorage (a union under the new shape).
  //   - `pick_b`:  picks the active arm (true -> SourceB; false -> SourceA).
  //   - `make_a`:  callable `(SourceA::TempStorage&) -> SourceA`, invoked only when `!pick_b`.
  //   - `make_b`:  callable `(SourceB::TempStorage&) -> SourceB`, invoked only when `pick_b`.
  //
  // The inactive factory is captured but never called -- ptxas eliminates its body
  // along with any work it would have done (input-iterator dereferences, ref-binding,
  // etc.) as dead code under the `if constexpr` / runtime branch.
  template <typename MakeA, typename MakeB>
  _CCCL_DEVICE _CCCL_FORCEINLINE multi_source_data_source(TempStorage& ts, bool pick_b, MakeA make_a, MakeB make_b)
      : pick_source_b(pick_b)
  {
    if constexpr (_temp_storage_is_empty)
    {
      // Both children publish empty TempStorage; pass on-stack empties of the
      // matching child type so the factory's ref binding works. Folded away.
      typename SourceA::TempStorage a_dummy{};
      typename SourceB::TempStorage b_dummy{};
      if (pick_b)
      {
        ::new (static_cast<void*>(&active_source.b)) SourceB(make_b(b_dummy));
      }
      else
      {
        ::new (static_cast<void*>(&active_source.a)) SourceA(make_a(a_dummy));
      }
    }
    else
    {
      auto& inner = ts.Alias();
      if (pick_b)
      {
        ::new (static_cast<void*>(&active_source.b)) SourceB(make_b(inner.b));
      }
      else
      {
        ::new (static_cast<void*>(&active_source.a)) SourceA(make_a(inner.a));
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE ~multi_source_data_source()
  {
    if (pick_source_b)
    {
      active_source.b.~SourceB();
    }
    else
    {
      active_source.a.~SourceA();
    }
  }

  // The active-source / handle unions inherit deletion of copy/move from any
  // child that has it (notably `async_to_shared_data_source`'s embedded
  // `BlockLoadToShared`). Mark them deleted explicitly so a future caller
  // accidentally routing through a copy / move-assignment site fails at
  // declaration rather than at a confusing template instantiation error
  // inside the union.
  multi_source_data_source(const multi_source_data_source&)            = delete;
  multi_source_data_source(multi_source_data_source&&)                 = delete;
  multi_source_data_source& operator=(const multi_source_data_source&) = delete;
  multi_source_data_source& operator=(multi_source_data_source&&)      = delete;

  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT tile_base)
  {
    if (pick_source_b)
    {
      active_source.b.set_tile_base(tile_base);
    }
    else
    {
      active_source.a.set_tile_base(tile_base);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& s)
  {
    full_load_handle out;
    out.pick_b = pick_source_b;
    if constexpr (_scratch_storage_is_empty)
    {
      // Both children's scratch is empty; their `submit_load` doesn't touch
      // the slot in that case anyway, so we pass on-stack stubs that the
      // compiler folds away.
      typename SourceA::ScratchStorage a_dummy{};
      typename SourceB::ScratchStorage b_dummy{};
      if (pick_source_b)
      {
        ::new (static_cast<void*>(&out.h.b))
          typename SourceB::full_load_handle(active_source.b.submit_load(b_dummy));
      }
      else
      {
        ::new (static_cast<void*>(&out.h.a))
          typename SourceA::full_load_handle(active_source.a.submit_load(a_dummy));
      }
    }
    else
    {
      auto& inner = s.Alias();
      if (pick_source_b)
      {
        ::new (static_cast<void*>(&out.h.b))
          typename SourceB::full_load_handle(active_source.b.submit_load(inner.b));
      }
      else
      {
        ::new (static_cast<void*>(&out.h.a))
          typename SourceA::full_load_handle(active_source.a.submit_load(inner.a));
      }
    }
    return out;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& s, OffsetT num_items)
  {
    partial_load_handle out;
    out.pick_b = pick_source_b;
    if constexpr (_scratch_storage_is_empty)
    {
      typename SourceA::ScratchStorage a_dummy{};
      typename SourceB::ScratchStorage b_dummy{};
      if (pick_source_b)
      {
        ::new (static_cast<void*>(&out.h.b))
          typename SourceB::partial_load_handle(active_source.b.submit_load(b_dummy, num_items));
      }
      else
      {
        ::new (static_cast<void*>(&out.h.a))
          typename SourceA::partial_load_handle(active_source.a.submit_load(a_dummy, num_items));
      }
    }
    else
    {
      auto& inner = s.Alias();
      if (pick_source_b)
      {
        ::new (static_cast<void*>(&out.h.b))
          typename SourceB::partial_load_handle(active_source.b.submit_load(inner.b, num_items));
      }
      else
      {
        ::new (static_cast<void*>(&out.h.a))
          typename SourceA::partial_load_handle(active_source.a.submit_load(inner.a, num_items));
      }
    }
    return out;
  }

  // On-demand single-item gather. Dispatches to whichever underlying source is
  // active (`pick_source_b` is set once at construction, so the branch is constant
  // within a kernel launch and ptxas eliminates the dead arm).
  _CCCL_DEVICE _CCCL_FORCEINLINE value_t gather_one(int item_idx) const
  {
    return pick_source_b ? active_source.b.gather_one(item_idx) : active_source.a.gather_one(item_idx);
  }

private:
  _ActiveSourceStorage active_source;
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
