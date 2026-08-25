// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private foundation building blocks: reserve callbacks, generative-iterator
//! trait, and `TileDataSource` specializations.
//!
//! Layout in this header (mirrors the dependency order):
//!   1. Generative-iterator trait (`is_generative_iterator`, `is_generative_iterator_v`)
//!      in `cub::detail`.
//!   2. Reserve callbacks (`atomic_reserve_range_op`, `back_grow_capped_reserve_op`)
//!      in `cub::detail::topk`.
//!   3. `tile_load_kind` enum -- the policy knob spanning the `BlockLoad` variants.
//!   4. The `TileDataSource` specializations (in `cub::detail::topk`):
//!        - `direct_data_source`              gmem -> registers, no smem
//!        - `sync_block_load_data_source`     wraps `cub::BlockLoad`
//!        - `multi_source_data_source`        runtime-switched two-source adapter
//!   5. The `make_tile_data_source` factory which redirects `cuda::counting_iterator`
//!      to `direct_data_source` regardless of the configured `tile_load_kind`.

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
// 1. Generative-iterator trait.
//
// Intentionally limited to `cuda::counting_iterator`. Recursion through adaptors
// (e.g. transform_iterator over a counting_iterator) is out of scope.
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
// 2. Reserve callbacks.
//
// Both follow the `(base, granted) operator()(n)` contract with the static
// `may_grant_less` trait. Stateless function objects with empty storage.
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
// 3. `tile_load_kind` -- the policy knob.
//
// Spans the `BlockLoadAlgorithm` choices. The factory below picks the concrete
// TileDataSource specialization from this enum.
//---------------------------------------------------------------------

enum class tile_load_kind
{
  direct,
  block_load_direct,
  block_load_vectorize,
};

// Mapping from `tile_load_kind` to `cub::BlockLoadAlgorithm`, used by the data source factory.
template <tile_load_kind Kind>
struct sync_block_load_algo;

template <>
struct sync_block_load_algo<tile_load_kind::block_load_direct>
{
  static constexpr CUB_NS_QUALIFIER::BlockLoadAlgorithm value = CUB_NS_QUALIFIER::BLOCK_LOAD_DIRECT;
};
template <>
struct sync_block_load_algo<tile_load_kind::block_load_vectorize>
{
  static constexpr CUB_NS_QUALIFIER::BlockLoadAlgorithm value = CUB_NS_QUALIFIER::BLOCK_LOAD_VECTORIZE;
};

//---------------------------------------------------------------------
// 4. `TileDataSource` specializations.
//
// Contract:
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

  // No persistent state and no per-tile scratch. Publishing the canonical empty marker
  // lets transitive empty-storage detection work downstream.
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

  // Re-target to a new input iterator without disturbing persistent state, so an agent can
  // reuse one long-lived source across segments instead of reconstructing it.
  _CCCL_DEVICE _CCCL_FORCEINLINE void set_input(InputIt input_it)
  {
    it = input_it;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& /*scratch*/)
  {
    return full_load_handle{it + tile_base};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& /*scratch*/, OffsetT num_items)
  {
    return partial_load_handle{it + tile_base, num_items};
  }

  // On-demand single-item gather for the lazy value-load path: fetch only the values that
  // will actually be written instead of loading the full per-thread array up front. BLOCKED
  // layout: thread `t` owns items `[t*IPT, (t+1)*IPT)` of the tile. The caller must not gather
  // past `num_thread_items` on partial tiles.
  _CCCL_DEVICE _CCCL_FORCEINLINE value_t gather_one(int item_idx) const
  {
    const OffsetT idx = tile_base + static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + item_idx;
    return it[idx];
  }

  // No-op: direct reads have no persistent smem state. Provided so agents can call
  // `invalidate()` unconditionally regardless of the tile-load kind.
  _CCCL_DEVICE _CCCL_FORCEINLINE void invalidate() {}

private:
  InputIt it;
  OffsetT tile_base{};
};

// 4.2 sync_block_load_data_source -- wraps `cub::BlockLoad`. ScratchStorage holds the
// underlying BlockLoad's TempStorage.
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

  // Re-target to a new input iterator. `cub::BlockLoad` carries no persistent state, so this is
  // just an iterator swap; lets an agent reuse one source across segments.
  _CCCL_DEVICE _CCCL_FORCEINLINE void set_input(InputIt input_it)
  {
    it = input_it;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& scratch)
  {
    return full_load_handle{it + tile_base, &scratch};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& scratch, OffsetT num_items)
  {
    return partial_load_handle{it + tile_base, &scratch, num_items};
  }

  // No-op: `cub::BlockLoad` carries no persistent smem state between calls.
  _CCCL_DEVICE _CCCL_FORCEINLINE void invalidate() {}

private:
  InputIt it;
  OffsetT tile_base{};
};

// 4.3 multi_source_data_source -- runtime-switched two-source adapter.
//
// Both underlying sources are alive: the multi-source delegates `set_tile_base` to both
// and `submit_load` / `gather_one` to whichever arm `pick_source_b` selects.
//
// Ownership: the agent owns both child sources and their per-source `TempStorage` slots and
// constructs each child against its own slot; the multi-source only borrows references to
// them. It publishes a `ScratchStorage` (it owns the per-tile alias decision between the two
// arms' scratch slots) but no `TempStorage` of its own.
//
// Taking children by reference (rather than by value) composes with non-copyable /
// non-movable children. The multi-source is itself non-copyable / non-movable to keep the
// lifetime contract symmetric.
//
// Lifetime contract: the agent guarantees both child references outlive the multi-source.
template <typename SourceA, typename SourceB, typename OffsetT = ::cuda::std::int64_t>
class multi_source_data_source
{
public:
  using value_t = typename SourceA::value_t;
  static_assert(::cuda::std::is_same_v<value_t, typename SourceB::value_t>,
                "multi_source_data_source requires both sources to share value_t");

  // This class intentionally publishes no `TempStorage`: per-source persistent state is owned
  // by the agent as two separate `SourceA` / `SourceB` `TempStorage` allocations.

  // Only one source is active per submit/complete window, so the two scratch slots alias via a
  // union. The union is wrapped in `cub::Uninitialized<>` so callers can place `ScratchStorage`
  // in `__shared__` despite the alternatives' non-trivial ctors / dtors.
  //
  // When both children publish an empty ScratchStorage, collapse to `empty_storage_t` so
  // consumers see the empty signal across class boundaries.

private:
  static constexpr bool _scratch_storage_is_empty =
    is_empty_storage_v<typename SourceA::ScratchStorage> && is_empty_storage_v<typename SourceB::ScratchStorage>;

  union _ScratchStorageInner
  {
    typename SourceA::ScratchStorage a;
    typename SourceB::ScratchStorage b;
  };
  struct _ScratchStorageWrapped : CUB_NS_QUALIFIER::Uninitialized<_ScratchStorageInner>
  {};

public:
  using ScratchStorage = ::cuda::std::conditional_t<_scratch_storage_is_empty, empty_storage_t, _ScratchStorageWrapped>;

  // Tagged-union load handles. Only the arm matching `pick_b` is ever initialized; the runtime
  // tag is constant within a launch, so the compiler eliminates the dead branch.
  //
  // The union has no-op ctor/dtor so the arms need not be default-constructible, and the handle
  // manages the active arm's lifetime explicitly:
  //   - Not default-constructible (would leave both arms unbuilt).
  //   - Not copy-constructible.
  //   - Move-constructible via an explicit ctor that placement-news the active arm.
  // Construction goes through tagged ctors (`from_a_t{}` / `from_b_t{}`), each of which
  // placement-news exactly one arm; `submit_load` returns a prvalue so the call site gets
  // C++17 mandatory copy elision.
  struct from_a_t
  {};
  struct from_b_t
  {};

  struct full_load_handle
  {
    using a_t = typename SourceA::full_load_handle;
    using b_t = typename SourceB::full_load_handle;

    union _H
    {
      a_t a;
      b_t b;
      _CCCL_DEVICE _CCCL_FORCEINLINE _H() {}
      _CCCL_DEVICE _CCCL_FORCEINLINE ~_H() {}
    } h;
    bool pick_b;

    template <typename... Args>
    _CCCL_DEVICE _CCCL_FORCEINLINE explicit full_load_handle(from_a_t, Args&&... args)
        : pick_b(false)
    {
      ::new (static_cast<void*>(&h.a)) a_t(::cuda::std::forward<Args>(args)...);
    }

    template <typename... Args>
    _CCCL_DEVICE _CCCL_FORCEINLINE explicit full_load_handle(from_b_t, Args&&... args)
        : pick_b(true)
    {
      ::new (static_cast<void*>(&h.b)) b_t(::cuda::std::forward<Args>(args)...);
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle(full_load_handle&& other) noexcept
        : pick_b(other.pick_b)
    {
      if (pick_b)
      {
        ::new (static_cast<void*>(&h.b)) b_t(::cuda::std::move(other.h.b));
      }
      else
      {
        ::new (static_cast<void*>(&h.a)) a_t(::cuda::std::move(other.h.a));
      }
    }

    full_load_handle()                                   = delete;
    full_load_handle(const full_load_handle&)            = delete;
    full_load_handle& operator=(const full_load_handle&) = delete;
    full_load_handle& operator=(full_load_handle&&)      = delete;

    _CCCL_DEVICE _CCCL_FORCEINLINE ~full_load_handle()
    {
      if (pick_b)
      {
        h.b.~b_t();
      }
      else
      {
        h.a.~a_t();
      }
    }

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
    using a_t = typename SourceA::partial_load_handle;
    using b_t = typename SourceB::partial_load_handle;

    union _H
    {
      a_t a;
      b_t b;
      _CCCL_DEVICE _CCCL_FORCEINLINE _H() {}
      _CCCL_DEVICE _CCCL_FORCEINLINE ~_H() {}
    } h;
    bool pick_b;

    template <typename... Args>
    _CCCL_DEVICE _CCCL_FORCEINLINE explicit partial_load_handle(from_a_t, Args&&... args)
        : pick_b(false)
    {
      ::new (static_cast<void*>(&h.a)) a_t(::cuda::std::forward<Args>(args)...);
    }

    template <typename... Args>
    _CCCL_DEVICE _CCCL_FORCEINLINE explicit partial_load_handle(from_b_t, Args&&... args)
        : pick_b(true)
    {
      ::new (static_cast<void*>(&h.b)) b_t(::cuda::std::forward<Args>(args)...);
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle(partial_load_handle&& other) noexcept
        : pick_b(other.pick_b)
    {
      if (pick_b)
      {
        ::new (static_cast<void*>(&h.b)) b_t(::cuda::std::move(other.h.b));
      }
      else
      {
        ::new (static_cast<void*>(&h.a)) a_t(::cuda::std::move(other.h.a));
      }
    }

    partial_load_handle()                                      = delete;
    partial_load_handle(const partial_load_handle&)            = delete;
    partial_load_handle& operator=(const partial_load_handle&) = delete;
    partial_load_handle& operator=(partial_load_handle&&)      = delete;

    _CCCL_DEVICE _CCCL_FORCEINLINE ~partial_load_handle()
    {
      if (pick_b)
      {
        h.b.~b_t();
      }
      else
      {
        h.a.~a_t();
      }
    }

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

  // Take both child sources by reference. The agent owns the underlying objects; the
  // multi-source just borrows for delegation. This composes with non-copyable / non-movable
  // children without forcing a by-value ctor path.
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE multi_source_data_source(SourceA& a, SourceB& b, bool pick_b)
      : source_a(a)
      , source_b(b)
      , pick_source_b(pick_b)
  {}

  // Copy/move construction is implicitly available (memberwise copy of the two references +
  // the `bool`); only the references are copied, never the children.
  //
  // Copy/move *assignment* is implicitly deleted because reference members can't be re-bound
  // after construction. The explicit `= delete` below is documentation only: rebinding the
  // multi-source to a different pair of children is not part of the API.
  multi_source_data_source& operator=(const multi_source_data_source&) = delete;
  multi_source_data_source& operator=(multi_source_data_source&&)      = delete;

  // Both sources alive -- propagate `set_tile_base` to both.
  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT tile_base)
  {
    source_a.set_tile_base(tile_base);
    source_b.set_tile_base(tile_base);
  }

  // Re-target both child sources to new inputs and switch the active arm without reconstructing
  // them, so a long-lived multi-source can be re-pointed at a new segment's iterators/buffer.
  // The child references are mutated through, not rebound, so the deleted-assignment contract is
  // unaffected.
  template <typename InputItA, typename InputItB>
  _CCCL_DEVICE _CCCL_FORCEINLINE void set_inputs(InputItA input_a, InputItB input_b, bool pick_b)
  {
    source_a.set_input(input_a);
    source_b.set_input(input_b);
    pick_source_b = pick_b;
  }

  // Each `return` is a prvalue invocation of a tagged ctor, so C++17 mandatory copy elision
  // builds the handle directly in the caller's slot. When the aggregate `ScratchStorage` is
  // `empty_storage_t` (both children empty) the union doesn't exist, so we pass stack-local
  // stubs the compiler folds away.
  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& s)
  {
    if constexpr (_scratch_storage_is_empty)
    {
      typename SourceA::ScratchStorage a_dummy{};
      typename SourceB::ScratchStorage b_dummy{};
      if (pick_source_b)
      {
        return full_load_handle{from_b_t{}, source_b.submit_load(b_dummy)};
      }
      return full_load_handle{from_a_t{}, source_a.submit_load(a_dummy)};
    }
    else
    {
      auto& inner = s.Alias();
      if (pick_source_b)
      {
        return full_load_handle{from_b_t{}, source_b.submit_load(inner.b)};
      }
      return full_load_handle{from_a_t{}, source_a.submit_load(inner.a)};
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
        return partial_load_handle{from_b_t{}, source_b.submit_load(b_dummy, num_items)};
      }
      return partial_load_handle{from_a_t{}, source_a.submit_load(a_dummy, num_items)};
    }
    else
    {
      auto& inner = s.Alias();
      if (pick_source_b)
      {
        return partial_load_handle{from_b_t{}, source_b.submit_load(inner.b, num_items)};
      }
      return partial_load_handle{from_a_t{}, source_a.submit_load(inner.a, num_items)};
    }
  }

  // On-demand single-item gather. Dispatches to whichever source is active; `pick_source_b`
  // is constant within a launch, so the dead arm is eliminated.
  _CCCL_DEVICE _CCCL_FORCEINLINE value_t gather_one(int item_idx) const
  {
    return pick_source_b ? source_b.gather_one(item_idx) : source_a.gather_one(item_idx);
  }

  // Propagate the (optional) reset step to both arms. A no-op for direct / sync_block_load
  // sources; called on both arms because both children are alive.
  _CCCL_DEVICE _CCCL_FORCEINLINE void invalidate()
  {
    source_a.invalidate();
    source_b.invalidate();
  }

private:
  SourceA& source_a;
  SourceB& source_b;
  bool pick_source_b;
};

//---------------------------------------------------------------------
// 5. `make_tile_data_source` factory.
//
// Picks the concrete TileDataSource for the given `tile_load_kind`. Generative iterators
// (today: `cuda::counting_iterator`) are statically downgraded to `direct_data_source`
// regardless of the configured kind -- coalescing buys nothing for an iterator that does
// not live in memory.
//---------------------------------------------------------------------

// Tag-dispatch helper: factory_impl<Kind, IsGen> picks the data source type.
template <tile_load_kind Kind, bool IsGenerative>
struct factory_impl;

template <tile_load_kind Kind>
struct factory_impl<Kind, /*IsGenerative=*/true>
{
  template <typename It, int BlockThreads, int ItemsPerThread, typename OffsetT>
  using data_source_t = direct_data_source<It, BlockThreads, ItemsPerThread, OffsetT>;
};

template <>
struct factory_impl<tile_load_kind::direct, /*IsGenerative=*/false>
{
  template <typename It, int BlockThreads, int ItemsPerThread, typename OffsetT>
  using data_source_t = direct_data_source<It, BlockThreads, ItemsPerThread, OffsetT>;
};

#define _CUB_DETAIL_TOPK_SYNC_BL_FACTORY(Kind)                                                                   \
  template <>                                                                                                    \
  struct factory_impl<Kind, /*IsGenerative=*/false>                                                              \
  {                                                                                                              \
    template <typename It, int BlockThreads, int ItemsPerThread, typename OffsetT>                               \
    using data_source_t =                                                                                        \
      sync_block_load_data_source<It, BlockThreads, ItemsPerThread, sync_block_load_algo<Kind>::value, OffsetT>; \
  }

_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_direct);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_vectorize);

#undef _CUB_DETAIL_TOPK_SYNC_BL_FACTORY

// Type alias selecting the concrete TileDataSource type for the given configuration.
// `OffsetT` defaults to `int64_t` to handle large problems; per-call overflow can be
// avoided by passing `int32_t` for known-small inputs.
template <typename It,
          tile_load_kind ConfiguredKind,
          int BlockThreads,
          int ItemsPerThread,
          typename OffsetT = ::cuda::std::int64_t>
using tile_data_source_t =
  typename factory_impl<ConfiguredKind,
                        CUB_NS_QUALIFIER::detail::is_generative_iterator_v<::cuda::std::remove_cv_t<It>>>::
    template data_source_t<It, BlockThreads, ItemsPerThread, OffsetT>;

template <typename It,
          tile_load_kind ConfiguredKind,
          int BlockThreads,
          int ItemsPerThread,
          typename OffsetT = ::cuda::std::int64_t>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto make_tile_data_source(
  It it, typename tile_data_source_t<It, ConfiguredKind, BlockThreads, ItemsPerThread, OffsetT>::TempStorage& state)
{
  using data_source_t = tile_data_source_t<It, ConfiguredKind, BlockThreads, ItemsPerThread, OffsetT>;
  return data_source_t{it, state};
}
} // namespace topk
} // namespace detail

CUB_NAMESPACE_END
