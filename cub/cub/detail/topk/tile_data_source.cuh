// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private foundation building blocks: storage helpers, reserve callbacks,
//! generative-iterator trait, and `TileDataSource` specializations.
//!
//! Architecture overview:
//! `[topk-building-blocks-architecture_2c7af1d3.plan.md]`. This header co-locates the
//! pieces of phase P1 (foundation) and the async TMA `TileDataSource` (architecture §7);
//! once the foundation stabilizes a future split into smaller headers is cheap.
//!
//! Layout in this header (mirrors the dependency order):
//!   1. Compositional storage helpers (`phase_union<Tuple>`, `phase_aggregate<Tuple>`,
//!      and the typed accessor `at<I>`) -- architecture §2.1.1; in `cub::detail`.
//!   2. Generative-iterator trait (`is_generative_iterator`, `is_generative_iterator_v`)
//!      -- architecture §7.5; in `cub::detail`.
//!   3. Reserve callbacks (`atomic_reserve_range_op`, `back_grow_capped_reserve_op`)
//!      -- architecture §8; in `cub::detail::topk`.
//!   4. `tile_load_kind` enum -- the unified policy knob spanning sync `BlockLoad`
//!      variants and async TMA -- architecture §2.4; in `cub::detail::topk`.
//!   5. The four `TileDataSource` specializations (in `cub::detail::topk`):
//!        - `direct_data_source`              gmem -> registers, no smem
//!        - `sync_block_load_data_source`     wraps `cub::BlockLoad`
//!        - `async_to_shared_data_source`     wraps `cub::detail::BlockLoadToShared`
//!        - `multi_source_data_source`        runtime-switched two-source adapter
//!   6. The `make_tile_data_source` factory which applies §7.5 to redirect
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
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__fwd/iterator.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail
{

//---------------------------------------------------------------------
// 1. Compositional storage helpers (architecture §2.1.1).
//
// `phase_union<Tuple>`     -- recursive tagged-union; tenants are sequential in time
//                             (separated by __syncthreads()) and alias each other in smem.
// `phase_aggregate<Tuple>` -- typed-slot aggregate; tenants coexist (alive simultaneously)
//                             and do NOT alias.
// `at<I>(arena&)`          -- compile-time accessor matching cuda::std::get<I> style.
//---------------------------------------------------------------------

template <typename Tuple>
struct phase_union;

template <>
struct phase_union<::cuda::std::tuple<>>
{};

template <typename T0, typename... Rest>
struct phase_union<::cuda::std::tuple<T0, Rest...>>
{
  // Trivial special members: phase_union must be trivially constructible so that
  // `__shared__ phase_union<...>` is legal (no dynamic initialization). Both union
  // alternatives are typed but never have their constructors / destructors run; the
  // agent decides lifetimes via the brokering protocol.
  union
  {
    T0 head;
    phase_union<::cuda::std::tuple<Rest...>> rest;
  };
};

template <int I, typename Head, typename... Rest>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto& at(phase_union<::cuda::std::tuple<Head, Rest...>>& a)
{
  if constexpr (I == 0)
  {
    return a.head;
  }
  else
  {
    return at<I - 1>(a.rest);
  }
}

template <typename Tuple>
struct phase_aggregate;

template <typename... Ts>
struct phase_aggregate<::cuda::std::tuple<Ts...>>
{
  ::cuda::std::tuple<Ts...> slots;
};

template <int I, typename... Ts>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto& at(phase_aggregate<::cuda::std::tuple<Ts...>>& a)
{
  return ::cuda::std::get<I>(a.slots);
}

//---------------------------------------------------------------------
// 2. Generative-iterator trait (architecture §7.5).
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
// 3. Reserve callbacks (architecture §8).
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
    const OffsetT base = static_cast<OffsetT>(atomicAdd(counter, n));
    return {base, n};
  }
};

template <typename OffsetT>
struct back_grow_capped_reserve_op
{
  static constexpr bool may_grant_less = true;

  OffsetT* counter;
  OffsetT back_anchor;
  OffsetT cap;

  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::pair<OffsetT, OffsetT> operator()(OffsetT n) const
  {
    // Advance the global counter by the unclamped n (so subsequent blocks compute the
    // right `prev`) and locally clamp to writable items. `base` is the per-block
    // forward write base such that the union of all blocks fills [back_anchor - cap,
    // back_anchor) with no gaps.
    const OffsetT prev     = static_cast<OffsetT>(atomicAdd(counter, n));
    const OffsetT writable = (cap > prev) ? static_cast<OffsetT>(cap - prev) : OffsetT{0};
    const OffsetT granted  = (n < writable) ? n : writable;
    const OffsetT base     = static_cast<OffsetT>(back_anchor - prev - granted);
    return {base, granted};
  }
};

//---------------------------------------------------------------------
// 4. `tile_load_kind` -- the unified policy knob (architecture §2.4).
//
// Spans the sync `BlockLoadAlgorithm` choices (so no information is lost vs the legacy
// `load_algorithm` policy entry) plus the async TMA path. The factory below picks the
// concrete TileDataSource specialization from this enum.
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

namespace detail_tds
{
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
  static constexpr CUB_NS_QUALIFIER::BlockLoadAlgorithm value =
    CUB_NS_QUALIFIER::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED;
};
} // namespace detail_tds

//---------------------------------------------------------------------
// 5. `TileDataSource` specializations (architecture §7.4).
//
// Contract per architecture §7.2:
//   - Construct with `(InputIt it, TempStorage& state)`.
//   - `set_tile_base(OffsetT)` advances the global offset of the next load.
//   - `submit_load(ScratchStorage&)`            -> full_load_handle.
//   - `submit_load(ScratchStorage&, OffsetT n)` -> partial_load_handle.
//   - Each handle has `complete_load(value_t (&out)[ItemsPerThread])`.
// Default arrangement is BLOCKED: thread t gets items [t*IPT, (t+1)*IPT) of the window.
//---------------------------------------------------------------------

// 5.1 direct_data_source -- no smem; per-thread `it[base + t*IPT + j]`. Hot path.
template <typename InputIt, int BlockThreads, int ItemsPerThread, typename OffsetT = ::cuda::std::int64_t>
class direct_data_source
{
public:
  using value_t = CUB_NS_QUALIFIER::detail::it_value_t<InputIt>;

  struct TempStorage
  {};
  struct ScratchStorage
  {};

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

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE direct_data_source(InputIt it, TempStorage& /*state*/)
      : it_(it)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT tile_base)
  {
    tile_base_ = tile_base;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& /*scratch*/)
  {
    return full_load_handle{it_ + tile_base_};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& /*scratch*/, OffsetT num_items)
  {
    return partial_load_handle{it_ + tile_base_, num_items};
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
    const OffsetT idx = tile_base_ + static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + item_idx;
    return it_[idx];
  }

private:
  InputIt it_;
  OffsetT tile_base_{};
};

// 5.2 sync_block_load_data_source -- wraps `cub::BlockLoad`. ScratchStorage holds the
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

  struct TempStorage
  {};
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

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE sync_block_load_data_source(InputIt it, TempStorage& /*state*/)
      : it_(it)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT tile_base)
  {
    tile_base_ = tile_base;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& scratch)
  {
    return full_load_handle{it_ + tile_base_, &scratch};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& scratch, OffsetT num_items)
  {
    return partial_load_handle{it_ + tile_base_, &scratch, num_items};
  }

private:
  InputIt it_;
  OffsetT tile_base_{};
};

// 5.3 async_to_shared_data_source -- wraps `cub::detail::BlockLoadToShared` (TMA on
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

  _CCCL_DEVICE _CCCL_FORCEINLINE async_to_shared_data_source(InputIt it, TempStorage& state)
      : it_(it)
      , loader_(state.barrier)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT tile_base)
  {
    tile_base_ = tile_base;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& scratch)
  {
    ::cuda::std::span<char> dst{scratch.buf, sizeof(scratch.buf)};
    ::cuda::std::span<const value_t> src{it_ + tile_base_, static_cast<::cuda::std::size_t>(tile_items)};
    auto span  = loader_.template CopyAsync<value_t, GmemAlign>(dst, src);
    auto token = loader_.Commit();
    return full_load_handle{span, ::cuda::std::move(token), &loader_};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& scratch, OffsetT num_items)
  {
    ::cuda::std::span<char> dst{scratch.buf, sizeof(scratch.buf)};
    ::cuda::std::span<const value_t> src{it_ + tile_base_, static_cast<::cuda::std::size_t>(num_items)};
    auto span  = loader_.template CopyAsync<value_t, GmemAlign>(dst, src);
    auto token = loader_.Commit();
    return partial_load_handle{span, ::cuda::std::move(token), num_items, &loader_};
  }

  // Make the persistent mbarrier reusable. Caller (the agent) pairs this with its own
  // lifetime end; the primitive itself does the necessary __syncthreads internally.
  _CCCL_DEVICE _CCCL_FORCEINLINE void invalidate()
  {
    loader_.Invalidate();
  }

private:
  InputIt it_;
  loader_t loader_;
  OffsetT tile_base_{};
};

// 5.4 multi_source_data_source -- runtime-switched two-source adapter. Both underlying
// sources are alive (`TempStorage` is the aggregate of both); only one is active per
// submit/complete window so their `ScratchStorage`s alias via `phase_union`.
template <typename SourceA, typename SourceB, typename OffsetT = ::cuda::std::int64_t>
class multi_source_data_source
{
public:
  using value_t = typename SourceA::value_t;
  static_assert(::cuda::std::is_same_v<value_t, typename SourceB::value_t>,
                "multi_source_data_source requires both sources to share value_t");

  struct TempStorage
  {
    typename SourceA::TempStorage a;
    typename SourceB::TempStorage b;
  };

  using ScratchStorage = CUB_NS_QUALIFIER::detail::phase_union<
    ::cuda::std::tuple<typename SourceA::ScratchStorage, typename SourceB::ScratchStorage>>;

  // Tagged-union load handles. Both alternatives are alive in the small POD; only the
  // one matching `pick_b_` is initialized via the underlying source's submit, and only
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

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE multi_source_data_source(SourceA a, SourceB b, bool pick_b)
      : a_(a)
      , b_(b)
      , pick_b_(pick_b)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT tile_base)
  {
    a_.set_tile_base(tile_base);
    b_.set_tile_base(tile_base);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& s)
  {
    if (pick_b_)
    {
      return full_load_handle{{}, b_.submit_load(CUB_NS_QUALIFIER::detail::at<1>(s)), true};
    }
    return full_load_handle{a_.submit_load(CUB_NS_QUALIFIER::detail::at<0>(s)), {}, false};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE partial_load_handle submit_load(ScratchStorage& s, OffsetT num_items)
  {
    if (pick_b_)
    {
      return partial_load_handle{{}, b_.submit_load(CUB_NS_QUALIFIER::detail::at<1>(s), num_items), true};
    }
    return partial_load_handle{a_.submit_load(CUB_NS_QUALIFIER::detail::at<0>(s), num_items), {}, false};
  }

  // On-demand single-item gather. Dispatches to whichever underlying source is
  // active (`pick_b_` is set once at construction, so the branch is constant
  // within a kernel launch and ptxas eliminates the dead arm).
  _CCCL_DEVICE _CCCL_FORCEINLINE value_t gather_one(int item_idx) const
  {
    return pick_b_ ? b_.gather_one(item_idx) : a_.gather_one(item_idx);
  }

private:
  SourceA a_;
  SourceB b_;
  bool pick_b_;
};

//---------------------------------------------------------------------
// 6. `make_tile_data_source` factory (architecture §7.5).
//
// Picks the concrete TileDataSource for the given `tile_load_kind`. Generative iterators
// (today: `cuda::counting_iterator`) are statically downgraded to `direct_data_source`
// regardless of the configured kind -- coalescing/TMA buy nothing for an iterator that
// doesn't live in memory.
//---------------------------------------------------------------------

namespace detail_tds
{
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

#define _CUB_DETAIL_TOPK_SYNC_BL_FACTORY(Kind)                                                                   \
  template <>                                                                                                    \
  struct factory_impl<Kind, /*IsGenerative=*/false>                                                              \
  {                                                                                                              \
    template <typename It,                                                                                       \
              int BlockThreads,                                                                                  \
              int ItemsPerThread,                                                                                \
              typename OffsetT,                                                                                  \
              ::cuda::std::size_t /*GmemAlign*/>                                                                 \
    using data_source_t =                                                                                        \
      sync_block_load_data_source<It, BlockThreads, ItemsPerThread, sync_block_load_algo<Kind>::value, OffsetT>; \
  }

_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_direct);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_striped);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_vectorize);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_transpose);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_warp_transpose);
_CUB_DETAIL_TOPK_SYNC_BL_FACTORY(tile_load_kind::block_load_warp_transpose_timesliced);

#undef _CUB_DETAIL_TOPK_SYNC_BL_FACTORY

} // namespace detail_tds

// Type alias selecting the concrete TileDataSource type for the given configuration.
// `OffsetT` defaults to `int64_t` to handle large problems; per-call overflow can be
// avoided by passing `int32_t` for known-small inputs.
template <typename It,
          tile_load_kind ConfiguredKind,
          int BlockThreads,
          int ItemsPerThread,
          typename OffsetT              = ::cuda::std::int64_t,
          ::cuda::std::size_t GmemAlign = alignof(CUB_NS_QUALIFIER::detail::it_value_t<It>)>
using tile_data_source_t = typename detail_tds::factory_impl<
  ConfiguredKind,
  CUB_NS_QUALIFIER::detail::is_generative_iterator_v<::cuda::std::remove_cv_t<It>>>::
  template data_source_t<It, BlockThreads, ItemsPerThread, OffsetT, GmemAlign>;

template <typename It,
          tile_load_kind ConfiguredKind,
          int BlockThreads,
          int ItemsPerThread,
          typename OffsetT              = ::cuda::std::int64_t,
          ::cuda::std::size_t GmemAlign = alignof(CUB_NS_QUALIFIER::detail::it_value_t<It>)>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
make_tile_data_source(It it,
                      typename tile_data_source_t<It, ConfiguredKind, BlockThreads, ItemsPerThread, OffsetT, GmemAlign>::
                        TempStorage& state)
{
  using data_source_t = tile_data_source_t<It, ConfiguredKind, BlockThreads, ItemsPerThread, OffsetT, GmemAlign>;
  return data_source_t{it, state};
}

} // namespace topk
} // namespace detail

CUB_NAMESPACE_END
