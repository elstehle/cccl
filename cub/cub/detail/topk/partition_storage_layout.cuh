// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Agent-side smem layout helper for the top-k partition / filter primitives.
//!
//! `partition_storage_layout` aliases the three temporally-disjoint smem
//! footprints of a CTA's partition pass:
//!
//!   1. The keys-source's per-tile `ScratchStorage` (active during the load
//!      phase, consumed by `complete_load`).
//!   2. The partition primitive's `ScratchStorage` (active during classify +
//!      scatter; itself embeds the value-source's `ScratchStorage`, see
//!      `block_partition.cuh`).
//!   3. An optional `PrefixSumT` arena used by the agent's terminal
//!      kth-bucket scan (only reached after the partition's `epilogue()`).
//!
//! The layout has two shapes, picked by whether the partition class carries
//! persistent state across `partition()` calls:
//!
//!   `NeedsPersistent == false` (the non-accumulating partition / filter
//!     classes; their `TempStorage` is empty): the three slots above alias
//!     a single byte-arena, `union { keys_source_scratch | prefix_sum |
//!     partition_scratch }`.
//!
//!   `NeedsPersistent == true`  (the accumulating partition / filter
//!     classes; their `TempStorage` carries per-stream slot buffers +
//!     counters across all tiles): `partition_state` is in its own arena
//!     and aliases with `prefix_sum` (only safe because prefix_sum runs
//!     after the partition's terminal `epilogue()` on the last block); the
//!     per-tile scratch arena is `union { keys_source_scratch |
//!     partition_scratch }`.
//!
//! All four slots are accessed through the four `get_*()` member functions
//! so the agent's call sites are layout-agnostic.
//!
//! This helper used to live next to the partition class definitions in
//! `block_partition.cuh`; it has been moved here so the partition primitives
//! can stay agnostic about prefix-sum / keys-source concerns -- those are
//! agent responsibilities.

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
#include <cub/util_type.cuh>

#include <cuda/std/__type_traits/is_empty.h>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{

//---------------------------------------------------------------------
// `partition_storage_layout` -- agent-side smem layout helper.
//
// Picks the right storage shape for the union (`partition_t::TempStorage`,
// prefix-sum state, load scratch, partition scratch) based on whether the
// partition class carries persistent state across calls. See the header doc
// for the lifetime / aliasing argument.
//---------------------------------------------------------------------

namespace partition_storage_layout_detail
{
// Inner storage for the non-persistent layout. `Uninitialized<>` wraps it so
// the agent can place it directly in `__shared__` regardless of the inner
// types' constructors -- the alternative members are themselves typically
// `Uninitialized<>`-wrapped (e.g. `multi_source_data_source::ScratchStorage`
// is a wrapped union); this outer wrapper keeps that pattern composable
// when arbitrary inner types are plugged in.
template <typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
union nonpersistent_scratch_inner
{
  KeysSourceScratchT keys_source_scratch;
  PrefixSumT prefix_sum;
  typename PartitionT::ScratchStorage partition_scratch;
};

// Persistent layout's two arenas. The `persistent_inner` arena lives across
// all tiles (partition_state's lifetime); `scratch_inner` is per-tile. They
// alias by virtue of `prefix_sum` running strictly after the partition's
// `epilogue()` on the last block.
template <typename PartitionT, typename PrefixSumT>
union persistent_arena_inner
{
  typename PartitionT::TempStorage partition_state;
  PrefixSumT prefix_sum;
};

template <typename PartitionT, typename KeysSourceScratchT>
union persistent_scratch_inner
{
  KeysSourceScratchT keys_source_scratch;
  typename PartitionT::ScratchStorage partition_scratch;
};
} // namespace partition_storage_layout_detail

template <bool NeedsPersistent, typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
struct partition_storage_layout;

template <typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
struct partition_storage_layout</*NeedsPersistent=*/false, PartitionT, KeysSourceScratchT, PrefixSumT>
{
  CUB_NS_QUALIFIER::Uninitialized<partition_storage_layout_detail::nonpersistent_scratch_inner<
    PartitionT,
    KeysSourceScratchT,
    PrefixSumT>>
    scratch;

  // Empty TempStorage -- accessed only for parity with the persistent layout. Sized
  // to zero by the compiler (PartitionT::TempStorage is `struct{}`).
  typename PartitionT::TempStorage partition_state;

  _CCCL_DEVICE _CCCL_FORCEINLINE typename PartitionT::TempStorage& get_partition_state()
  {
    return partition_state;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE PrefixSumT& get_prefix_sum()
  {
    return scratch.Alias().prefix_sum;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE KeysSourceScratchT& get_keys_source_scratch()
  {
    return scratch.Alias().keys_source_scratch;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE typename PartitionT::ScratchStorage& get_partition_scratch()
  {
    return scratch.Alias().partition_scratch;
  }
};

template <typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
struct partition_storage_layout</*NeedsPersistent=*/true, PartitionT, KeysSourceScratchT, PrefixSumT>
{
  CUB_NS_QUALIFIER::Uninitialized<partition_storage_layout_detail::persistent_arena_inner<PartitionT, PrefixSumT>>
    persistent;
  CUB_NS_QUALIFIER::Uninitialized<partition_storage_layout_detail::persistent_scratch_inner<PartitionT, KeysSourceScratchT>>
    scratch;

  _CCCL_DEVICE _CCCL_FORCEINLINE typename PartitionT::TempStorage& get_partition_state()
  {
    return persistent.Alias().partition_state;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE PrefixSumT& get_prefix_sum()
  {
    return persistent.Alias().prefix_sum;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE KeysSourceScratchT& get_keys_source_scratch()
  {
    return scratch.Alias().keys_source_scratch;
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE typename PartitionT::ScratchStorage& get_partition_scratch()
  {
    return scratch.Alias().partition_scratch;
  }
};

// Agent-friendly alias that auto-derives `NeedsPersistent` from
// `cuda::std::is_empty_v<typename PartitionT::TempStorage>`. The partition class
// itself doesn't have to expose any explicit "needs-persistent-state" trait --
// the non-accumulating primitives' `TempStorage = struct{}` is empty (so we get
// the 3-way union), and the accumulating variants' `TempStorage` is non-empty
// (so we get the persistent + scratch layout). Empty-struct vs. wrapped-data is
// a clean signal because `cub::Uninitialized<T>` carries a `DeviceWord
// storage[N]` member, which makes `is_empty_v` correctly return `false` on the
// wrapper.
template <typename PartitionT, typename KeysSourceScratchT, typename PrefixSumT>
using partition_storage_layout_for_t =
  partition_storage_layout<!::cuda::std::is_empty_v<typename PartitionT::TempStorage>,
                           PartitionT,
                           KeysSourceScratchT,
                           PrefixSumT>;

} // namespace detail::topk

CUB_NAMESPACE_END
