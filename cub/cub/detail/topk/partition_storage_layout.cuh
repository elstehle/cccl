// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Agent-side smem layout helper for the top-k partition / filter primitives.
//!
//! `partition_storage_layout` aliases the temporally-disjoint smem footprints of a CTA's
//! partition pass -- the keys-source's per-tile `ScratchStorage` (active during load) and the
//! partition primitive's `ScratchStorage` (active during classify + scatter, itself embedding
//! the value-source's `ScratchStorage`; see `block_partition.cuh`) -- in one byte-arena:
//! `union { keys_source_scratch | partition_scratch }`. Slots are accessed through the
//! `get_*()` member functions so call sites are layout-agnostic.

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
// `partition_storage_layout` -- agent-side smem layout helper; see the header doc
// for the lifetime / aliasing argument.
//---------------------------------------------------------------------

namespace partition_storage_layout_detail
{
// `Uninitialized<>` wraps the union so the agent can place it directly in `__shared__`
// regardless of the inner types' constructors (the members are themselves typically
// `Uninitialized<>`-wrapped, e.g. `multi_source_data_source::ScratchStorage`).
template <typename PartitionT, typename KeysSourceScratchT>
union nonpersistent_scratch_inner
{
  KeysSourceScratchT keys_source_scratch;
  typename PartitionT::ScratchStorage partition_scratch;
};
} // namespace partition_storage_layout_detail

template <bool NeedsPersistent, typename PartitionT, typename KeysSourceScratchT>
struct partition_storage_layout;

template <typename PartitionT, typename KeysSourceScratchT>
struct partition_storage_layout</*NeedsPersistent=*/false, PartitionT, KeysSourceScratchT>
{
  CUB_NS_QUALIFIER::Uninitialized<
    partition_storage_layout_detail::nonpersistent_scratch_inner<PartitionT, KeysSourceScratchT>>
    scratch;

  // Empty TempStorage -- sized to zero by the compiler (PartitionT::TempStorage is `struct{}`).
  typename PartitionT::TempStorage partition_state;

  _CCCL_DEVICE _CCCL_FORCEINLINE typename PartitionT::TempStorage& get_partition_state()
  {
    return partition_state;
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

// Agent-friendly alias that derives `NeedsPersistent` from
// `cuda::std::is_empty_v<typename PartitionT::TempStorage>`: an empty `TempStorage` (e.g.
// `struct{}`) selects the non-persistent union. (`cub::Uninitialized<T>` carries a
// `DeviceWord storage[N]` member, so `is_empty_v` correctly returns `false` on the wrapper.)
template <typename PartitionT, typename KeysSourceScratchT>
using partition_storage_layout_for_t =
  partition_storage_layout<!::cuda::std::is_empty_v<typename PartitionT::TempStorage>, PartitionT, KeysSourceScratchT>;
} // namespace detail::topk

CUB_NAMESPACE_END
