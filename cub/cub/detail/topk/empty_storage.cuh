// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Empty-storage convention for the top-k primitive subtree.
//!
//! Two pieces:
//!
//!   - `empty_storage_t` -- canonical marker type for "this `TempStorage` /
//!     `ScratchStorage` carries no smem state". Storage classes inside the
//!     top-k subtree that have nothing to allocate should publish their
//!     storage as `using TempStorage = empty_storage_t;` (or
//!     `ScratchStorage`). The marker has a no-op `Alias()` so consumer code
//!     that uniformly does `buffer.Alias().<member>` keeps compiling whether
//!     `buffer` is wrapped in `cub::Uninitialized<>` (non-empty case) or is
//!     `empty_storage_t` itself (empty case).
//!
//!   - `is_empty_storage_v<T>` -- permissive trait. Returns `true` when `T`
//!     is the canonical marker *or* an empty struct (so legacy `struct {}`
//!     declarations still work without migration). Consumers gate
//!     `__syncthreads()` and other empty-storage-only setup work on this
//!     trait.
//!
//! Conventional pattern at composition sites (e.g. `multi_source_data_source`,
//! the partition `phase_t` unions, the agent's outer arm unions):
//!
//! @code
//!   private:
//!     static constexpr bool _is_empty =
//!         is_empty_storage_v<typename A::ScratchStorage>
//!      && is_empty_storage_v<typename B::ScratchStorage>;
//!     union _inner { ... };
//!     struct _wrapped : ::cub::Uninitialized<_inner> {};
//!   public:
//!     using ScratchStorage =
//!       ::cuda::std::conditional_t<_is_empty, empty_storage_t, _wrapped>;
//! @endcode
//!
//! This makes the trait transitive: when a composite's children are all
//! empty, the composite publishes itself as empty too. Consumers then see
//! the empty signal across class boundaries even after `Uninitialized<>`
//! wrapping (which would otherwise hide it, since `Uninitialized<empty>`
//! still carries a 1-byte `DeviceWord storage[N]`).
//!
//! Scope: top-k primitives only. We do *not* migrate `BlockLoad` or other
//! shared CUB primitives -- the permissive `is_empty_v<T>` arm of the trait
//! catches their `struct {}`-style empty TempStorage declarations
//! automatically.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{

//! Canonical marker for an empty `TempStorage` / `ScratchStorage` in the top-k
//! subtree. Trivially default-constructible, trivially copyable, trivially
//! destructible -- safe to declare as `__shared__` directly. The no-op
//! `Alias()` lets consumer code that does `buffer.Alias().<member>` keep
//! compiling whether `buffer` is `Uninitialized<inner>` (non-empty case) or
//! `empty_storage_t` (empty case); in the empty case the consumer is expected
//! to gate the access via `if constexpr (!is_empty_storage_v<...>)` and never
//! actually read members through this Alias().
struct empty_storage_t
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE empty_storage_t& Alias()
  {
    return *this;
  }
};

//! Permissive empty-storage trait.
//!
//! Returns `true` when `T` is the canonical `empty_storage_t` marker *or*
//! a class type with no non-static data members (the latter catches legacy
//! `struct {}`-style TempStorage declarations such as the ones used in
//! BlockLoad, BlockScan, etc., without forcing those primitives to migrate
//! to the marker).
template <typename T>
inline constexpr bool is_empty_storage_v =
     ::cuda::std::is_same_v<T, empty_storage_t>
  || ::cuda::std::is_empty_v<T>;

} // namespace detail::topk

CUB_NAMESPACE_END
