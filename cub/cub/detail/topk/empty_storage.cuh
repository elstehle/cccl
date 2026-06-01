// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Empty-storage convention for the top-k primitive subtree.
//!
//!   - `empty_storage_t` -- canonical marker for "this `TempStorage` / `ScratchStorage`
//!     carries no smem state". Its no-op `Alias()` lets consumer code that uniformly does
//!     `buffer.Alias().<member>` keep compiling whether `buffer` is `cub::Uninitialized<>`
//!     (non-empty case) or `empty_storage_t` itself (empty case).
//!   - `is_empty_storage_v<T>` -- permissive trait: `true` when `T` is the marker *or* an
//!     empty struct (so legacy `struct {}` declarations work without migration). Consumers
//!     gate `__syncthreads()` and other empty-storage-only setup on it.
//!
//! Conventional pattern at composition sites:
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
//! This makes the trait transitive: a composite whose children are all empty publishes itself
//! as empty too, so the signal survives `Uninitialized<>` wrapping (which would otherwise hide
//! it, since `Uninitialized<empty>` still carries a 1-byte `DeviceWord storage[N]`).

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

//! Canonical marker for an empty `TempStorage` / `ScratchStorage` in the top-k subtree.
//! Trivially constructible/copyable/destructible -- safe to declare as `__shared__` directly.
//! The no-op `Alias()` keeps `buffer.Alias().<member>` compiling in both the `Uninitialized<>`
//! and empty cases; in the empty case the consumer must gate the access with
//! `if constexpr (!is_empty_storage_v<...>)` and never actually read through this `Alias()`.
struct empty_storage_t
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE empty_storage_t& Alias()
  {
    return *this;
  }
};

//! Permissive empty-storage trait: `true` when `T` is the canonical `empty_storage_t` marker
//! *or* a class type with no non-static data members (the latter catches legacy `struct {}`
//! TempStorage declarations, e.g. in BlockLoad / BlockScan, without forcing them to migrate).
template <typename T>
inline constexpr bool is_empty_storage_v =
     ::cuda::std::is_same_v<T, empty_storage_t>
  || ::cuda::std::is_empty_v<T>;

} // namespace detail::topk

CUB_NAMESPACE_END
