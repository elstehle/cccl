// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
// Move register to uniform register

// For int32_t and uint32_t, we can use the CREDUX instruction, which is coupled and has a constant latency.
// For 64-bit types, we still use __shfl_sync

[[nodiscard]] _CCCL_DEVICE_API inline int makeWarpUniform(int x)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __reduce_min_sync(~0, x);), (return x;));
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t makeWarpUniform(::cuda::std::uint32_t x)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __reduce_min_sync(~0, x);), (return x;));
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint64_t makeWarpUniform(::cuda::std::uint64_t x)
{
  // Split into two 32-bit halves so we can route each through the 32-bit overload, which uses
  // `__reduce_min_sync` (CREDUX) on Hopper+. Empirically, ptxas's `R2UR` heuristic on Blackwell
  // does not recognize the result of `SHFL.IDX` with `srcLane=0` as warp-uniform, but does
  // recognize the result of `__reduce_*_sync`. Using two CREDUX dispatches per 64-bit value
  // therefore lets downstream consumers of the result be eligible for uniform-register
  // promotion. Falls back to a plain `__shfl_sync` on sub-Hopper (where the 32-bit overload
  // is a no-op identity anyway, so we can't compose).
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (const auto lo = makeWarpUniform(static_cast<::cuda::std::uint32_t>(x));
     const auto hi = makeWarpUniform(static_cast<::cuda::std::uint32_t>(x >> 32));
     return (static_cast<::cuda::std::uint64_t>(hi) << 32) | lo;),
    (return __shfl_sync(~0, x, 0);));
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int64_t makeWarpUniform(::cuda::std::int64_t x)
{
  return static_cast<::cuda::std::int64_t>(makeWarpUniform(static_cast<::cuda::std::uint64_t>(x)));
}

// Pointer overload: round-trip through the 64-bit integer broadcast so any
// pointer-typed warp-uniform value (per-segment metadata pointer, dereferenced
// iterator value when the iterator is a raw pointer, etc.) becomes eligible
// for ptxas's R2UR promotion downstream.
template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE_API inline _Tp* makeWarpUniform(_Tp* p)
{
  const auto bits = ::cuda::std::bit_cast<::cuda::std::uint64_t>(p);
  return ::cuda::std::bit_cast<_Tp*>(makeWarpUniform(bits));
}

// Bool overload: route through the 32-bit integer broadcast so warp-uniform
// predicates feed UISETP downstream rather than per-thread predicates.
[[nodiscard]] _CCCL_DEVICE_API inline bool makeWarpUniform(bool b)
{
  return makeWarpUniform(static_cast<::cuda::std::uint32_t>(b)) != 0u;
}

// Generic overload for trivially-copyable types that don't match any of the
// specific overloads above (e.g. small enums, iterator wrappers like
// `cuda::transform_output_iterator`, `cuda::constant_iterator`).
//
// Decomposes the value into a sequence of 32-bit chunks, broadcasts each
// chunk through the 32-bit `makeWarpUniform` (CREDUX on sm_90+), and
// reconstructs. Works for any size as long as `_Tp` is trivially copyable
// (and we round the byte count up to a multiple of 4 with zero-padding for
// the broadcast -- the padding bytes never get written back to the result).
//
// Requires `_Tp` to be default-constructible so we can build the output
// before `memcpy`-ing the broadcast bits into it. Trivially-default-ctible
// is the standard for trivially-copyable types we care about (iterators,
// PODs, small functor structs).
template <typename _Tp,
          ::cuda::std::enable_if_t<::cuda::std::is_trivially_copyable_v<_Tp> && !::cuda::std::is_pointer_v<_Tp>
                                     && !::cuda::std::is_integral_v<_Tp> && !::cuda::std::is_same_v<_Tp, bool>,
                                   int> = 0>
[[nodiscard]] _CCCL_DEVICE_API inline _Tp makeWarpUniform(_Tp value)
{
  constexpr ::cuda::std::size_t kNumChunks = (sizeof(_Tp) + 3u) / 4u;
  ::cuda::std::uint32_t chunks[kNumChunks]{};
  __builtin_memcpy(chunks, &value, sizeof(_Tp));
  for (::cuda::std::size_t i = 0; i < kNumChunks; ++i)
  {
    chunks[i] = makeWarpUniform(chunks[i]);
  }
  _Tp out{};
  __builtin_memcpy(&out, chunks, sizeof(_Tp));
  return out;
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END
