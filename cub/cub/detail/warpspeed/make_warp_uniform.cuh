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

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
// Move register to uniform register. For uint32_t we use the CREDUX instruction
// (`__reduce_min_sync`) on Hopper+, which is coupled and has a constant latency; on older
// architectures it degrades to an identity.

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t makeWarpUniform(::cuda::std::uint32_t x)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __reduce_min_sync(~0, x);), (return x;));
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END
