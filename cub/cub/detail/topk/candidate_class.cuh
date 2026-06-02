// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Classification of a top-k candidate against the running k-th key boundary. Shared by the
//! single-problem and batched top-k implementations: the partition / filter primitives return it,
//! and the candidate-identification ops produce it.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
enum class candidate_class
{
  // The given candidate is definitely amongst the top-k items
  selected,
  // The given candidate may or may not be amongst the top-k items
  candidate,
  // The given candidate is definitely not amongst the top-k items
  rejected
};
} // namespace detail::topk

CUB_NAMESPACE_END
