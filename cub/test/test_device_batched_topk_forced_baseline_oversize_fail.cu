// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__execution/tune.h>
#include <cuda/argument>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>

#include <iostream>

// Verifies the strict oversize-segment diagnostic for a *deterministic* request in cub::DeviceBatchedTopK (the
// segment-size static_assert in cub/device/device_batched_topk.cuh).
//
// This test used to pin the opposite behaviour: that forcing the baseline backend on a segment no worker tile covers is
// a compile error. It no longer is -- the baseline backend escalates such segments to its multi-CTA-per-segment path,
// which is now the supported route for them. What remains a compile error is asking for a segment size past the cluster
// backend's competitive range (2^21) *while also* requiring a deterministic result set: only the cluster backend is
// deterministic, so no backend can serve that combination. Built without CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT to
// exercise the strict path.
int main()
{
  namespace ex = cuda::execution;

  int** d_keys_in  = nullptr;
  int** d_keys_out = nullptr;
  // 2^22 exceeds the cluster backend's competitive range (2^21), which is the cap that still applies once a
  // deterministic result set is requested.
  auto segment_sizes = cuda::args::constant<(1 << 22)>{};
  auto k_arg         = cuda::args::constant<3>{};
  auto num_segments  = cuda::args::immediate{cuda::std::int64_t{2}};

  auto requirements =
    ex::require(ex::determinism::gpu_to_gpu, ex::tie_break::prefer_smaller_index, ex::output_ordering::unsorted);
  auto env = cuda::std::execution::env{requirements};
  // expected-error {{"exceeds the maximum supported segment size"}}

  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segments, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxKeys failed with status: " << error << '\n';
  }
}
