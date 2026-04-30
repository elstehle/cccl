// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include <cub/block/block_scan.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_device.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/concepts>

CUB_NAMESPACE_BEGIN
namespace detail::topk
{
_CCCL_HOST_DEVICE_API constexpr int calc_bits_per_pass(int key_size)
{
  switch (key_size)
  {
    case 1:
    default:
      return 8;
    case 2:
    case 4:
    case 8:
      return 11;
  }
}

template <class KeyT>
_CCCL_HOST_DEVICE_API constexpr int calc_bits_per_pass()
{
  return calc_bits_per_pass(int{sizeof(KeyT)});
}

// Map a `BlockLoadAlgorithm` enum value to the equivalent `tile_load_kind` value, so
// per-arch defaults expressed in terms of `BlockLoadAlgorithm` (the legacy convention)
// preserve the same load behavior under the unified `tile_load_kind` knob.
[[nodiscard]] _CCCL_API constexpr tile_load_kind block_load_algorithm_to_tile_load_kind(BlockLoadAlgorithm algo)
{
  return (algo == BLOCK_LOAD_DIRECT)            ? tile_load_kind::block_load_direct
       : (algo == BLOCK_LOAD_STRIPED)           ? tile_load_kind::block_load_striped
       : (algo == BLOCK_LOAD_VECTORIZE)         ? tile_load_kind::block_load_vectorize
       : (algo == BLOCK_LOAD_TRANSPOSE)         ? tile_load_kind::block_load_transpose
       : (algo == BLOCK_LOAD_WARP_TRANSPOSE)    ? tile_load_kind::block_load_warp_transpose
                                                 : tile_load_kind::block_load_warp_transpose_timesliced;
}

struct topk_policy
{
  int threads_per_block;
  int items_per_thread;
  int bits_per_pass;
  BlockLoadAlgorithm load_algorithm;
  BlockScanAlgorithm scan_algorithm;
  BlockPartitionStrategy partition_strategy = BlockPartitionStrategy::Atomics;
  // Architecture §2.4: unifies sync `BlockLoadAlgorithm` choices and adds async TMA.
  // Defaults to the equivalent of `load_algorithm` so per-arch tunings expressed in
  // terms of the legacy enum preserve their behavior without explicit migration.
  tile_load_kind keys_tile_load_kind = block_load_algorithm_to_tile_load_kind(BLOCK_LOAD_VECTORIZE);
  // Selects how `BlockPartition` materializes the per-item classification: either
  // precompute a `classes[ItemsPerThread]` array up front (smaller code, +1 register
  // pass over the keys), or recompute it inline at each scatter use-site (frees the
  // array's registers; pays for one extra `identify_candidates_op` evaluation per
  // item). `inlined` is only honored when `partition_strategy` is `Atomics`. Placed
  // last so existing positional initializers (which omit it) keep working.
  BlockPartitionClassifyMode classify_mode = BlockPartitionClassifyMode::precomputed;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const topk_policy& lhs, const topk_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.bits_per_pass == rhs.bits_per_pass && lhs.load_algorithm == rhs.load_algorithm
        && lhs.scan_algorithm == rhs.scan_algorithm && lhs.partition_strategy == rhs.partition_strategy
        && lhs.classify_mode == rhs.classify_mode && lhs.keys_tile_load_kind == rhs.keys_tile_load_kind;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const topk_policy& lhs, const topk_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const topk_policy& p)
  {
    return os << "topk_policy { .threads_per_block = " << p.threads_per_block
              << ", .items_per_thread = " << p.items_per_thread << ", .bits_per_pass = " << p.bits_per_pass
              << ", .load_algorithm = " << p.load_algorithm << ", .scan_algorithm = " << p.scan_algorithm
              << ", .partition_strategy = " << static_cast<int>(p.partition_strategy)
              << ", .classify_mode = " << static_cast<int>(p.classify_mode)
              << ", .keys_tile_load_kind = " << static_cast<int>(p.keys_tile_load_kind) << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept topk_policy_selector = policy_selector<T, topk_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int key_size;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> topk_policy
  {
    constexpr int nominal_4b_items_per_thread = 4;
    const int bits_per_pass                   = calc_bits_per_pass(key_size);

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      // Try to load 16 bytes per thread: int64 -> 2, int32 -> 4, int16 -> 8.
      const int items_per_thread = ::cuda::std::max(1, nominal_4b_items_per_thread * 4 / key_size);
      return topk_policy{512,
                         items_per_thread,
                         bits_per_pass,
                         BLOCK_LOAD_VECTORIZE,
                         BLOCK_SCAN_WARP_SCANS,
                         BlockPartitionStrategy::Atomics,
                         block_load_algorithm_to_tile_load_kind(BLOCK_LOAD_VECTORIZE)};
    }

    // Default tuning used on older architectures.
    const int items_per_thread =
      ::cuda::std::clamp(nominal_4b_items_per_thread * 4 / key_size, 1, nominal_4b_items_per_thread);
    return topk_policy{512,
                       items_per_thread,
                       bits_per_pass,
                       BLOCK_LOAD_VECTORIZE,
                       BLOCK_SCAN_WARP_SCANS,
                       BlockPartitionStrategy::Atomics,
                       block_load_algorithm_to_tile_load_kind(BLOCK_LOAD_VECTORIZE)};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(topk_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename KeyT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> topk_policy
  {
    constexpr auto policies = policy_selector{int{sizeof(KeyT)}};
    return policies(cc);
  }
};
} // namespace detail::topk
CUB_NAMESPACE_END
