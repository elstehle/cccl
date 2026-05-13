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
#include <cub/detail/topk/block_filter.cuh>
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

// Selects how the value channel propagates through the candidate buffer.
// `indexed` may be preferable for smallish `k` and/or wider value types: the
// candidate buffer stores `OffsetT` indices into the user's input iterator,
// and full values are only gathered (and written through to the output
// iterator) once a candidate is classified as selected.
// The choice is implemented purely as dispatch-side iterator rewiring: in
// `indexed` mode the agent receives a `cuda::counting_iterator` for the value
// input and a `cuda::transform_output_iterator` for the value output, so the
// agent itself remains unaware of the mode.
enum class value_materialization_mode
{
  // candidate buffer stores `OffsetT` indices; values are gathered from the user's input iterator at write time
  indexed,
  // candidate buffer stores full `value_in_t` items
  materialized,
};

struct topk_policy
{
  int threads_per_block;
  int items_per_thread;
  int bits_per_pass;

  // Algorithm used to load each tile of keys (covers both `BlockLoadAlgorithm`
  // variants and async-TMA loads under a single enum).
  tile_load_kind keys_tile_load_kind;

  // Scan algorithm used in the `finalize pass` epilogue, computing prefix sum over the histogram bins.
  BlockScanAlgorithm scan_algorithm;

  // Three independent strategy knobs, one per pass.
  //   - `buffered_partition_strategy` is a `BlockPartitionStrategy` value -- the
  //     four non-accumulating values select `BlockPartition<...>`, and
  //     `AccumulatingCandidates` selects `BlockPartitionAccumulatingCandidates`.
  //   - `early_stop_filter_strategy` is a `BlockFilterStrategy` value -- the four
  //     non-accumulating values select `BlockFilter<...>`, and `AccumulatingFilter`
  //     selects `BlockFilterAccumulating`. The early-stop pass operates as a
  //     1-stream filter (the candidate-side machinery is statically elided), so it
  //     has its own enum independent of the buffered-pass partition enum.
  //   - `last_filter_partition_strategy` accepts any `BlockPartitionStrategy` value
  //     (including `AccumulatingCandidates`).
  BlockPartitionStrategy buffered_partition_strategy    = BlockPartitionStrategy::AtomicsInlinedClassify;
  BlockFilterStrategy early_stop_filter_strategy        = BlockFilterStrategy::AtomicsInlinedClassify;
  BlockPartitionStrategy last_filter_partition_strategy = BlockPartitionStrategy::AtomicsInlinedClassify;

  // Smem-slot count for the accumulating partition / filter variants' per-stream
  // buffer. Only consulted when `buffered_partition_strategy == AccumulatingCandidates`
  // and/or `early_stop_filter_strategy == AccumulatingFilter`. Ignored otherwise.
  int accumulating_buffer_capacity = 256;

  value_materialization_mode value_materialization = value_materialization_mode::indexed;

  // When `true`, the partitioning loop skips loading the full tile of values data, gathering only values of
  // non-rejected items.
  bool lazy_value_load = true;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const topk_policy& lhs, const topk_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.bits_per_pass == rhs.bits_per_pass && lhs.keys_tile_load_kind == rhs.keys_tile_load_kind
        && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.buffered_partition_strategy == rhs.buffered_partition_strategy
        && lhs.early_stop_filter_strategy == rhs.early_stop_filter_strategy
        && lhs.last_filter_partition_strategy == rhs.last_filter_partition_strategy
        && lhs.accumulating_buffer_capacity == rhs.accumulating_buffer_capacity
        && lhs.value_materialization == rhs.value_materialization && lhs.lazy_value_load == rhs.lazy_value_load;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const topk_policy& lhs, const topk_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const topk_policy& p)
  {
    return os
        << "topk_policy { .threads_per_block = " << p.threads_per_block
        << ", .items_per_thread = " << p.items_per_thread
        << ", .bits_per_pass = " << p.bits_per_pass
        << ", .keys_tile_load_kind = " << static_cast<int>(p.keys_tile_load_kind)
        << ", .scan_algorithm = " << p.scan_algorithm
        << ", .buffered_partition_strategy = " << static_cast<int>(p.buffered_partition_strategy)
        << ", .early_stop_filter_strategy = " << static_cast<int>(p.early_stop_filter_strategy)
        << ", .last_filter_partition_strategy = " << static_cast<int>(p.last_filter_partition_strategy)
        << ", .accumulating_buffer_capacity = " << p.accumulating_buffer_capacity
        << ", .value_materialization = " << static_cast<int>(p.value_materialization)
        << ", .lazy_value_load = " << (p.lazy_value_load ? "true" : "false") << " }";
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
      return topk_policy{
        512, items_per_thread, bits_per_pass, tile_load_kind::block_load_vectorize, BLOCK_SCAN_WARP_SCANS};
    }

    // Default tuning used on older architectures.
    const int items_per_thread =
      ::cuda::std::clamp(nominal_4b_items_per_thread * 4 / key_size, 1, nominal_4b_items_per_thread);
    return topk_policy{
      512, items_per_thread, bits_per_pass, tile_load_kind::block_load_vectorize, BLOCK_SCAN_WARP_SCANS};
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
