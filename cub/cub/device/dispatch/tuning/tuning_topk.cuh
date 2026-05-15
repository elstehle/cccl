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
  //     three non-accumulating values select one of `BlockPartition{Atomics,Staged,
  //     SharedMem}`, and `AccumulatingCandidates` selects
  //     `BlockPartitionAccumulatingCandidates`.
  //   - `early_stop_filter_strategy` is a `BlockFilterStrategy` value -- the three
  //     non-accumulating values select one of `BlockFilter{Atomics,Staged,
  //     SharedMem}`, and `AccumulatingFilter` selects `BlockFilterAccumulating`.
  //     The early-stop pass operates as a 1-stream filter (the candidate-side
  //     machinery is statically elided), so it has its own enum independent of the
  //     buffered-pass partition enum.
  //   - `last_filter_partition_strategy` accepts any `BlockPartitionStrategy` value
  //     (including `AccumulatingCandidates`).
  BlockPartitionStrategy buffered_partition_strategy    = BlockPartitionStrategy::Atomics;
  BlockFilterStrategy early_stop_filter_strategy        = BlockFilterStrategy::Atomics;
  BlockPartitionStrategy last_filter_partition_strategy = BlockPartitionStrategy::Atomics;

  // Smem-slot count for the accumulating partition / filter variants' per-stream
  // buffer. Only consulted when `buffered_partition_strategy == AccumulatingCandidates`
  // and/or `early_stop_filter_strategy == AccumulatingFilter`. Ignored otherwise.
  int accumulating_buffer_capacity = 256;

  value_materialization_mode value_materialization = value_materialization_mode::indexed;

  // When `true`, the partitioning loop skips loading the full tile of values data
  // upfront. Instead it only gathers values of non-rejected items via the source's
  // `gather_one()` operation. Honored by all three non-accumulating partition /
  // filter classes; the accumulating variants implement their own value path and
  // ignore this flag.
  bool lazy_value_load = true;

  // When `true`, the per-pass classification (identify_candidates_op /
  // identify_selected_op) is recomputed at each per-item scatter use-site rather
  // than materialized into a `classes[]` / `kept[]` register array up front. The
  // tradeoff is register pressure (precomputed) vs recomputation cost (inlined).
  // Honored by all three non-accumulating partition / filter classes; the
  // accumulating variants always classify inline by design.
  bool inlined_classify = true;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const topk_policy& lhs, const topk_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.bits_per_pass == rhs.bits_per_pass && lhs.keys_tile_load_kind == rhs.keys_tile_load_kind
        && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.buffered_partition_strategy == rhs.buffered_partition_strategy
        && lhs.early_stop_filter_strategy == rhs.early_stop_filter_strategy
        && lhs.last_filter_partition_strategy == rhs.last_filter_partition_strategy
        && lhs.accumulating_buffer_capacity == rhs.accumulating_buffer_capacity
        && lhs.value_materialization == rhs.value_materialization && lhs.lazy_value_load == rhs.lazy_value_load
        && lhs.inlined_classify == rhs.inlined_classify;
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
        << ", .lazy_value_load = " << (p.lazy_value_load ? "true" : "false")
        << ", .inlined_classify = " << (p.inlined_classify ? "true" : "false") << " }";
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

  // Strategy choice rationale -- see the design notes in this file for full detail.
  //
  // Top-k's three streams (buffered-pass candidates, early-stop selected, last-filter
  // selected + ties) all benefit from cross-tile batching of the reserve-op atomics.
  // The buffered-pass stream is the most sensitive: ~`tile_items / num_buckets` items
  // per tile (typically 1-2 items per 2048-item tile at `bits_per_pass = 11`), so a
  // per-candidate `atomicAdd` to the global reserve counter is mostly atomic latency
  // on a near-empty stream. `BlockPartitionAccumulatingCandidates` collapses that to
  // one cooperative flush per ~`accumulating_buffer_capacity` items. The early-stop
  // filter pass and the dedicated last-filter pass have dense streams instead, where
  // the same idea reduces the per-item-atomic cost on the hot output path.
  //
  // The smem cost of the Accumulating variants is a persistent buffer of
  // ~`Capacity * (sizeof(KeyT) + sizeof(IndexedValueT))` plus a few counter words
  // (~1-3 KiB at the default capacity of 256). With the smem-reuse refactor in the
  // partition / histogram agents, this buffer aliases with the kth-bucket scan
  // scratch (`arms.prefix_sum`) and disappears from the kernel's smem budget on
  // typical key sizes -- the previous `Atomics`-only kernels were sized by the
  // larger of `prefix_sum` or the per-tile scratch anyway.
  //
  // The alternative `Staged` / `SharedMem` strategies trade `~tile_items * sizeof(KeyT)`
  // of per-tile smem for fully coalesced stores; on the sparse buffered-pass stream
  // they end up paying for an arena that's mostly empty, and on the dense streams
  // they consume the smem that the agent refactor freed. We do not pick them by
  // default; they remain available for user-supplied policy selectors.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> topk_policy
  {
    constexpr int nominal_4b_items_per_thread = 4;
    const int bits_per_pass                   = calc_bits_per_pass(key_size);

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      // Try to load 16 bytes per thread: int64 -> 2, int32 -> 4, int16 -> 8.
      constexpr int threads_per_block = 512;
      const int items_per_thread      = ::cuda::std::max(1, nominal_4b_items_per_thread * 4 / key_size);
      const int tile_items            = threads_per_block * items_per_thread;
      // Hopper+ has the smem budget (~228 KiB / SM) to absorb a per-block
      // accumulating buffer the size of a full tile. Sizing the buffer at
      // `tile_items` collapses each accumulating primitive's overflow loop to
      // `max_flush_iters == 2`, which NVCC straight-lines and which cuts the
      // loop-carried liveness of the per-thread `positions[]` array roughly in
      // half. On 4-byte keys the buffer costs ~16 KiB of smem; with the agent
      // `arms.prefix_sum` alias landed in the smem-reuse refactor, that buffer
      // overlaps the kth-bucket scratch and effectively replaces the previous
      // 8-KiB scan slot. Cross-tile batching of sparse buffered-pass candidates
      // still works -- the counter accumulates over many tiles before any
      // single tile can fill the buffer.
      return topk_policy{
        /*.threads_per_block             =*/ threads_per_block,
        /*.items_per_thread              =*/ items_per_thread,
        /*.bits_per_pass                 =*/ bits_per_pass,
        /*.keys_tile_load_kind           =*/ tile_load_kind::block_load_vectorize,
        /*.scan_algorithm                =*/ BLOCK_SCAN_WARP_SCANS,
        /*.buffered_partition_strategy   =*/ BlockPartitionStrategy::AccumulatingCandidates,
        /*.early_stop_filter_strategy    =*/ BlockFilterStrategy::AccumulatingFilter,
        /*.last_filter_partition_strategy=*/ BlockPartitionStrategy::AccumulatingCandidates,
        /*.accumulating_buffer_capacity  =*/ tile_items};
    }

    // Default tuning used on older architectures: keep the smaller 256-slot
    // accumulating buffer (smem-constrained pre-Hopper). The overflow loop is
    // still compile-time-bounded by `max_flush_iters`; it just won't collapse
    // to a 2-iteration form here.
    const int items_per_thread =
      ::cuda::std::clamp(nominal_4b_items_per_thread * 4 / key_size, 1, nominal_4b_items_per_thread);
    return topk_policy{
      /*.threads_per_block             =*/ 512,
      /*.items_per_thread              =*/ items_per_thread,
      /*.bits_per_pass                 =*/ bits_per_pass,
      /*.keys_tile_load_kind           =*/ tile_load_kind::block_load_vectorize,
      /*.scan_algorithm                =*/ BLOCK_SCAN_WARP_SCANS,
      /*.buffered_partition_strategy   =*/ BlockPartitionStrategy::AccumulatingCandidates,
      /*.early_stop_filter_strategy    =*/ BlockFilterStrategy::AccumulatingFilter,
      /*.last_filter_partition_strategy=*/ BlockPartitionStrategy::AccumulatingCandidates};
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
