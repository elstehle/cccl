// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cub/block/block_store.cuh>
#include <cub/detail/topk/block_filter.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/device/dispatch/tuning/tuning_topk.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN
namespace detail::batched_topk
{
struct epilogue_policy
{
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const epilogue_policy& lhs, const epilogue_policy& rhs)
  {
    return lhs.items_per_thread == rhs.items_per_thread && lhs.load_algorithm == rhs.load_algorithm
        && lhs.store_algorithm == rhs.store_algorithm && lhs.scan_algorithm == rhs.scan_algorithm;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const epilogue_policy& lhs, const epilogue_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const epilogue_policy& p)
  {
    return os
        << "epilogue_policy { .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .store_algorithm = " << p.store_algorithm << ", .scan_algorithm = " << p.scan_algorithm << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct worker_policy
{
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  BlockStoreAlgorithm store_algorithm;

  epilogue_policy epilogue;

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const worker_policy& lhs, const worker_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.store_algorithm == rhs.store_algorithm
        && lhs.epilogue == rhs.epilogue;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const worker_policy& lhs, const worker_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const worker_policy& p)
  {
    return os << "worker_policy { .threads_per_block = " << p.threads_per_block
              << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
              << ", .store_algorithm = " << p.store_algorithm << ", .epilogue = " << p.epilogue << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

// Tuning policy for the multi-CTA-per-segment top-k kernels (histogram / filter / last-filter).
struct multi_worker_policy
{
  int threads_per_block;
  int items_per_thread;
  int bits_per_pass;

  // Algorithm used to load each tile of keys (covers `BlockLoadAlgorithm` variants and async-TMA)
  detail::topk::tile_load_kind keys_tile_load_kind;

  // Scan algorithm used in the `finalize_pass` epilogue, computing prefix sum over the histogram bins.
  BlockScanAlgorithm scan_algorithm;

  // Partition / filter strategies used for each of the three scenarios.
  // Strategy used for writing candidates to the temp buffer and selected to the user-provided iterator.
  detail::topk::block_partition_strategy buffered_partition_strategy;
  // Strategy used for writing both candidates and selected items to the user-provided iterator
  detail::topk::block_filter_strategy early_stop_filter_strategy;
  // During the last pass, a capped number of candidates (tied keys) goes back-to-front to the user-iterator, while selected are appended front-to-back.
  detail::topk::block_partition_strategy last_filter_partition_strategy;

  // Smem-slot count for the accumulating partition / filter variants' per-stream
  // buffer. Reused as the candidate-stream buffer capacity for `SpeculativeBoth`
  // and the selected-stream buffer capacity for `SpeculativeFilter`. Ignored by
  // the non-accumulating strategies.
  // TODO (elstehle): Remove from initial version
  int accumulating_buffer_capacity;

  // Smem-slot count for the selected-stream buffer of the `SpeculativeBoth`
  // partition strategy. `0` short-circuits the selected smem buffer to pure
  // per-item global atomics for the selected stream. Ignored by every other strategy.
  // TODO (elstehle): Remove from initial version
  int speculative_selected_buffer_capacity;

  // Whether to materialize values into the temp storage or use indexed top-k with an on-the-fly gather.
  detail::topk::value_materialization_mode value_materialization;

  // When `true`, the partitioning loop skips loading the full tile of values data upfront.
  bool lazy_value_load;

  // When `true`, the per-pass classification computed scatter use-site rather than materialized into a `classes[]` array up front.
  bool inlined_classify;

  // Number of consecutive tiles a single CTA processes before grid-striding to the next chunk
  // (`gridDim.x * tiles_per_chunk` apart). Used by every multi-CTA-per-segment kernel
  // (histogram / filter / last_filter) as the inner-loop count of the kernel's nested
  // grid-stride loop:
  //   - histogram / filter: when consecutive tiles belong to the same segment, the per-segment
  //     smem histogram is initialized once at the top of the run and merged into the
  //     per-segment global histogram once at the bottom, amortizing init / merge across the
  //     chunk. A chunk that crosses a segment boundary flushes the current segment's smem
  //     histogram (when applicable for the mode) and re-initializes for the new segment.
  //   - last_filter: no histogram, but the same chunking still amortizes the per-segment state
  //     resolution (binary search + counter / iterator dereferences) across same-segment tiles.
  // Set to `1` to fall back to one-tile-per-grid-stride.
  int tiles_per_chunk;

  // Experimental knob: split the partial-tile responsibility off of the histogram kernel.
  //
  //   - `false` (default, original behavior): the histogram kernel walks all tiles of the
  //     queue (full tiles + the trailing partial tile of each segment); the
  //     finalize-histogram kernel does only the prefix-sum + bucket-finder epilogue.
  //   - `true`: the histogram kernel processes **only** full tiles. The trailing partial
  //     tile of each segment (if any) is loaded + binned by the finalize-histogram kernel
  //     (one CTA per segment) directly into that segment's global histogram, right before
  //     the prefix-sum + bucket-finder runs.
  //
  // The motivation is kernel-code streamlining: with `true`, the histogram kernel has no
  // partial-tile load path, no partial-tile bin-extract loop, and no `process_partial`
  // predicate -- everything the compiler sees per inner iteration is a full-tile load. The
  // partial tile becomes one extra small loop in the (already serialized at 1 CTA per
  // segment) finalize kernel.
  bool full_tiles_only_histogram;

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const multi_worker_policy& lhs, const multi_worker_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block //
        && lhs.items_per_thread == rhs.items_per_thread //
        && lhs.bits_per_pass == rhs.bits_per_pass //
        && lhs.keys_tile_load_kind == rhs.keys_tile_load_kind //
        && lhs.scan_algorithm == rhs.scan_algorithm //
        && lhs.buffered_partition_strategy == rhs.buffered_partition_strategy //
        && lhs.early_stop_filter_strategy == rhs.early_stop_filter_strategy //
        && lhs.last_filter_partition_strategy == rhs.last_filter_partition_strategy //
        && lhs.accumulating_buffer_capacity == rhs.accumulating_buffer_capacity //
        && lhs.speculative_selected_buffer_capacity == rhs.speculative_selected_buffer_capacity //
        && lhs.value_materialization == rhs.value_materialization //
        && lhs.lazy_value_load == rhs.lazy_value_load //
        && lhs.inlined_classify == rhs.inlined_classify //
        && lhs.tiles_per_chunk == rhs.tiles_per_chunk //
        && lhs.full_tiles_only_histogram == rhs.full_tiles_only_histogram;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const multi_worker_policy& lhs, const multi_worker_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const multi_worker_policy& p)
  {
    return os << "multi_worker_policy { .threads_per_block = " << p.threads_per_block
              << ", .items_per_thread = " << p.items_per_thread //
              << ", .bits_per_pass = " << p.bits_per_pass //
              << ", .keys_tile_load_kind = " << static_cast<int>(p.keys_tile_load_kind) //
              << ", .scan_algorithm = " << p.scan_algorithm //
              << ", .buffered_partition_strategy = " << static_cast<int>(p.buffered_partition_strategy) //
              << ", .early_stop_filter_strategy = " << static_cast<int>(p.early_stop_filter_strategy) //
              << ", .last_filter_partition_strategy = " << static_cast<int>(p.last_filter_partition_strategy) //
              << ", .accumulating_buffer_capacity = " << p.accumulating_buffer_capacity //
              << ", .speculative_selected_buffer_capacity = " << p.speculative_selected_buffer_capacity //
              << ", .value_materialization = " << static_cast<int>(p.value_materialization) //
              << ", .lazy_value_load = " << (p.lazy_value_load ? "true" : "false") //
              << ", .inlined_classify = " << (p.inlined_classify ? "true" : "false") //
              << ", .tiles_per_chunk = " << p.tiles_per_chunk //
              << ", .full_tiles_only_histogram = " << (p.full_tiles_only_histogram ? "true" : "false") //
              << " }";
  }
#endif // _CCCL_HOSTED()
};

struct batched_topk_policy
{
  // The list of per-segment agent policies is ordered by decreasing tile size. At compile time, the smallest policy
  // whose tile size still covers the upper bound of the segment size is selected.
  ::cuda::std::array<worker_policy, 6> worker_per_segment_policies;
  multi_worker_policy multi_worker_per_segment_policy;

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const batched_topk_policy& lhs, const batched_topk_policy& rhs)
  {
    return lhs.worker_per_segment_policies == rhs.worker_per_segment_policies
        && lhs.multi_worker_per_segment_policy == rhs.multi_worker_per_segment_policy;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const batched_topk_policy& lhs, const batched_topk_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const batched_topk_policy& p)
  {
    os << "batched_topk_policy { .worker_per_segment_policies = { ";
    for (::cuda::std::size_t i = 0; i < p.worker_per_segment_policies.size(); ++i)
    {
      if (i != 0)
      {
        os << ", ";
      }
      os << p.worker_per_segment_policies[i];
    }
    return os << " }, .multi_worker_per_segment_policy = " << p.multi_worker_per_segment_policy << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept batched_topk_policy_selector = policy_selector<T, batched_topk_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  // Size of the key type, in bytes. Used to size the multi-CTA-per-segment tuning
  int key_size = sizeof(int);

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> batched_topk_policy
  {
    constexpr auto load_alg  = BLOCK_LOAD_WARP_TRANSPOSE;
    constexpr auto store_alg = BLOCK_STORE_WARP_TRANSPOSE;
    constexpr auto scan_alg  = BLOCK_SCAN_WARP_SCANS;
    constexpr auto epilogue  = epilogue_policy{16, load_alg, store_alg, scan_alg};

    // Stand-alone multi-CTA-per-segment tuning. Mirrors the single-problem
    // `tuning_topk.cuh::policy_selector` computation of `items_per_thread`: target 16 B per
    // thread on Hopper+ (`max(1, ...)`, no upper cap) and clamp to `nominal_4b_items_per_thread`
    // on older architectures. The rest of the multi-worker policy is currently cc-independent,
    // so the cc branch lives only on this one knob and the policy literal below is shared.
    constexpr int nominal_4b_items_per_thread = 4;
    const int multi_items_per_thread          = (cc >= ::cuda::compute_capability{9, 0})
                                                  ? ::cuda::std::max(1, nominal_4b_items_per_thread * 4 / key_size)
                                                  : ::cuda::std::clamp(nominal_4b_items_per_thread * 4 / key_size,
                                                             1,
                                                             nominal_4b_items_per_thread);
    const int multi_bits_per_pass             = detail::topk::calc_bits_per_pass(key_size);

    return batched_topk_policy{
      {{
        worker_policy{256, 64, load_alg, store_alg, epilogue},
        worker_policy{256, 32, load_alg, store_alg, epilogue},
        worker_policy{256, 16, load_alg, store_alg, epilogue},
        worker_policy{256, 8, load_alg, store_alg, epilogue},
        worker_policy{256, 4, load_alg, store_alg, epilogue},
        worker_policy{128, 2, load_alg, store_alg, epilogue},
      }},
      multi_worker_policy{
        /*.threads_per_block                    =*/512,
        /*.items_per_thread                     =*/multi_items_per_thread,
        /*.bits_per_pass                        =*/multi_bits_per_pass,
        /*.keys_tile_load_kind                  =*/detail::topk::tile_load_kind::block_load_vectorize,
        /*.scan_algorithm                       =*/BLOCK_SCAN_WARP_SCANS,
        /*.buffered_partition_strategy          =*/detail::topk::block_partition_strategy::atomics,
        /*.early_stop_filter_strategy           =*/detail::topk::block_filter_strategy::atomics,
        /*.last_filter_partition_strategy       =*/detail::topk::block_partition_strategy::atomics,
        /*.accumulating_buffer_capacity         =*/256,
        /*.speculative_selected_buffer_capacity =*/128,
        /*.value_materialization                =*/detail::topk::value_materialization_mode::indexed,
        /*.lazy_value_load                      =*/true,
        /*.inlined_classify                     =*/true,
        /*.tiles_per_chunk                      =*/8,
        /*.full_tiles_only_histogram            =*/true}};
  }
};

template <typename KeyT, typename ValueT, typename SegmentSizeT, ::cuda::std::int64_t MaxK>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> batched_topk_policy
  {
    return policy_selector{int{sizeof(KeyT)}}(cc);
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(batched_topk_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()
} // namespace detail::batched_topk

CUB_NAMESPACE_END
