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
// Whether the per-segment candidate buffer stores value indices (gathered lazily) or full values.
enum class value_materialization_mode
{
  indexed, // candidate buffer stores `OffsetT` indices; values gathered from the input iterator at write time
  materialized // candidate buffer stores full `value_t` items
};

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

  // Algorithm used to load each tile of keys (`BlockLoadAlgorithm` variants).
  detail::topk::tile_load_kind keys_tile_load_kind;

  // Scan algorithm used to prefix-sum the histogram bins when finding the k-th bucket.
  BlockScanAlgorithm scan_algorithm;

  // Whether to materialize values into the temp storage or use indexed top-k with an on-the-fly gather.
  value_materialization_mode value_materialization;

  // When `true`, the partitioning loop skips loading the full tile of values data upfront.
  bool lazy_value_load;

  // When `true`, the per-pass classification is computed at the scatter use-site rather than
  // materialized into a `classes[]` array up front.
  bool inlined_classify;

  // Number of consecutive tiles a single CTA processes before grid-striding to the next chunk
  // (`gridDim.x * tiles_per_chunk` apart), used by every multi-CTA-per-segment kernel as the
  // inner-loop count. Amortizes per-segment setup across same-segment tiles: histogram / filter
  // init + merge the per-segment smem histogram once per chunk (re-initializing at segment
  // boundaries); last_filter amortizes the per-segment state resolution (binary search + counter /
  // iterator dereferences). Set to `1` for one-tile-per-grid-stride.
  int tiles_per_chunk;

  // Splits the partial-tile responsibility off the histogram kernel.
  //   - `false`: the histogram kernel walks all tiles (full + each segment's trailing partial);
  //     the finalize-histogram kernel does only the prefix-sum + bucket-finder epilogue.
  //   - `true`: the histogram kernel processes only full tiles; each segment's trailing partial
  //     is loaded + binned by the finalize-histogram kernel (one CTA per segment) into the global
  //     histogram, right before the prefix-sum + bucket-finder. Keeps the histogram kernel's inner
  //     loop a pure full-tile load.
  bool full_tiles_only_histogram;

  // Same idea as `full_tiles_only_histogram`, for the filter kernels. When `true`,
  // `agent_batched_topk_filter_partition::run()` skips its slow-path `dispatch_tile<false>` call;
  // each segment's trailing partial tile is processed by `device_segmented_topk_finalize_filter_kernel`
  // (one CTA per segment) via `process_partial_for_segment`, before the finalize prefix-sum +
  // bucket-finder. Each filter mode (early_stop / buffered / unbuffered) handles its own partial there.
  bool full_tiles_only_filter;

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const multi_worker_policy& lhs, const multi_worker_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block //
        && lhs.items_per_thread == rhs.items_per_thread //
        && lhs.bits_per_pass == rhs.bits_per_pass //
        && lhs.keys_tile_load_kind == rhs.keys_tile_load_kind //
        && lhs.scan_algorithm == rhs.scan_algorithm //
        && lhs.value_materialization == rhs.value_materialization //
        && lhs.lazy_value_load == rhs.lazy_value_load //
        && lhs.inlined_classify == rhs.inlined_classify //
        && lhs.tiles_per_chunk == rhs.tiles_per_chunk //
        && lhs.full_tiles_only_histogram == rhs.full_tiles_only_histogram //
        && lhs.full_tiles_only_filter == rhs.full_tiles_only_filter;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const multi_worker_policy& lhs, const multi_worker_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const multi_worker_policy& p)
  {
    return os
        << "multi_worker_policy { .threads_per_block = " << p.threads_per_block
        << ", .items_per_thread = " << p.items_per_thread //
        << ", .bits_per_pass = " << p.bits_per_pass //
        << ", .keys_tile_load_kind = " << static_cast<int>(p.keys_tile_load_kind) //
        << ", .scan_algorithm = " << p.scan_algorithm //
        << ", .value_materialization = " << static_cast<int>(p.value_materialization) //
        << ", .lazy_value_load = " << (p.lazy_value_load ? "true" : "false") //
        << ", .inlined_classify = " << (p.inlined_classify ? "true" : "false") //
        << ", .tiles_per_chunk = " << p.tiles_per_chunk //
        << ", .full_tiles_only_histogram = " << (p.full_tiles_only_histogram ? "true" : "false") //
        << ", .full_tiles_only_filter = " << (p.full_tiles_only_filter ? "true" : "false") //
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
    const int multi_items_per_thread =
      (cc >= ::cuda::compute_capability{9, 0})
        ? ::cuda::std::max(1, nominal_4b_items_per_thread * 4 / key_size)
        : ::cuda::std::clamp(nominal_4b_items_per_thread * 4 / key_size, 1, nominal_4b_items_per_thread);
    const int multi_bits_per_pass = detail::topk::calc_bits_per_pass(key_size);

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
        /*.value_materialization                =*/value_materialization_mode::indexed,
        /*.lazy_value_load                      =*/true,
        /*.inlined_classify                     =*/true,
        /*.tiles_per_chunk                      =*/8,
        /*.full_tiles_only_histogram            =*/true,
        /*.full_tiles_only_filter               =*/true}};
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
