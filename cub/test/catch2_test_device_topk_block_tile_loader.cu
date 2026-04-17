// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Unit tests for the top-k-private BlockTileLoader and grid_stride_queue. Exercises
//! the default grid_stride_queue / sync_load_strategy combo -- the only combo shipped
//! today; atomic-counter / UGETNEXTWORKID / segmented queues and async TMA loading are
//! deferred plan Q9/Q10 follow-ups and will reuse this test file as they come online.

#include <cub/agent/agent_topk.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <c2h/catch2_test_helper.h>

namespace topk = cub::detail::topk;

using policy_t = topk::AgentTopKPolicy<128, 4, 8, cub::BLOCK_LOAD_DIRECT, cub::BLOCK_SCAN_WARP_SCANS>;

// Minimal processor: records each item into a histogram indexed by the item's value.
// Also tracks whether on_segment_change was called (exercises the SFINAE hook detection).
struct recording_processor
{
  static constexpr int items_per_thread = policy_t::items_per_thread;

  unsigned int* d_value_histogram;
  int* d_segment_change_count;

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile(
    const int (&items)[items_per_thread], unsigned int /*thread_offset*/, int num_thread_items)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      if (j < num_thread_items)
      {
        atomicAdd(d_value_histogram + items[j], 1u);
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void on_segment_change(int /*segment_id*/)
  {
    if (threadIdx.x == 0)
    {
      atomicAdd(d_segment_change_count, 1);
    }
  }
};

__global__ void consume_kernel(const int* d_keys_in,
                               unsigned int num_items,
                               unsigned int* d_value_histogram,
                               int* d_segment_change_count)
{
  using loader_t = topk::BlockTileLoader<policy_t, int, unsigned int>;

  __shared__ typename loader_t::_TempStorage loader_storage;
  loader_t loader(loader_storage, num_items);

  recording_processor proc{d_value_histogram, d_segment_change_count};
  loader.consume(proc, d_keys_in, num_items);
}

C2H_TEST("BlockTileLoader processes every item exactly once via grid_stride_queue",
         "[block][topk]")
{
  const auto num_items_sel = GENERATE(
    // Exactly 1 tile
    policy_t::block_threads * policy_t::items_per_thread,
    // Multiple full tiles
    policy_t::block_threads * policy_t::items_per_thread * 5,
    // Partial last tile
    policy_t::block_threads * policy_t::items_per_thread * 3 + 17,
    // Tiny input (< 1 tile)
    123);

  const unsigned int num_items = static_cast<unsigned int>(num_items_sel);

  // Value range is the histogram size; use [0, max_val) so keys fit in the histogram.
  constexpr unsigned int max_val = 1024;
  std::vector<int> h_keys(num_items);
  for (unsigned int i = 0; i < num_items; ++i)
  {
    h_keys[i] = static_cast<int>(i % max_val);
  }

  thrust::device_vector<int> d_keys(h_keys.begin(), h_keys.end());
  thrust::device_vector<unsigned int> d_histogram(max_val, 0);
  thrust::device_vector<int> d_segment_change_count(1, 0);

  const auto grid_size = GENERATE(1u, 4u, 32u);

  consume_kernel<<<grid_size, policy_t::block_threads>>>(
    thrust::raw_pointer_cast(d_keys.data()),
    num_items,
    thrust::raw_pointer_cast(d_histogram.data()),
    thrust::raw_pointer_cast(d_segment_change_count.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Build expected histogram on host.
  std::vector<unsigned int> expected_histogram(max_val, 0);
  for (unsigned int i = 0; i < num_items; ++i)
  {
    ++expected_histogram[i % max_val];
  }

  thrust::host_vector<unsigned int> got_histogram = d_histogram;
  REQUIRE(std::equal(got_histogram.begin(), got_histogram.end(), expected_histogram.begin()));

  // grid_stride_queue has crosses_segment_boundary() == false, so the hook must never fire.
  REQUIRE(d_segment_change_count[0] == 0);
}

// Queue that reports a segment boundary on the very first tile -- exercises the
// on_segment_change hook detection without needing a full segmented queue impl.
struct single_boundary_queue
{
  static constexpr unsigned int sentinel = ::cuda::std::numeric_limits<unsigned int>::max();

  unsigned int next_block_;
  unsigned int grid_size_;
  unsigned int total_blocks_;
  bool first_tile_ = true; // report the boundary exactly once, before the first tile

  _CCCL_DEVICE _CCCL_FORCEINLINE single_boundary_queue(unsigned int total_blocks)
      : next_block_(static_cast<unsigned int>(blockIdx.x))
      , grid_size_(static_cast<unsigned int>(gridDim.x))
      , total_blocks_(total_blocks)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int next_tile_id()
  {
    const unsigned int ret = next_block_;
    if (ret >= total_blocks_)
    {
      return sentinel;
    }
    next_block_ += grid_size_;
    return ret;
  }

  // Returns true only for the first tile; the loader calls this once per tile to decide
  // whether to invoke the optional on_segment_change hook.
  _CCCL_DEVICE _CCCL_FORCEINLINE bool crosses_segment_boundary()
  {
    const bool result = first_tile_;
    first_tile_       = false;
    return result;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE int current_segment() const
  {
    return 42;
  }
};

__global__ void consume_segmented_kernel(const int* d_keys_in,
                                         unsigned int num_items,
                                         unsigned int* d_value_histogram,
                                         int* d_segment_change_count)
{
  using loader_t =
    topk::BlockTileLoader<policy_t, int, unsigned int, single_boundary_queue, topk::sync_load_strategy>;

  __shared__ typename loader_t::_TempStorage loader_storage;
  const unsigned int total_blocks = ::cuda::ceil_div(num_items, static_cast<unsigned int>(loader_t::tile_items));
  loader_t loader(loader_storage, single_boundary_queue{total_blocks});

  recording_processor proc{d_value_histogram, d_segment_change_count};
  loader.consume(proc, d_keys_in, num_items);
}

C2H_TEST("BlockTileLoader invokes on_segment_change when the queue reports a boundary",
         "[block][topk]")
{
  const unsigned int num_items =
    policy_t::block_threads * policy_t::items_per_thread * 3; // 3 full tiles

  std::vector<int> h_keys(num_items);
  for (unsigned int i = 0; i < num_items; ++i)
  {
    h_keys[i] = 0;
  }
  thrust::device_vector<int> d_keys(h_keys.begin(), h_keys.end());
  thrust::device_vector<unsigned int> d_histogram(1, 0);
  thrust::device_vector<int> d_segment_change_count(1, 0);

  consume_segmented_kernel<<<1, policy_t::block_threads>>>(
    thrust::raw_pointer_cast(d_keys.data()),
    num_items,
    thrust::raw_pointer_cast(d_histogram.data()),
    thrust::raw_pointer_cast(d_segment_change_count.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // All items processed exactly once.
  REQUIRE(d_histogram[0] == num_items);
  // Hook fired exactly once (on the first tile; no boundary on subsequent tiles).
  REQUIRE(d_segment_change_count[0] == 1);
}
