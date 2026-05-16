// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Unit tests for the top-k-private `block_partition_accumulating_candidates` primitive
//! in `cub/detail/topk/block_partition_accumulating.cuh`. Drives multiple tiles
//! through a single block, exercising:
//!   - cross-tile smem accumulation when the per-tile reservation count stays
//!     below the buffer capacity (no in-tile flush; the terminal `epilogue()`
//!     drains whatever's in the buffer).
//!   - in-tile multi-round overflow when the cumulative reservation count
//!     exceeds capacity (the cooperative flush + position-renumbering path).
//!   - both `LazyValueLoad` modes (forced off for keys-only).
//!   - keys-only and paired keys+values.
//!   - one config with a trailing partial tile.
//!
//! The early-stop/selected-buffering counterpart lives in
//! `catch2_test_device_topk_block_filter_accumulating.cu` -- it tests
//! `block_filter_accumulating`, the dedicated single-stream filter primitive.
//!
//! Test scope is bounded to the `atomic_reserve_range_op` reserve op (matching
//! the agent's `buffered`-mode pass). The `back_grow_capped_reserve_op`
//! cap-clamp interaction is intentionally not exercised in this prototype test
//! (the agent's last-filter pass continues to use plain `BlockPartition`).

#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/block_partition_accumulating.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <algorithm>
#include <vector>

#include <c2h/catch2_test_helper.h>

namespace topk = cub::detail::topk;

//---------------------------------------------------------------------
// Test scaffolding
//---------------------------------------------------------------------

// Driver classifier: returns the per-item class read out of a global `classes[]`
// array, indexed by the key value (which we set up to equal the item's global
// position into the flat input buffer).
struct driver_identify_op
{
  const ::cuda::std::int8_t* d_classes;

  _CCCL_DEVICE _CCCL_FORCEINLINE topk::candidate_class operator()(int key) const
  {
    return static_cast<topk::candidate_class>(d_classes[key]);
  }
};

// Counts every candidate-classified key into a global `callback_count`. For the
// Selected variant this should never fire (HasCandidates == false); for the
// Candidates variant it should equal the eligible candidate count across all tiles.
struct counting_callback_op
{
  unsigned int* d_count;

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(int /*key*/) const
  {
    atomicAdd(d_count, 1u);
  }
};

//---------------------------------------------------------------------
// Kernel: drives `num_tiles` tiles through one block of
// `block_partition_accumulating_candidates`, then calls `epilogue()`.
//---------------------------------------------------------------------

template <int BlockThreads, int ItemsPerThread, int BufferCapacity, bool KeysOnly, bool LazyValueLoad>
__global__ void acc_partition_kernel(
  const int* d_keys_in,
  const int* d_values_in,
  const ::cuda::std::int8_t* d_classes_in,
  int num_tiles,
  int last_tile_items, // = tile_items if the last tile is full
  int* d_sel_keys,
  int* d_cand_keys,
  int* d_sel_vals,
  int* d_cand_vals,
  unsigned int* d_sel_counter,
  unsigned int* d_cand_counter,
  unsigned int* d_callback_count)
{
  static constexpr int tile_items = BlockThreads * ItemsPerThread;

  using value_ds_t = topk::direct_data_source<const int*, BlockThreads, ItemsPerThread>;

  using value_sinks_t = topk::value_channel_sinks_t<int*, int*, ::cuda::std::identity, ::cuda::std::identity>;
  using value_channel_sinks_tuple_t =
    ::cuda::std::conditional_t<KeysOnly, ::cuda::std::tuple<>, ::cuda::std::tuple<value_sinks_t>>;
  using value_types_tuple_t =
    ::cuda::std::conditional_t<KeysOnly, ::cuda::std::tuple<>, ::cuda::std::tuple<int>>;
  using value_sources_tuple_t =
    ::cuda::std::conditional_t<KeysOnly, ::cuda::std::tuple<>, ::cuda::std::tuple<value_ds_t>>;

  using sel_reserve_op_t  = topk::atomic_reserve_range_op<unsigned int>;
  using cand_reserve_op_t = topk::atomic_reserve_range_op<unsigned int>;
  using xform_t           = ::cuda::std::identity;

  using partition_t = topk::block_partition_accumulating_candidates<
    BlockThreads,
    ItemsPerThread,
    BufferCapacity,
    int,
    unsigned int,
    unsigned int,
    sel_reserve_op_t,
    cand_reserve_op_t,
    xform_t,
    xform_t,
    int*,
    int*,
    driver_identify_op,
    counting_callback_op,
    value_channel_sinks_tuple_t,
    value_types_tuple_t,
    LazyValueLoad>;

  // `partition_t::TempStorage` is an `Uninitialized<>` wrapper internally, so a
  // bare `__shared__` declaration is legal -- no manual wrapping needed.
  __shared__ typename partition_t::TempStorage partition_ts;
  __shared__ typename partition_t::ScratchStorage scratch;

  // Build the sinks tuple (captured by ctor).
  auto make_sinks = [&] {
    if constexpr (KeysOnly)
    {
      return ::cuda::std::tuple<>{};
    }
    else
    {
      return ::cuda::std::tuple<value_sinks_t>{
        value_sinks_t{d_sel_vals, d_cand_vals, ::cuda::std::identity{}, ::cuda::std::identity{}}};
    }
  };
  value_channel_sinks_tuple_t sinks = make_sinks();

  sel_reserve_op_t reserve_sel{d_sel_counter};
  cand_reserve_op_t reserve_cand{d_cand_counter};
  xform_t key_transform{};

  driver_identify_op identify_op{d_classes_in};
  counting_callback_op callback_op{d_callback_count};

  partition_t partition{
    partition_ts,
    reserve_sel,
    reserve_cand,
    key_transform,
    key_transform,
    d_sel_keys,
    d_cand_keys,
    sinks,
    identify_op,
    callback_op};

  // --- per-tile loop --------------------------------------------------
  for (int tile_id = 0; tile_id < num_tiles; ++tile_id)
  {
    const bool is_last_tile = (tile_id == num_tiles - 1);
    const int items_in_tile = is_last_tile ? last_tile_items : tile_items;
    const int tile_base     = tile_id * tile_items;

    // Load keys for the tile.
    int keys[ItemsPerThread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const int local_idx  = static_cast<int>(threadIdx.x) * ItemsPerThread + j;
      const int global_idx = tile_base + local_idx;
      keys[j]              = (local_idx < items_in_tile) ? d_keys_in[global_idx] : 0;
    }

    // Build the per-call sources tuple (set_tile_base for the current tile).
    typename value_ds_t::TempStorage val_state{};
    value_ds_t val_ds{d_values_in, val_state};
    val_ds.set_tile_base(tile_base);
    auto make_sources = [&] {
      if constexpr (KeysOnly)
      {
        return ::cuda::std::tuple<>{};
      }
      else
      {
        return ::cuda::std::tuple<value_ds_t>{val_ds};
      }
    };
    value_sources_tuple_t sources = make_sources();

    __syncthreads();

    if (items_in_tile == tile_items)
    {
      partition.partition(scratch, keys, sources);
    }
    else
    {
      partition.partition(scratch, keys, items_in_tile, sources);
    }
  }

  // Terminal flush: drain whatever's left in the smem buffer.
  partition.epilogue();
}

//---------------------------------------------------------------------
// Host-side helpers
//---------------------------------------------------------------------

// Build a deterministic class pattern over `num_items` slots (same recipe as the
// BlockPartition test): every selected_every-th item is selected, every
// candidate_every-th item (that isn't selected) is candidate, everything else is
// rejected.
static std::vector<::cuda::std::int8_t> make_classes(int num_items, int selected_every, int candidate_every)
{
  std::vector<::cuda::std::int8_t> out(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    if (i % selected_every == 0)
    {
      out[i] = static_cast<::cuda::std::int8_t>(topk::candidate_class::selected);
    }
    else if (i % candidate_every == 0)
    {
      out[i] = static_cast<::cuda::std::int8_t>(topk::candidate_class::candidate);
    }
    else
    {
      out[i] = static_cast<::cuda::std::int8_t>(topk::candidate_class::rejected);
    }
  }
  return out;
}

// Run one configuration of the accumulating test.
template <int BlockThreads, int ItemsPerThread, int BufferCapacity, bool KeysOnly, bool LazyValueLoad>
void run_accumulating_test(int num_tiles, int last_tile_items, int selected_every, int candidate_every)
{
  static constexpr int tile_items = BlockThreads * ItemsPerThread;

  REQUIRE(last_tile_items > 0);
  REQUIRE(last_tile_items <= tile_items);
  REQUIRE(num_tiles >= 1);

  const int total_items = (num_tiles - 1) * tile_items + last_tile_items;

  // Construct the input vectors. keys[i] = i (so the driver_identify_op can recover
  // the per-item class from the key value); values[i] = i + 1000.
  std::vector<int> keys(total_items);
  std::vector<int> values(total_items);
  for (int i = 0; i < total_items; ++i)
  {
    keys[i]   = i;
    values[i] = i + 1000;
  }
  auto classes = make_classes(total_items, selected_every, candidate_every);

  // Host-side golden expectations.
  std::vector<int> expected_selected_keys;
  std::vector<int> expected_selected_vals;
  std::vector<int> expected_candidate_keys;
  std::vector<int> expected_candidate_vals;
  unsigned int expected_callback_count = 0;
  for (int i = 0; i < total_items; ++i)
  {
    const auto c = static_cast<topk::candidate_class>(classes[i]);
    if (c == topk::candidate_class::selected)
    {
      expected_selected_keys.push_back(keys[i]);
      expected_selected_vals.push_back(values[i]);
    }
    else if (c == topk::candidate_class::candidate)
    {
      expected_candidate_keys.push_back(keys[i]);
      expected_candidate_vals.push_back(values[i]);
      ++expected_callback_count;
    }
  }

  thrust::device_vector<int> d_keys_in(keys.begin(), keys.end());
  thrust::device_vector<int> d_values_in(values.begin(), values.end());
  thrust::device_vector<::cuda::std::int8_t> d_classes_in(classes.begin(), classes.end());

  const int out_capacity = total_items + 1;
  thrust::device_vector<int> d_sel_keys(out_capacity, 0);
  thrust::device_vector<int> d_cand_keys(out_capacity, 0);
  thrust::device_vector<int> d_sel_vals(out_capacity, 0);
  thrust::device_vector<int> d_cand_vals(out_capacity, 0);
  thrust::device_vector<unsigned int> d_sel_cnt(1, 0);
  thrust::device_vector<unsigned int> d_cand_cnt(1, 0);
  thrust::device_vector<unsigned int> d_callback_cnt(1, 0);

  acc_partition_kernel<BlockThreads, ItemsPerThread, BufferCapacity, KeysOnly, LazyValueLoad><<<1, BlockThreads>>>(
    thrust::raw_pointer_cast(d_keys_in.data()),
    thrust::raw_pointer_cast(d_values_in.data()),
    thrust::raw_pointer_cast(d_classes_in.data()),
    num_tiles,
    last_tile_items,
    thrust::raw_pointer_cast(d_sel_keys.data()),
    thrust::raw_pointer_cast(d_cand_keys.data()),
    thrust::raw_pointer_cast(d_sel_vals.data()),
    thrust::raw_pointer_cast(d_cand_vals.data()),
    thrust::raw_pointer_cast(d_sel_cnt.data()),
    thrust::raw_pointer_cast(d_cand_cnt.data()),
    thrust::raw_pointer_cast(d_callback_cnt.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Counter assertions.
  REQUIRE(d_sel_cnt[0] == expected_selected_keys.size());
  REQUIRE(d_cand_cnt[0] == expected_candidate_keys.size());
  REQUIRE(d_callback_cnt[0] == expected_callback_count);

  // Output-set assertions (writes are order-independent across atomic reservations).
  std::vector<int> got_sel_keys(d_sel_keys.begin(), d_sel_keys.begin() + d_sel_cnt[0]);
  std::sort(got_sel_keys.begin(), got_sel_keys.end());
  std::sort(expected_selected_keys.begin(), expected_selected_keys.end());
  REQUIRE(got_sel_keys == expected_selected_keys);

  std::vector<int> got_cand_keys(d_cand_keys.begin(), d_cand_keys.begin() + d_cand_cnt[0]);
  std::sort(got_cand_keys.begin(), got_cand_keys.end());
  std::sort(expected_candidate_keys.begin(), expected_candidate_keys.end());
  REQUIRE(got_cand_keys == expected_candidate_keys);

  if constexpr (!KeysOnly)
  {
    std::vector<int> got_sel_vals(d_sel_vals.begin(), d_sel_vals.begin() + d_sel_cnt[0]);
    std::sort(got_sel_vals.begin(), got_sel_vals.end());
    std::sort(expected_selected_vals.begin(), expected_selected_vals.end());
    REQUIRE(got_sel_vals == expected_selected_vals);

    std::vector<int> got_cand_vals(d_cand_vals.begin(), d_cand_vals.begin() + d_cand_cnt[0]);
    std::sort(got_cand_vals.begin(), got_cand_vals.end());
    std::sort(expected_candidate_vals.begin(), expected_candidate_vals.end());
    REQUIRE(got_cand_vals == expected_candidate_vals);
  }
}

//---------------------------------------------------------------------
// Test cases
//---------------------------------------------------------------------

C2H_TEST("block_partition_accumulating_candidates accumulates across tiles below capacity", "[block][topk][accumulating]")
{
  // Tile size = 64 * 4 = 256. Buffer capacity = 256 (one full tile worth). Per-tile
  // candidate count is roughly tile_items / candidate_every ~= 256/13 ~= 19; with 4
  // tiles we get ~80 candidates total -- well below the 256-slot capacity. The
  // terminal `epilogue()` is what flushes them.
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 256;
  constexpr int NumTiles       = 4;
  constexpr int LastTileItems  = BlockThreads * ItemsPerThread; // full last tile
  constexpr int SelectedEvery  = 7;
  constexpr int CandidateEvery = 13;

  run_accumulating_test<BlockThreads, ItemsPerThread, BufferCapacity, /*KeysOnly=*/false, /*LazyValueLoad=*/false>(
    NumTiles, LastTileItems, SelectedEvery, CandidateEvery);
}

C2H_TEST("block_partition_accumulating_candidates triggers in-tile overflow loop", "[block][topk][accumulating]")
{
  // Buffer capacity = 8 (much smaller than per-tile candidate count). Per-tile
  // candidate count ~= 256/3 ~= 85 with candidate_every = 3 (and selected_every very
  // sparse). Each tile triggers many flush rounds.
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 8;
  constexpr int NumTiles       = 3;
  constexpr int LastTileItems  = BlockThreads * ItemsPerThread;
  constexpr int SelectedEvery  = 17;
  constexpr int CandidateEvery = 3;

  run_accumulating_test<BlockThreads, ItemsPerThread, BufferCapacity, /*KeysOnly=*/false, /*LazyValueLoad=*/false>(
    NumTiles, LastTileItems, SelectedEvery, CandidateEvery);
}

C2H_TEST("block_partition_accumulating_candidates keys-only", "[block][topk][accumulating]")
{
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 64;
  constexpr int NumTiles       = 3;
  constexpr int LastTileItems  = BlockThreads * ItemsPerThread;

  run_accumulating_test<BlockThreads, ItemsPerThread, BufferCapacity, /*KeysOnly=*/true, /*LazyValueLoad=*/false>(
    NumTiles, LastTileItems, /*selected_every=*/5, /*candidate_every=*/7);
}

C2H_TEST("block_partition_accumulating_candidates lazy value load", "[block][topk][accumulating]")
{
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 32;
  constexpr int NumTiles       = 3;
  constexpr int LastTileItems  = BlockThreads * ItemsPerThread;

  run_accumulating_test<BlockThreads, ItemsPerThread, BufferCapacity, /*KeysOnly=*/false, /*LazyValueLoad=*/true>(
    NumTiles, LastTileItems, /*selected_every=*/5, /*candidate_every=*/7);
}

C2H_TEST("block_partition_accumulating_candidates partial trailing tile", "[block][topk][accumulating]")
{
  // Last tile has 173 valid items out of tile_items (256). The classify loop must
  // force OOB items to `rejected` and the per-item bound check must hold.
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 64;
  constexpr int NumTiles       = 3;
  constexpr int LastTileItems  = 173;

  run_accumulating_test<BlockThreads, ItemsPerThread, BufferCapacity, /*KeysOnly=*/false, /*LazyValueLoad=*/false>(
    NumTiles, LastTileItems, /*selected_every=*/5, /*candidate_every=*/7);
}
