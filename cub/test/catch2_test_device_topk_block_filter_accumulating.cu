// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Unit tests for the top-k-private `block_filter_accumulating` primitive in
//! `cub/detail/topk/block_filter_accumulating.cuh`. Drives multiple tiles
//! through a single block, exercising:
//!   - cross-tile smem accumulation when the per-tile reservation count stays
//!     below the buffer capacity (no in-tile flush; the terminal `epilogue()`
//!     drains whatever's in the buffer).
//!   - in-tile multi-round overflow when the cumulative reservation count
//!     exceeds capacity (the cooperative flush + position-renumbering path).
//!   - both `LazyValueLoad` modes (forced off for keys-only).
//!   - keys-only and paired keys+values.
//!   - one config with a trailing partial tile.

#include <cub/detail/topk/block_filter.cuh>
#include <cub/detail/topk/block_filter_accumulating.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/utility>

#include <algorithm>
#include <vector>

#include <c2h/catch2_test_helper.h>

namespace topk = cub::detail::topk;

//---------------------------------------------------------------------
// Test scaffolding
//---------------------------------------------------------------------

// Driver predicate: returns `true` if the per-item "selected" bit is set in a
// global table indexed by the key value (the test fills keys with their global
// index so the predicate can recover the index).
struct driver_identify_selected_op
{
  const ::cuda::std::uint8_t* d_keep;

  _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(int key) const
  {
    return d_keep[key] != 0;
  }
};

//---------------------------------------------------------------------
// Kernel: drives `num_tiles` tiles through one block of
// `block_filter_accumulating`, then calls `epilogue()`.
//---------------------------------------------------------------------

template <int BlockThreads, int ItemsPerThread, int BufferCapacity, bool KeysOnly, bool LazyValueLoad>
__global__ void acc_filter_kernel(
  const int* d_keys_in,
  const int* d_values_in,
  const ::cuda::std::uint8_t* d_keep_in,
  int num_tiles,
  int last_tile_items, // = tile_items if the last tile is full
  int* d_sel_keys,
  int* d_sel_vals,
  unsigned int* d_sel_counter)
{
  static constexpr int tile_items = BlockThreads * ItemsPerThread;

  using value_ds_t = topk::direct_data_source<const int*, BlockThreads, ItemsPerThread>;

  using value_sinks_t = topk::value_channel_sinks_filter_t<int*>;
  using value_channel_sinks_or_null_t =
    ::cuda::std::conditional_t<KeysOnly, cub::NullType, value_sinks_t>;
  using value_t_t =
    ::cuda::std::conditional_t<KeysOnly, cub::NullType, int>;

  using sel_reserve_op_t = topk::atomic_reserve_range_op<unsigned int>;
  using xform_t          = ::cuda::std::identity;

  using filter_t = topk::block_filter_accumulating<
    BlockThreads,
    ItemsPerThread,
    BufferCapacity,
    int,
    unsigned int,
    sel_reserve_op_t,
    xform_t,
    int*,
    driver_identify_selected_op,
    value_channel_sinks_or_null_t,
    value_t_t,
    LazyValueLoad>;

  // `filter_t::TempStorage` is an `Uninitialized<>` wrapper internally, so a
  // bare `__shared__` declaration is legal -- no manual wrapping needed.
  __shared__ typename filter_t::TempStorage filter_ts;
  __shared__ typename filter_t::ScratchStorage scratch;

  // Build the sinks (captured by ctor).
  auto make_sinks = [&] {
    if constexpr (KeysOnly)
    {
      return cub::NullType{};
    }
    else
    {
      return value_sinks_t{d_sel_vals};
    }
  };
  auto sinks = make_sinks();

  sel_reserve_op_t reserve_sel{d_sel_counter};
  xform_t key_transform{};
  driver_identify_selected_op identify_op{d_keep_in};

  filter_t filter{filter_ts, reserve_sel, key_transform, d_sel_keys, sinks, identify_op};

  // --- per-tile loop --------------------------------------------------
  for (int tile_id = 0; tile_id < num_tiles; ++tile_id)
  {
    const bool is_last_tile = (tile_id == num_tiles - 1);
    const int items_in_tile = is_last_tile ? last_tile_items : tile_items;
    const int tile_base     = tile_id * tile_items;

    int keys[ItemsPerThread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const int local_idx  = static_cast<int>(threadIdx.x) * ItemsPerThread + j;
      const int global_idx = tile_base + local_idx;
      keys[j]              = (local_idx < items_in_tile) ? d_keys_in[global_idx] : 0;
    }

    typename value_ds_t::TempStorage val_state{};
    value_ds_t val_ds{d_values_in, val_state};
    val_ds.set_tile_base(tile_base);
    auto make_source = [&] {
      if constexpr (KeysOnly)
      {
        return cub::NullType{};
      }
      else
      {
        return val_ds;
      }
    };
    auto value_source = make_source();

    __syncthreads();

    if (items_in_tile == tile_items)
    {
      filter.partition(scratch, keys, value_source);
    }
    else
    {
      filter.partition(scratch, keys, items_in_tile, value_source);
    }
  }

  // Terminal flush: drain whatever's left in the smem buffer.
  filter.epilogue();
}

//---------------------------------------------------------------------
// Host-side helpers
//---------------------------------------------------------------------

static std::vector<::cuda::std::uint8_t> make_keep(int num_items, int keep_every)
{
  std::vector<::cuda::std::uint8_t> out(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    out[i] = (i % keep_every == 0) ? ::cuda::std::uint8_t{1} : ::cuda::std::uint8_t{0};
  }
  return out;
}

template <int BlockThreads, int ItemsPerThread, int BufferCapacity, bool KeysOnly, bool LazyValueLoad>
void run_accumulating_filter_test(int num_tiles, int last_tile_items, int keep_every)
{
  static constexpr int tile_items = BlockThreads * ItemsPerThread;

  REQUIRE(last_tile_items > 0);
  REQUIRE(last_tile_items <= tile_items);
  REQUIRE(num_tiles >= 1);

  const int total_items = (num_tiles - 1) * tile_items + last_tile_items;

  std::vector<int> keys(total_items);
  std::vector<int> values(total_items);
  for (int i = 0; i < total_items; ++i)
  {
    keys[i]   = i;
    values[i] = i + 1000;
  }
  auto keep = make_keep(total_items, keep_every);

  std::vector<int> expected_selected_keys;
  std::vector<int> expected_selected_vals;
  for (int i = 0; i < total_items; ++i)
  {
    if (keep[i])
    {
      expected_selected_keys.push_back(keys[i]);
      expected_selected_vals.push_back(values[i]);
    }
  }

  thrust::device_vector<int> d_keys_in(keys.begin(), keys.end());
  thrust::device_vector<int> d_values_in(values.begin(), values.end());
  thrust::device_vector<::cuda::std::uint8_t> d_keep_in(keep.begin(), keep.end());

  const int out_capacity = total_items + 1;
  thrust::device_vector<int> d_sel_keys(out_capacity, 0);
  thrust::device_vector<int> d_sel_vals(out_capacity, 0);
  thrust::device_vector<unsigned int> d_sel_cnt(1, 0);

  acc_filter_kernel<BlockThreads, ItemsPerThread, BufferCapacity, KeysOnly, LazyValueLoad><<<1, BlockThreads>>>(
    thrust::raw_pointer_cast(d_keys_in.data()),
    thrust::raw_pointer_cast(d_values_in.data()),
    thrust::raw_pointer_cast(d_keep_in.data()),
    num_tiles,
    last_tile_items,
    thrust::raw_pointer_cast(d_sel_keys.data()),
    thrust::raw_pointer_cast(d_sel_vals.data()),
    thrust::raw_pointer_cast(d_sel_cnt.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(d_sel_cnt[0] == expected_selected_keys.size());

  std::vector<int> got_sel_keys(d_sel_keys.begin(), d_sel_keys.begin() + d_sel_cnt[0]);
  std::sort(got_sel_keys.begin(), got_sel_keys.end());
  std::sort(expected_selected_keys.begin(), expected_selected_keys.end());
  REQUIRE(got_sel_keys == expected_selected_keys);

  if constexpr (!KeysOnly)
  {
    std::vector<int> got_sel_vals(d_sel_vals.begin(), d_sel_vals.begin() + d_sel_cnt[0]);
    std::sort(got_sel_vals.begin(), got_sel_vals.end());
    std::sort(expected_selected_vals.begin(), expected_selected_vals.end());
    REQUIRE(got_sel_vals == expected_selected_vals);
  }
}

//---------------------------------------------------------------------
// Test cases
//---------------------------------------------------------------------

C2H_TEST("block_filter_accumulating accumulates across tiles below capacity", "[block][topk][accumulating]")
{
  // Tile size = 64 * 4 = 256. Buffer capacity = 256 (one full tile worth). Per-tile
  // kept count is roughly tile_items / keep_every ~= 256/13 ~= 19; with 4 tiles we
  // get ~80 selected total -- well below the 256-slot capacity. The terminal
  // `epilogue()` is what flushes them.
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 256;
  constexpr int NumTiles       = 4;
  constexpr int LastTileItems  = BlockThreads * ItemsPerThread;
  constexpr int KeepEvery      = 13;

  run_accumulating_filter_test<BlockThreads,
                               ItemsPerThread,
                               BufferCapacity,
                               /*KeysOnly=*/false,
                               /*LazyValueLoad=*/false>(NumTiles, LastTileItems, KeepEvery);
}

C2H_TEST("block_filter_accumulating triggers in-tile overflow loop", "[block][topk][accumulating]")
{
  // Buffer capacity = 8 (much smaller than per-tile kept count). Per-tile kept
  // count ~= 256/3 ~= 85. Each tile triggers many flush rounds.
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 8;
  constexpr int NumTiles       = 3;
  constexpr int LastTileItems  = BlockThreads * ItemsPerThread;
  constexpr int KeepEvery      = 3;

  run_accumulating_filter_test<BlockThreads,
                               ItemsPerThread,
                               BufferCapacity,
                               /*KeysOnly=*/false,
                               /*LazyValueLoad=*/false>(NumTiles, LastTileItems, KeepEvery);
}

C2H_TEST("block_filter_accumulating keys-only", "[block][topk][accumulating]")
{
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 64;
  constexpr int NumTiles       = 3;
  constexpr int LastTileItems  = BlockThreads * ItemsPerThread;

  run_accumulating_filter_test<BlockThreads,
                               ItemsPerThread,
                               BufferCapacity,
                               /*KeysOnly=*/true,
                               /*LazyValueLoad=*/false>(NumTiles, LastTileItems, /*keep_every=*/5);
}

C2H_TEST("block_filter_accumulating lazy value load", "[block][topk][accumulating]")
{
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 32;
  constexpr int NumTiles       = 3;
  constexpr int LastTileItems  = BlockThreads * ItemsPerThread;

  run_accumulating_filter_test<BlockThreads,
                               ItemsPerThread,
                               BufferCapacity,
                               /*KeysOnly=*/false,
                               /*LazyValueLoad=*/true>(NumTiles, LastTileItems, /*keep_every=*/5);
}

C2H_TEST("block_filter_accumulating partial trailing tile", "[block][topk][accumulating]")
{
  // Last tile has 173 valid items out of tile_items (256). The classify loop
  // must drop OOB items and the per-item bound check must hold.
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int BufferCapacity = 64;
  constexpr int NumTiles       = 3;
  constexpr int LastTileItems  = 173;

  run_accumulating_filter_test<BlockThreads,
                               ItemsPerThread,
                               BufferCapacity,
                               /*KeysOnly=*/false,
                               /*LazyValueLoad=*/false>(NumTiles, LastTileItems, /*keep_every=*/5);
}
