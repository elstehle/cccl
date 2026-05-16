// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Unit tests for the top-k-private `BlockPartitionSpeculative` primitive in
//! `cub/detail/topk/block_partition_speculative.cuh`. Drives multiple tiles
//! through a single block, exercising:
//!   - the `SelectedBufferCapacity > 0` path: both streams accumulate
//!     speculatively; both cooperative flushes (and overflow drains) are
//!     exercised.
//!   - the `SelectedBufferCapacity == 0` short-circuit: the selected stream
//!     degrades to pure-Atomics behaviour; only the candidate-stream smem
//!     buffer is in play.
//!   - cross-tile smem accumulation when the per-tile reservation counts
//!     stay below the respective capacities (terminal `epilogue()` flushes).
//!   - in-tile overflow on either stream (per-item global-atomic drain +
//!     single cooperative flush).
//!   - both `LazyValueLoad` modes (forced off for keys-only).
//!   - keys-only and paired keys+values.
//!   - one config with a trailing partial tile.

#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/block_partition_speculative.cuh>
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

struct driver_identify_op
{
  const ::cuda::std::int8_t* d_classes;

  _CCCL_DEVICE _CCCL_FORCEINLINE topk::candidate_class operator()(int key) const
  {
    return static_cast<topk::candidate_class>(d_classes[key]);
  }
};

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
// `BlockPartitionSpeculative`, then calls `epilogue()`.
//---------------------------------------------------------------------

template <int BlockThreads,
          int ItemsPerThread,
          int CandidateBufferCapacity,
          int SelectedBufferCapacity,
          bool KeysOnly,
          bool LazyValueLoad>
__global__ void spec_partition_kernel(
  const int* d_keys_in,
  const int* d_values_in,
  const ::cuda::std::int8_t* d_classes_in,
  int num_tiles,
  int last_tile_items,
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

  using partition_t = topk::BlockPartitionSpeculative<
    BlockThreads,
    ItemsPerThread,
    CandidateBufferCapacity,
    SelectedBufferCapacity,
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

  __shared__ typename partition_t::TempStorage partition_ts;
  __shared__ typename partition_t::ScratchStorage scratch;

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
      partition.Partition(scratch, keys, sources);
    }
    else
    {
      partition.Partition(scratch, keys, items_in_tile, sources);
    }
  }

  partition.epilogue();
}

//---------------------------------------------------------------------
// Host-side helpers
//---------------------------------------------------------------------

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

template <int BlockThreads,
          int ItemsPerThread,
          int CandidateBufferCapacity,
          int SelectedBufferCapacity,
          bool KeysOnly,
          bool LazyValueLoad>
void run_speculative_partition_test(int num_tiles, int last_tile_items, int selected_every, int candidate_every)
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
  auto classes = make_classes(total_items, selected_every, candidate_every);

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

  spec_partition_kernel<BlockThreads, ItemsPerThread, CandidateBufferCapacity, SelectedBufferCapacity, KeysOnly, LazyValueLoad>
    <<<1, BlockThreads>>>(
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

  REQUIRE(d_sel_cnt[0] == expected_selected_keys.size());
  REQUIRE(d_cand_cnt[0] == expected_candidate_keys.size());
  REQUIRE(d_callback_cnt[0] == expected_callback_count);

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

C2H_TEST("BlockPartitionSpeculative both streams accumulate across tiles below capacity", "[block][topk][speculative]")
{
  // Tile = 64*4 = 256. Cand cap = 256, sel cap = 128. Per-tile cand count ~=
  // 256/13 = 19; per-tile sel count ~= 256/7 = 36. Over 4 tiles: ~80 cands,
  // ~145 sels -> the selected buffer fills (sel total > 128), exercising one
  // cooperative selected flush + the residual epilogue() drain. Candidate
  // buffer stays below capacity -> terminal flush only.
  constexpr int BlockThreads      = 64;
  constexpr int ItemsPerThread    = 4;
  constexpr int CandidateCapacity = 256;
  constexpr int SelectedCapacity  = 128;
  constexpr int NumTiles          = 4;
  constexpr int LastTileItems     = BlockThreads * ItemsPerThread;
  constexpr int SelectedEvery     = 7;
  constexpr int CandidateEvery    = 13;

  run_speculative_partition_test<BlockThreads,
                                 ItemsPerThread,
                                 CandidateCapacity,
                                 SelectedCapacity,
                                 /*KeysOnly=*/false,
                                 /*LazyValueLoad=*/false>(NumTiles, LastTileItems, SelectedEvery, CandidateEvery);
}

C2H_TEST("BlockPartitionSpeculative triggers in-tile candidate overflow drain", "[block][topk][speculative]")
{
  // Cand cap = 8, but per-tile cand count ~= 256/3 ~= 85. Most candidates drain
  // via per-item global atomics. Sel cap = 256, sel sparse.
  constexpr int BlockThreads      = 64;
  constexpr int ItemsPerThread    = 4;
  constexpr int CandidateCapacity = 8;
  constexpr int SelectedCapacity  = 256;
  constexpr int NumTiles          = 3;
  constexpr int LastTileItems     = BlockThreads * ItemsPerThread;
  constexpr int SelectedEvery     = 17;
  constexpr int CandidateEvery    = 3;

  run_speculative_partition_test<BlockThreads,
                                 ItemsPerThread,
                                 CandidateCapacity,
                                 SelectedCapacity,
                                 /*KeysOnly=*/false,
                                 /*LazyValueLoad=*/false>(NumTiles, LastTileItems, SelectedEvery, CandidateEvery);
}

C2H_TEST("BlockPartitionSpeculative triggers in-tile selected overflow drain", "[block][topk][speculative]")
{
  // Sel cap = 8, but per-tile sel count ~= 256/3 ~= 85. Most selected drain
  // via per-item global atomics. Cand cap = 256, cand sparse.
  constexpr int BlockThreads      = 64;
  constexpr int ItemsPerThread    = 4;
  constexpr int CandidateCapacity = 256;
  constexpr int SelectedCapacity  = 8;
  constexpr int NumTiles          = 3;
  constexpr int LastTileItems     = BlockThreads * ItemsPerThread;
  constexpr int SelectedEvery     = 3;
  constexpr int CandidateEvery    = 17;

  run_speculative_partition_test<BlockThreads,
                                 ItemsPerThread,
                                 CandidateCapacity,
                                 SelectedCapacity,
                                 /*KeysOnly=*/false,
                                 /*LazyValueLoad=*/false>(NumTiles, LastTileItems, SelectedEvery, CandidateEvery);
}

C2H_TEST("BlockPartitionSpeculative selected-bypass (SelCap=0) routes selected through atomics", "[block][topk][speculative]")
{
  // SelectedBufferCapacity = 0: the selected stream goes pure-Atomics; only
  // the candidate stream uses the smem buffer + cooperative flush. Sel dense,
  // cand sparse.
  constexpr int BlockThreads      = 64;
  constexpr int ItemsPerThread    = 4;
  constexpr int CandidateCapacity = 128;
  constexpr int SelectedCapacity  = 0;
  constexpr int NumTiles          = 3;
  constexpr int LastTileItems     = BlockThreads * ItemsPerThread;
  constexpr int SelectedEvery     = 3;
  constexpr int CandidateEvery    = 17;

  run_speculative_partition_test<BlockThreads,
                                 ItemsPerThread,
                                 CandidateCapacity,
                                 SelectedCapacity,
                                 /*KeysOnly=*/false,
                                 /*LazyValueLoad=*/false>(NumTiles, LastTileItems, SelectedEvery, CandidateEvery);
}

C2H_TEST("BlockPartitionSpeculative keys-only", "[block][topk][speculative]")
{
  constexpr int BlockThreads      = 64;
  constexpr int ItemsPerThread    = 4;
  constexpr int CandidateCapacity = 64;
  constexpr int SelectedCapacity  = 32;
  constexpr int NumTiles          = 3;
  constexpr int LastTileItems     = BlockThreads * ItemsPerThread;

  run_speculative_partition_test<BlockThreads,
                                 ItemsPerThread,
                                 CandidateCapacity,
                                 SelectedCapacity,
                                 /*KeysOnly=*/true,
                                 /*LazyValueLoad=*/false>(
    NumTiles, LastTileItems, /*selected_every=*/5, /*candidate_every=*/7);
}

C2H_TEST("BlockPartitionSpeculative lazy value load", "[block][topk][speculative]")
{
  constexpr int BlockThreads      = 64;
  constexpr int ItemsPerThread    = 4;
  constexpr int CandidateCapacity = 32;
  constexpr int SelectedCapacity  = 32;
  constexpr int NumTiles          = 3;
  constexpr int LastTileItems     = BlockThreads * ItemsPerThread;

  run_speculative_partition_test<BlockThreads,
                                 ItemsPerThread,
                                 CandidateCapacity,
                                 SelectedCapacity,
                                 /*KeysOnly=*/false,
                                 /*LazyValueLoad=*/true>(
    NumTiles, LastTileItems, /*selected_every=*/5, /*candidate_every=*/7);
}

C2H_TEST("BlockPartitionSpeculative partial trailing tile", "[block][topk][speculative]")
{
  // Last tile has 173 valid items out of tile_items (256).
  constexpr int BlockThreads      = 64;
  constexpr int ItemsPerThread    = 4;
  constexpr int CandidateCapacity = 64;
  constexpr int SelectedCapacity  = 32;
  constexpr int NumTiles          = 3;
  constexpr int LastTileItems     = 173;

  run_speculative_partition_test<BlockThreads,
                                 ItemsPerThread,
                                 CandidateCapacity,
                                 SelectedCapacity,
                                 /*KeysOnly=*/false,
                                 /*LazyValueLoad=*/false>(
    NumTiles, LastTileItems, /*selected_every=*/5, /*candidate_every=*/7);
}
