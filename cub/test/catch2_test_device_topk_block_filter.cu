// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Unit tests for the top-k-private `BlockFilter` primitive in
//! `cub/detail/topk/block_filter.cuh`. Sweeps strategy x KeysOnly x full/partial.
//!
//! The test exercises the "safe-both" interface: the filter is constructed with
//! its sinks (reserve op, transform, output iterator, value channel sink) AND
//! its `identify_selected_op` predicate at the top of the kernel, and
//! `partition()` is called per tile with just the per-tile data + a bare
//! `cuda::std::tuple<TileDataSource...>` of value sources. After the call the
//! kernel invokes `partition.epilogue()`, which is a `_CCCL_FORCEINLINE` no-op
//! for `BlockFilter` (the accumulating sister class has a separate test).
//!
//! The strategy sweep covers the four non-accumulating values of
//! `BlockFilterStrategy`:
//!   Atomics, Staged, SharedMem -- crossed with `InlinedClassify` in {false, true}.

#include <cub/detail/topk/block_filter.cuh>
#include <cub/detail/topk/block_filter_accumulating.cuh>
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
// Test scaffolding: identify_selected_op (unary predicate -> bool).
//---------------------------------------------------------------------

// Driver predicate: returns `true` if the per-item "selected" bit is set in a
// global table indexed by the key value (the test fills keys with their global
// index so the predicate can recover the index).
struct driver_identify_selected_op
{
  const ::cuda::std::uint8_t* d_keep;
  int key_offset;

  _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(int key) const
  {
    const int i = key - key_offset;
    return d_keep[i] != 0;
  }
};

//---------------------------------------------------------------------
// Kernel: drives one tile through BlockFilter (safe-both interface).
//---------------------------------------------------------------------

template <int BlockThreads, int ItemsPerThread, topk::BlockFilterStrategy Strategy, bool InlinedClassify, bool KeysOnly>
__global__ void filter_kernel(
  const int* d_keys_in,
  const int* d_values_in,
  const ::cuda::std::uint8_t* d_keep_in,
  int num_items,
  int key_offset,
  int* d_sel_keys,
  int* d_sel_vals,
  unsigned int* d_sel_counter)
{
  using value_ds_t = topk::direct_data_source<const int*, BlockThreads, ItemsPerThread>;

  // Single-stream sink-side bundle.
  using value_sinks_t = topk::value_channel_sinks_filter_t<int*, ::cuda::std::identity>;
  using value_channel_sinks_tuple_t =
    ::cuda::std::conditional_t<KeysOnly, ::cuda::std::tuple<>, ::cuda::std::tuple<value_sinks_t>>;
  using value_types_tuple_t =
    ::cuda::std::conditional_t<KeysOnly, ::cuda::std::tuple<>, ::cuda::std::tuple<int>>;
  using value_data_source_scratch_types_tuple_t =
    ::cuda::std::conditional_t<KeysOnly,
                               ::cuda::std::tuple<>,
                               ::cuda::std::tuple<typename value_ds_t::ScratchStorage>>;

  // Per-call sources tuple: bare tuple of TileDataSource. Empty when keys-only.
  using value_sources_tuple_t =
    ::cuda::std::conditional_t<KeysOnly, ::cuda::std::tuple<>, ::cuda::std::tuple<value_ds_t>>;

  using sel_reserve_op_t = topk::atomic_reserve_range_op<unsigned int>;
  using xform_t          = ::cuda::std::identity;

  using filter_t = topk::strategy_to_filter_class_t<
    Strategy,
    BlockThreads,
    ItemsPerThread,
    /*AccumulatingBufferCapacity=*/0,
    int,
    unsigned int,
    sel_reserve_op_t,
    xform_t,
    int*,
    driver_identify_selected_op,
    value_channel_sinks_tuple_t,
    value_types_tuple_t,
    value_data_source_scratch_types_tuple_t,
    /*LazyValueLoad=*/false,
    InlinedClassify>;

  __shared__ typename filter_t::TempStorage filter_ts;
  __shared__ typename filter_t::ScratchStorage scratch;

  // BLOCKED arrangement: thread t gets items [t*IPT, (t+1)*IPT).
  int keys[ItemsPerThread];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int j = 0; j < ItemsPerThread; ++j)
  {
    const int idx = static_cast<int>(threadIdx.x) * ItemsPerThread + j;
    keys[j]       = (idx < num_items) ? d_keys_in[idx] : 0;
  }

  // Per-call sources tuple: stack-local data source for the value channel.
  typename value_ds_t::TempStorage val_state{};
  value_ds_t val_ds{d_values_in, val_state};
  val_ds.set_tile_base(0);
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

  // Sinks tuple (captured by ctor).
  auto make_sinks = [&] {
    if constexpr (KeysOnly)
    {
      return ::cuda::std::tuple<>{};
    }
    else
    {
      return ::cuda::std::tuple<value_sinks_t>{value_sinks_t{d_sel_vals, ::cuda::std::identity{}}};
    }
  };
  value_channel_sinks_tuple_t sinks = make_sinks();

  driver_identify_selected_op identify_op{d_keep_in, key_offset};
  xform_t key_transform{};
  sel_reserve_op_t reserve_sel{d_sel_counter};

  filter_t filter{filter_ts, reserve_sel, key_transform, d_sel_keys, sinks, identify_op};

  if (num_items == BlockThreads * ItemsPerThread)
  {
    filter.partition(scratch, keys, sources);
  }
  else
  {
    filter.partition(scratch, keys, num_items, sources);
  }

  // No-op for BlockFilter; present for parity with BlockFilterAccumulating.
  filter.epilogue();
}

//---------------------------------------------------------------------
// Helpers
//---------------------------------------------------------------------

// Build a deterministic keep pattern over `num_items` slots: every
// `keep_every`-th item is kept, everything else is dropped.
static std::vector<::cuda::std::uint8_t> make_keep(int num_items, int keep_every)
{
  std::vector<::cuda::std::uint8_t> out(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    out[i] = (i % keep_every == 0) ? ::cuda::std::uint8_t{1} : ::cuda::std::uint8_t{0};
  }
  return out;
}

// %PARAM% TEST_STRAT strat 0:1:2:3:4:5
// (Strategy, InlinedClassify) cross product covering all six non-accumulating
// combinations:
//   0 = atomics + precomputed-classify
//   1 = atomics + inlined-classify
//   2 = staged + precomputed-classify
//   3 = staged + inlined-classify
//   4 = shared_mem + precomputed-classify
//   5 = shared_mem + inlined-classify
#if TEST_STRAT == 0
constexpr topk::BlockFilterStrategy kStrategy = topk::BlockFilterStrategy::Atomics;
constexpr bool kInlinedClassify               = false;
#elif TEST_STRAT == 1
constexpr topk::BlockFilterStrategy kStrategy = topk::BlockFilterStrategy::Atomics;
constexpr bool kInlinedClassify               = true;
#elif TEST_STRAT == 2
constexpr topk::BlockFilterStrategy kStrategy = topk::BlockFilterStrategy::Staged;
constexpr bool kInlinedClassify               = false;
#elif TEST_STRAT == 3
constexpr topk::BlockFilterStrategy kStrategy = topk::BlockFilterStrategy::Staged;
constexpr bool kInlinedClassify               = true;
#elif TEST_STRAT == 4
constexpr topk::BlockFilterStrategy kStrategy = topk::BlockFilterStrategy::SharedMem;
constexpr bool kInlinedClassify               = false;
#else
constexpr topk::BlockFilterStrategy kStrategy = topk::BlockFilterStrategy::SharedMem;
constexpr bool kInlinedClassify               = true;
#endif

// Drive a single (Strategy, KeysOnly, full|partial) configuration.
template <int BlockThreads, int ItemsPerThread, bool KeysOnly>
void run_filter_test(
  int num_items,
  const std::vector<int>& keys,
  const std::vector<int>& values,
  const std::vector<::cuda::std::uint8_t>& keep)
{
  constexpr int tile_items = BlockThreads * ItemsPerThread;
  REQUIRE(num_items <= tile_items);

  // Host-side golden expectations.
  std::vector<int> expected_selected_keys;
  std::vector<int> expected_selected_vals;
  for (int i = 0; i < num_items; ++i)
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

  const int out_capacity = tile_items + 1;
  thrust::device_vector<int> d_sel_keys(out_capacity, 0);
  thrust::device_vector<int> d_sel_vals(out_capacity, 0);
  thrust::device_vector<unsigned int> d_sel_cnt(1, 0);

  filter_kernel<BlockThreads, ItemsPerThread, kStrategy, kInlinedClassify, KeysOnly><<<1, BlockThreads>>>(
    thrust::raw_pointer_cast(d_keys_in.data()),
    thrust::raw_pointer_cast(d_values_in.data()),
    thrust::raw_pointer_cast(d_keep_in.data()),
    num_items,
    /*key_offset=*/0,
    thrust::raw_pointer_cast(d_sel_keys.data()),
    thrust::raw_pointer_cast(d_sel_vals.data()),
    thrust::raw_pointer_cast(d_sel_cnt.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Counter assertion.
  REQUIRE(d_sel_cnt[0] == expected_selected_keys.size());

  // Output-set assertion (atomic strategies are order-independent).
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

C2H_TEST("BlockFilter keys-only filters a full tile", "[block][topk]")
{
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;

  std::vector<int> keys(tile_items);
  std::vector<int> values(tile_items, 0);
  for (int i = 0; i < tile_items; ++i)
  {
    keys[i] = i;
  }
  auto keep = make_keep(tile_items, /*keep_every=*/3);

  run_filter_test<BlockThreads, ItemsPerThread, /*KeysOnly=*/true>(tile_items, keys, values, keep);
}

C2H_TEST("BlockFilter paired filters a full tile", "[block][topk]")
{
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;

  std::vector<int> keys(tile_items);
  std::vector<int> values(tile_items);
  for (int i = 0; i < tile_items; ++i)
  {
    keys[i]   = i;
    values[i] = i + 1000;
  }
  auto keep = make_keep(tile_items, /*keep_every=*/3);

  run_filter_test<BlockThreads, ItemsPerThread, /*KeysOnly=*/false>(tile_items, keys, values, keep);
}

C2H_TEST("BlockFilter partial tile leaves OOB items unscattered", "[block][topk]")
{
  constexpr int BlockThreads   = 32;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;

  // Partial tile with `tile_items - 17` valid items. The filter's `partition()`
  // partial overload computes per-thread valid counts; OOB items are forced to
  // `false` and never appear in the output.
  const int num_items = tile_items - 17;

  std::vector<int> keys(tile_items, 0);
  std::vector<int> values(tile_items, 0);
  for (int i = 0; i < tile_items; ++i)
  {
    keys[i]   = i;
    values[i] = i + 1000;
  }
  std::vector<::cuda::std::uint8_t> keep(tile_items, 0);
  for (int i = 0; i < num_items; ++i)
  {
    keep[i] = (i % 3 == 0) ? ::cuda::std::uint8_t{1} : ::cuda::std::uint8_t{0};
  }

  run_filter_test<BlockThreads, ItemsPerThread, /*KeysOnly=*/false>(num_items, keys, values, keep);
}
