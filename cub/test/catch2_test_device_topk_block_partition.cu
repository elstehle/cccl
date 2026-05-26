// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Unit tests for the top-k-private `BlockPartition` primitive in
//! `cub/detail/topk/block_partition.cuh`. Sweeps strategy x KeysOnly x reserve op x
//! full/partial.
//!
//! The test exercises the "safe-both" interface: the partition is constructed with
//! its sinks (reserve ops, transforms, output iterators, value channel sinks) and
//! its classify hooks (identify op, candidate callback) at the top of the kernel,
//! and `partition()` is called per tile with just the per-tile data + the live
//! value `TileDataSource` (or `cub::NullType` for keys-only). After the tile loop
//! the kernel calls `partition.epilogue()`, which is a `_CCCL_FORCEINLINE` no-op
//! for `BlockPartition` (the accumulating sister class has a separate test).
//!
//! `BlockPartition` always operates as a true 2-way partition (HasCandidates is
//! baked in). The single-stream "filter" path lives in `BlockFilter` and has its
//! own dedicated test file (`catch2_test_device_topk_block_filter.cu`).
//!
//! The strategy sweep covers all four non-accumulating values of
//! `block_partition_strategy`:
//!   Atomics, Staged, SharedMem -- crossed with `InlinedClassify` in {false, true}.

#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/block_partition_accumulating.cuh>
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
// Test scaffolding: identify_candidates_op, candidate callback, value channel.
//---------------------------------------------------------------------

// Driver classifier: returns the per-item class read out of a global `classes[]`
// array. Indexed by a per-item global index passed via a closure-captured base.
struct driver_identify_op
{
  const ::cuda::std::int8_t* d_classes;

  // The classify path of BlockPartition only sees `key`; we encode the per-item
  // global index by interpreting the key value as the index into d_classes. The
  // test vectors are built so that `keys[i] == i + key_offset_`; we recover i.
  int key_offset;

  _CCCL_DEVICE _CCCL_FORCEINLINE topk::candidate_class operator()(int key) const
  {
    const int i = key - key_offset;
    return static_cast<topk::candidate_class>(d_classes[i]);
  }
};

// Counts every candidate-classified key into a global `callback_count` (atomic). Used
// to assert the architecture §10.2 invariant that the callback fires
// `eligible_candidate_count` times -- including ones the cap subsequently drops.
struct counting_callback_op
{
  unsigned int* d_count;

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(int /*key*/) const
  {
    atomicAdd(d_count, 1u);
  }
};

struct noop_callback_op
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(const T&) const
  {}
};

//---------------------------------------------------------------------
// Kernel: drives one tile through BlockPartition (safe-both interface).
//---------------------------------------------------------------------

template <int BlockThreads,
          int ItemsPerThread,
          topk::block_partition_strategy Strategy,
          bool InlinedClassify,
          bool KeysOnly,
          bool BackGrowCapped>
__global__ void partition_kernel(
  const int* d_keys_in,
  const int* d_values_in,
  const ::cuda::std::int8_t* d_classes_in,
  int num_items,
  int key_offset,
  int* d_sel_keys,
  int* d_cand_keys,
  int* d_sel_vals,
  int* d_cand_vals,
  unsigned int* d_sel_counter,
  unsigned int* d_cand_counter,
  unsigned int* d_callback_count,
  unsigned int back_anchor,
  unsigned int cap)
{
  using value_ds_t = topk::direct_data_source<const int*, BlockThreads, ItemsPerThread>;

  using value_sinks_t = topk::value_channel_sinks_t<int*, int*>;
  using value_channel_sinks_or_null_t =
    ::cuda::std::conditional_t<KeysOnly, cub::NullType, value_sinks_t>;
  using value_t_t =
    ::cuda::std::conditional_t<KeysOnly, cub::NullType, int>;
  using value_data_source_scratch_t =
    ::cuda::std::conditional_t<KeysOnly, cub::NullType, typename value_ds_t::ScratchStorage>;

  using sel_reserve_op_t = topk::atomic_reserve_range_op<unsigned int>;
  using cand_reserve_op_t =
    ::cuda::std::conditional_t<BackGrowCapped,
                               topk::back_grow_capped_reserve_op<unsigned int>,
                               topk::atomic_reserve_range_op<unsigned int>>;

  using xform_t = ::cuda::std::identity;

  using partition_t = topk::strategy_to_partition_class_t<
    Strategy,
    BlockThreads,
    ItemsPerThread,
    /*AccumulatingBufferCapacity=*/0,
    /*SpeculativeSelectedBufferCapacity=*/0,
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
    value_channel_sinks_or_null_t,
    value_t_t,
    value_data_source_scratch_t,
    /*LazyValueLoad=*/false,
    InlinedClassify>;

  __shared__ typename partition_t::TempStorage partition_ts;
  __shared__ typename partition_t::ScratchStorage scratch;

  // BLOCKED arrangement: thread t gets items [t*IPT, (t+1)*IPT).
  int keys[ItemsPerThread];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int j = 0; j < ItemsPerThread; ++j)
  {
    const int idx = static_cast<int>(threadIdx.x) * ItemsPerThread + j;
    keys[j]       = (idx < num_items) ? d_keys_in[idx] : 0;
  }

  // Build the per-call value source. `direct_data_source` has no default ctor and
  // its TempStorage is empty so we hand it a stack-local sink. Under keys-only the
  // primitive holds the reference but never reads through it.
  typename value_ds_t::TempStorage val_state{};
  value_ds_t val_ds{d_values_in, val_state};
  val_ds.set_tile_base(0);
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

  // Build the sinks (captured by ctor).
  auto make_sinks = [&] {
    if constexpr (KeysOnly)
    {
      return cub::NullType{};
    }
    else
    {
      return value_sinks_t{d_sel_vals, d_cand_vals};
    }
  };
  auto sinks = make_sinks();

  driver_identify_op identify_op{d_classes_in, key_offset};
  counting_callback_op callback_op{d_callback_count};
  xform_t key_transform{};

  // Build mode-dependent reserve ops.
  sel_reserve_op_t reserve_sel{d_sel_counter};
  cand_reserve_op_t reserve_cand = [&]() -> cand_reserve_op_t {
    if constexpr (BackGrowCapped)
    {
      return cand_reserve_op_t{d_cand_counter, back_anchor, cap};
    }
    else
    {
      return cand_reserve_op_t{d_cand_counter};
    }
  }();

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

  if (num_items == BlockThreads * ItemsPerThread)
  {
    partition.partition(scratch, keys, value_source);
  }
  else
  {
    partition.partition(scratch, keys, num_items, value_source);
  }

  // No-op for BlockPartition; present for parity with the accumulating sister class.
  partition.epilogue();
}

//---------------------------------------------------------------------
// Helpers
//---------------------------------------------------------------------

// Build a deterministic class pattern over `num_items` slots:
//  every selected_every-th item is selected,
//  every candidate_every-th item (that isn't selected) is candidate,
//  everything else is rejected.
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
constexpr topk::block_partition_strategy kStrategy = topk::block_partition_strategy::atomics;
constexpr bool kInlinedClassify                  = false;
#elif TEST_STRAT == 1
constexpr topk::block_partition_strategy kStrategy = topk::block_partition_strategy::atomics;
constexpr bool kInlinedClassify                  = true;
#elif TEST_STRAT == 2
constexpr topk::block_partition_strategy kStrategy = topk::block_partition_strategy::staged;
constexpr bool kInlinedClassify                  = false;
#elif TEST_STRAT == 3
constexpr topk::block_partition_strategy kStrategy = topk::block_partition_strategy::staged;
constexpr bool kInlinedClassify                  = true;
#elif TEST_STRAT == 4
constexpr topk::block_partition_strategy kStrategy = topk::block_partition_strategy::shared_mem;
constexpr bool kInlinedClassify                  = false;
#else
constexpr topk::block_partition_strategy kStrategy = topk::block_partition_strategy::shared_mem;
constexpr bool kInlinedClassify                  = true;
#endif

// Drive a single (Strategy, KeysOnly, full|partial) atomic configuration:
// full when num_items == tile_items.
template <int BlockThreads, int ItemsPerThread, bool KeysOnly>
void run_atomic_partition_test(
  int num_items,
  const std::vector<int>& keys,
  const std::vector<int>& values,
  const std::vector<::cuda::std::int8_t>& classes)
{
  constexpr int tile_items = BlockThreads * ItemsPerThread;
  REQUIRE(num_items <= tile_items);

  // Host-side golden expectations.
  std::vector<int> expected_selected_keys;
  std::vector<int> expected_candidate_keys;
  std::vector<int> expected_selected_vals;
  std::vector<int> expected_candidate_vals;
  unsigned int expected_callback_count = 0;
  for (int i = 0; i < num_items; ++i)
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

  const int out_capacity = tile_items + 1;
  thrust::device_vector<int> d_sel_keys(out_capacity, 0);
  thrust::device_vector<int> d_cand_keys(out_capacity, 0);
  thrust::device_vector<int> d_sel_vals(out_capacity, 0);
  thrust::device_vector<int> d_cand_vals(out_capacity, 0);
  thrust::device_vector<unsigned int> d_sel_cnt(1, 0);
  thrust::device_vector<unsigned int> d_cand_cnt(1, 0);
  thrust::device_vector<unsigned int> d_callback_cnt(1, 0);

  partition_kernel<BlockThreads, ItemsPerThread, kStrategy, kInlinedClassify, KeysOnly, /*BackGrowCapped=*/false>
    <<<1, BlockThreads>>>(
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_classes_in.data()),
      num_items,
      /*key_offset=*/0,
      thrust::raw_pointer_cast(d_sel_keys.data()),
      thrust::raw_pointer_cast(d_cand_keys.data()),
      thrust::raw_pointer_cast(d_sel_vals.data()),
      thrust::raw_pointer_cast(d_cand_vals.data()),
      thrust::raw_pointer_cast(d_sel_cnt.data()),
      thrust::raw_pointer_cast(d_cand_cnt.data()),
      thrust::raw_pointer_cast(d_callback_cnt.data()),
      /*back_anchor=*/0u,
      /*cap=*/0u);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Counter assertions.
  REQUIRE(d_sel_cnt[0] == expected_selected_keys.size());
  REQUIRE(d_cand_cnt[0] == expected_candidate_keys.size());
  REQUIRE(d_callback_cnt[0] == expected_callback_count);

  // Output-set assertions (atomic strategies are order-independent).
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

template <int BlockThreads, int ItemsPerThread, bool KeysOnly>
void run_back_grow_capped_partition_test(
  int num_items,
  const std::vector<int>& keys,
  const std::vector<int>& values,
  const std::vector<::cuda::std::int8_t>& classes,
  unsigned int back_anchor,
  unsigned int cap)
{
  constexpr int tile_items = BlockThreads * ItemsPerThread;
  REQUIRE(num_items <= tile_items);

  std::vector<int> expected_selected_keys;
  std::vector<int> eligible_candidate_keys;
  unsigned int expected_callback_count = 0;
  for (int i = 0; i < num_items; ++i)
  {
    const auto c = static_cast<topk::candidate_class>(classes[i]);
    if (c == topk::candidate_class::selected)
    {
      expected_selected_keys.push_back(keys[i]);
    }
    else if (c == topk::candidate_class::candidate)
    {
      eligible_candidate_keys.push_back(keys[i]);
      ++expected_callback_count;
    }
  }

  thrust::device_vector<int> d_keys_in(keys.begin(), keys.end());
  thrust::device_vector<int> d_values_in(values.begin(), values.end());
  thrust::device_vector<::cuda::std::int8_t> d_classes_in(classes.begin(), classes.end());

  // Selected and candidate share a single combined output of size `back_anchor`,
  // matching how agent_topk's last_filter passes `d_keys_out` for both streams.
  const int sentinel = -1;
  thrust::device_vector<int> d_combined_keys(back_anchor, sentinel);
  thrust::device_vector<int> d_combined_vals(back_anchor, sentinel);
  thrust::device_vector<unsigned int> d_sel_cnt(1, 0);
  thrust::device_vector<unsigned int> d_cand_cnt(1, 0);
  thrust::device_vector<unsigned int> d_callback_cnt(1, 0);

  partition_kernel<BlockThreads, ItemsPerThread, kStrategy, kInlinedClassify, KeysOnly, /*BackGrowCapped=*/true>
    <<<1, BlockThreads>>>(
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_classes_in.data()),
      num_items,
      /*key_offset=*/0,
      thrust::raw_pointer_cast(d_combined_keys.data()),
      thrust::raw_pointer_cast(d_combined_keys.data()),
      thrust::raw_pointer_cast(d_combined_vals.data()),
      thrust::raw_pointer_cast(d_combined_vals.data()),
      thrust::raw_pointer_cast(d_sel_cnt.data()),
      thrust::raw_pointer_cast(d_cand_cnt.data()),
      thrust::raw_pointer_cast(d_callback_cnt.data()),
      back_anchor,
      cap);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(d_sel_cnt[0] == expected_selected_keys.size());
  // Candidate counter is bumped by the unclamped count (architecture §8.2).
  REQUIRE(d_cand_cnt[0] == expected_callback_count);
  REQUIRE(d_callback_cnt[0] == expected_callback_count);

  std::vector<int> host_combined(d_combined_keys.begin(), d_combined_keys.end());

  // Selected items at the front.
  std::vector<int> got_sel_keys(host_combined.begin(), host_combined.begin() + d_sel_cnt[0]);
  std::sort(got_sel_keys.begin(), got_sel_keys.end());
  std::sort(expected_selected_keys.begin(), expected_selected_keys.end());
  REQUIRE(got_sel_keys == expected_selected_keys);

  const unsigned int cand_written = std::min<unsigned int>(d_cand_cnt[0], cap);
  std::vector<int> got_cand_keys(host_combined.end() - cand_written, host_combined.end());
  std::sort(got_cand_keys.begin(), got_cand_keys.end());
  std::sort(eligible_candidate_keys.begin(), eligible_candidate_keys.end());
  REQUIRE(got_cand_keys.size() == cand_written);
  REQUIRE(std::includes(
    eligible_candidate_keys.begin(), eligible_candidate_keys.end(), got_cand_keys.begin(), got_cand_keys.end()));

  // Slots between the front (selected) region and the back (candidate) region must
  // remain at their sentinel value -- back-write must not bleed past the cap.
  const unsigned int back_start = back_anchor - cand_written;
  for (unsigned int i = d_sel_cnt[0]; i < back_start; ++i)
  {
    REQUIRE(host_combined[i] == sentinel);
  }
}

//---------------------------------------------------------------------
// Test cases: full sweep
//---------------------------------------------------------------------

C2H_TEST("BlockPartition keys-only partitions a full tile", "[block][topk]")
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
  auto classes = make_classes(tile_items, /*selected_every=*/3, /*candidate_every=*/5);

  run_atomic_partition_test<BlockThreads, ItemsPerThread, /*KeysOnly=*/true>(tile_items, keys, values, classes);
}

C2H_TEST("BlockPartition paired partitions a full tile", "[block][topk]")
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
  auto classes = make_classes(tile_items, 3, 5);

  run_atomic_partition_test<BlockThreads, ItemsPerThread, /*KeysOnly=*/false>(tile_items, keys, values, classes);
}

C2H_TEST("BlockPartition partial tile leaves OOB items unscattered", "[block][topk]")
{
  constexpr int BlockThreads   = 32;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;

  // Partial tile with `tile_items - 17` valid items. The partition's `partition()`
  // partial overload computes per-thread valid counts; OOB items are forced to
  // `rejected` and never appear in either output.
  const int num_items = tile_items - 17;

  std::vector<int> keys(tile_items, 0);
  std::vector<int> values(tile_items, 0);
  for (int i = 0; i < tile_items; ++i)
  {
    // Fill all slots so OOB items are distinguishable from in-bounds ones if the
    // implementation accidentally classifies them as non-rejected.
    keys[i]   = i;
    values[i] = i + 1000;
  }
  std::vector<::cuda::std::int8_t> classes(
    tile_items, static_cast<::cuda::std::int8_t>(topk::candidate_class::rejected));
  for (int i = 0; i < num_items; ++i)
  {
    classes[i] = (i % 3 == 0)
                 ? static_cast<::cuda::std::int8_t>(topk::candidate_class::selected)
                 : ((i % 5 == 0) ? static_cast<::cuda::std::int8_t>(topk::candidate_class::candidate)
                                 : static_cast<::cuda::std::int8_t>(topk::candidate_class::rejected));
  }

  run_atomic_partition_test<BlockThreads, ItemsPerThread, /*KeysOnly=*/false>(num_items, keys, values, classes);
}

C2H_TEST("BlockPartition back_grow_capped reserve op clamps candidate writes "
         "and stacks them at the back",
         "[block][topk]")
{
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;
  constexpr bool KeysOnly      = false;

  std::vector<int> keys(tile_items);
  std::vector<int> values(tile_items);
  for (int i = 0; i < tile_items; ++i)
  {
    // Keys must equal their index for the driver_identify_op to work (it recovers the
    // per-item index from the key value to look up the per-item class).
    keys[i]   = i;
    values[i] = i + 1000;
  }
  auto classes = make_classes(tile_items, /*selected_every=*/4, /*candidate_every=*/3);

  // back_anchor sized so the back range comfortably holds all eligible candidates.
  // cap is set well below the eligible count so the cap-clamp path is exercised.
  run_back_grow_capped_partition_test<BlockThreads, ItemsPerThread, KeysOnly>(
    tile_items, keys, values, classes, /*back_anchor=*/300u, /*cap=*/12u);
}
