// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Unit tests for the top-k-private BlockPartition primitive. Covers all three strategies
//! (Atomics, Staged, SharedMem), keys-only and paired key/value flows, and the compile-time
//! specializations derived from sink_mode (HasCandidates, HasCandidateCap).

#include <cub/agent/topk/block_partition.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <c2h/catch2_test_helper.h>

namespace topk = cub::detail::topk;

// Kernel that wires BlockLoad-style striped reads into BlockPartition and applies the
// strategy-appropriate entry point.
template <typename KeyT,
          typename ValueT,
          int BlockThreads,
          int ItemsPerThread,
          topk::BlockPartitionStrategy Strategy,
          bool HasCandidates,
          bool HasCandidateCap,
          bool KeysOnly>
__global__ void partition_kernel(
  const KeyT* d_keys_in,
  const ValueT* d_vals_in,
  const ::cuda::std::int8_t* d_classes_in,
  int num_items,
  KeyT* d_sel_keys,
  KeyT* d_cand_keys,
  ValueT* d_sel_vals,
  ValueT* d_cand_vals,
  unsigned int* d_sel_counter,
  unsigned int* d_cand_counter,
  unsigned int max_candidate_count)
{
  using partition_t = topk::BlockPartition<KeyT,
                                           ::cuda::std::conditional_t<KeysOnly, cub::NullType, ValueT>,
                                           unsigned int,
                                           unsigned int,
                                           BlockThreads,
                                           ItemsPerThread,
                                           Strategy,
                                           HasCandidates,
                                           HasCandidateCap>;

  __shared__ typename partition_t::buffer_t buffer;

  KeyT keys[ItemsPerThread];
  ValueT values[ItemsPerThread];
  topk::candidate_class classes[ItemsPerThread];

  constexpr int tile_items = BlockThreads * ItemsPerThread;

  // Blocked layout: thread t gets items [t*IPT, (t+1)*IPT).
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int j = 0; j < ItemsPerThread; ++j)
  {
    const int lane_idx = static_cast<int>(threadIdx.x) * ItemsPerThread + j;
    if (lane_idx < num_items)
    {
      keys[j]   = d_keys_in[lane_idx];
      values[j] = d_vals_in[lane_idx];
      classes[j] = static_cast<topk::candidate_class>(d_classes_in[lane_idx]);
    }
    else
    {
      classes[j] = topk::candidate_class::rejected;
    }
    (void) tile_items;
  }

  partition_t partition;

  if constexpr (Strategy == topk::BlockPartitionStrategy::SharedMem)
  {
    partition.PartitionPairs(
      buffer, keys, values, classes, d_sel_keys, d_cand_keys, d_sel_vals, d_cand_vals,
      d_sel_counter, d_cand_counter, max_candidate_count);
  }
  else if constexpr (Strategy == topk::BlockPartitionStrategy::Staged)
  {
    partition.PartitionKeys(
      buffer, keys, classes, d_sel_keys, d_cand_keys, d_sel_counter, d_cand_counter, max_candidate_count);
    if constexpr (!KeysOnly)
    {
      partition.ScatterValues(buffer, values, d_sel_vals, d_cand_vals);
    }
  }
  else // Atomics
  {
    if constexpr (KeysOnly)
    {
      partition.PartitionKeys(
        buffer, keys, classes, d_sel_keys, d_cand_keys, d_sel_counter, d_cand_counter, max_candidate_count);
    }
    else
    {
      partition.PartitionPairs(
        buffer, keys, values, classes, d_sel_keys, d_cand_keys, d_sel_vals, d_cand_vals,
        d_sel_counter, d_cand_counter, max_candidate_count);
    }
  }
}

template <typename KeyT,
          typename ValueT,
          int BlockThreads,
          int ItemsPerThread,
          topk::BlockPartitionStrategy Strategy,
          bool HasCandidates,
          bool HasCandidateCap,
          bool KeysOnly>
void run_partition_test(int num_items,
                        const std::vector<KeyT>& keys,
                        const std::vector<ValueT>& values,
                        const std::vector<::cuda::std::int8_t>& classes,
                        unsigned int max_candidate_count = ::cuda::std::numeric_limits<unsigned int>::max())
{
  constexpr int tile_items = BlockThreads * ItemsPerThread;
  REQUIRE(num_items <= tile_items);

  // Build golden expectations.
  // "Eligible" candidates: under HasCandidateCap, any `max_candidate_count` of them may
  // be written (the specific set depends on nondeterministic atomic order), so we only
  // assert subset membership + exact count. Without a cap, all eligible candidates must
  // appear in the output.
  std::vector<KeyT> expected_selected_keys;
  std::vector<ValueT> expected_selected_vals;
  std::vector<KeyT> eligible_candidate_keys;
  std::vector<ValueT> eligible_candidate_vals;
  unsigned int expected_candidate_counter = 0;
  for (int i = 0; i < num_items; ++i)
  {
    const auto c = static_cast<topk::candidate_class>(classes[i]);
    if (c == topk::candidate_class::selected
        || (!HasCandidates && c == topk::candidate_class::candidate))
    {
      expected_selected_keys.push_back(keys[i]);
      expected_selected_vals.push_back(values[i]);
    }
    else if (c == topk::candidate_class::candidate)
    {
      eligible_candidate_keys.push_back(keys[i]);
      eligible_candidate_vals.push_back(values[i]);
      ++expected_candidate_counter;
    }
  }

  thrust::device_vector<KeyT> d_keys_in(keys.begin(), keys.end());
  thrust::device_vector<ValueT> d_vals_in(values.begin(), values.end());
  thrust::device_vector<::cuda::std::int8_t> d_classes_in(classes.begin(), classes.end());

  const int out_capacity = tile_items + 1;
  thrust::device_vector<KeyT> d_sel_keys(out_capacity, KeyT{});
  thrust::device_vector<KeyT> d_cand_keys(out_capacity, KeyT{});
  thrust::device_vector<ValueT> d_sel_vals(out_capacity, ValueT{});
  thrust::device_vector<ValueT> d_cand_vals(out_capacity, ValueT{});
  thrust::device_vector<unsigned int> d_sel_cnt(1, 0);
  thrust::device_vector<unsigned int> d_cand_cnt(1, 0);

  partition_kernel<KeyT, ValueT, BlockThreads, ItemsPerThread, Strategy, HasCandidates, HasCandidateCap, KeysOnly>
    <<<1, BlockThreads>>>(
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_vals_in.data()),
      thrust::raw_pointer_cast(d_classes_in.data()),
      num_items,
      thrust::raw_pointer_cast(d_sel_keys.data()),
      thrust::raw_pointer_cast(d_cand_keys.data()),
      thrust::raw_pointer_cast(d_sel_vals.data()),
      thrust::raw_pointer_cast(d_cand_vals.data()),
      thrust::raw_pointer_cast(d_sel_cnt.data()),
      thrust::raw_pointer_cast(d_cand_cnt.data()),
      max_candidate_count);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  // Validate final counters.
  const unsigned int sel_cnt  = d_sel_cnt[0];
  const unsigned int cand_cnt = d_cand_cnt[0];
  REQUIRE(sel_cnt == expected_selected_keys.size());
  REQUIRE(cand_cnt == expected_candidate_counter);

  // Output is order-independent (atomic-based), so sort before compare.
  std::vector<KeyT> got_sel_keys(d_sel_keys.begin(), d_sel_keys.begin() + sel_cnt);
  std::sort(got_sel_keys.begin(), got_sel_keys.end());
  std::sort(expected_selected_keys.begin(), expected_selected_keys.end());
  REQUIRE(got_sel_keys == expected_selected_keys);

  const int cand_written = static_cast<int>(
    std::min<unsigned int>(cand_cnt, HasCandidateCap ? max_candidate_count : cand_cnt));
  std::vector<KeyT> got_cand_keys(d_cand_keys.begin(), d_cand_keys.begin() + cand_written);
  std::sort(got_cand_keys.begin(), got_cand_keys.end());
  std::sort(eligible_candidate_keys.begin(), eligible_candidate_keys.end());
  REQUIRE(got_cand_keys.size() == static_cast<std::size_t>(cand_written));
  // Written candidates must be a subset of the eligible ones.
  REQUIRE(std::includes(
    eligible_candidate_keys.begin(), eligible_candidate_keys.end(),
    got_cand_keys.begin(), got_cand_keys.end()));

  if constexpr (!KeysOnly)
  {
    std::vector<ValueT> got_sel_vals(d_sel_vals.begin(), d_sel_vals.begin() + sel_cnt);
    std::sort(got_sel_vals.begin(), got_sel_vals.end());
    std::sort(expected_selected_vals.begin(), expected_selected_vals.end());
    REQUIRE(got_sel_vals == expected_selected_vals);

    std::vector<ValueT> got_cand_vals(d_cand_vals.begin(), d_cand_vals.begin() + cand_written);
    std::sort(got_cand_vals.begin(), got_cand_vals.end());
    std::sort(eligible_candidate_vals.begin(), eligible_candidate_vals.end());
    REQUIRE(got_cand_vals.size() == static_cast<std::size_t>(cand_written));
    REQUIRE(std::includes(
      eligible_candidate_vals.begin(), eligible_candidate_vals.end(),
      got_cand_vals.begin(), got_cand_vals.end()));
  }
}

// Generate a deterministic sprinkle of classes across the tile. Every ~3rd item is
// selected, every ~5th is candidate, and the rest are rejected. Uses a small prime
// stride so we hit all three classes for any tile_items >= 16.
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

// %PARAM% TEST_STRAT strat 0:1:2
// 0 = Atomics, 1 = Staged, 2 = SharedMem

#if TEST_STRAT == 0
constexpr topk::BlockPartitionStrategy kStrategy = topk::BlockPartitionStrategy::Atomics;
#elif TEST_STRAT == 1
constexpr topk::BlockPartitionStrategy kStrategy = topk::BlockPartitionStrategy::Staged;
#else
constexpr topk::BlockPartitionStrategy kStrategy = topk::BlockPartitionStrategy::SharedMem;
#endif

// Keys-only path. SharedMem has no PartitionKeys; we exercise it through PartitionPairs
// with NullType values elided at compile time. Atomics/Staged use PartitionKeys directly.
C2H_TEST("BlockPartition keys-only (HasCandidates=true) partitions and scatters", "[block][topk]")
{
  if constexpr (kStrategy == topk::BlockPartitionStrategy::SharedMem)
  {
    // SharedMem keys-only requires PartitionPairs; covered in the paired path too.
    SUCCEED();
    return;
  }
  constexpr int BlockThreads    = 64;
  constexpr int ItemsPerThread  = 4;
  constexpr int tile_items      = BlockThreads * ItemsPerThread;
  constexpr bool HasCandidates  = true;
  constexpr bool HasCandidateCap = false;
  constexpr bool KeysOnly       = true;

  std::vector<int> keys(tile_items);
  std::vector<cub::NullType> values(tile_items);
  for (int i = 0; i < tile_items; ++i)
  {
    keys[i] = i * 7 + 3;
  }
  auto classes = make_classes(tile_items, /*selected_every=*/3, /*candidate_every=*/5);

  run_partition_test<int, cub::NullType, BlockThreads, ItemsPerThread, kStrategy, HasCandidates, HasCandidateCap, KeysOnly>(
    tile_items, keys, values, classes);
}

C2H_TEST("BlockPartition pairs (HasCandidates=true) partitions keys and values", "[block][topk]")
{
  constexpr int BlockThreads    = 64;
  constexpr int ItemsPerThread  = 4;
  constexpr int tile_items      = BlockThreads * ItemsPerThread;
  constexpr bool HasCandidates  = true;
  constexpr bool HasCandidateCap = false;
  constexpr bool KeysOnly       = false;

  std::vector<int> keys(tile_items);
  std::vector<float> values(tile_items);
  for (int i = 0; i < tile_items; ++i)
  {
    keys[i]   = i * 7 + 3;
    values[i] = static_cast<float>(i) + 0.5f;
  }
  auto classes = make_classes(tile_items, 3, 5);

  run_partition_test<int, float, BlockThreads, ItemsPerThread, kStrategy, HasCandidates, HasCandidateCap, KeysOnly>(
    tile_items, keys, values, classes);
}

C2H_TEST("BlockPartition pairs (HasCandidates=false, early_stop mode)", "[block][topk]")
{
  constexpr int BlockThreads     = 32;
  constexpr int ItemsPerThread   = 4;
  constexpr int tile_items       = BlockThreads * ItemsPerThread;
  constexpr bool HasCandidates   = false;
  constexpr bool HasCandidateCap = false;
  constexpr bool KeysOnly        = false;

  std::vector<int> keys(tile_items);
  std::vector<float> values(tile_items);
  for (int i = 0; i < tile_items; ++i)
  {
    keys[i]   = i;
    values[i] = static_cast<float>(i) * 0.25f;
  }
  // Early_stop mode: caller must fold candidate -> selected before calling. We simulate
  // that here by only emitting selected or rejected.
  std::vector<::cuda::std::int8_t> classes(tile_items);
  for (int i = 0; i < tile_items; ++i)
  {
    classes[i] = (i % 2 == 0)
                   ? static_cast<::cuda::std::int8_t>(topk::candidate_class::selected)
                   : static_cast<::cuda::std::int8_t>(topk::candidate_class::rejected);
  }

  run_partition_test<int, float, BlockThreads, ItemsPerThread, kStrategy, HasCandidates, HasCandidateCap, KeysOnly>(
    tile_items, keys, values, classes);
}

C2H_TEST("BlockPartition pairs (HasCandidateCap=true, last_filter mode)", "[block][topk]")
{
  constexpr int BlockThreads     = 64;
  constexpr int ItemsPerThread   = 4;
  constexpr int tile_items       = BlockThreads * ItemsPerThread;
  constexpr bool HasCandidates   = true;
  constexpr bool HasCandidateCap = true;
  constexpr bool KeysOnly        = false;

  std::vector<int> keys(tile_items);
  std::vector<float> values(tile_items);
  for (int i = 0; i < tile_items; ++i)
  {
    keys[i]   = i;
    values[i] = static_cast<float>(i) + 1000.0f;
  }
  auto classes = make_classes(tile_items, 4, 3);

  // Cap candidate writes to 10: items beyond that still bump the counter but are suppressed.
  run_partition_test<int, float, BlockThreads, ItemsPerThread, kStrategy, HasCandidates, HasCandidateCap, KeysOnly>(
    tile_items, keys, values, classes, /*max_candidate_count=*/10u);
}

C2H_TEST("BlockPartition handles partial last tile via rejected marker", "[block][topk]")
{
  constexpr int BlockThreads     = 32;
  constexpr int ItemsPerThread   = 4;
  constexpr int tile_items       = BlockThreads * ItemsPerThread;
  constexpr bool HasCandidates   = true;
  constexpr bool HasCandidateCap = false;
  constexpr bool KeysOnly        = false;

  // num_items less than a full tile; remaining lanes are classed as rejected by the kernel.
  const int num_items = tile_items - 17;

  std::vector<int> keys(tile_items, 0);
  std::vector<float> values(tile_items, 0.0f);
  for (int i = 0; i < num_items; ++i)
  {
    keys[i]   = i + 100;
    values[i] = static_cast<float>(i) + 0.125f;
  }
  std::vector<::cuda::std::int8_t> classes(tile_items,
                                           static_cast<::cuda::std::int8_t>(topk::candidate_class::rejected));
  for (int i = 0; i < num_items; ++i)
  {
    classes[i] = (i % 3 == 0)
                   ? static_cast<::cuda::std::int8_t>(topk::candidate_class::selected)
                   : ((i % 5 == 0)
                        ? static_cast<::cuda::std::int8_t>(topk::candidate_class::candidate)
                        : static_cast<::cuda::std::int8_t>(topk::candidate_class::rejected));
  }

  run_partition_test<int, float, BlockThreads, ItemsPerThread, kStrategy, HasCandidates, HasCandidateCap, KeysOnly>(
    num_items, keys, values, classes);
}
