// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_topk.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/sort.h>

#include <algorithm>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKKeys, topk_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKMinKeys, topk_min_keys);

using key_types       = c2h::type_list<cuda::std::uint16_t, cuda::std::uint32_t, cuda::std::uint64_t>;
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;

C2H_TEST("DeviceTopK::TopKKeys: Basic testing", "[keys][topk][device]", key_types, num_items_types)
{
  using key_t       = c2h::get<0, TestType>;
  using num_items_t = c2h::get<1, TestType>;

  // Set input size
  constexpr num_items_t min_num_items = 1 << 10;
  constexpr num_items_t max_num_items = 1 << 15;
  const num_items_t num_items         = GENERATE_COPY(take(5, random(min_num_items, max_num_items)));

  // Set the k value
  constexpr num_items_t min_k = 1 << 3;
  constexpr num_items_t max_k = 1 << 5;
  const num_items_t k         = GENERATE_COPY(take(5, random(min_k, max_k)));

  // Allocate the device memory
  c2h::device_vector<key_t> keys_in(num_items);
  c2h::device_vector<key_t> keys_out(k);

  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in);

  const bool is_descending = GENERATE(false, true);

  // Run the device-wide API
  if (!is_descending)
  {
    topk_min_keys(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_keys(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  // Sort the entire input data as result reference
  c2h::host_vector<key_t> h_keys_in(keys_in);
  c2h::host_vector<key_t> host_results;
  host_results.resize(keys_out.size());
  if (is_descending)
  {
    std::partial_sort_copy(
      h_keys_in.begin(), h_keys_in.end(), host_results.begin(), host_results.end(), std::greater<key_t>());
  }
  else
  {
    std::partial_sort_copy(
      h_keys_in.begin(), h_keys_in.end(), host_results.begin(), host_results.end(), std::less<key_t>());
  }
  // Since the results of API TopKMinKeys() and TopKKeys() are not-sorted
  // We need to sort the results first.
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());
  }
  else
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());
  }

  REQUIRE(host_results == device_results);
}

C2H_TEST("DeviceTopK::TopKKeys: works with iterators", "[keys][topk][device]", key_types, num_items_types)
{
  using key_t       = c2h::get<0, TestType>;
  using num_items_t = c2h::get<1, TestType>;

  // Set input size
  constexpr num_items_t min_num_items = 1 << 10;
  constexpr num_items_t max_num_items = 1 << 15;
  const num_items_t num_items         = GENERATE_COPY(take(5, random(min_num_items, max_num_items)));

  // Set the k value
  constexpr num_items_t min_k = 1 << 3;
  constexpr num_items_t max_k = 1 << 5;
  const num_items_t k         = GENERATE_COPY(take(5, random(min_k, max_k)));

  // Prepare input and output
  auto keys_in = thrust::make_counting_iterator(key_t{});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));

  // Run the device-wide API
  const bool is_descending = GENERATE(false, true);
  if (!is_descending)
  {
    topk_min_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  // Verify results

  /// Since the results of API TopKMinKeys() and TopKKeys() are not-sorted
  /// We need to sort the results first.
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());
  }
  else
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());
  }

  bool res = true;
  if (is_descending)
  {
    auto keys_expected_it = thrust::make_reverse_iterator(keys_in + num_items);
    for (num_items_t i = 0; i < k; i++)
    {
      if (device_results[i] != keys_expected_it[i])
      {
        res = false;
      }
    }
  }
  else
  {
    auto keys_expected_it = keys_in;
    for (num_items_t i = 0; i < k; i++)
    {
      if (device_results[i] != keys_expected_it[i])
      {
        res = false;
      }
    }
  }

  REQUIRE(res == true);
}
