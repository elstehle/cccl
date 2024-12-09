// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_topk.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>

#include <algorithm>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKPairs, topk_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKMinPairs, topk_min_pairs);

/**
 * Custom comparator that compares a tuple type's first element using `operator <`.
 */
struct compare_op_t
{
  bool is_descending{};
  compare_op_t(bool is_descending)
      : is_descending(is_descending)
  {}
  /**
   * We need to be able to have two different types for lhs and rhs, as the call to std::stable_sort with a
   * zip-iterator, will pass a thrust::tuple for lhs and a tuple_of_iterator_references for rhs.
   */
  template <typename LhsT, typename RhsT>
  __host__ __device__ bool operator()(const LhsT& lhs, const RhsT& rhs) const
  {
    bool res;
    if (is_descending)
    {
      res = thrust::get<0>(lhs) > thrust::get<0>(rhs);
      if (thrust::get<0>(lhs) == thrust::get<0>(rhs))
      {
        res = thrust::get<1>(lhs) < thrust::get<1>(rhs);
      }
    }
    else
    {
      res = thrust::get<0>(lhs) < thrust::get<0>(rhs);
      if (thrust::get<0>(lhs) == thrust::get<0>(rhs))
      {
        res = thrust::get<1>(lhs) < thrust::get<1>(rhs);
      }
    }
    return res;
  }
};

template <typename key_t, typename value_t, typename num_items_t>
void sort_keys_and_values(
  c2h::host_vector<key_t>& h_keys, c2h::host_vector<value_t>& h_values, num_items_t num_items, bool is_descending)
{
  compare_op_t comp{is_descending};
  auto zipped_it = thrust::make_zip_iterator(h_keys.begin(), h_values.begin());
  std::stable_sort(zipped_it, zipped_it + num_items, comp);
}

using value_types     = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;

CUB_TEST("DeviceTopK::TopKPairs: Basic testing", "[pairs][topk][device]", value_types, num_items_types)
{
  using key_t       = cuda::std::uint32_t;
  using value_t     = c2h::get<0, TestType>;
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
  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(k);

  c2h::device_vector<value_t> in_values(num_items);
  c2h::device_vector<value_t> out_values(k);

  const int num_key_seeds   = 1;
  const int num_value_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);
  c2h::gen(CUB_SEED(num_value_seeds), in_values);

  const bool select_min    = GENERATE(false, true);
  const bool is_descending = !select_min;

  // Run the device-wide API
  if (select_min)
  {
    topk_min_pairs(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()),
      num_items,
      k);
  }
  else
  {
    topk_pairs(thrust::raw_pointer_cast(in_keys.data()),
               thrust::raw_pointer_cast(out_keys.data()),
               thrust::raw_pointer_cast(in_values.data()),
               thrust::raw_pointer_cast(out_values.data()),
               num_items,
               k);
  }

  // Sort the entire input data as result referece
  c2h::host_vector<key_t> h_in_keys(in_keys);
  c2h::host_vector<value_t> h_in_values(in_values);
  sort_keys_and_values(h_in_keys, h_in_values, num_items, is_descending);

  // Since the results of API TopKMinPairs() and TopKPairs() are not-sorted
  // We need to sort the results first.
  c2h::host_vector<key_t> h_out_keys(out_keys);
  c2h::host_vector<value_t> h_out_values(out_values);
  sort_keys_and_values(h_out_keys, h_out_values, k, is_descending);

  // i for results from gpu (TopKMinPairs() and TopKPairs()); j for reference results
  num_items_t i = 0, j = 0;
  bool res = true;
  while (i < k && j < num_items)
  {
    if (h_out_keys[i] == h_in_keys[j])
    {
      if (h_out_values[i] == h_in_values[j])
      {
        i++;
        j++;
      }
      else if (h_out_values[i] > h_in_values[j])
      {
        // Since the results of API TopKMinPairs() and TopKPairs() are not stable.
        // There might be multiple items equaling to the value of kth element,
        // any of them can appear in the results. We need to find them from the input data.
        j++;
      }
      else
      {
        res = false;
        break;
      }
    }
    else
    {
      res = false;
      break;
    }
  }
  REQUIRE(res == true);
}
