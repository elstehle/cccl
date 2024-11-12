/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

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

template <typename key_t, typename value_t>
struct comparator_t
{
  const key_t* key_arr;
  const value_t* value_arr;
  bool is_descending{};

  comparator_t(key_t* key_arr, value_t* value_arr, bool is_descending)
      : key_arr(key_arr)
      , value_arr(value_arr)
      , is_descending(is_descending)
  {}

  bool operator()(std::size_t a, std::size_t b)
  {
    bool res;
    if (is_descending)
    {
      if (key_arr[a] > key_arr[b])
      {
        res = true;
      }
      else if ((key_arr[a] == key_arr[b]) && (value_arr[a] < value_arr[b]))
      {
        res = true;
      }
      else
      {
        res = false;
      }
    }
    else
    {
      if (key_arr[a] < key_arr[b])
      {
        res = true;
      }
      else if ((key_arr[a] == key_arr[b]) && (value_arr[a] < value_arr[a]))
      {
        res = true;
      }
      else
      {
        res = false;
      }
    }

    return res;
  }
};

template <typename key_t, typename value_t>
void sort_keys_and_values(c2h::device_vector<key_t>& keys,
                          c2h::device_vector<value_t>& values,
                          std::pair<c2h::host_vector<key_t>, c2h::host_vector<value_t>>& results,
                          bool is_descending)
{
  c2h::host_vector<key_t> h_keys(keys);
  c2h::host_vector<value_t> h_values(values);

  c2h::host_vector<std::size_t> h_permutation(keys.size());
  thrust::sequence(h_permutation.begin(), h_permutation.end());
  comparator_t<key_t, value_t> comp{
    thrust::raw_pointer_cast(h_keys.data()), thrust::raw_pointer_cast(h_values.data()), is_descending};
  std::stable_sort(h_permutation.begin(), h_permutation.end(), comp);

  thrust::gather(h_permutation.cbegin(),
                 h_permutation.cend(),
                 thrust::make_zip_iterator(h_keys.cbegin(), h_values.cbegin()),
                 thrust::make_zip_iterator(results.first.begin(), results.second.begin()));
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
  std::pair<c2h::host_vector<key_t>, c2h::host_vector<value_t>> in_results;
  in_results.first.resize(in_keys.size());
  in_results.second.resize(in_keys.size());
  sort_keys_and_values(in_keys, in_values, in_results, is_descending);

  // Since the results of API TopKMinPairs() and TopKPairs() are not-sorted
  // We need to sort the results first.
  std::pair<c2h::host_vector<key_t>, c2h::host_vector<value_t>> out_results;
  out_results.first.resize(out_keys.size());
  out_results.second.resize(out_keys.size());
  sort_keys_and_values(out_keys, out_values, out_results, is_descending);

  // i for results from gpu (TopKMinPairs() and TopKPairs()); j for reference results
  num_items_t i = 0, j = 0;
  bool res = true;
  while (i < k && j < num_items)
  {
    if (out_results.first[i] == in_results.first[j])
    {
      if (out_results.second[i] == in_results.second[j])
      {
        i++;
        j++;
      }
      else if (out_results.second[i] > in_results.second[j])
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
