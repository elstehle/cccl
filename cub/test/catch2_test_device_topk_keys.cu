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
#include <thrust/sort.h>

#include <algorithm>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKKeys, topk_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKMinKeys, topk_min_keys);

using key_types       = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;

CUB_TEST("DeviceTopK::TopKKeys: Basic testing", "[keys][topk][device]", key_types, num_items_types)
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
  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(k);

  const int num_key_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);

  const bool select_min    = GENERATE(false, true);
  const bool is_descending = !select_min;

  // Run the device-wide API
  if (select_min)
  {
    topk_min_keys(thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()), num_items, k);
  }
  else
  {
    topk_keys(thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()), num_items, k);
  }

  // Sort the entire input data as result referece
  c2h::host_vector<key_t> h_in_keys(in_keys);
  c2h::host_vector<key_t> host_results;
  host_results.resize(out_keys.size());
  if (is_descending)
  {
    std::partial_sort_copy(
      h_in_keys.begin(), h_in_keys.end(), host_results.begin(), host_results.end(), std::greater<key_t>());
  }
  else
  {
    std::partial_sort_copy(
      h_in_keys.begin(), h_in_keys.end(), host_results.begin(), host_results.end(), std::less<key_t>());
  }
  // Since the results of API TopKMinKeys() and TopKKeys() are not-sorted
  // We need to sort the results first.
  c2h::host_vector<key_t> device_results(out_keys);
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
