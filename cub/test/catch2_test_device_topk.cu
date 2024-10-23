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
#include <cub/util_type.cuh>

#include <thrust/memory.h>

#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new> // bad_alloc

#include "catch2_large_array_sort_helper.cuh"
#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKPairs, topk_pairs);

using custom_value_t = c2h::custom_type_t<c2h::equal_comparable_t>;
// using value_types    = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;

// cub::detail::ChooseOffsetsT only selected 32/64 bit unsigned types:
// using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using value_types     = c2h::type_list<cuda::std::uint32_t>;
using num_items_types = c2h::type_list<cuda::std::uint32_t>;

CUB_TEST("DeviceTopK::TopKPairs: Basic testing", "[pairs][topk][device]", value_types, num_items_types)
{
  using key_t       = cuda::std::uint32_t;
  using value_t     = c2h::get<0, TestType>;
  using num_items_t = c2h::get<1, TestType>;

  constexpr num_items_t min_num_items = 1 << 5;
  constexpr num_items_t max_num_items = 1 << 20;
  const num_items_t num_items =
    GENERATE_COPY(num_items_t{0}, num_items_t{1}, take(5, random(min_num_items, max_num_items)));

  constexpr num_items_t min_k = 1 << 3;
  constexpr num_items_t max_k = 1 << 15;
  const num_items_t k         = GENERATE_COPY(num_items_t{0}, num_items_t{1}, take(5, random(min_k, max_k)));

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);

  c2h::device_vector<value_t> in_values(k);
  c2h::device_vector<value_t> out_values(k);

  const int num_key_seeds   = 1;
  const int num_value_seeds = 1;
  c2h::gen(CUB_SEED(num_key_seeds), in_keys);
  c2h::gen(CUB_SEED(num_value_seeds), in_values);

  const bool is_descending = GENERATE(false, true);

  if (is_descending)
  {
    // sort_pairs_descending(
    //   thrust::raw_pointer_cast(in_keys.data()),
    //   thrust::raw_pointer_cast(out_keys.data()),
    //   thrust::raw_pointer_cast(in_values.data()),
    //   thrust::raw_pointer_cast(out_values.data()),
    //   num_items,
    //   begin_bit<key_t>(),
    //   end_bit<key_t>());
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

  auto refs        = radix_sort_reference(in_keys, in_values, is_descending);
  auto& ref_keys   = refs.first;
  auto& ref_values = refs.second;

  int num_equaling_kth   = 0;
  num_items_t kth_offset = k - 1;
  key_t kth_element      = ref_keys[kth_offset];
  key_t pre_kth_element  = kth_element;
  while (pre_kth_element == kth_element && kth_offset != 0)
  {
    num_equaling_kth++;
    pre_kth_element = ref_keys[kth_offset];
    kth_offset--;
  }

  c2h::device_vector<value_t> out_values_unde(out_values.begin(), out_values.end() - num_equaling_kth);
  c2h::host_vector<value_t> ref_values_unde(ref_values.begin(), ref_values.end() - num_equaling_kth);

  REQUIRE(ref_keys == out_keys);
  REQUIRE(ref_values_unde == out_values_unde);
}
