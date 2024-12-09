// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_topk.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 4:32:4
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 256:1024:256

#if !TUNE_BASE
template <class KeyInT, class NumItemT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = TUNE_ITEMS_PER_THREAD;

    static constexpr int ITEMS_PER_THREAD =
      CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(KeyInT))));

    static constexpr int BITS_PER_PASS = cub::detail::topk::calc_bits_per_pass<KeyInT>(); //@TODO:add tuning axis?
    static constexpr int COFFICIENT_FOR_BUFFER = 128; //@TODO:add tuning axis?

    using TopKPolicyT =
      cub::AgentTopKPolicy<TUNE_THREADS_PER_BLOCK,
                           ITEMS_PER_THREAD,
                           BITS_PER_PASS,
                           COFFICIENT_FOR_BUFFER,
                           cub::BLOCK_LOAD_DIRECT,
                           cub::BLOCK_HISTO_ATOMIC,
                           cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <typename KeyT, typename NumItemT>
void topk_keys(nvbench::state& state, nvbench::type_list<KeyT, NumItemT>)
{
  using key_input_it_t  = const KeyT*;
  using key_output_it_t = KeyT*;
  using num_items_t     = NumItemT;
  // using select_op_t        = less_then_t<T>;
  // using equality_op_t      = cub::NullType;

  constexpr bool select_min = false;

#if !TUNE_BASE
  using policy_t = policy_hub_t<KeyT, NumItemT>;
  using dispatch_t =
    cub::DispatchTopK<key_input_it_t, key_output_it_t, cub::NullType*, cub::NullType*, num_items_t, select_min, policy_t>;
#else // TUNE_BASE
  using dispatch_t =
    cub::DispatchTopK<key_input_it_t, key_output_it_t, cub::NullType*, cub::NullType*, num_items_t, select_min>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements          = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto selected_elements = static_cast<std::size_t>(state.get_int64("SelectedElements{io}"));

  // If possible, do not initialize the input data in the benchmark function.
  // Instead, use the gen function.
  thrust::device_vector<KeyT> in_keys  = generate(elements);
  thrust::device_vector<KeyT> out_keys = generate(selected_elements);
  key_input_it_t d_keys_in             = thrust::raw_pointer_cast(in_keys.data());
  key_output_it_t d_keys_out           = thrust::raw_pointer_cast(out_keys.data());

  // optionally add memory usage to the state
  //  Calling `state.add_element_count(num_elements)` with the number of input
  //  items will report the item throughput rate in elements-per-second.
  //
  //  Calling `state.add_global_memory_reads<T>(num_elements)` and/or
  //  `state.add_global_memory_writes<T>(num_elements)` will report global device
  //  memory throughput as a percentage of the current device's peak global memory
  //  bandwidth, and also in bytes-per-second.
  //
  //  All of these methods take an optional second `column_name` argument, which
  //  will add a new column to the output with the reported element count / buffer
  //  size and column name.
  state.add_element_count(elements, "NumElements");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_writes<KeyT>(selected_elements, "OutputKeys");

  // allocate temporary storage
  std::size_t temp_size;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_keys_in,
    d_keys_out,
    static_cast<cub::NullType*>(nullptr),
    static_cast<cub::NullType*>(nullptr),
    elements,
    selected_elements,
    0);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  // run the algorithm
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_keys_in,
      d_keys_out,
      static_cast<cub::NullType*>(nullptr),
      static_cast<cub::NullType*>(nullptr),
      elements,
      selected_elements,
      launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(topk_keys, NVBENCH_TYPE_AXES(fundamental_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "NumItemT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SelectedElements{io}", nvbench::range(3, 15, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
