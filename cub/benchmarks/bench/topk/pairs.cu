// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-time toggle: when non-zero, the benchmark routes the workload through the
// segmented-batch dispatch (`cub::detail::batched_topk::dispatch`) configured with a single
// segment instead of `cub::DeviceTopK`. Intended as a rough comparison of the segmented-batch
// implementation against the single-problem dispatch on equivalent workloads.
#ifndef CUB_BENCH_TOPK_USE_BATCHED
#  define CUB_BENCH_TOPK_USE_BATCHED 1
#endif

#include <cub/device/device_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_KEYS_TILE_LOAD_KIND ld 0:2:1

#if !TUNE_BASE && !CUB_BENCH_TOPK_USE_BATCHED
template <class KeyInT>
struct policy_selector_t
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::topk::topk_policy
  {
#  if TUNE_KEYS_TILE_LOAD_KIND == 0
    constexpr auto keys_tile_load_kind = cub::detail::topk::tile_load_kind::block_load_direct;
#  elif TUNE_KEYS_TILE_LOAD_KIND == 1
    constexpr auto keys_tile_load_kind = cub::detail::topk::tile_load_kind::block_load_warp_transpose;
#  elif TUNE_KEYS_TILE_LOAD_KIND == 2
    constexpr auto keys_tile_load_kind = cub::detail::topk::tile_load_kind::block_load_vectorize;
#  endif

    constexpr int nominal_4b_items_per_thread = TUNE_ITEMS_PER_THREAD;
    constexpr int items_per_thread            = cuda::std::max(1, (nominal_4b_items_per_thread * 4 / sizeof(KeyInT)));
    return cub::detail::topk::topk_policy{
      TUNE_THREADS_PER_BLOCK,
      items_per_thread,
      cub::detail::topk::calc_bits_per_pass<KeyInT>(),
      keys_tile_load_kind,
      cub::BLOCK_SCAN_WARP_SCANS};
  }
};
#endif // !TUNE_BASE && !CUB_BENCH_TOPK_USE_BATCHED

template <typename KeyT, typename ValueT, typename OffsetT, typename OutOffsetT>
void topk_pairs(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT, OutOffsetT>)
{
  // Retrieve axis parameters
  const auto elements          = static_cast<size_t>(state.get_int64("Elements{io}"));
  const auto selected_elements = static_cast<size_t>(state.get_int64("SelectedElements"));
  const bit_entropy entropy    = str_to_entropy(state.get_string("Entropy"));

  // Skip benchmarks at runtime
  if (selected_elements >= elements)
  {
    state.skip("We only support the case where the variable SelectedElements is smaller than the variable "
               "Elements{io}.");
    return;
  }

  thrust::device_vector<KeyT> in_keys     = generate(elements, entropy);
  thrust::device_vector<ValueT> in_values = generate(elements);
  thrust::device_vector<KeyT> out_keys(selected_elements, thrust::no_init);
  thrust::device_vector<ValueT> out_values(selected_elements, thrust::no_init);

  const KeyT* d_keys_in     = thrust::raw_pointer_cast(in_keys.data());
  KeyT* d_keys_out          = thrust::raw_pointer_cast(out_keys.data());
  const ValueT* d_values_in = thrust::raw_pointer_cast(in_values.data());
  ValueT* d_values_out      = thrust::raw_pointer_cast(out_values.data());

  state.add_element_count(elements, "NumElements");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_reads<ValueT>(elements, "InputValues");
  state.add_global_memory_writes<KeyT>(selected_elements, "OutputKeys");
  state.add_global_memory_writes<ValueT>(selected_elements, "OutputVales");

#if CUB_BENCH_TOPK_USE_BATCHED
  // Wrap the input/output pointers in the iterator-of-iterators expected by the segmented-batch
  // dispatch. With a single segment the outer iterator always dereferences to the same key /
  // value pointer, so `constant_iterator{ptr}` exactly models what the API needs.
  // `OffsetT`/`OutOffsetT` are unused on this path (the batched dispatch derives its own offset
  // types internally).
  auto d_keys_in_it    = ::cuda::make_constant_iterator(d_keys_in);
  auto d_keys_out_it   = ::cuda::make_constant_iterator(d_keys_out);
  auto d_values_in_it  = ::cuda::make_constant_iterator(d_values_in);
  auto d_values_out_it = ::cuda::make_constant_iterator(d_values_out);

  // Static upper bounds matching the maxima of the benchmark axes (lower bounds left at the
  // parameter-type defaults). These tighten the candidate-buffer sizing in the dispatch and
  // feed into compile-time policy resolution. Must be kept in sync with the axis ranges below.
  constexpr ::cuda::std::int64_t max_elements          = ::cuda::std::int64_t{1} << 28;
  constexpr ::cuda::std::int64_t max_selected_elements = ::cuda::std::int64_t{1} << 23;
  constexpr ::cuda::std::int64_t max_num_segments      = 1;

  cub::detail::batched_topk::segment_size_uniform<0, max_elements> segment_sizes_param{
    static_cast<::cuda::std::int64_t>(elements)};
  cub::detail::batched_topk::k_uniform<1, max_selected_elements> k_param{
    static_cast<::cuda::std::int64_t>(selected_elements)};
  // NOTE: explicit ctor value is required -- `uniform_discrete_param`'s default ctor leaves
  // `value` uninitialized, even when the template only allows a single option. Without this
  // the dispatch runtime-reads garbage and silently picks `select::min`. See
  // `cub/cub/detail/segmented_params.cuh::uniform_discrete_param`.
  cub::detail::batched_topk::select_direction_static<cub::detail::topk::select::max> direction_param{
    cub::detail::topk::select::max};
  cub::detail::batched_topk::num_segments_uniform<1, max_num_segments> num_segments_param{::cuda::std::int64_t{1}};
  cub::detail::batched_topk::total_num_items_guarantee<1, max_elements> total_items_param{
    static_cast<::cuda::std::int64_t>(elements)};

  size_t temp_size{};
  cub::detail::batched_topk::dispatch(
    nullptr,
    temp_size,
    d_keys_in_it,
    d_keys_out_it,
    d_values_in_it,
    d_values_out_it,
    segment_sizes_param,
    k_param,
    direction_param,
    num_segments_param,
    total_items_param);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::batched_topk::dispatch(
      temp_storage,
      temp_size,
      d_keys_in_it,
      d_keys_out_it,
      d_values_in_it,
      d_values_out_it,
      segment_sizes_param,
      k_param,
      direction_param,
      num_segments_param,
      total_items_param,
      launch.get_stream());
  });
#else // CUB_BENCH_TOPK_USE_BATCHED
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted)
#  if !TUNE_BASE
      ,
    cuda::execution::tune(policy_selector_t<KeyT>{})
#  endif // !TUNE_BASE
  };

  // Allocate temporary storage
  size_t temp_size{};
  cub::DeviceTopK::MaxPairs(
    nullptr,
    temp_size,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    static_cast<OffsetT>(elements),
    static_cast<OutOffsetT>(selected_elements),
    env);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env_with_stream = cuda::std::execution::env{cuda::stream_ref{launch.get_stream().get_stream()}, env};
    cub::DeviceTopK::MaxPairs(
      temp_storage,
      temp_size,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      static_cast<OffsetT>(elements),
      static_cast<OutOffsetT>(selected_elements),
      env_with_stream);
  });
#endif // CUB_BENCH_TOPK_USE_BATCHED
}

NVBENCH_BENCH_TYPES(topk_pairs, NVBENCH_TYPE_AXES(integral_types, integral_types, offset_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}", "OutOffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SelectedElements", nvbench::range(3, 23, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201", "0.000"});
