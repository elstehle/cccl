// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_batched_topk.cuh>
#include <cub/device/device_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__execution/tune.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_BLOCK_LOAD_ALGORITHM ld 0:2:1

#if !TUNE_BASE
template <class KeyInT>
struct policy_selector_t
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::topk::topk_policy
  {
#  if TUNE_BLOCK_LOAD_ALGORITHM == 0
    constexpr auto load_alg = cub::BLOCK_LOAD_DIRECT;
#  elif TUNE_BLOCK_LOAD_ALGORITHM == 1
    constexpr auto load_alg = cub::BLOCK_LOAD_WARP_TRANSPOSE;
#  elif TUNE_BLOCK_LOAD_ALGORITHM == 2
    constexpr auto load_alg = cub::BLOCK_LOAD_VECTORIZE;
#  endif

    constexpr int nominal_4b_items_per_thread = TUNE_ITEMS_PER_THREAD;
    constexpr int items_per_thread            = cuda::std::max(1, (nominal_4b_items_per_thread * 4 / sizeof(KeyInT)));
    return cub::detail::topk::topk_policy{
      TUNE_THREADS_PER_BLOCK,
      items_per_thread,
      load_alg,
      cub::BLOCK_SCAN_WARP_SCANS,
      cub::detail::topk::calc_bits_per_pass<KeyInT>()};
  }
};
#endif // !TUNE_BASE

// Development aid: the same workload can be run through cub::DeviceBatchedTopK with a single segment, so the batched
// implementation can be compared against the single-problem one on identical inputs. Exposed as the runtime "Impl"
// axis below (only under TUNE_BASE, so tuning runs keep sweeping exactly one algorithm).
//
// A single segment of up to `max_elements` items puts the batched request past the cluster backend's competitive
// range, so it routes to the baseline backend's multi-CTA-per-segment path on every architecture -- which is the point
// of the comparison.
//
// These bounds are compile-time and must cover the largest value each axis produces; keep them in sync with the axis
// ranges at the bottom of the file. Note the batched dispatch sizes its per-segment candidate buffers from the *bound*
// rather than the runtime size, so every point allocates as if it were the largest one. That is a fixed setup cost
// outside the timed region, but it does mean temporary storage here is not representative.
constexpr ::cuda::std::int64_t max_elements          = ::cuda::std::int64_t{1} << 28;
constexpr ::cuda::std::int64_t max_selected_elements = ::cuda::std::int64_t{1} << 23;

template <typename KeyT, typename OffsetT, typename OutOffsetT>
void topk_keys(nvbench::state& state, nvbench::type_list<KeyT, OffsetT, OutOffsetT>)
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

  thrust::device_vector<KeyT> in_keys = generate(elements, entropy);
  thrust::device_vector<KeyT> out_keys(selected_elements, thrust::no_init);
  const KeyT* d_keys_in = thrust::raw_pointer_cast(in_keys.data());
  KeyT* d_keys_out      = thrust::raw_pointer_cast(out_keys.data());

  state.add_element_count(elements, "NumElements");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_writes<KeyT>(selected_elements, "OutputKeys");

  // TODO(bgruber): call cub::DeviceTopK::MaxKeys with a the caching_allocator_t once we have an env-overload without
  // temporary storage
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted)
#if !TUNE_BASE
      ,
    cuda::execution::tune(policy_selector_t<KeyT>{})
#endif // !TUNE_BASE
  };

#if TUNE_BASE
  const bool use_batched = state.get_string("Impl") == "batched";
#else // tuning sweeps the single-problem policy, so keep that the only algorithm under test
  constexpr bool use_batched = false;
#endif // TUNE_BASE

  if (use_batched)
  {
    // Iterator-of-iterators over a single segment: the outer iterator always yields the same key pointer.
    auto d_keys_in_it  = cuda::constant_iterator{d_keys_in};
    auto d_keys_out_it = cuda::constant_iterator{d_keys_out};

    const auto seg_arg = cuda::args::immediate{static_cast<OffsetT>(elements),
                                               cuda::args::bounds<OffsetT{0}, static_cast<OffsetT>(max_elements)>()};
    const auto k_arg =
      cuda::args::immediate{static_cast<OutOffsetT>(selected_elements),
                            cuda::args::bounds<OutOffsetT{0}, static_cast<OutOffsetT>(max_selected_elements)>()};
    const auto ns_arg = cuda::args::immediate{cuda::std::int64_t{1}};

    // The batched entry requires determinism and tie_break to be acknowledged together, so it cannot reuse `env`.
    // Tuning is not forwarded either: `policy_selector_t` above returns a single-problem policy.
    auto batched_env = cuda::std::execution::env{
      cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                               cuda::execution::tie_break::unspecified,
                               cuda::execution::output_ordering::unsorted)};

    size_t temp_size{};
    cub::DeviceBatchedTopK::MaxKeys(
      nullptr, temp_size, d_keys_in_it, d_keys_out_it, seg_arg, k_arg, ns_arg, batched_env);
    thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
    auto* temp_storage = thrust::raw_pointer_cast(temp.data());

    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      auto env_with_stream =
        cuda::std::execution::env{cuda::stream_ref{launch.get_stream().get_stream()}, batched_env};
      cub::DeviceBatchedTopK::MaxKeys(
        temp_storage, temp_size, d_keys_in_it, d_keys_out_it, seg_arg, k_arg, ns_arg, env_with_stream);
    });
    return;
  }

  // Allocate temporary storage
  size_t temp_size{};
  cub::DeviceTopK::MaxKeys(
    nullptr,
    temp_size,
    d_keys_in,
    d_keys_out,
    static_cast<OffsetT>(elements),
    static_cast<OutOffsetT>(selected_elements),
    env);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env_with_stream = cuda::std::execution::env{cuda::stream_ref{launch.get_stream().get_stream()}, env};
    cub::DeviceTopK::MaxKeys(
      temp_storage,
      temp_size,
      d_keys_in,
      d_keys_out,
      static_cast<OffsetT>(elements),
      static_cast<OutOffsetT>(selected_elements),
      env_with_stream);
  });
}

NVBENCH_BENCH_TYPES(topk_keys, NVBENCH_TYPE_AXES(fundamental_types, offset_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "OffsetT{ct}", "OutOffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("SelectedElements", nvbench::range(3, 23, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201", "0.000"})
#if TUNE_BASE
  .add_string_axis("Impl", {"single", "batched"})
#endif // TUNE_BASE
  ;
