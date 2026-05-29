// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items
//! from sequences of unordered data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_topk.cuh>
#include <cub/detail/cc_dispatch.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_topk_common.cuh>
#include <cub/device/dispatch/tuning/tuning_topk.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_output_iterator.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{

template <typename PolicySelector, typename KeyInputIteratorT, typename OffsetT, typename OutOffsetT, typename ExtractBinOpT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceTopKHistogramKernel(
    _CCCL_GRID_CONSTANT const KeyInputIteratorT d_keys_in,
    counter<it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
    _CCCL_GRID_CONSTANT OffsetT* const histogram,
    _CCCL_GRID_CONSTANT const OffsetT num_items,
    _CCCL_GRID_CONSTANT const OutOffsetT k,
    ExtractBinOpT extract_bin_op,
    _CCCL_GRID_CONSTANT const int pass,
    _CCCL_GRID_CONSTANT const bool reset_histogram)
{
  static constexpr topk_policy policy = current_policy<PolicySelector>();
  using agent_topk_policy_t =
    AgentTopKPolicy<policy.threads_per_block,
                    policy.items_per_thread,
                    policy.bits_per_pass,
                    policy.scan_algorithm,
                    policy.keys_tile_load_kind,
                    policy.accumulating_buffer_capacity,
                    policy.speculative_selected_buffer_capacity>;
  using agent_t = AgentTopKHistogram<agent_topk_policy_t, KeyInputIteratorT, ExtractBinOpT, OffsetT, OutOffsetT>;

  __shared__ typename agent_t::TempStorage temp_storage;

  // Two last-block-only callbacks update the device-side `counter`:
  //   `counter_update_fn` runs on thread 0 before the prefix-sum.
  //   `on_kth_bucket`     runs on the single thread that owns the bucket
  //                       containing the kth element, i.e. inside
  //                       `block_identify_kth_bucket::find_kth_bucket`.
  // `load_from_candidates_buffer` stays at its zero-initialized `false`; the dispatch-side
  // `cudaMemsetAsync` over the counter blob is the source of truth for that bit at pass 0.
  auto counter_update_fn = [counter, num_items] {
    counter->num_candidates_in      = num_items;
    counter->num_candidates_written = 0;
  };
  auto on_kth_bucket =
    [counter, pass](OutOffsetT current_k, int bin_index, OffsetT num_selected, OffsetT num_candidates) {
      counter->k                  = static_cast<OutOffsetT>(current_k - num_selected);
      counter->num_candidates_out = num_candidates;
      set_kth_key_bits<policy.bits_per_pass>(counter->kth_key_bits, pass, static_cast<unsigned int>(bin_index));
    };

  agent_t(temp_storage, d_keys_in, num_items, k, extract_bin_op)
    .invoke(&counter->finished_block_cnt, histogram, reset_histogram, counter_update_fn, on_kth_bucket);
}

// Filter kernel covering passes 1..num_passes-1. The kernel reads counter state once,
// computes the sink_mode (early_stop / buffered / unbuffered), and dispatches to the
// matching agent instantiation. The unbuffered "scout" mode is handled by the
// shared `AgentTopKHistogram` (driven by a candidate-filter predicate), while the
// other two modes use `agent_topk_filter_partition`. All three TempStorages share
// the same __shared__ buffer via a union. The last-filter pass is handled by
// `DeviceTopKLastFilterKernel`.
template <typename PolicySelector,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyT,
          typename ValueInT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceTopKFilterKernel(
    _CCCL_GRID_CONSTANT const KeyInputIteratorT d_keys_in,
    _CCCL_GRID_CONSTANT const KeyOutputIteratorT d_keys_out,
    _CCCL_GRID_CONSTANT const ValueInputIteratorT d_values_in,
    _CCCL_GRID_CONSTANT const ValueOutputIteratorT d_values_out,
    _CCCL_GRID_CONSTANT KeyT* const in_key_buf,
    _CCCL_GRID_CONSTANT ValueInT* const in_val_buf,
    _CCCL_GRID_CONSTANT KeyT* const out_key_buf,
    _CCCL_GRID_CONSTANT ValueInT* const out_val_buf,
    counter<it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
    _CCCL_GRID_CONSTANT OffsetT* const histogram,
    _CCCL_GRID_CONSTANT const OutOffsetT k,
    _CCCL_GRID_CONSTANT const OffsetT buffer_length,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op,
    _CCCL_GRID_CONSTANT const int pass,
    _CCCL_GRID_CONSTANT const bool reset_histogram)
{
  static constexpr topk_policy policy = current_policy<PolicySelector>();
  using agent_topk_policy_t =
    AgentTopKPolicy<policy.threads_per_block,
                    policy.items_per_thread,
                    policy.bits_per_pass,
                    policy.scan_algorithm,
                    policy.keys_tile_load_kind,
                    policy.accumulating_buffer_capacity,
                    policy.speculative_selected_buffer_capacity>;

  static constexpr block_partition_strategy buffered_part_strat = policy.buffered_partition_strategy;
  static constexpr block_filter_strategy early_stop_filter_strat = policy.early_stop_filter_strategy;
  static constexpr bool lazy_value_load = policy.lazy_value_load;
  static constexpr bool inlined_classify = policy.inlined_classify;

  // Single agent type handling both `early_stop` and `buffered` modes; the
  // agent does a runtime two-way `if` over the `mode` arg of `run()`.
  // The unbuffered scout mode is a separate `AgentTopKHistogram` instantiation
  // driven by a candidate-filter predicate. Both agents' `_TempStorage` types
  // share the same __shared__ buffer via a union.
  using agent_fp_t = agent_topk_filter_partition<
    agent_topk_policy_t,
    KeyInputIteratorT,
    KeyOutputIteratorT,
    ValueInputIteratorT,
    ValueOutputIteratorT,
    ExtractBinOpT,
    IdentifyCandidatesOpT,
    OffsetT,
    OutOffsetT,
    buffered_part_strat,
    early_stop_filter_strat,
    lazy_value_load,
    inlined_classify>;

  // Unbuffered scout mode: drive the histogram agent with a candidate-filter
  // predicate that wraps `identify_candidates_op`. See the single-source
  // invariant comment on `AgentTopKHistogram` for why this is sound -- the
  // unbuffered branch is reachable only when `current_len > buffer_length`,
  // which by induction implies no prior pass wrote to `in_key_buf`, so we
  // always load from `d_keys_in`.
  using filter_op_t = topk_candidate_filter_op<IdentifyCandidatesOpT>;
  using agent_ub_t =
    AgentTopKHistogram<agent_topk_policy_t, KeyInputIteratorT, ExtractBinOpT, OffsetT, OutOffsetT, filter_op_t>;

  union all_modes_ts_t
  {
    typename agent_fp_t::TempStorage fp;
    typename agent_ub_t::TempStorage ub;
  };
  __shared__ all_modes_ts_t temp_storage;

  // Read counter state once at entry. The cross-pass scalar fields share a single cache line.
  const OutOffsetT current_k             = counter->k;
  const OffsetT current_len              = counter->num_candidates_out;
  const OffsetT input_length             = counter->num_candidates_in;
  const bool load_from_candidates_buffer = counter->load_from_candidates_buffer;

  // Universal early exit: the `early_stop` epilogue (or any prior pass that took it) writes
  // `num_candidates_in = 0` to signal that all subsequent passes should be no-ops.
  if (input_length == 0)
  {
    return;
  }

  const bool early_stop  = (current_len == static_cast<OffsetT>(current_k));
  const bool will_buffer = !early_stop && (current_len <= buffer_length);

  ValueInT* effective_in_val_buf  = load_from_candidates_buffer ? in_val_buf : nullptr;
  KeyT* effective_out_key_buf   = will_buffer ? out_key_buf : nullptr;
  ValueInT* effective_out_val_buf = will_buffer ? out_val_buf : nullptr;

  // Two last-block-only callbacks update the device-side `counter`. Shared across all three
  // filter modes:
  //   `counter_update_fn` runs on thread 0 before the prefix-sum.
  //   `on_kth_bucket`     runs on the single thread that owns the bucket containing the kth
  //                       element, i.e. inside `block_identify_kth_bucket::find_kth_bucket`.
  // The three branches map to the three filter modes:
  //   - early_stop : write `num_candidates_in = 0` as the universal early-exit signal for
  //                  subsequent passes; the rest of the cross-pass state is intentionally left
  //                  stale (it is never read again before the next early-exit short-circuit).
  //   - buffered   : the buffer now holds `current_len` candidates; flip
  //                  `load_from_candidates_buffer` so the next pass loads from it. Once flipped,
  //                  the flag sticks (the candidate set is monotonically non-increasing).
  //   - unbuffered : nothing to update. `num_candidates_in` already equals `num_items` (set by
  //                  the histogram pass and preserved across the unbuffered chain), and
  //                  `load_from_candidates_buffer` stays false.
  auto counter_update_fn = [counter, current_len, early_stop, will_buffer] {
    if (early_stop)
    {
      counter->num_candidates_in = 0;
    }
    else if (will_buffer)
    {
      counter->num_candidates_in           = current_len;
      counter->load_from_candidates_buffer = true;
      counter->num_candidates_written      = 0;
    }
  };
  auto on_kth_bucket =
    [counter, pass](OutOffsetT current_k, int bin_index, OffsetT num_selected, OffsetT num_candidates) {
      counter->k                  = static_cast<OutOffsetT>(current_k - num_selected);
      counter->num_candidates_out = num_candidates;
      set_kth_key_bits<policy.bits_per_pass>(counter->kth_key_bits, pass, static_cast<unsigned int>(bin_index));
    };

  if (early_stop || will_buffer)
  {
    // Both output-writing modes go through one agent type. The runtime `mode`
    // arg selects between the `BlockFilter[Accumulating]` (early_stop) and
    // `BlockPartition[Accumulating]` (buffered) primitive paths inside the agent.
    const sink_mode mode = early_stop ? sink_mode::early_stop : sink_mode::buffered;
    agent_fp_t agent(
      temp_storage.fp,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      in_key_buf,
      effective_in_val_buf,
      &counter->num_selected_written,
      input_length,
      load_from_candidates_buffer,
      extract_bin_op,
      identify_candidates_op,
      histogram);
    agent.run(
      &counter->finished_block_cnt,
      current_k,
      reset_histogram,
      counter_update_fn,
      on_kth_bucket,
      mode,
      effective_out_key_buf,
      effective_out_val_buf,
      &counter->num_candidates_written);
  }
  else
  {
    // Unbuffered scout pass: histogram-only, gated by a candidate filter. The single-source
    // invariant on `AgentTopKHistogram` (see its block comment) is now structurally enforced:
    // we only reach this branch when the previous pass did not write to the candidate buffer,
    // so `load_from_candidates_buffer == false` and there is no buffer to load from.
    filter_op_t filter_op{identify_candidates_op};
    agent_ub_t agent(temp_storage.ub, d_keys_in, input_length, current_k, extract_bin_op, filter_op);
    agent.invoke(&counter->finished_block_cnt, histogram, reset_histogram, counter_update_fn, on_kth_bucket);
  }
}

// Dedicated last-filter kernel. Runs `agent_topk_last_filter` only; it scatters surviving
// "selected" candidates to the front of `d_keys_out` and the remaining `kth`-class
// candidates to the back of `d_keys_out` (capped at `num_of_kth_needed`). The agent
// neither accumulates a histogram nor finalizes the pass, so this kernel touches no
// histogram smem.
template <typename PolicySelector,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyT,
          typename ValueInT,
          typename IdentifyCandidatesOpT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceTopKLastFilterKernel(
    _CCCL_GRID_CONSTANT const KeyInputIteratorT d_keys_in,
    _CCCL_GRID_CONSTANT const KeyOutputIteratorT d_keys_out,
    _CCCL_GRID_CONSTANT const ValueInputIteratorT d_values_in,
    _CCCL_GRID_CONSTANT const ValueOutputIteratorT d_values_out,
    _CCCL_GRID_CONSTANT KeyT* const in_key_buf,
    _CCCL_GRID_CONSTANT ValueInT* const in_val_buf,
    counter<it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
    _CCCL_GRID_CONSTANT const OutOffsetT k,
    IdentifyCandidatesOpT identify_candidates_op)
{
  static constexpr topk_policy policy = current_policy<PolicySelector>();
  using agent_topk_policy_t =
    AgentTopKPolicy<policy.threads_per_block,
                    policy.items_per_thread,
                    policy.bits_per_pass,
                    policy.scan_algorithm,
                    policy.keys_tile_load_kind,
                    policy.accumulating_buffer_capacity,
                    policy.speculative_selected_buffer_capacity>;

  static constexpr block_partition_strategy part_strat = policy.last_filter_partition_strategy;
  static constexpr bool lazy_value_load              = policy.lazy_value_load;
  static constexpr bool inlined_classify             = policy.inlined_classify;

  using agent_lf_t = agent_topk_last_filter<
    agent_topk_policy_t,
    KeyInputIteratorT,
    KeyOutputIteratorT,
    ValueInputIteratorT,
    ValueOutputIteratorT,
    IdentifyCandidatesOpT,
    OffsetT,
    OutOffsetT,
    part_strat,
    lazy_value_load,
    inlined_classify>;

  __shared__ typename agent_lf_t::TempStorage temp_storage;

  const OffsetT input_length             = counter->num_candidates_in;
  const bool load_from_candidates_buffer = counter->load_from_candidates_buffer;

  if (input_length == 0)
  {
    return;
  }

  ValueInT* effective_in_val_buf = load_from_candidates_buffer ? in_val_buf : nullptr;

  const OutOffsetT num_of_kth_needed = static_cast<OutOffsetT>(counter->k);

  agent_lf_t agent(
    temp_storage,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    in_key_buf,
    effective_in_val_buf,
    &counter->num_selected_written,
    input_length,
    load_from_candidates_buffer,
    identify_candidates_op);
  agent.run(&counter->num_ties_written_to_back, k, num_of_kth_needed);
}

//! @tparam SelectDirection
//!   Determines whether to select the smallest or largest K elements.
//!
//! @tparam KeyInputIteratorT
//!   **[inferred]** Random-access input iterator type for reading input keys @iterator
//!
//! @tparam KeyOutputIteratorT
//!   **[inferred]** Random-access output iterator type for writing output keys @iterator
//!
//! @tparam ValueInputIteratorT
//!   **[inferred]** Random-access input iterator type for reading input values @iterator
//!
//! @tparam ValueOutputIteratorT
//!   **[inferred]** Random-access input iterator type for writing output values @iterator
//!
//! @tparam OffsetT
//!  Data Type for variables: num_items
//!
//! @tparam OutOffsetT
//!  Data Type for variables: k
//!
//! @tparam DecomposerT
//!   Implementation detail, do not specify directly, requirements on the content of this type are subject to breaking
//!   change.
template <select SelectDirection,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          typename DecomposerT           = detail::identity_decomposer_t,
          typename PolicySelector        = policy_selector_from_types<it_value_t<KeyInputIteratorT>>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyInputIteratorT d_keys_in,
  KeyOutputIteratorT d_keys_out,
  const ValueInputIteratorT d_values_in,
  ValueOutputIteratorT d_values_out,
  OffsetT num_items,
  OutOffsetT k,
  DecomposerT decomposer,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

#if _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::stringstream ss;
                 ss << policy_selector(cc);
                 _CubLog("Dispatching DeviceTopK to compute capability %d.%d with tuning: %s\n",
                         cc.major_cap(),
                         cc.minor_cap(),
                         ss.str().c_str());
               }))
#endif // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)

  return dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) {
    static constexpr topk_policy active_policy = policy_getter();
    using key_t                             = it_value_t<KeyInputIteratorT>;
    using value_t                           = it_value_t<ValueInputIteratorT>;
    static constexpr bool keys_only            = ::cuda::std::is_same_v<value_t, NullType>;

    // Resolve the value-channel materialization mode. The agent / kernels are iterator-
    // agnostic; the only behavioral difference between `indexed` and
    // `materialized` is which iterator types the kernels are instantiated with
    // (and, transitively, the per-record size of the candidate back-buffer).
    //
    //   indexed      -- the value channel is rewired so the candidate buffer
    //                   stores `OffsetT` indices; the value-output iterator is
    //                   wrapped in a `cuda::transform_output_iterator` that
    //                   gathers from the user's input iterator on each write
    //                   (matches `main`'s behavior; the default).
    //   materialized -- value channel uses the user's iterators directly; the
    //                   candidate buffer holds full `value_t` records.
    //
    // Forced to `materialized` when keys_only so the existing keys-only path
    // (which never instantiates the value channel) keeps all of its types
    // pointing at the original `NullType*` iterators.
    static constexpr bool indexed =
      !keys_only && active_policy.value_materialization == value_materialization_mode::indexed;
    using effective_value_iterators_t =
      effective_value_iterators<indexed, ValueInputIteratorT, ValueOutputIteratorT, OffsetT>;
    using effective_value_t        = typename effective_value_iterators_t::value_t;
    using effective_value_input_it_t  = typename effective_value_iterators_t::in_t;
    using effective_value_output_it_t = typename effective_value_iterators_t::out_t;

    // atomicAdd does not implement overloads for all integer types, so we limit OffsetT to uint32_t or unsigned long
    // long
    static_assert(
      ::cuda::std::is_same_v<OffsetT, ::cuda::std::uint32_t> || ::cuda::std::is_same_v<OffsetT, unsigned long long>,
      "The top-k algorithm is limited to unsigned offset types retrieved from choose_offset_t<T>.");

    // atomicAdd does not implement overloads for all integer types, so we limit OffsetT to uint32_t or unsigned long
    // long
    static_assert(::cuda::std::is_same_v<OutOffsetT, ::cuda::std::uint32_t>
                    || ::cuda::std::is_same_v<OutOffsetT, unsigned long long>,
                  "The top-k algorithm is limited to unsigned offset types retrieved from choose_offset_t<T>.");

    // TODO (elstehle): consider making this part of the env-based API
    // The algorithm allocates a double-buffer for intermediate results of size
    // num_items/coefficient_for_candidate_buffer
    static constexpr OffsetT coefficient_for_candidate_buffer = 128;
    constexpr int threads_per_block                           = active_policy.threads_per_block;
    constexpr int items_per_thread                            = active_policy.items_per_thread;
    constexpr int bits_per_pass                               = active_policy.bits_per_pass;
    constexpr int tile_size                                   = threads_per_block * items_per_thread;
    const auto num_tiles      = static_cast<unsigned int>(::cuda::ceil_div(num_items, tile_size));
    const int total_bits      = detail::radix::traits_t<key_t>::default_end_bit(decomposer);
    const int num_passes      = calc_num_passes<bits_per_pass>(total_bits);
    constexpr int num_buckets = 1 << bits_per_pass;

    // Define operators
    using identify_candidates_op = identify_candidates_op_t<key_t, SelectDirection, bits_per_pass, DecomposerT>;
    using extract_bin_op         = extract_bin_op_t<key_t, SelectDirection, bits_per_pass, DecomposerT>;

    // We are capping k at a maximum of num_items
    using common_offset_t = ::cuda::std::common_type_t<OffsetT, OutOffsetT>;
    k = static_cast<OutOffsetT>((::cuda::std::min) (common_offset_t{k}, static_cast<common_offset_t>(num_items)));

    // Construct the effective value-channel iterators. On the indexed path
    // these wrap the user's iterators so the kernels see a generative
    // `counting_iterator<OffsetT>` for input and a
    // `transform_output_iterator{d_values_out, topk_index_gather_op{d_values_in}}`
    // for output -- the wrapper turns every `out[pos] = idx` the agent issues
    // into `d_values_out[pos] = d_values_in[idx]`. On the materialized /
    // keys-only paths the lambdas return the user iterators unchanged.
    auto effective_d_values_in = [&]() -> effective_value_input_it_t {
      if constexpr (indexed)
      {
        return effective_value_input_it_t{OffsetT{0}};
      }
      else
      {
        return d_values_in;
      }
    }();
    auto effective_d_values_out = [&]() -> effective_value_output_it_t {
      if constexpr (indexed)
      {
        using gather_op_t = topk_index_gather_op<ValueInputIteratorT>;
        return effective_value_output_it_t{d_values_out, gather_op_t{d_values_in}};
      }
      else
      {
        return d_values_out;
      }
    }();

    // Specify temporary storage allocation requirements
    using counter_t             = counter<key_t, OffsetT, OutOffsetT>;
    const size_t size_counter   = sizeof(counter_t);
    const size_t size_histogram = num_buckets * sizeof(OffsetT);
    const OffsetT candidate_buffer_length =
      (::cuda::std::max) (OffsetT{1}, num_items / coefficient_for_candidate_buffer);

    constexpr int allocations_array_size            = keys_only ? 4 : 6;
    size_t allocation_sizes[allocations_array_size] = {
      size_counter,
      size_histogram,
      candidate_buffer_length * sizeof(key_t),
      candidate_buffer_length * sizeof(key_t)};
    if constexpr (!keys_only)
    {
      // `effective_value_t` is `OffsetT` on the indexed path and the user's
      // value type on the materialized path -- shrinking the candidate buffer
      // when values are wider than offsets is the whole point of indexed mode.
      allocation_sizes[4] = candidate_buffer_length * sizeof(effective_value_t);
      allocation_sizes[5] = candidate_buffer_length * sizeof(effective_value_t);
    }

    // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
    void* allocations[allocations_array_size] = {};
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage allocation
      return cudaSuccess;
    }

    // Init the buffer for descriptor and histogram. Zeroing the counter blob is also the source
    // of truth for `counter::load_from_candidates_buffer` at pass 0: it must observe `false`
    // before the first filter pass runs, which is guaranteed by this memset.
    if (const auto error = CubDebug(launcher_factory.MemsetAsync(
          allocations[0], 0, static_cast<char*>(allocations[2]) - static_cast<char*>(allocations[0]), stream)))
    {
      return error;
    }

    // Get grid size for scanning tiles
    int num_sms = 0;
    if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(num_sms)))
    {
      return error;
    }

    auto topk_kernel = DeviceTopKFilterKernel<
      PolicySelector,
      KeyInputIteratorT,
      KeyOutputIteratorT,
      effective_value_input_it_t,
      effective_value_output_it_t,
      OffsetT,
      OutOffsetT,
      key_t,
      effective_value_t,
      extract_bin_op,
      identify_candidates_op>;

    int main_kernel_blocks_per_sm = 0;
    if (const auto error =
          CubDebug(launcher_factory.MaxSmOccupancy(main_kernel_blocks_per_sm, topk_kernel, threads_per_block)))
    {
      return error;
    }
    const auto main_kernel_max_occupancy = static_cast<unsigned int>(main_kernel_blocks_per_sm * num_sms);
    const auto topk_grid_size            = (::cuda::std::min) (main_kernel_max_occupancy, num_tiles);

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking topk_kernel<<<%d, %d, 0, "
            "%lld>>>(), %d items per thread, %d SM occupancy\n",
            topk_grid_size,
            threads_per_block,
            (long long) stream,
            items_per_thread,
            main_kernel_blocks_per_sm);
#endif // CUB_DEBUG_LOG

    // Initialize address variables
    counter_t* counter = static_cast<counter_t*>(allocations[0]);
    OffsetT* histogram = static_cast<decltype(histogram)>(allocations[1]);

    // Pass 0: dedicated histogram-only kernel over the full input
    {
      auto histogram_kernel =
        DeviceTopKHistogramKernel<PolicySelector, KeyInputIteratorT, OffsetT, OutOffsetT, extract_bin_op>;

      int histogram_kernel_blocks_per_sm = 0;
      if (const auto error = CubDebug(
            launcher_factory.MaxSmOccupancy(histogram_kernel_blocks_per_sm, histogram_kernel, threads_per_block)))
      {
        return error;
      }
      const auto histogram_kernel_max_occupancy = static_cast<unsigned int>(histogram_kernel_blocks_per_sm * num_sms);
      const auto histogram_grid_size            = (::cuda::std::min) (histogram_kernel_max_occupancy, num_tiles);

      extract_bin_op extract_op(0, total_bits, decomposer);
      if (const auto error = CubDebug(
            launcher_factory(histogram_grid_size, threads_per_block, 0, stream)
              .doit(histogram_kernel, d_keys_in, counter, histogram, num_items, k, extract_op, 0, num_passes != 1)))
      {
        return error;
      }
    }

    // Passes 1..num_passes-1: fused filter + histogram kernel
    // Current() = input buffer (read), Alternate() = output buffer (write)
    DoubleBuffer<key_t> key_bufs(static_cast<key_t*>(allocations[3]), static_cast<key_t*>(allocations[2]));
    DoubleBuffer<effective_value_t> val_bufs;
    if constexpr (!keys_only)
    {
      val_bufs = DoubleBuffer<effective_value_t>(
        static_cast<effective_value_t*>(allocations[5]), static_cast<effective_value_t*>(allocations[4]));
    }

    int pass = 1;
    for (; pass < num_passes; pass++)
    {
      extract_bin_op extract_op(pass, total_bits, decomposer);
      identify_candidates_op identify_op(&counter->kth_key_bits, pass, total_bits, decomposer);

      if (const auto error = CubDebug(
            launcher_factory(topk_grid_size, threads_per_block, 0, stream)
              .doit(topk_kernel,
                    d_keys_in,
                    d_keys_out,
                    effective_d_values_in,
                    effective_d_values_out,
                    key_bufs.Current(),
                    val_bufs.Current(),
                    key_bufs.Alternate(),
                    val_bufs.Alternate(),
                    counter,
                    histogram,
                    k,
                    candidate_buffer_length,
                    extract_op,
                    identify_op,
                    pass,
                    pass != num_passes - 1)))
      {
        return error;
      }

      key_bufs.selector ^= 1;
      if constexpr (!keys_only)
      {
        val_bufs.selector ^= 1;
      }
    }

    // Last filter pass: dedicated DeviceTopKLastFilterKernel running agent_topk_last_filter only.
    identify_candidates_op identify_op(&counter->kth_key_bits, pass, total_bits, decomposer);
    auto last_filter_kernel = DeviceTopKLastFilterKernel<
      PolicySelector,
      KeyInputIteratorT,
      KeyOutputIteratorT,
      effective_value_input_it_t,
      effective_value_output_it_t,
      OffsetT,
      OutOffsetT,
      key_t,
      effective_value_t,
      identify_candidates_op>;

    int last_filter_kernel_blocks_per_sm = 0;
    if (const auto error = CubDebug(
          launcher_factory.MaxSmOccupancy(last_filter_kernel_blocks_per_sm, last_filter_kernel, threads_per_block)))
    {
      return error;
    }
    const auto last_filter_kernel_max_occupancy = static_cast<unsigned int>(last_filter_kernel_blocks_per_sm * num_sms);
    const auto last_filter_grid_size            = (::cuda::std::min) (last_filter_kernel_max_occupancy, num_tiles);
    if (const auto error = CubDebug(
          launcher_factory(last_filter_grid_size, threads_per_block, 0, stream)
            .doit(last_filter_kernel,
                  d_keys_in,
                  d_keys_out,
                  effective_d_values_in,
                  effective_d_values_out,
                  key_bufs.Current(),
                  val_bufs.Current(),
                  counter,
                  k,
                  identify_op)))
    {
      return error;
    }

    return cudaSuccess;
  });
}
} // namespace detail::topk

CUB_NAMESPACE_END
