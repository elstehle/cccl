// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Kernel entry point for device-wide batched top-k.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_batched_topk.cuh>
#include <cub/agent/agent_batched_topk_cluster.cuh>
#include <cub/agent/agent_topk_common.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_topk_common.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>

#include <cuda/__cmath/round_up.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// Assert-free search shared by `resolve_worker_policy_device` and the backend coverage predicate. Returns the
// index of the smallest worker policy whose tile size still covers the upper bound on segment size AND whose
// instantiated agent's shared memory usage fits within the static shared memory limit (max_smem_per_block), or -1 if
// none does. Kept separate from `resolve_worker_policy_device` so callers can query coverage as a bool without
// tripping that trait's hard `static_assert`.
template <typename PolicyGetter, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_covering_policy_index
{
private:
  struct policy_t
  {
    worker_policy worker_per_segment_policy;
    multi_worker_policy multi_worker_per_segment_policy;
  };
  static constexpr ::cuda::std::int64_t max_segment_size = ::cuda::args::__traits<SegmentSizeParameterT>::highest;
  static constexpr topk_policy active_policy             = PolicyGetter{}();

  template <int Index>
  [[nodiscard]] static constexpr int find_index()
  {
    if constexpr (Index >= active_policy.baseline.worker_per_segment_policies.size())
    {
      return -1;
    }
    else
    {
      constexpr worker_policy wp = active_policy.baseline.worker_per_segment_policies[Index];
      constexpr auto tile_size   = ::cuda::std::int64_t{wp.threads_per_block} * wp.items_per_thread;

      struct policy_getter_17 // TODO(bgruber): drop this in C++20 and pass wp directly
      {
        _CCCL_HOST_DEVICE_API constexpr auto operator()() const
        {
          return policy_t{active_policy.baseline.worker_per_segment_policies[Index],
                          active_policy.baseline.multi_worker_per_segment_policy};
        }
      };
      using candidate_agent_t  = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
      constexpr bool covers    = tile_size >= max_segment_size;
      constexpr bool fits_smem = sizeof(typename candidate_agent_t::TempStorage) <= max_smem_per_block;
      constexpr int next       = find_index<Index + 1>();
      if constexpr (covers && fits_smem)
      {
        return next >= 0 ? next : Index;
      }
      else
      {
        return next;
      }
    }
  }

public:
  static constexpr int value = find_index<0>();
};

// True iff some one-worker-per-segment policy covers the statically-known maximum segment size within the shared-memory
// limit. Used by the backend selector to decide whether the baseline backend can serve the batch with the
// worker-per-segment path *alone*; when false the batch still runs on the baseline backend, with oversize segments
// escalated to the multi-CTA-per-segment path (see `find_fitting_policy_index`).
template <typename PolicyGetter, typename SegmentSizeParameterT, typename... AgentParamsT>
inline constexpr bool baseline_can_cover_v =
  find_covering_policy_index<PolicyGetter, SegmentSizeParameterT, AgentParamsT...>::value >= 0;

// Fallback search for the case no worker policy *covers* the segment-size upper bound: returns the index of the
// *largest* worker policy whose instantiated agent still fits static shared memory, or -1 if none fits at all.
// `worker_per_segment_policies` is ordered by decreasing tile size, so the first policy that fits is the largest one.
// The worker then treats any segment larger than that tile size as oversize and enqueues it onto the large-segment
// queue that the multi-CTA-per-segment kernels drain -- which is why picking the *largest* fitting tile matters: it
// keeps as many segments as possible on the cheaper single-CTA path.
template <typename PolicyGetter, typename SegmentSizeParameterT, typename... AgentParamsT>
struct find_fitting_policy_index
{
private:
  struct policy_t
  {
    worker_policy worker_per_segment_policy;
    multi_worker_policy multi_worker_per_segment_policy;
  };
  static constexpr topk_policy active_policy = PolicyGetter{}();

  template <int Index>
  [[nodiscard]] static constexpr int find_index()
  {
    if constexpr (Index >= active_policy.baseline.worker_per_segment_policies.size())
    {
      return -1;
    }
    else
    {
      struct policy_getter_17 // TODO(bgruber): drop this in C++20 and pass the policy directly
      {
        _CCCL_HOST_DEVICE_API constexpr auto operator()() const
        {
          return policy_t{active_policy.baseline.worker_per_segment_policies[Index],
                          active_policy.baseline.multi_worker_per_segment_policy};
        }
      };
      using candidate_agent_t = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
      if constexpr (sizeof(typename candidate_agent_t::TempStorage) <= max_smem_per_block)
      {
        return Index;
      }
      else
      {
        return find_index<Index + 1>();
      }
    }
  }

public:
  static constexpr int value = find_index<0>();
};

// The worker-per-segment policy index the kernel and the host dispatch both resolve to: the smallest *covering*
// policy when one exists (every segment fits one CTA, so nothing is escalated), otherwise the largest shared-memory
// fitting policy (oversize segments escalate to the multi-CTA path). Host and device must agree on this, because the
// worker agent computes each escalated segment's tile count from the resolved policy's multi-worker tile size and the
// multi-CTA kernels grid-stride over exactly those tiles.
template <typename PolicyGetter, typename SegmentSizeParameterT, typename... AgentParamsT>
inline constexpr int worker_policy_index_v =
  find_covering_policy_index<PolicyGetter, SegmentSizeParameterT, AgentParamsT...>::value >= 0
    ? find_covering_policy_index<PolicyGetter, SegmentSizeParameterT, AgentParamsT...>::value
    : find_fitting_policy_index<PolicyGetter, SegmentSizeParameterT, AgentParamsT...>::value;

// Resolves the worker-per-segment policy and agent type the kernel instantiates, via `worker_policy_index_v`: the
// smallest covering policy where one exists, else the largest shared-memory fitting one (oversize segments then
// escalate to the multi-CTA-per-segment path). `PolicyGetter` is a nullary constant-expression getter returning the
// resolved `topk_policy` (e.g. the resolved-CC policy from `dispatch_compute_cap`); use this form when you already have
// the resolved policy. Prefer the `resolve_worker_policy_device` alias below when you have a `PolicySelector`.
template <typename PolicyGetter, typename SegmentSizeParameterT, typename... AgentParamsT>
struct resolve_worker_policy_for_getter
{
private:
  struct policy_t
  {
    worker_policy worker_per_segment_policy;
    multi_worker_policy multi_worker_per_segment_policy;
  };
  static constexpr topk_policy active_policy = PolicyGetter{}();
  static constexpr int selected_index = worker_policy_index_v<PolicyGetter, SegmentSizeParameterT, AgentParamsT...>;

public:
  // Only reachable when *no* worker policy fits static shared memory -- not merely when none covers the segment size,
  // which is a supported configuration served by escalating to the multi-CTA-per-segment path. The smallest policy is
  // 128x2 keys, so this fires only for a key/value type whose block primitives cannot fit even that.
  static_assert(selected_index >= 0,
                "cub::DeviceBatchedTopK: no baseline worker policy fits within the static shared-memory limit for this "
                "key (and value) type. Use a narrower key/value type, or supply a tuning override with a smaller "
                "worker tile.");
  static constexpr policy_t policy = {active_policy.baseline.worker_per_segment_policies[selected_index],
                                      active_policy.baseline.multi_worker_per_segment_policy};

  struct policy_getter_17 // TODO(bgruber): drop this in C++20 and pass policy directly
  {
    _CCCL_HOST_DEVICE_API constexpr auto operator()() const
    {
      return policy;
    }
  };
  using agent_t = agent_batched_topk_worker_per_segment<policy_getter_17, AgentParamsT...>;
};

// `PolicySelector`-based form: resolves the policy for this compilation's CC via `current_policy<PolicySelector>()`
// (per-`__CUDA_ARCH__` device-side; the host default CC host-side). Device consumers (kernel body, launch-bounds
// helpers) use this. The host baseline arm instead uses `resolve_worker_policy_for_getter` with the
// resolved-CC getter so its policy choice matches the device kernel per CC.
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
struct resolve_worker_policy_device
{
private:
#if _CCCL_HAS_CONCEPTS()
  static_assert(topk_policy_selector<PolicySelector>);
#endif

  struct active_policy_getter_17 // TODO(bgruber): drop this in C++20 and pass policy directly
  {
    _CCCL_HOST_DEVICE_API constexpr auto operator()() const
    {
      return current_policy<PolicySelector>();
    }
  };
  using impl_t =
    resolve_worker_policy_for_getter<active_policy_getter_17, SegmentSizeParameterT, AgentParamsT...>;

public:
  static constexpr auto policy = impl_t::policy;
  using agent_t                = typename impl_t::agent_t;
};

// -----------------------------------------------------------------------------
// Single kernel symbol hosting both backends
// -----------------------------------------------------------------------------
// There is exactly one kernel symbol per instantiation. Its body selects the active backend device-side via
// `current_policy<PolicySelector>()` (evaluated per `__CUDA_ARCH__` pass), so each target architecture compiles only
// the backend the selector picks for it -- honoring CUB's "one kernel per arch/problem" rule while still supporting a
// multi-architecture fatbin whose per-arch choice differs. The host still branches its launch configuration (grid,
// shared memory, cluster dimensions) per backend, but both host arms launch this same symbol.

// Backend-specific kernel arguments. The unused struct is passed default-constructed (all-null / zero) to the arm the
// selector does not pick; passing it costs nothing (a few grid-constant scalars) and keeps a single kernel signature.
template <typename NumSegmentsValueT, typename LargeSegmentTileOffsetT>
struct baseline_kernel_args
{
  batched_topk_counters<NumSegmentsValueT>* d_counters   = nullptr;
  NumSegmentsValueT* d_large_segments_ids                = nullptr;
  LargeSegmentTileOffsetT* d_large_segments_tile_offsets = nullptr;
};

struct cluster_kernel_args
{
  ::cuda::std::uint32_t max_block_resident_items = 0;
};

// -----------------------------------------------------------------------------
// Launch-bounds helpers
// -----------------------------------------------------------------------------
// The two backends use different `__launch_bounds__` shapes (baseline: just threads_per_block; cluster: threads plus a
// min-blocks-per-SM and an optional max-blocks-per-cluster cap). We resolve all three per architecture from the
// selected policy. `resolve_worker_policy_device` (which carries a hard `static_assert`) is only ever touched
// inside the `backend == baseline` branch, so an oversize bound routed to the cluster/unsupported backend never trips
// it.
_CCCL_EXEC_CHECK_DISABLE
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int topk_threads_per_block_helper() noexcept
{
  constexpr auto policy = current_policy<PolicySelector>();
  if constexpr (policy.backend == topk_algorithm::baseline)
  {
    return resolve_worker_policy_device<PolicySelector, SegmentSizeParameterT, AgentParamsT...>::policy
      .worker_per_segment_policy.threads_per_block;
  }
  else if constexpr (policy.backend == topk_algorithm::cluster)
  {
    return policy.cluster.threads_per_block;
  }
  else
  {
    // unsupported: harmless positive default; the host never launches this arm.
    return 128;
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <typename PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int topk_min_blocks_per_sm_helper() noexcept
{
  constexpr auto policy = current_policy<PolicySelector>();
  if constexpr (policy.backend == topk_algorithm::cluster)
  {
    return policy.cluster.min_blocks_per_sm;
  }
  else
  {
    // baseline / unsupported: no minimum-blocks constraint.
    return 0;
  }
}

// Third `__launch_bounds__` argument (`maxBlocksPerCluster`): the cluster policy's `max_blocks_per_cluster` cap. The
// host arm launches a dynamic cluster width, so this is the only compile-time width hint `ptxas` sees, and
// `launch_cluster_arm` clamps the launch to `<= max_blocks_per_cluster`. `0` disables the cap.
_CCCL_EXEC_CHECK_DISABLE
template <typename PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int topk_max_blocks_per_cluster_helper() noexcept
{
  constexpr auto policy = current_policy<PolicySelector>();
  if constexpr (policy.backend == topk_algorithm::cluster)
  {
    return policy.cluster.max_blocks_per_cluster;
  }
  else
  {
    // baseline / unsupported: not a cluster launch, so no cluster-width cap.
    return 0;
  }
}

// Variable templates force constant evaluation of the helpers, otherwise nvcc reports a "bad attribute argument
// substitution" error on the `__launch_bounds__` below (same pattern as `transform_kernel`).
template <typename PolicySelector, typename SegmentSizeParameterT, typename... AgentParamsT>
inline constexpr int topk_threads_per_block =
  topk_threads_per_block_helper<PolicySelector, SegmentSizeParameterT, AgentParamsT...>();

template <typename PolicySelector>
inline constexpr int topk_min_blocks_per_sm = topk_min_blocks_per_sm_helper<PolicySelector>();

template <typename PolicySelector>
inline constexpr int topk_max_blocks_per_cluster = topk_max_blocks_per_cluster_helper<PolicySelector>();

// Hands the cluster agent its resolved sub-policy as a type (C++17 has no class-type NTTP).
// TODO(bgruber): drop this in C++20 and pass `policy.cluster` by value.
template <typename PolicySelector>
struct cluster_policy_getter
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()() const
  {
    return current_policy<PolicySelector>().cluster;
  }
};

// -----------------------------------------------------------------------------
// Global kernel entry point (single symbol for both backends)
// -----------------------------------------------------------------------------
// Launch bounds: only `topk_threads_per_block` takes the full kernel type list (its baseline branch runs the
// covering-policy search); min/max-blocks depend on `PolicySelector` alone. The parentheses around
// `topk_threads_per_block<...>` hide its template commas from the fixed-arity `_CCCL_LAUNCH_BOUNDS_CLUSTER(a, b, c)`.
template <typename PolicySelector,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename LargeSegmentTileOffsetT,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak>
_CCCL_LAUNCH_BOUNDS_CLUSTER((topk_threads_per_block<PolicySelector,
                                                    SegmentSizeParameterT,
                                                    KeyInputItItT,
                                                    KeyOutputItItT,
                                                    ValueInputItItT,
                                                    ValueOutputItItT,
                                                    SegmentSizeParameterT,
                                                    KParameterT,
                                                    SelectDirectionParameterT,
                                                    NumSegmentsParameterT,
                                                    LargeSegmentTileOffsetT>),
                            topk_min_blocks_per_sm<PolicySelector>,
                            topk_max_blocks_per_cluster<PolicySelector>) _CCCL_KERNEL_ATTRIBUTES void
device_batched_topk_kernel(
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  baseline_kernel_args<typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type, LargeSegmentTileOffsetT>
    base_args,
  [[maybe_unused]] cluster_kernel_args clus_args)
{
  constexpr auto policy = current_policy<PolicySelector>();

  if constexpr (policy.backend == topk_algorithm::baseline)
  {
    using agent_t = typename resolve_worker_policy_device<
      PolicySelector,
      SegmentSizeParameterT,
      KeyInputItItT,
      KeyOutputItItT,
      ValueInputItItT,
      ValueOutputItItT,
      SegmentSizeParameterT,
      KParameterT,
      SelectDirectionParameterT,
      NumSegmentsParameterT,
      LargeSegmentTileOffsetT>::agent_t;

    // No tile-size-covers-segment-size assertion here: a bound the worker tile cannot cover is a supported
    // configuration, with those segments escalated to the multi-CTA-per-segment kernels. The shared-memory fit is the
    // only hard requirement, and `resolve_worker_policy_device` already asserts a fitting policy exists.
    static_assert(sizeof(typename agent_t::TempStorage) <= max_smem_per_block,
                  "Static shared memory per block must not exceed 48KB limit.");

    __shared__ typename agent_t::TempStorage temp_storage;

    agent_t agent(
      temp_storage,
      d_key_segments_it,
      d_key_segments_out_it,
      d_value_segments_it,
      d_value_segments_out_it,
      segment_sizes,
      k,
      select_directions,
      num_segments,
      base_args.d_counters,
      base_args.d_large_segments_ids,
      base_args.d_large_segments_tile_offsets);

    agent.Process();
  }
  else if constexpr (policy.backend == topk_algorithm::cluster)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (using agent_t = batched_topk_cluster::agent_batched_topk_cluster<
         cluster_policy_getter<PolicySelector>,
         Determinism,
         TieBreak,
         KeyInputItItT,
         KeyOutputItItT,
         ValueInputItItT,
         ValueOutputItItT,
         SegmentSizeParameterT,
         KParameterT,
         SelectDirectionParameterT,
         NumSegmentsParameterT>;

       // A `tune`d override with an oversized static footprint (e.g. a large `bits_per_pass` histogram) fails here
       // rather than as an opaque ptxas error. Only the static footprint is checked: the dynamic block-tile slots may
       // exceed the static shared-memory cap via opt-in.
       static_assert(sizeof(typename agent_t::TempStorage) <= max_smem_per_block,
                     "Static shared memory per block must not exceed 48KB limit.");

       __shared__ typename agent_t::TempStorage temp_storage;
       extern __shared__ char topk_cluster_smem[];
       char* key_slots = topk_cluster_smem;
       // Align the base up to `slot_alignment` (>= load_align) so every bulk-copy destination gets the same
       // `load_align` alignment the gmem sources have (peak TMA throughput on Hopper). The layout reserves
       // `base_padding_bytes` for this.
       {
         ::cuda::std::uint32_t smem32 = __cvta_generic_to_shared(key_slots);
         smem32 = ::cuda::round_up(smem32, static_cast<::cuda::std::uint32_t>(agent_t::slot_alignment));
         asm("" : "+r"(smem32));
         key_slots = static_cast<char*>(__cvta_shared_to_generic(smem32));
       }

       agent_t agent(
         temp_storage,
         d_key_segments_it,
         d_key_segments_out_it,
         d_value_segments_it,
         d_value_segments_out_it,
         segment_sizes,
         k,
         select_directions,
         num_segments,
         key_slots,
         clus_args.max_block_resident_items);

       agent.Process();),
      // Cluster-policy kernels are only ever launched on SM90+, so the sub-SM90 device pass is unreachable at runtime.
      (_CCCL_UNREACHABLE();));
  }
  else
  {
    // topk_algorithm::unsupported: the host arm returns cudaErrorNotSupported before launching, so this never
    // runs.
    return;
  }
}

// Zeroes the per-segment counter array and the per-segment global histogram slabs, which the multi-CTA-per-segment
// passes require to start at 0 (the counters establish `load_from_candidates_buffer == false` at pass 0; the histogram
// slabs are accumulated into with `atomicAdd`).
//
// This replaces two `cudaMemsetAsync` calls with a single kernel for two reasons: it is one device operation instead of
// two on a pipeline whose cost is dominated by operation count, and unlike a memset it can participate in programmatic
// dependent launch, letting the histogram kernel that follows load and bin its tiles into shared memory while this one
// is still running (it only has to wait before merging into the global slabs).
template <int ThreadsPerBlock, typename SegCounterT, typename OffsetT>
__launch_bounds__(ThreadsPerBlock) _CCCL_KERNEL_ATTRIBUTES
  void device_batched_topk_init_kernel(SegCounterT* d_seg_counters,
                                       OffsetT* d_seg_histograms,
                                       ::cuda::std::uint64_t num_counter_words,
                                       ::cuda::std::uint64_t num_histogram_bins)
{
  // The counter array is zeroed as raw 32-bit words rather than through its element type: it is a plain trivially
  // copyable aggregate whose size is a multiple of its 128-byte alignment, so this is both well-defined and lets the
  // two loops share one grid-stride shape.
  static_assert(sizeof(SegCounterT) % sizeof(::cuda::std::uint32_t) == 0,
                "counter type must be zeroable as 32-bit words");
  auto* const counter_words = reinterpret_cast<::cuda::std::uint32_t*>(d_seg_counters);

  const auto tid    = static_cast<::cuda::std::uint64_t>(blockIdx.x) * ThreadsPerBlock + threadIdx.x;
  const auto stride = static_cast<::cuda::std::uint64_t>(gridDim.x) * ThreadsPerBlock;

  for (auto i = tid; i < num_counter_words; i += stride)
  {
    counter_words[i] = ::cuda::std::uint32_t{0};
  }
  for (auto i = tid; i < num_histogram_bins; i += stride)
  {
    d_seg_histograms[i] = OffsetT{0};
  }

  // Release the dependent grid as soon as this CTA's zeroing stores are issued; it has no trailing work.
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
}

//---------------------------------------------------------------------
// Segmented multi-CTA-per-segment top-k kernels.
//
// Three kernels mirror the single-problem `DeviceTopK{Histogram,Filter,LastFilter}Kernel` trio in
// `cub/device/dispatch/dispatch_topk.cuh`: each resolves `multi_worker_per_segment_policy`, lifts it
// into an `AgentTopKPolicy<...>`, instantiates the matching segmented agent, and calls `agent.run(...)`.
// `SelectDirection` is a template NTTP from the host-side `dispatch_discrete`; the kernels are
// otherwise direction-agnostic.
//---------------------------------------------------------------------

namespace topk_seg_kernel_detail
{
// Helper: build an `AgentTopKPolicy<...>` from the multi-worker policy of a given selector.
template <typename PolicySelector>
struct multi_worker_agent_policy_lift
{
  static constexpr topk_policy bp          = current_policy<PolicySelector>();
  static constexpr multi_worker_policy mw = bp.baseline.multi_worker_per_segment_policy;
  using type                              = detail::batched_topk::
    AgentTopKPolicy<mw.threads_per_block, mw.items_per_thread, mw.bits_per_pass, mw.scan_algorithm, mw.keys_tile_load_kind>;
};

// Lift `multi_worker_per_segment_policy.tiles_per_chunk` to a compile-time constant for the
// kernels' grid-stride loop bound. Kept separate from `multi_worker_agent_policy_lift` because
// `AgentTopKPolicy` does not carry this knob -- it is consumed only by the kernels' loop, not the
// agents' smem layouts or template logic.
template <typename PolicySelector>
struct tiles_per_chunk
{
  static constexpr int value = current_policy<PolicySelector>().baseline.multi_worker_per_segment_policy.tiles_per_chunk;
};

// Lift `multi_worker_per_segment_policy.full_tiles_only_histogram` to a compile-time boolean.
// When `true`, the histogram kernel skips the partial-tile path and the finalize-histogram kernel
// loads + bins each segment's trailing partial into the global histogram before the prefix-sum +
// bucket-finder.
template <typename PolicySelector>
struct full_tiles_only_histogram
{
  static constexpr bool value =
    current_policy<PolicySelector>().baseline.multi_worker_per_segment_policy.full_tiles_only_histogram;
};

// Same lift, for the filter kernel. When `true`, the filter kernel skipped the trailing
// partial tile of every segment; the finalize-filter kernel re-injects each segment's
// partial via `agent_batched_topk_filter_partition::process_partial_for_segment` before
// running the prefix-sum + bucket-finder.
template <typename PolicySelector>
struct full_tiles_only_filter
{
  static constexpr bool value = current_policy<PolicySelector>().baseline.multi_worker_per_segment_policy.full_tiles_only_filter;
};
} // namespace topk_seg_kernel_detail

template <typename PolicySelector,
          typename KeyInputItItT,
          typename SegmentSizeParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename LargeSegmentsCountItT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT,
          typename SegmentCountT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().baseline.multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_batched_topk_histogram_kernel(
    const KeyInputItItT d_key_segments_it,
    const SegmentSizeParameterT segment_sizes,
    const SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* const d_large_segments_tile_offsets,
    OffsetT* const d_segment_histograms,
    const LargeSegmentsCountItT large_segments_count_it,
    const ExtractBinOpT extract_bin_op)
{
  // `large_segments_count_it` is either a raw pointer into the mixed-path
  // `batched_topk_counters::large_segments_count` (written by the worker-per-segment kernel's
  // atomicAdd enqueue) or a `transform_iterator` returning the host-known `num_segments_val` for
  // the all-large path; the kernel does not need to know which. The raw iterator + tile-offsets
  // array flow into the agent, which dereferences them inside `run` / `resolve_queue_idx`.
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;

  using agent_t = agent_batched_topk_histogram<
    agent_topk_policy_t,
    KeyInputItItT,
    ExtractBinOpT,
    SegmentSizeParameterT,
    SegmentIdProviderT,
    LargeSegmentTileOffsetT,
    OffsetT,
    LargeSegmentsCountItT,
    SegmentCountT>;

  __shared__ typename agent_t::TempStorage temp_storage;

  agent_t agent(
    temp_storage,
    d_key_segments_it,
    segment_sizes,
    segment_id_provider,
    d_large_segments_tile_offsets,
    d_segment_histograms,
    extract_bin_op,
    large_segments_count_it);

  // The chunk-level grid-stride loop lives inside `agent.run` so per-segment cached state (smem
  // histogram, segment-end bound, segment pointers / scalars) persists across chunks. When a CTA's
  // run stays inside one segment this collapses to one `init_histogram` + one `merge_histogram`,
  // matching the single-problem agent's cost model; multi-segment workloads pay init/merge per
  // (CTA, segment-stretch). `TilesPerChunk` is a compile-time NTTP so the loop bounds and per-CTA
  // stride arithmetic are known at codegen time.
  static constexpr int tiles_per_chunk = topk_seg_kernel_detail::tiles_per_chunk<PolicySelector>::value;
  agent.template run<tiles_per_chunk>();
}

// Per-segment epilogue kernel for the histogram pass. Runs after
// `device_batched_topk_histogram_kernel` finishes (host-side launch ordering on the same stream
// ensures all its CTAs retire first). One CTA per large segment in a grid-strided loop: prefix-sums
// that segment's global histogram, finds the bucket containing the k-th key, updates the per-segment
// counter, and (optionally) zeros the histogram slab for the next pass. Splitting this out keeps the
// per-tile `finalize_pass` cost off the histogram CTAs, at the cost of one extra kernel launch per pass.
template <typename PolicySelector,
          typename KeyInputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentsCountItT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().baseline.multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_batched_topk_finalize_histogram_kernel(
    const KeyInputItItT d_key_segments_it,
    const SegmentSizeParameterT segment_sizes,
    const KParameterT k_param,
    const SegmentIdProviderT segment_id_provider,
    // See the `device_batched_topk_histogram_kernel` doc for why we mark the pointer
    // (not the pointee) grid-constant.
    detail::batched_topk::counter<KeyT, OffsetT, OutOffsetT>* const d_segment_counters,
    OffsetT* const d_segment_histograms,
    const LargeSegmentsCountItT large_segments_count_it,
    const ExtractBinOpT extract_bin_op,
    const int pass,
    const bool reset_histogram)
{
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;
  static constexpr int block_threads    = agent_topk_policy_t::block_threads;
  static constexpr int items_per_thread = agent_topk_policy_t::items_per_thread;
  static constexpr int bits_per_pass    = agent_topk_policy_t::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;

  // Mirrors `topk_seg_kernel_detail::full_tiles_only_histogram<PolicySelector>::value`.
  // When `true`, the companion histogram kernel skipped the trailing partial tile of every
  // segment; this kernel's responsibility, before running the prefix-sum + bucket-finder, is
  // to load + bin that partial tile directly into the segment's global histogram slab.
  static constexpr bool process_partial = topk_seg_kernel_detail::full_tiles_only_histogram<PolicySelector>::value;

  using counter_t                   = detail::batched_topk::counter<KeyT, OffsetT, OutOffsetT>;
  using block_identify_kth_bucket_t = detail::batched_topk::
    block_identify_kth_bucket<block_threads, bits_per_pass, agent_topk_policy_t::scan_algorithm, OffsetT, OutOffsetT>;

  // In `process_partial` mode the trailing partial tile is staged into an smem histogram (primed
  // from the global slab), read into blocked registers, and fed to the bucket-finder from registers
  // -- avoiding the global atomic-adds and the global histogram re-read. The staged histogram is
  // dead before the bucket-finder reuses the smem for its scan, so the two alias via a union. In
  // the default mode only the bucket-finder storage is needed.
  using tile_histogram_t = detail::batched_topk::tile_histogram<block_threads, num_buckets, OffsetT, ExtractBinOpT>;
  union staged_storage_t
  {
    typename tile_histogram_t::TempStorage staged_histogram;
    typename block_identify_kth_bucket_t::TempStorage prefix_sum;
  };
  union plain_storage_t
  {
    typename block_identify_kth_bucket_t::TempStorage prefix_sum;
  };
  __shared__ ::cuda::std::conditional_t<process_partial, staged_storage_t, plain_storage_t> temp_storage;

  const queue_segment_count_t num_large_segments =
    static_cast<queue_segment_count_t>(*large_segments_count_it);

  // Grid-stride loop over queue slots. One CTA owns one segment for the duration of that
  // segment's epilogue; CTAs are independent and write to disjoint counter / histogram slabs.
  using queue_idx_t = queue_segment_count_t;
  for (queue_idx_t queue_idx = static_cast<queue_idx_t>(blockIdx.x); queue_idx < num_large_segments;
       queue_idx += static_cast<queue_idx_t>(gridDim.x))
  {
    const auto segment_id      = segment_id_provider[queue_idx];
    const OffsetT num_items =
      static_cast<OffsetT>(params::__get_and_clamp_param_to_nonnegative(segment_sizes, segment_id));
    counter_t* segment_counter = d_segment_counters + queue_idx;
    OffsetT* segment_histogram = d_segment_histograms + queue_idx * num_buckets;

    // Clip `k` to the segment's input size (the histogram agent's clip comment explains why).
    const OutOffsetT k =
      (::cuda::std::min) (static_cast<OutOffsetT>(params::__get_and_clamp_param_to_nonnegative(k_param, segment_id)),
                          static_cast<OutOffsetT>(num_items));

    // Per-segment counter update + kth-bucket find. `on_kth_bucket` writes the kth bucket's
    // bin index into the counter's `kth_key_bits` for the next pass to consume, and decrements
    // `k` by the count of already-selected items.
    auto on_kth_bucket =
      [segment_counter, pass](OutOffsetT current_k, int bin_index, OffsetT num_selected, OffsetT num_candidates) {
        segment_counter->k                  = static_cast<OutOffsetT>(current_k - num_selected);
        segment_counter->num_candidates_out = num_candidates;
        detail::batched_topk::set_kth_key_bits<bits_per_pass>(
          segment_counter->kth_key_bits, pass, static_cast<unsigned int>(bin_index));
      };

    if (threadIdx.x == 0)
    {
      segment_counter->num_candidates_in      = num_items;
      segment_counter->num_candidates_written = 0;
    }

    if constexpr (process_partial)
    {
      // `process_partial` mode: prime the smem histogram from the global slab (full-tile counts),
      // add the trailing partial tile via fast smem atomics, then read into registers for the
      // bucket-finder.
      tile_histogram_t hist{temp_storage.staged_histogram, extract_bin_op};

      const OffsetT num_full_tiles = num_items / static_cast<OffsetT>(tile_items);
      const OffsetT partial_items  = num_items - num_full_tiles * static_cast<OffsetT>(tile_items);

      // Load the trailing partial tile into registers *before* touching anything the preceding kernel produced. It
      // reads this segment's input keys, which that kernel also only reads, so it carries no dependency -- under PDL
      // this load overlaps with the primary's tail. Ordering it after `load_from` (which reads the global histogram
      // slab the primary fills) would serialize it behind the whole primary instead.
      KeyT items[items_per_thread];
      int num_thread_items = 0;
      if (partial_items > OffsetT{0})
      {
        const auto inner_key_it = d_key_segments_it[segment_id];
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);
        // Blocked per-thread load of the trailing partial tile (matches the histogram agent's
        // `add_partial` arrangement); out-of-range lanes load nothing and are not binned.
        const OffsetT thread_base = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < items_per_thread; ++j)
        {
          const OffsetT idx = thread_base + static_cast<OffsetT>(j);
          if (idx < num_items)
          {
            items[j] = inner_key_it[idx];
          }
        }
        num_thread_items =
          (thread_base >= num_items)
            ? 0
            : static_cast<int>((::cuda::std::min) (static_cast<OffsetT>(items_per_thread), num_items - thread_base));
      }

      // First read of the primary's output.
      _CCCL_PDL_GRID_DEPENDENCY_SYNC();
      hist.load_from(segment_histogram);
      __syncthreads();

      if (partial_items > OffsetT{0})
      {
        hist.add_partial(items, num_thread_items);
        __syncthreads();
      }

      // Drain the staged histogram into this thread's blocked register chunk (direct smem read, no
      // transpose), then run the bucket-finder against registers. The staged-histogram smem aliases
      // the bucket-finder's scan storage, so the read must complete before the scan overwrites it.
      OffsetT thread_histogram[block_identify_kth_bucket_t::bins_per_thread];
      block_identify_kth_bucket_t::load_blocked(hist.data(), thread_histogram);
      __syncthreads();
      block_identify_kth_bucket_t{temp_storage.prefix_sum}.find_kth_bucket(thread_histogram, k, on_kth_bucket);
    }
    else
    {
      // No partial-tile work to hoist here, so the dependency is honored right at the first read.
      _CCCL_PDL_GRID_DEPENDENCY_SYNC();
      block_identify_kth_bucket_t{temp_storage.prefix_sum}.find_kth_bucket(segment_histogram, k, on_kth_bucket);
    }

    if (reset_histogram)
    {
      // Zero the per-segment histogram slab so the next pass starts clean. The two
      // `__syncthreads()` bracket the reset against the kth-bucket primitive's smem reuse and
      // against the next iteration's load.
      __syncthreads();
      detail::batched_topk::init_histogram<block_threads, num_buckets>(segment_histogram);
    }

    // Separate iterations work on independent counter / histogram slabs but share the smem
    // `temp_storage.prefix_sum` arena; barrier between iterations.
    __syncthreads();
  }
}

template <typename PolicySelector,
          detail::topk::select SelectDirection,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename LargeSegmentsCountItT,
          typename DecomposerT,
          typename OffsetT,
          typename OutOffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().baseline.multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_batched_topk_filter_kernel(
    const KeyInputItItT d_key_segments_it,
    const KeyOutputItItT d_key_segments_out_it,
    const ValueInputItItT d_value_segments_it,
    const ValueOutputItItT d_value_segments_out_it,
    const SegmentSizeParameterT segment_sizes,
    const SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* const d_large_segments_tile_offsets,
    // See the `device_batched_topk_histogram_kernel` doc for why we mark the pointer
    // (not the pointee) grid-constant.
    detail::batched_topk::counter<it_value_t<it_value_t<KeyInputItItT>>, OffsetT, OutOffsetT>* const d_segment_counters,
    OffsetT* const d_segment_histograms,
    it_value_t<it_value_t<KeyInputItItT>>* const d_segment_in_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_in_val_buf,
    it_value_t<it_value_t<KeyInputItItT>>* const d_segment_out_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_out_val_buf,
    const OffsetT candidate_buffer_length,
    const OffsetT candidate_buffer_coefficient,
    const LargeSegmentsCountItT large_segments_count_it,
    const int pass,
    const int total_bits,
    const DecomposerT decomposer)
{
  using key_t = it_value_t<it_value_t<KeyInputItItT>>;
  // See the histogram kernel for the rationale behind reading `total_large_tiles` from the
  // sentinel slot and `large_segments_count` through an iterator. Narrowed to `queue_segment_count_t`
  // so the agent's `resolve_queue_idx` `UpperBound` + indexing stay 32-bit when the count fits.
  const queue_segment_count_t num_large_segments =
    static_cast<queue_segment_count_t>(*large_segments_count_it);
  // Pointer to the sentinel slot of the per-segment tile-offset table; the agent dereferences
  // it lazily at the grid-stride loop boundary instead of materialising the value into a
  // long-lived register at kernel entry. See the agent's `run` doc for the register-pressure
  // motivation.
  const LargeSegmentTileOffsetT* const d_total_large_tiles = &d_large_segments_tile_offsets[num_large_segments];
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;

  static constexpr topk_policy bp          = current_policy<PolicySelector>();
  static constexpr multi_worker_policy mw = bp.baseline.multi_worker_per_segment_policy;
  static constexpr bool lazy_value_load   = mw.lazy_value_load;
  static constexpr bool inlined_classify  = mw.inlined_classify;
  using extract_bin_op_t =
    detail::batched_topk::extract_bin_op_t<key_t, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;
  using identify_candidates_op_t = detail::topk::
    identify_candidates_op_t<key_t, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;

  using agent_t = agent_batched_topk_filter_partition<
    agent_topk_policy_t,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    extract_bin_op_t,
    identify_candidates_op_t,
    DecomposerT,
    SegmentSizeParameterT,
    NumSegmentsParameterT,
    SegmentIdProviderT,
    LargeSegmentTileOffsetT,
    OffsetT,
    OutOffsetT,
    lazy_value_load,
    inlined_classify>;

  __shared__ typename agent_t::TempStorage temp_storage;
  const extract_bin_op_t extract_bin_op{pass, total_bits, decomposer};
  // `identify_candidates_op_t` is constructed inside the agent's `run()` after the on-device
  // binary search resolves the per-segment counter (and thus the per-segment `kth_key_bits`
  // pointer). The agent stores `(pass, total_bits, decomposer)` plus the per-segment counter
  // pointer to rebuild it.

  agent_t agent(
    temp_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    segment_id_provider,
    d_large_segments_tile_offsets,
    d_segment_counters,
    d_segment_histograms,
    d_segment_in_key_buf,
    d_segment_in_val_buf,
    d_segment_out_key_buf,
    d_segment_out_val_buf,
    extract_bin_op,
    total_bits,
    decomposer,
    candidate_buffer_length,
    candidate_buffer_coefficient,
    num_large_segments);

  // Grid-stride loop lives inside `agent.run<TilesPerChunk>(pass)`; the kernel materialises the
  // policy's `tiles_per_chunk` knob and hands off. The per-segment epilogue (counter update +
  // prefix-sum + bucket-finder + optional histogram reset) is done by
  // `device_batched_topk_finalize_filter_kernel`, launched on the same stream right after this kernel.
  (void) d_total_large_tiles;
  static constexpr int tiles_per_chunk = topk_seg_kernel_detail::tiles_per_chunk<PolicySelector>::value;
  agent.template run<tiles_per_chunk>(pass);
}

// Per-segment epilogue kernel for the filter pass. Runs after
// `device_batched_topk_filter_kernel` finishes (host-side launch ordering on the same stream).
// One CTA per large segment: prefix-sums the per-segment global histogram (skipping early_stop
// segments), finds the bucket containing the k-th key, updates the per-segment counter, and
// (optionally) zeros the histogram slab for the next pass.
//
// The `early_stop` / `will_buffer` mode discovered by the filter pass per segment is recomputed
// here from the same counter fields the filter agent read at entry; the filter kernel does not
// modify those fields, so the two kernels stay in lock-step without an extra device-side flag.
template <typename PolicySelector,
          detail::topk::select SelectDirection,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename LargeSegmentsCountItT,
          typename DecomposerT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().baseline.multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_batched_topk_finalize_filter_kernel(
    const KeyInputItItT d_key_segments_it,
    const KeyOutputItItT d_key_segments_out_it,
    const ValueInputItItT d_value_segments_it,
    const ValueOutputItItT d_value_segments_out_it,
    const SegmentSizeParameterT segment_sizes,
    const SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* const d_large_segments_tile_offsets,
    // See the `device_batched_topk_histogram_kernel` doc for why we mark the pointer
    // (not the pointee) grid-constant.
    detail::batched_topk::counter<KeyT, OffsetT, OutOffsetT>* const d_segment_counters,
    OffsetT* const d_segment_histograms,
    KeyT* const d_segment_in_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_in_val_buf,
    KeyT* const d_segment_out_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_out_val_buf,
    const LargeSegmentsCountItT large_segments_count_it,
    const OffsetT candidate_buffer_length,
    const OffsetT candidate_buffer_coefficient,
    const int pass,
    const int total_bits,
    const DecomposerT decomposer,
    const bool reset_histogram)
{
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;
  static constexpr int block_threads = agent_topk_policy_t::block_threads;
  static constexpr int bits_per_pass = agent_topk_policy_t::bits_per_pass;
  static constexpr int num_buckets   = 1 << bits_per_pass;

  // Mirrors `topk_seg_kernel_detail::full_tiles_only_filter<PolicySelector>::value`. When
  // `true`, the companion filter kernel skipped the trailing partial tile of every
  // segment; this kernel's responsibility, before running the per-segment prefix-sum +
  // bucket-finder, is to call `agent.process_partial_for_segment(queue_idx, pass)` to
  // re-inject the partial-tile contribution via the appropriate mode-specific partition
  // primitive.
  static constexpr bool process_partial = topk_seg_kernel_detail::full_tiles_only_filter<PolicySelector>::value;

  using counter_t                   = detail::batched_topk::counter<KeyT, OffsetT, OutOffsetT>;
  using block_identify_kth_bucket_t = detail::batched_topk::
    block_identify_kth_bucket<block_threads, bits_per_pass, agent_topk_policy_t::scan_algorithm, OffsetT, OutOffsetT>;

  // Partial-tile processing instantiates the same filter agent the filter kernel uses --
  // same `extract_bin_op_t` / `identify_candidates_op_t`, same partition primitives, same
  // smem layout. Only the entry method differs (`process_partial_for_segment` vs `run`).
  static constexpr topk_policy bp          = current_policy<PolicySelector>();
  static constexpr multi_worker_policy mw = bp.baseline.multi_worker_per_segment_policy;
  static constexpr bool lazy_value_load   = mw.lazy_value_load;
  static constexpr bool inlined_classify  = mw.inlined_classify;

  using extract_bin_op_t =
    detail::batched_topk::extract_bin_op_t<KeyT, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;
  using identify_candidates_op_t = detail::topk::
    identify_candidates_op_t<KeyT, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;

  using filter_agent_t = agent_batched_topk_filter_partition<
    agent_topk_policy_t,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    extract_bin_op_t,
    identify_candidates_op_t,
    DecomposerT,
    SegmentSizeParameterT,
    NumSegmentsParameterT,
    SegmentIdProviderT,
    LargeSegmentTileOffsetT,
    OffsetT,
    OutOffsetT,
    lazy_value_load,
    inlined_classify>;

  // Union the agent's per-tile smem (~9 KB for the buffered arm: smem histogram + keys
  // source state + partition arena) with the prefix-sum scratch. Partial processing
  // touches `agent_storage` first; after `__syncthreads()` the bytes are reused for
  // `prefix_sum`.
  __shared__ union
  {
    typename filter_agent_t::TempStorage agent_storage;
    typename block_identify_kth_bucket_t::TempStorage prefix_sum;
  } temp_storage;

  // The agent's constructor is cheap (member-init of pointers + iterators) so we always
  // build it; ptxas drops the unused args when `process_partial == false`.
  const queue_segment_count_t num_large_segments =
    static_cast<queue_segment_count_t>(*large_segments_count_it);
  const extract_bin_op_t extract_bin_op{pass, total_bits, decomposer};
  filter_agent_t agent{
    temp_storage.agent_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    segment_id_provider,
    d_large_segments_tile_offsets,
    d_segment_counters,
    d_segment_histograms,
    d_segment_in_key_buf,
    d_segment_in_val_buf,
    d_segment_out_key_buf,
    d_segment_out_val_buf,
    extract_bin_op,
    total_bits,
    decomposer,
    candidate_buffer_length,
    candidate_buffer_coefficient,
    num_large_segments};

  using queue_idx_t = queue_segment_count_t;
  for (queue_idx_t queue_idx = static_cast<queue_idx_t>(blockIdx.x); queue_idx < num_large_segments;
       queue_idx += static_cast<queue_idx_t>(gridDim.x))
  {
    const auto segment_id      = segment_id_provider[queue_idx];
    const OffsetT num_items =
      static_cast<OffsetT>(params::__get_and_clamp_param_to_nonnegative(segment_sizes, segment_id));
    counter_t* segment_counter = d_segment_counters + queue_idx;
    OffsetT* segment_histogram = d_segment_histograms + queue_idx * num_buckets;

    // Conservative placement, unlike the histogram finalize above: this kernel's partial-tile work runs inside the
    // filter agent, which touches the counters and the candidate buffers the preceding filter kernel wrote, so there
    // is no independent prologue to hoist above the dependency. Only the launch and this CTA's index resolution
    // overlap with the primary.
    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    const OutOffsetT current_k         = segment_counter->k;
    const OffsetT current_len          = segment_counter->num_candidates_out;
    const OffsetT counter_input_length = segment_counter->num_candidates_in;

    // Skip empty segments (universal early-exit) -- match the filter agent's same check.
    if (counter_input_length == 0)
    {
      __syncthreads();
      continue;
    }

    // Phase 1: trailing partial-tile work (only when the filter kernel skipped it). The
    // agent reads the per-pass counter fields itself; this must run BEFORE the per-mode
    // counter update below so it sees the same `current_k` / `num_candidates_out` /
    // `load_from_candidates_buffer` the filter agent did, and so picks the same mode.
    if constexpr (process_partial)
    {
      __syncthreads();
      agent.process_partial_for_segment(queue_idx, pass);
      __syncthreads();
    }

    // Recompute the mode the filter pass took for this segment, from the same counter fields
    // the filter agent read at entry. Same expressions, same operands, so we stay in lock-step
    // without an extra device-side flag.
    const bool early_stop  = (current_len == static_cast<OffsetT>(current_k));
    const bool will_buffer = !early_stop && (current_len <= candidate_buffer_length)
                          && (current_len <= num_items / candidate_buffer_coefficient);

    // Per-mode counter update:
    //   - early_stop : write `num_candidates_in = 0` (universal early-exit for the next pass).
    //   - buffered   : write `num_candidates_in = current_len`, flip
    //                  `load_from_candidates_buffer` to true, reset `num_candidates_written`.
    //   - unbuffered : no counter writes.
    if (threadIdx.x == 0)
    {
      if (early_stop)
      {
        segment_counter->num_candidates_in = 0;
      }
      else if (will_buffer)
      {
        segment_counter->num_candidates_in           = current_len;
        segment_counter->load_from_candidates_buffer = true;
        segment_counter->num_candidates_written      = 0;
      }
    }

    // For early_stop the histogram is meaningless (the agent did not touch it). Skip the
    // prefix-sum + bucket-finder entirely. For buffered / unbuffered, both branches updated
    // the per-segment global histogram, so both run the kth-bucket scan.
    if (!early_stop)
    {
      auto on_kth_bucket =
        [segment_counter, pass](OutOffsetT k_cb, int bin_index, OffsetT num_selected, OffsetT num_candidates) {
          segment_counter->k                  = static_cast<OutOffsetT>(k_cb - num_selected);
          segment_counter->num_candidates_out = num_candidates;
          detail::batched_topk::set_kth_key_bits<bits_per_pass>(
            segment_counter->kth_key_bits, pass, static_cast<unsigned int>(bin_index));
        };

      __syncthreads();
      block_identify_kth_bucket_t{temp_storage.prefix_sum}.find_kth_bucket(segment_histogram, current_k, on_kth_bucket);

      if (reset_histogram)
      {
        __syncthreads();
        detail::batched_topk::init_histogram<block_threads, num_buckets>(segment_histogram);
      }
    }

    // Separate iterations work on independent counter / histogram slabs but share the smem
    // `temp_storage.prefix_sum` arena; barrier between iterations.
    __syncthreads();
  }
}

template <typename PolicySelector,
          detail::topk::select SelectDirection,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT,
          typename SegmentIdProviderT,
          typename LargeSegmentTileOffsetT,
          typename LargeSegmentsCountItT,
          typename DecomposerT,
          typename OffsetT,
          typename OutOffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().baseline.multi_worker_per_segment_policy.threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void device_batched_topk_last_filter_kernel(
    const KeyInputItItT d_key_segments_it,
    const KeyOutputItItT d_key_segments_out_it,
    const ValueInputItItT d_value_segments_it,
    const ValueOutputItItT d_value_segments_out_it,
    const SegmentSizeParameterT segment_sizes,
    const KParameterT k_param,
    const SegmentIdProviderT segment_id_provider,
    const LargeSegmentTileOffsetT* const d_large_segments_tile_offsets,
    // See the `device_batched_topk_histogram_kernel` doc for why we mark the pointer
    // (not the pointee) grid-constant.
    detail::batched_topk::counter<it_value_t<it_value_t<KeyInputItItT>>, OffsetT, OutOffsetT>* const d_segment_counters,
    it_value_t<it_value_t<KeyInputItItT>>* const d_segment_in_key_buf,
    it_value_t<it_value_t<ValueInputItItT>>* const d_segment_in_val_buf,
    const OffsetT candidate_buffer_length,
    const LargeSegmentsCountItT large_segments_count_it,
    const int pass,
    const int total_bits,
    const DecomposerT decomposer)
{
  using key_t = it_value_t<it_value_t<KeyInputItItT>>;
  // Materialise the queue-shape `num_large_segments` so the agent can hold it as a member
  // (the agent re-derives `d_total_large_tiles` from `d_large_segments_tile_offsets +
  // num_large_segments` itself on entry to `run`). Narrowed to `queue_segment_count_t` to keep the
  // agent's `resolve_queue_idx` `UpperBound` + indexing 32-bit when the count fits.
  const queue_segment_count_t num_large_segments =
    static_cast<queue_segment_count_t>(*large_segments_count_it);
  using agent_topk_policy_t = typename topk_seg_kernel_detail::multi_worker_agent_policy_lift<PolicySelector>::type;

  static constexpr topk_policy bp          = current_policy<PolicySelector>();
  static constexpr multi_worker_policy mw = bp.baseline.multi_worker_per_segment_policy;
  static constexpr bool lazy_value_load   = mw.lazy_value_load;
  static constexpr bool inlined_classify  = mw.inlined_classify;

  using identify_candidates_op_t = detail::topk::
    identify_candidates_op_t<key_t, SelectDirection, agent_topk_policy_t::bits_per_pass, DecomposerT>;

  using agent_t = agent_batched_topk_last_filter<
    agent_topk_policy_t,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    identify_candidates_op_t,
    DecomposerT,
    SegmentSizeParameterT,
    KParameterT,
    NumSegmentsParameterT,
    SegmentIdProviderT,
    LargeSegmentTileOffsetT,
    OffsetT,
    OutOffsetT,
    lazy_value_load,
    inlined_classify>;

  __shared__ typename agent_t::TempStorage temp_storage;

  agent_t agent(
    temp_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    k_param,
    segment_id_provider,
    d_large_segments_tile_offsets,
    d_segment_counters,
    d_segment_in_key_buf,
    d_segment_in_val_buf,
    pass,
    total_bits,
    decomposer,
    candidate_buffer_length,
    num_large_segments);

  // Grid-stride loop lives inside `agent.run<TilesPerChunk>()`; the kernel materialises the policy's
  // `tiles_per_chunk` knob and hands off.
  static constexpr int tiles_per_chunk = topk_seg_kernel_detail::tiles_per_chunk<PolicySelector>::value;
  agent.template run<tiles_per_chunk>();
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
