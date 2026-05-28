// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in device-wide topK.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_topk_common.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/detail/topk/block_filter.cuh>
#include <cub/detail/topk/block_filter_accumulating.cuh>
#include <cub/detail/topk/block_filter_speculative.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/detail/topk/block_partition_accumulating.cuh>
#include <cub/detail/topk/partition_storage_layout.cuh>
#include <cub/detail/topk/block_partition_speculative.cuh>
#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
//---------------------------------------------------------------------
// AgentTopKHistogram: histogram agent shared by pass 0 and the unbuffered
// filter passes. Owns _TempStorage, instantiates a TileDataSource for the
// keys stream (per `AgentTopKPolicyT::keys_tile_load_kind`), and runs the
// §4.3 tile loop with full/partial dispatch into `process_tile`.
//
// Each input key is gated by the `FilterOpT` predicate before contributing
// to the histogram. The pass-0 caller leaves the default
// `topk_pass_through_filter_op` in place; the unbuffered filter caller
// supplies a `topk_candidate_filter_op` that wraps its
// `identify_candidates_op`.
//
// Single-source invariant for the unbuffered usage: the unbuffered filter
// branch is reached only when the surviving candidate count exceeds the
// back-buffer capacity (`current_len > buffer_length`). The candidate set is
// monotonically non-increasing across passes, so once an unbuffered pass is
// taken, every prior pass was also unbuffered, which means no candidate has
// ever been written to `in_key_buf`. Therefore the unbuffered path always
// loads from the original `d_keys_in`, and a single `tile_data_source_t<
// KeyInputIteratorT, ...>` is sufficient -- we don't need the
// `multi_source_data_source` machinery the partition agent uses. If we ever
// allow unbuffered passes to follow buffered ones (e.g. an adaptive
// strategy that pauses output writes mid-decode), this agent will need to
// be re-templated on the keys data source so the caller can hand in a
// multi_source over `(d_keys_in, in_key_buf)`.
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT,
          typename FilterOpT = topk_pass_through_filter_op>
struct AgentTopKHistogram
{
  using key_in_t = it_value_t<KeyInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;

  // The tile data source
  using keys_source_t =
  tile_data_source_t<KeyInputIteratorT, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;

  // Block-level primitive that takes the global histogram and identifies bin that the k-th item falls into
  using block_identify_kth_bucket_t =
    block_identify_kth_bucket<block_threads, bits_per_pass, AgentTopKPolicyT::scan_algorithm, OffsetT, OutOffsetT>;

  // Smem reuse plan. The agent's smem accesses split into three temporally
  // disjoint phases, separated by the `__syncthreads()` / `__threadfence()` /
  // `__syncthreads_or` chain that bridges the tile loop and `finalize_pass`:
  //
  //   Phase 1 -- tile loop. Concurrently uses `histogram` (atomicAdd target),
  //              `keys_source_state` (persistent data-source state, e.g. the
  //              TMA mbarrier), and `keys_source_scratch` (BlockLoad / TMA
  //              staging buffer).
  //   Phase 2 -- `merge_histogram`. Reads `histogram` only. The post-loop
  //              `__syncthreads()` makes phase-1 atomicAdds visible; the
  //              load-state / load-scratch are unused from here on.
  //   Phase 3 -- `block_identify_kth_bucket` on the last block only. Uses
  //              `prefix_sum` exclusively. The `__threadfence()` +
  //              `__syncthreads_or` in `finalize_pass` separates phase 2
  //              from phase 3, so the phase-1+2 storage is dead by then.
  //
  // We fold phases 1+2 into one named arm of an outer union and put
  // `prefix_sum` in the other. The total smem requirement becomes
  // `max(H + S + C, P)` (with `H` = histogram, `S` = keys_source_state,
  // `C` = keys_source_scratch, `P` = prefix_sum) instead of
  // `H + S + max(C, P)`. The refactor never increases the total (each arm is
  // a subset of the original linear layout) and saves up to `H + S` bytes per
  // block when `P` dominates `C` -- the common case for the default
  // `block_load_vectorize` policy, where `C` is empty.
  // Types-in-anonymous-unions are not allowed, so the phase-1+2 group is
  // declared at struct scope here and aliased through the anonymous union below.
  struct phase_load_t
  {
    OffsetT histogram[num_buckets]; // phases 1 + 2
    typename keys_source_t::TempStorage keys_source_state; // phase 1
    typename keys_source_t::ScratchStorage keys_source_scratch; // phase 1
  };

  struct _TempStorage
  {
    union
    {
      phase_load_t phase_load;
      typename block_identify_kth_bucket_t::TempStorage prefix_sum; // phase 3 (last block only)
    };
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& temp_storage;
  KeyInputIteratorT d_keys_in;
  OffsetT num_items;
  OutOffsetT k;
  ExtractBinOpT extract_bin_op;
  FilterOpT filter_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentTopKHistogram(
    TempStorage& ts,
    const KeyInputIteratorT d_keys_in,
    OffsetT num_items,
    OutOffsetT k,
    ExtractBinOpT extract_bin_op,
    FilterOpT filter_op = {})
      : temp_storage(ts.Alias())
      , d_keys_in(d_keys_in)
      , num_items(num_items)
      , k(k)
      , extract_bin_op(extract_bin_op)
      , filter_op(filter_op)
  {}

  // process_tile: filter + classify + per-bucket atomicAdd into smem histogram.
  // The full overload omits the per-item bound check; the partial overload
  // bound-checks against `num_thread_items`. The default `FilterOpT` returns a
  // constexpr `true`, so the predicate call folds away and the pass-0 inner
  // loop reduces to the original `extract_bin_op + atomicAdd` sequence.
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile_full(const key_in_t (&items)[items_per_thread])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      if (filter_op(items[j]))
      {
        const int bucket = extract_bin_op(items[j]);
        atomicAdd(temp_storage.phase_load.histogram + bucket, OffsetT{1});
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  process_tile_partial(const key_in_t (&items)[items_per_thread], int num_thread_items)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      if (j < num_thread_items && filter_op(items[j]))
      {
        const int bucket = extract_bin_op(items[j]);
        atomicAdd(temp_storage.phase_load.histogram + bucket, OffsetT{1});
      }
    }
  }

  // Both callbacks run only on the last finishing block:
  //   `counter_update_fn` runs on thread 0 (before the prefix-sum / bucket
  //     selection). The pass-0 caller passes a lambda that resets
  //     `num_candidates_in`/`num_candidates_written`; the unbuffered caller passes its
  //     mode-dependent counter update (early-stop vs non-early-stop logic).
  //   `on_kth_bucket`     runs on the single thread that owns the bucket
  //     containing the kth element. See `block_identify_kth_bucket::find_kth_bucket` for the
  //     callback signature.
  template <typename CounterUpdateFn, typename KthBucketFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  invoke(unsigned int* retired_block_counter,
         OffsetT* global_histogram,
         bool reset_histogram,
         CounterUpdateFn counter_update_fn,
         KthBucketFn on_kth_bucket)
  {
    init_histogram<block_threads, num_buckets>(temp_storage.phase_load.histogram);
    __syncthreads();

    keys_source_t keys_source{d_keys_in, temp_storage.phase_load.keys_source_state};

    // Split the tile space into full tiles and a possible single trailing
    // partial tile. The hot-path loop below processes only full tiles. The
    // partial tile, if any, is handled exactly once after the loop by the
    // unique block that would have owned it under the original grid-strided
    // schedule.
    const OffsetT num_full_tiles = num_items / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items  = num_items - num_full_tiles * static_cast<OffsetT>(tile_items);

    // --- full-tile loop --------------------------------------------------
    for (OffsetT tile_id = static_cast<OffsetT>(blockIdx.x); tile_id < num_full_tiles;
         tile_id += static_cast<OffsetT>(gridDim.x))
    {
      const OffsetT tile_base = tile_id * static_cast<OffsetT>(tile_items);

      keys_source.set_tile_base(tile_base);

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(temp_storage.phase_load.keys_source_scratch);
      h.complete_load(items);
      process_tile_full(items);
    }

    // --- trailing partial tile (handled by exactly one block) ------------
    //
    // Under the original grid-strided schedule the partial tile (tile_id =
    // num_full_tiles) was claimed by the block whose blockIdx.x equals
    // num_full_tiles % gridDim.x. We preserve that ownership.
    if (partial_items > 0)
    {
      const unsigned partial_owner = static_cast<unsigned>(num_full_tiles % static_cast<OffsetT>(gridDim.x));
      if (blockIdx.x == partial_owner)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);

        keys_source.set_tile_base(tile_base);

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(temp_storage.phase_load.keys_source_scratch, partial_items);
        h.complete_load(items);
        const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
        const int num_thread_items =
          (thread_offset >= num_items)
            ? 0
            : static_cast<int>((::cuda::std::min) (static_cast<OffsetT>(items_per_thread), num_items - thread_offset));
        process_tile_partial(items, num_thread_items);
      }
    }

    __syncthreads();
    merge_histogram<block_threads, num_buckets>(temp_storage.phase_load.histogram, global_histogram);

    // Last-block epilogue: per-mode counter update on thread 0, then the
    // prefix-sum + bucket selection whose `on_kth_bucket` callback writes the
    // discovered next-pass inputs into `counter`, then optional histogram
    // reset for the next pass.
    //
    // The `__threadfence()` + `__syncthreads_or` inside `finalize_pass` is the
    // boundary that switches the outer union's active arm from `phase_load`
    // (only alive for `merge_histogram`'s smem reads above) to `prefix_sum`.
    // After that boundary, no thread in this block reads
    // `temp_storage.phase_load.*` again, and the other (non-last) blocks have
    // already exited.
    auto epilogue_op = [&] {
      if (threadIdx.x == 0)
      {
        counter_update_fn();
      }
      block_identify_kth_bucket_t{temp_storage.prefix_sum}.find_kth_bucket(global_histogram, k, on_kth_bucket);
      if (reset_histogram)
      {
        // TODO (elstehle): We could skip this reset when we detect an early-stop condition. However, it would require a block-wide broadcast. This short-circuit needs to be evaluated experimentally. 
        init_histogram<block_threads, num_buckets>(global_histogram);
      }
    };

    finalize_pass(retired_block_counter, gridDim.x, epilogue_op);
  }
};


//---------------------------------------------------------------------
// agent_topk_filter_partition: agent for passes 1..num_passes-1 (the
// "filter passes" between the histogram pass and the dedicated last-filter
// pass).
//
// Handles BOTH `sink_mode::early_stop` and `sink_mode::buffered` in one
// agent type. The agent's `run()` takes a runtime `sink_mode mode` arg and
// dispatches over it with a two-way `if`:
//   - early_stop  -> constructs `BlockFilter[Accumulating]` (single-stream
//                    "filter" primitive) with sinks + identify-selected op
//                    captured at ctor.
//   - buffered    -> constructs `BlockPartition[Accumulating]` (two-stream
//                    "partition" primitive) with sinks + identify op +
//                    histogram callback captured at ctor.
// The mode-agnostic `drive_tile_loop<Primitive, StorageLayout>` helper then
// iterates the tile space and finishes with `primitive.epilogue()`. The
// "scout" mode that only builds a histogram is handled separately by
// `AgentTopKHistogram` driven by a candidate-filter predicate.
//
// Members hold the inputs/outputs that both modes consume; the buffered-only
// candidate outputs (`out_key_buf`, `out_val_buf`, `p_num_candidates_written`)
// are passed to `run()` rather than being stored on the agent.
//---------------------------------------------------------------------


template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          typename OffsetT,
          typename OutOffsetT,
          block_partition_strategy BufferedPartStrat = block_partition_strategy::atomics,
          block_filter_strategy EarlyStopFilterStrat = block_filter_strategy::atomics,
          bool LazyValueLoad = false,
          bool InlinedClassify             = false>
struct agent_topk_filter_partition
{
  using key_in_t   = it_value_t<KeyInputIteratorT>;
  using value_in_t = it_value_t<ValueInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  // The epilogue primitive used to find the k-th bucket to conclude the histogram stage
  using block_identify_kth_bucket_t =
    block_identify_kth_bucket<block_threads, bits_per_pass, AgentTopKPolicyT::scan_algorithm, OffsetT, OutOffsetT>;

  // Lazy value-load only matters when there's a value channel to load lazily.
  static constexpr bool effective_lazy_value_load = LazyValueLoad && !keys_only;

  // Keys data source: multi_source over (d_keys_in source, in_key_buf source).
  using key_source_input_t =
    tile_data_source_t<KeyInputIteratorT, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using key_source_buffer_t =
    tile_data_source_t<key_in_t*, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using keys_source_t = multi_source_data_source<key_source_input_t, key_source_buffer_t, OffsetT>;

  // Value channels: multi_source over (d_values_in, in_val_buf)
  using value_source_input_t = direct_data_source<ValueInputIteratorT, block_threads, items_per_thread, OffsetT>;
  using value_source_buffer_t = direct_data_source<value_in_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t = multi_source_data_source<value_source_input_t, value_source_buffer_t, OffsetT>;

  using val_out_t = ValueOutputIteratorT;
  // Buffered mode: the candidate iterators are `key_in_t*` / `value_in_t*` (the
  // back buffers). Early-stop mode has only the selected stream.
  using buffered_cand_val_out_t = value_in_t*;
  using buffered_cand_key_out_t = key_in_t*;

  // Buffered-mode sinks: 2-stream `value_channel_sinks_t` (selected + candidate).
  // `NullType` for keys-only; the block primitive captures the reference but
  // never reads through it under `keys_only`.
  using buffered_value_channel_sinks_concrete_t = value_channel_sinks_t<val_out_t, buffered_cand_val_out_t>;
  using buffered_value_channel_sinks_t =
    ::cuda::std::conditional_t<keys_only, NullType, buffered_value_channel_sinks_concrete_t>;

  // Early-stop sinks: 1-stream `value_channel_sinks_filter_t` (selected only).
  using early_stop_value_channel_sinks_concrete_t = value_channel_sinks_filter_t<val_out_t>;
  using early_stop_value_channel_sinks_t =
    ::cuda::std::conditional_t<keys_only, NullType, early_stop_value_channel_sinks_concrete_t>;

  // Value-channel type fed to the block primitive as `ValueT`. `NullType`
  // selects the keys-only paths inside the primitives (via their internal
  // `keys_only` constexpr).
  using agent_value_t = ::cuda::std::conditional_t<keys_only, NullType, value_in_t>;

  // Per-channel data-source `ScratchStorage`, supplied to the partition class so
  // it can size its `load` slot in the Staged / SharedMem scratch. `NullType`
  // collapses to a no-op slot in keys-only mode.
  using agent_value_data_source_scratch_t =
    ::cuda::std::conditional_t<keys_only, NullType, typename value_source_t::ScratchStorage>;

    // Offset types used to index into the selected and candidate iterators (and counter updates)
  using selected_offset_t  = OutOffsetT;
  using candidate_offset_t = OffsetT;

  // Reserve op types.
  using selected_reserve_op_t  = atomic_reserve_range_op<selected_offset_t>;
  using candidate_reserve_op_t = atomic_reserve_range_op<candidate_offset_t>;

  // Key-output transforms are always identity for top-k.

  // Callback / classify-hook types for the two modes.
  using histogram_callback_op_t = topk_histogram_callback_op<ExtractBinOpT, OffsetT>;

  // Specialized for early_stop: Any non-rejected item goes to the user-provided output iterator
  using identify_selected_op_t  = topk_identify_selected_op<IdentifyCandidatesOpT>;

  // The buffered-mode primitive (`BlockPartition`,
  // `block_partition_accumulating_candidates`, or `block_partition_speculative`).
  using buffered_partition_t = strategy_to_partition_class_t<
    BufferedPartStrat,
    block_threads,
    items_per_thread,
    AgentTopKPolicyT::accumulating_buffer_capacity,
    AgentTopKPolicyT::speculative_selected_buffer_capacity,
    key_in_t,
    selected_offset_t,
    candidate_offset_t,
    selected_reserve_op_t,
    candidate_reserve_op_t,
    KeyOutputIteratorT,
    buffered_cand_key_out_t,
    IdentifyCandidatesOpT,
    histogram_callback_op_t,
    buffered_value_channel_sinks_t,
    agent_value_t,
    agent_value_data_source_scratch_t,
    effective_lazy_value_load,
    InlinedClassify>;

  // The early-stop-mode primitive (`BlockFilter` or `block_filter_accumulating`).
  using early_stop_filter_t = strategy_to_filter_class_t<
    EarlyStopFilterStrat,
    block_threads,
    items_per_thread,
    AgentTopKPolicyT::accumulating_buffer_capacity,
    key_in_t,
    selected_offset_t,
    selected_reserve_op_t,
    KeyOutputIteratorT,
    identify_selected_op_t,
    early_stop_value_channel_sinks_t,
    agent_value_t,
    agent_value_data_source_scratch_t,
    effective_lazy_value_load,
    InlinedClassify>;

  // Per-mode storage layouts. Both expose the same `get_*()` accessors so the
  // mode-agnostic `drive_tile_loop` helper is layout-agnostic. Neither layout
  // reserves space for the kth-bucket prefix-sum scratch internally: `prefix_sum`
  // is hoisted out into the outer `_TempStorage` union below so it can alias
  // with the (much larger) phase-1+2 footprint. Passing `empty_prefix_sum_t`
  // as the `PrefixSumT` collapses the corresponding slot in each layout to a
  // 1-byte placeholder; the per-mode `get_prefix_sum()` accessors still
  // compile but are never called by this agent.
  struct empty_prefix_sum_t
  {};

  using buffered_storage_layout_t = partition_storage_layout_for_t<
    buffered_partition_t,
    typename keys_source_t::ScratchStorage,
    empty_prefix_sum_t>;

  using early_stop_storage_layout_t = partition_storage_layout_for_t<
    early_stop_filter_t,
    typename keys_source_t::ScratchStorage,
    empty_prefix_sum_t>;

  // Smem reuse plan. The agent's smem accesses split into temporally disjoint
  // arms separated by `__syncthreads()` / `__threadfence()` / `__syncthreads_or`
  // chains:
  //
  //   `sink_mode::buffered`:
  //     Phase 1+2 (every block) -- `histogram` (atomicAdd target, then merge
  //                                 source), `keys_source_state` (persistent
  //                                 data-source state), and the buffered
  //                                 `arena` (load + partition scratch, plus
  //                                 the partition's persistent state on the
  //                                 accumulating strategies).
  //     Phase 3   (last block)  -- `prefix_sum` only. The
  //                                 `__threadfence()` + `__syncthreads_or` in
  //                                 `finalize_pass` separates phase 2 from
  //                                 phase 3, so the phase-1+2 storage is
  //                                 dead by then.
  //
  //   `sink_mode::early_stop`:
  //     Whole pass              -- `keys_source_state` + early-stop `arena`.
  //                                 `histogram` and `prefix_sum` are unused.
  //
  // We place these three lifetime arms in an outer union. The previous layout
  // already aliased `partition_state` <-> `prefix_sum` (persistent-partition
  // case) and `keys_source_scratch` <-> `prefix_sum` (non-persistent case)
  // INSIDE the per-mode arena. Hoisting `prefix_sum` to the outer union
  // recovers those savings via the wider phase-1+2 buffered arm and
  // additionally frees `histogram + keys_source_state` on configurations
  // where `prefix_sum` dominates the load+partition scratch (the common case
  // for the default `block_load_vectorize` policy with the non-accumulating
  // partition strategy). The refactor never increases the total smem
  // requirement (each arm is a subset of the original linear layout).
  //
  // Both inner layouts have non-trivial special members (their inner phase
  // unions carry user-defined ctor/dtor), so the wrapping union itself needs
  // explicit ctor/dtor.
  struct _TempStorage
  {
    union arms_t
    {
      struct buffered_t
      {
        OffsetT histogram[num_buckets]; // phases 1 + 2
        // Per-child persistent state. `keys_source_t` (multi-source) does
        // not publish a `TempStorage`; the agent holds one `TempStorage` per
        // child source. See `tile_data_source.cuh::multi_source_data_source`.
        typename key_source_input_t::TempStorage key_src_input_state; // phase 1
        typename key_source_buffer_t::TempStorage key_src_buffer_state; // phase 1
        buffered_storage_layout_t arena; // phase 1
      } buffered;

      typename block_identify_kth_bucket_t::TempStorage prefix_sum; // phase 3 (last block only)

      struct early_stop_t
      {
        typename key_source_input_t::TempStorage key_src_input_state;
        typename key_source_buffer_t::TempStorage key_src_buffer_state;
        early_stop_storage_layout_t arena;
      } early_stop;
    } arms;
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& storage;

  // Inputs/outputs shared across both modes.
  KeyInputIteratorT d_keys_in;
  KeyOutputIteratorT d_keys_out;
  ValueInputIteratorT d_values_in;
  ValueOutputIteratorT d_values_out;
  key_in_t* in_key_buf;
  value_in_t* in_val_buf;

  OutOffsetT* p_num_selected_written;

  OffsetT input_length;
  bool load_from_candidates_buffer;

  ExtractBinOpT extract_bin_op;
  IdentifyCandidatesOpT identify_candidates_op;
  OffsetT* global_histogram;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_topk_filter_partition(
    TempStorage& temp_storage,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    key_in_t* in_key_buf,
    value_in_t* in_val_buf,
    OutOffsetT* p_num_selected_written,
    OffsetT input_length,
    bool load_from_candidates_buffer,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op,
    OffsetT* global_histogram)
      : storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , in_key_buf(in_key_buf)
      , in_val_buf(in_val_buf)
      , p_num_selected_written(p_num_selected_written)
      , input_length(input_length)
      , load_from_candidates_buffer(load_from_candidates_buffer)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
      , global_histogram(global_histogram)
  {}

private:
  // Per-mode sinks factories. Each returns a single sink (with values) or a
  // `NullType` placeholder (keys-only). The result is captured by reference at
  // the block primitive's ctor; under keys-only the primitive holds the
  // reference but never reads through it.
  _CCCL_DEVICE _CCCL_FORCEINLINE auto
  make_buffered_value_channel_sinks([[maybe_unused]] buffered_cand_val_out_t cand_val_out)
  {
    if constexpr (keys_only)
    {
      return NullType{};
    }
    else
    {
      return buffered_value_channel_sinks_concrete_t{d_values_out, cand_val_out};
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_early_stop_value_channel_sinks()
  {
    if constexpr (keys_only)
    {
      return NullType{};
    }
    else
    {
      return early_stop_value_channel_sinks_concrete_t{d_values_out};
    }
  }

  // Single mode-agnostic helper that iterates the tile space: full-tile loop +
  // possible trailing partial tile, both calling `primitive.partition(...)` with
  // just `(scratch, keys, [num_items,] value_source)`. Finishes with
  // `primitive.epilogue()`. The `arena` parameter is one of the two mode-specific
  // storage layouts; both expose the same `get_*()` accessors.
  //
  // The value-source children are declared at the helper's outer scope so
  // they outlive the per-tile `value_source` multi-source (which borrows
  // references to them). For `keys_only`, the locals are unused (`value_source`
  // is `NullType{}`) but their construction is trivial.
  template <typename Primitive, typename StorageLayout>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  drive_tile_loop(Primitive& primitive, StorageLayout& arena, keys_source_t& keys_source)
  {
    const OffsetT num_full_tiles = input_length / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items  = input_length - num_full_tiles * static_cast<OffsetT>(tile_items);

    [[maybe_unused]] typename value_source_input_t::TempStorage val_state_input{};
    [[maybe_unused]] typename value_source_buffer_t::TempStorage val_state_buffer{};
    [[maybe_unused]] value_source_input_t val_input{d_values_in, val_state_input};
    [[maybe_unused]] value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};

    // Full-tile loop iterations
    for (OffsetT tile_id = static_cast<OffsetT>(blockIdx.x); tile_id < num_full_tiles;
         tile_id += static_cast<OffsetT>(gridDim.x))
    {
      const OffsetT tile_base = tile_id * static_cast<OffsetT>(tile_items);

      keys_source.set_tile_base(tile_base);
      // Lambda returns a prvalue of `value_source_t` (or `NullType`).
      // Mandatory copy elision constructs `value_source` directly in the
      // outer slot; the multi-source's references bind to the *outer*
      // `val_input` / `val_buffer` declared above (loop-invariant lifetime).
      auto value_source = [&] {
        if constexpr (keys_only)
        {
          return NullType{};
        }
        else
        {
          return value_source_t{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
        }
      }();
      if constexpr (!keys_only)
      {
        value_source.set_tile_base(tile_base);
      }

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(arena.get_keys_source_scratch());
      h.complete_load(items);
      __syncthreads();

      primitive.partition(arena.get_partition_scratch(), items, value_source);
    }

    // Trailing partial tile (handled by exactly one block)
    if (partial_items > 0)
    {
      const unsigned partial_owner = static_cast<unsigned>(num_full_tiles % static_cast<OffsetT>(gridDim.x));
      if (blockIdx.x == partial_owner)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);

        keys_source.set_tile_base(tile_base);
        auto value_source = [&] {
          if constexpr (keys_only)
          {
            return NullType{};
          }
          else
          {
            return value_source_t{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
          }
        }();
        if constexpr (!keys_only)
        {
          value_source.set_tile_base(tile_base);
        }

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(arena.get_keys_source_scratch(), partial_items);
        h.complete_load(items);
        __syncthreads();

        primitive.partition(arena.get_partition_scratch(), items, partial_items, value_source);
      }
    }

    // Terminal flush of any remaining data that was kept on-chip until this point.
    primitive.epilogue();
  }

public:
  // --- entry point ----------------------------------------------------
  //
  // The buffered-mode-only outputs (`out_key_buf`, `out_val_buf`,
  // `p_num_candidates_written`) flow through the run signature; the early_stop
  // branch ignores them.
  //
  // `counter_update_fn` and `on_kth_bucket` are the two last-block-only
  // callbacks for `counter` writes; see the same-named docs on
  // `AgentTopKHistogram::invoke` and `block_identify_kth_bucket::find_kth_bucket`.
  template <typename CounterUpdateFn, typename KthBucketFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  run(unsigned int* retired_block_counter,
      OutOffsetT current_k,
      bool reset_histogram,
      CounterUpdateFn counter_update_fn,
      KthBucketFn on_kth_bucket,
      sink_mode mode,
      key_in_t* out_key_buf             = nullptr,
      value_in_t* out_val_buf           = nullptr,
      OffsetT* p_num_candidates_written = nullptr)
  {
    // Mode-shared stack-locals.
    selected_reserve_op_t reserve_sel{p_num_selected_written};

    if (mode == sink_mode::early_stop)
    {
      // Early-stop branch: no histogram is accumulated, no kth-bucket scan runs.
      // The filter primitive's `partition()` writes selected items directly to `d_keys_out` via `reserve_sel`.
      //
      // `keys_source_state` lives in this arm (so it can alias with the
      // buffered arm's footprint), hence the keys-source is constructed here
      // rather than at function scope.
      key_source_input_t key_src_input{d_keys_in, storage.arms.early_stop.key_src_input_state};
      key_source_buffer_t key_src_buffer{in_key_buf, storage.arms.early_stop.key_src_buffer_state};
      keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/load_from_candidates_buffer};

      identify_selected_op_t identify_selected{identify_candidates_op};
      auto value_channel_sinks = make_early_stop_value_channel_sinks();

      early_stop_filter_t filter{
        storage.arms.early_stop.arena.get_partition_state(),
        reserve_sel,
        d_keys_out,
        value_channel_sinks,
        identify_selected};

      drive_tile_loop(filter, storage.arms.early_stop.arena, keys_source);
    }
    else
    {
      // Buffered branch: accumulate a histogram over candidates, scatter
      // selected to `d_keys_out` and candidates to `out_key_buf`.
      key_source_input_t key_src_input{d_keys_in, storage.arms.buffered.key_src_input_state};
      key_source_buffer_t key_src_buffer{in_key_buf, storage.arms.buffered.key_src_buffer_state};
      keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/load_from_candidates_buffer};

      init_histogram<block_threads, num_buckets>(storage.arms.buffered.histogram);
      __syncthreads();

      buffered_cand_key_out_t cand_key_out = out_key_buf;
      buffered_cand_val_out_t cand_val_out = out_val_buf;
      candidate_reserve_op_t reserve_cand{p_num_candidates_written};
      histogram_callback_op_t histogram_cb{extract_bin_op, storage.arms.buffered.histogram};
      auto value_channel_sinks = make_buffered_value_channel_sinks(cand_val_out);

      buffered_partition_t partition{
        storage.arms.buffered.arena.get_partition_state(),
        reserve_sel,
        reserve_cand,
        d_keys_out,
        cand_key_out,
        value_channel_sinks,
        identify_candidates_op,
        histogram_cb};

      drive_tile_loop(partition, storage.arms.buffered.arena, keys_source);

      __syncthreads();
      merge_histogram<block_threads, num_buckets>(storage.arms.buffered.histogram, global_histogram);
    }

    // Last-block epilogue: per-mode counter update on thread 0, then the kth-bucket
    // scan + optional histogram reset (buffered branch only -- the early_stop
    // pass has nothing to finalize beyond the counter write).
    //
    // The `__threadfence()` + `__syncthreads_or` inside `finalize_pass` is the
    // boundary that switches the buffered branch's active arm from
    // `arms.buffered` (only alive for `merge_histogram`'s smem reads above) to
    // `arms.prefix_sum`. After that boundary, no thread in this block reads
    // `storage.arms.buffered.*` again, and the other (non-last) blocks have
    // already exited.
    auto epilogue_op = [&] {
      if (threadIdx.x == 0)
      {
        counter_update_fn();
      }
      if (mode == sink_mode::buffered)
      {
        block_identify_kth_bucket_t{storage.arms.prefix_sum}.find_kth_bucket(
          global_histogram, current_k, on_kth_bucket);
        if (reset_histogram)
        {
          init_histogram<block_threads, num_buckets>(global_histogram);
        }
      }
    };

    finalize_pass(retired_block_counter, gridDim.x, epilogue_op);
  }
};

//---------------------------------------------------------------------
// agent_topk_last_filter: dedicated agent for the final filter pass.
//
// Unlike `agent_topk_filter_partition` (which covers passes
// 1..num_passes-1), this agent never accumulates a histogram and never runs
// finalize_pass. Its smem footprint is correspondingly smaller: just the
// keys-source persistent state plus the partition's scratch buffer.
//
// The pass-specific outputs (`p_num_ties_written_to_back`, `k_total`, `num_of_kth_needed`)
// flow through `run()` rather than the constructor; they only become known
// after the previous pass has updated the device-side counter.
//
// Selected candidates land at the front of `d_keys_out` via `p_num_selected_written`;
// surplus "kth"-class candidates land at the back of `d_keys_out` via a
// `back_grow_capped_reserve_op` (cap = num_of_kth_needed, anchor = k_total).
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename IdentifyCandidatesOpT,
          typename OffsetT,
          typename OutOffsetT,
          block_partition_strategy PartStrat = block_partition_strategy::atomics,
          bool LazyValueLoad               = false,
          bool InlinedClassify             = false>
struct agent_topk_last_filter
{
  using key_in_t   = it_value_t<KeyInputIteratorT>;
  using value_in_t = it_value_t<ValueInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  // last_filter operates as a true 2-way partition (it both writes selected items
  // to the front of d_keys_out AND back-grow-cap writes "kth"-class candidates to
  // the back). `BlockPartition` (any non-accumulating strategy) and
  // `block_partition_accumulating_candidates` are both supported -- the cooperative
  // flush in the accumulating variant honors the `back_grow_capped_reserve_op`'s
  // `may_grant_less` semantics by dropping items beyond the granted count,
  // equivalent (modulo flush-chunk granularity) to the per-item drop the Atomics
  // strategy performs.

  static constexpr bool effective_lazy_value_load = LazyValueLoad && !keys_only;

  using selected_offset_t  = OutOffsetT;
  using candidate_offset_t = OutOffsetT;

  using key_source_input_t =
    tile_data_source_t<KeyInputIteratorT, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using key_source_buffer_t =
    tile_data_source_t<key_in_t*, AgentTopKPolicyT::keys_tile_load_kind, block_threads, items_per_thread, OffsetT>;
  using keys_source_t = multi_source_data_source<key_source_input_t, key_source_buffer_t, OffsetT>;

  using value_source_input_t = direct_data_source<ValueInputIteratorT, block_threads, items_per_thread, OffsetT>;
  using value_source_buffer_t = direct_data_source<value_in_t*, block_threads, items_per_thread, OffsetT>;
  using value_source_t = multi_source_data_source<value_source_input_t, value_source_buffer_t, OffsetT>;

  using val_out_t      = ValueOutputIteratorT;
  using cand_val_out_t = ValueOutputIteratorT; // selected and candidate share d_values_out

  // Sinks-bundle: concrete `value_channel_sinks_t` for values-mode, `NullType`
  // placeholder for keys-only.
  using value_channel_sinks_concrete_t = value_channel_sinks_t<val_out_t, cand_val_out_t>;
  using value_channel_sinks_or_null_t =
    ::cuda::std::conditional_t<keys_only, NullType, value_channel_sinks_concrete_t>;

  // Value-channel type fed to the block primitive as `ValueT`.
  using agent_value_t = ::cuda::std::conditional_t<keys_only, NullType, value_in_t>;

  // Per-channel data-source `ScratchStorage`, used to size the partition's
  // `load` slot in the Staged / SharedMem scratch.
  using agent_value_data_source_scratch_t =
    ::cuda::std::conditional_t<keys_only, NullType, typename value_source_t::ScratchStorage>;

  // Reserve-op types: selected stream uses an atomic reserve, candidate stream uses
  // back-grow-capped (writes ties at the back of the output).
  using selected_reserve_op_t  = atomic_reserve_range_op<selected_offset_t>;
  using candidate_reserve_op_t = back_grow_capped_reserve_op<candidate_offset_t>;


  using partition_t = strategy_to_partition_class_t<
    PartStrat,
    block_threads,
    items_per_thread,
    AgentTopKPolicyT::accumulating_buffer_capacity,
    AgentTopKPolicyT::speculative_selected_buffer_capacity,
    key_in_t,
    selected_offset_t,
    candidate_offset_t,
    selected_reserve_op_t,
    candidate_reserve_op_t,
    KeyOutputIteratorT,
    KeyOutputIteratorT,
    IdentifyCandidatesOpT,
    topk_noop_candidate_callback_op,
    value_channel_sinks_or_null_t,
    agent_value_t,
    agent_value_data_source_scratch_t,
    effective_lazy_value_load,
    InlinedClassify>;

  // last_filter's smem: keys-source persistent state plus the same partition layout
  // helper the filter agent uses. For `BlockPartition` the partition state is empty
  // (so the layout collapses to a 2-arm union of `keys_source_scratch` and
  // `partition_scratch`); for `block_partition_accumulating_candidates` the persistent
  // slot buffers live in `partition_state`. There's no `prefix_sum` here, so we
  // hand the layout helper an empty `prefix_sum` placeholder type.
  struct empty_prefix_sum_t
  {};

  using storage_layout_t = partition_storage_layout_for_t<
    partition_t,
    typename keys_source_t::ScratchStorage,
    empty_prefix_sum_t>;

  // Per-child persistent state. `keys_source_t` (multi-source) does not
  // publish a `TempStorage`; the agent holds one `TempStorage` per child
  // source.
  struct _TempStorage
  {
    typename key_source_input_t::TempStorage key_src_input_state;
    typename key_source_buffer_t::TempStorage key_src_buffer_state;
    storage_layout_t partition_arena;
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& storage;

  KeyInputIteratorT d_keys_in;
  KeyOutputIteratorT d_keys_out;
  ValueInputIteratorT d_values_in;
  ValueOutputIteratorT d_values_out;
  key_in_t* in_key_buf;
  value_in_t* in_val_buf;

  OutOffsetT* p_num_selected_written;
  OffsetT input_length;
  bool load_from_candidates_buffer;
  IdentifyCandidatesOpT identify_candidates_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_topk_last_filter(
    TempStorage& temp_storage,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    key_in_t* in_key_buf,
    value_in_t* in_val_buf,
    OutOffsetT* p_num_selected_written,
    OffsetT input_length,
    bool load_from_candidates_buffer,
    IdentifyCandidatesOpT identify_candidates_op)
      : storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , in_key_buf(in_key_buf)
      , in_val_buf(in_val_buf)
      , p_num_selected_written(p_num_selected_written)
      , input_length(input_length)
      , load_from_candidates_buffer(load_from_candidates_buffer)
      , identify_candidates_op(identify_candidates_op)
  {}

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE auto make_value_channel_sinks()
  {
    if constexpr (keys_only)
    {
      return NullType{};
    }
    else
    {
      // last_filter sends both selected and candidate values to `d_values_out`.
      return value_channel_sinks_concrete_t{d_values_out, d_values_out};
    }
  }

  // Note: `make_value_channel_sources` (previously returning a `value_source_t`
  // by value) was removed when `value_source_t` became non-movable. The value-
  // source children are now declared at the outer scope of `run()` and the
  // per-tile multi-source is constructed via an IIFE that returns a prvalue,
  // letting C++17 mandatory copy elision place it directly into the outer
  // `value_source` local.

  template <bool IsFull, typename ValueSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void do_partition(
    partition_t& partition,
    const key_in_t (&keys)[items_per_thread],
    OffsetT num_items_in_tile,
    ValueSourceT& value_source)
  {
    if constexpr (IsFull)
    {
      partition.partition(storage.partition_arena.get_partition_scratch(), keys, value_source);
    }
    else
    {
      partition.partition(storage.partition_arena.get_partition_scratch(), keys, num_items_in_tile, value_source);
    }
  }

public:
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  run(OutOffsetT* p_num_ties_written_to_back, OutOffsetT k_total, OutOffsetT num_of_kth_needed)
  {
    key_source_input_t key_src_input{d_keys_in, storage.key_src_input_state};
    key_source_buffer_t key_src_buffer{in_key_buf, storage.key_src_buffer_state};
    keys_source_t keys_source{key_src_input, key_src_buffer, /*pick_b=*/load_from_candidates_buffer};

    // Build the reserve ops + sinks once, before the tile loop. The partition object
    // captures these references and consults them at every flush.
    //
    // The back-grow-capped reserve op carries the precomputed `region_start = k_total -
    // num_of_kth_needed` rather than the back-region end anchor, so its per-call math
    // collapses from two subtracts to one add (see `back_grow_capped_reserve_op`).
    selected_reserve_op_t reserve_sel{p_num_selected_written};
    candidate_reserve_op_t reserve_cand{
      p_num_ties_written_to_back,
      static_cast<candidate_offset_t>(k_total - num_of_kth_needed),
      static_cast<candidate_offset_t>(num_of_kth_needed)};
    auto value_channel_sinks = make_value_channel_sinks();
    topk_noop_candidate_callback_op callback_op{};

    partition_t partition{
      storage.partition_arena.get_partition_state(),
      reserve_sel,
      reserve_cand,
      d_keys_out,
      d_keys_out,
      value_channel_sinks,
      identify_candidates_op,
      callback_op};

    // Split the tile space into full tiles and a possible single trailing
    // partial tile. The hot-path loop below processes only full tiles. The
    // partial tile, if any, is handled exactly once after the loop by the
    // unique block that would have owned it under the original grid-strided
    // schedule.
    const OffsetT num_full_tiles = input_length / static_cast<OffsetT>(tile_items);
    const OffsetT partial_items  = input_length - num_full_tiles * static_cast<OffsetT>(tile_items);

    // Per-tile value source: children declared at this outer scope so they
    // outlive the multi-source (which holds references to them). For
    // `keys_only`, these locals are unused but their construction is trivial.
    [[maybe_unused]] typename value_source_input_t::TempStorage val_state_input{};
    [[maybe_unused]] typename value_source_buffer_t::TempStorage val_state_buffer{};
    [[maybe_unused]] value_source_input_t val_input{d_values_in, val_state_input};
    [[maybe_unused]] value_source_buffer_t val_buffer{in_val_buf, val_state_buffer};

    // --- full-tile loop --------------------------------------------------
    for (OffsetT tile_id = static_cast<OffsetT>(blockIdx.x); tile_id < num_full_tiles;
         tile_id += static_cast<OffsetT>(gridDim.x))
    {
      const OffsetT tile_base = tile_id * static_cast<OffsetT>(tile_items);

      keys_source.set_tile_base(tile_base);
      auto value_source = [&] {
        if constexpr (keys_only)
        {
          return NullType{};
        }
        else
        {
          return value_source_t{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
        }
      }();
      if constexpr (!keys_only)
      {
        value_source.set_tile_base(tile_base);
      }

      __syncthreads();
      key_in_t items[items_per_thread];
      auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch());
      h.complete_load(items);
      __syncthreads();
      do_partition</*IsFull=*/true>(partition, items, /*num_items_in_tile=*/tile_items, value_source);
    }

    // --- trailing partial tile (handled by exactly one block) ------------
    if (partial_items > 0)
    {
      const unsigned partial_owner = static_cast<unsigned>(num_full_tiles % static_cast<OffsetT>(gridDim.x));
      if (blockIdx.x == partial_owner)
      {
        const OffsetT tile_base = num_full_tiles * static_cast<OffsetT>(tile_items);

        keys_source.set_tile_base(tile_base);
        auto value_source = [&] {
          if constexpr (keys_only)
          {
            return NullType{};
          }
          else
          {
            return value_source_t{val_input, val_buffer, /*pick_b=*/load_from_candidates_buffer};
          }
        }();
        if constexpr (!keys_only)
        {
          value_source.set_tile_base(tile_base);
        }

        __syncthreads();
        key_in_t items[items_per_thread];
        auto h = keys_source.submit_load(storage.partition_arena.get_keys_source_scratch(), partial_items);
        h.complete_load(items);
        __syncthreads();
        do_partition</*IsFull=*/false>(partition, items, partial_items, value_source);
      }
    }

    // Terminal partition flush. No-op on the non-accumulating strategies.
    partition.epilogue();
  }
};
} // namespace detail::topk
CUB_NAMESPACE_END
