// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Top-k-private block-level 2-way partition primitive. Classifies a per-thread register
//! array of keys (and optionally values) as `selected`, `candidate`, or `rejected` and
//! scatters selected items front-to-back and candidate items back-to-front into two
//! caller-supplied output iterators with independent global counters.
//!
//! Deferred follow-ups (see design plan Q9-Q13, tracked for future benchmark-driven work):
//!
//! * Q7 (partial) -- "always write values" Staged/SharedMem integration. Today
//!   `agent_topk_filter_partition` uses the same per-item atomic scatter that Atomics would
//!   emit, but inlined into `process_tile` with compile-time sink_mode gating (to preserve
//!   the legacy dispatch layout with candidate index buffers). Staged/SharedMem require
//!   switching the candidate buffers to carry (key, value) pairs (plan Q7) plus tuning-
//!   policy entries per strategy; the primitive is unit-tested and ready to plug in.
//! * Q9   -- async TMA tile loading via `async_load_strategy` backed by
//!   `BlockLoadToShared`. BlockTileLoader already has the `LoadStrategyT` template param;
//!   only `sync_load_strategy` is implemented today.
//! * Q10  -- atomic-counter / UGETNEXTWORKID / segmented `TileQueueT` variants. The
//!   template param is in place on BlockTileLoader and a `segmented_queue` exercising
//!   on_segment_change is covered by the tile-loader unit tests.
//! * Q11  -- Programmatic Dependent Launch (PDL) in `dispatch_topk.cuh` to hide the
//!   per-pass launch latency. Purely additive dispatch-level change.
//! * Q12  -- item bit-packing in Staged/SharedMem buffers for small KeyT+ValueT types.
//!   Private buffer_t detail; no interface change.
//! * Q13  -- vectorized cooperative store (STG.E.128) with runtime alignment check in the
//!   Staged/SharedMem flush loops.
//! * `kth_key_bits` register/smem caching at pass entry -- compiler likely already hoists
//!   via constant-cache; verify in SASS once other optimizations land.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{

//---------------------------------------------------------------------
// Shared types
//---------------------------------------------------------------------

// The three classes emitted by top-k's classifier. `rejected` is also the out-of-bounds
// marker the caller MUST set for partial last tiles; BlockPartition performs no num_valid
// checks internally.
enum class candidate_class
{
  selected,
  candidate,
  rejected
};

// Strategy selector for BlockPartition.
//
// Atomics   -- no smem; per-non-rejected-item global atomicAdd + scatter.
// Staged    -- smem scatter + cooperative coalesced store; 2 global atomics per block per pass.
//              Supports the split PartitionKeys / ScatterValues flow to decouple keys/values
//              in registers.
// SharedMem -- fused PartitionPairs only; keys and values live simultaneously in smem.
enum class BlockPartitionStrategy
{
  Atomics,
  Staged,
  SharedMem
};

//---------------------------------------------------------------------
// buffer_t building blocks
//
// The counters union lets Phase 1 (intra-tile smem scatter) use 32-bit `int` smem atomics
// -- faster than 64-bit smem atomics and always sufficient since tile_items fits in int --
// while Phase 2 (global base claim) uses the proper global offset types. The two phases
// are separated by __syncthreads so the reuse is safe.
//---------------------------------------------------------------------

template <typename SelectedOffsetT, typename CandidateOffsetT>
struct partition_counters
{
  union
  {
    int counters[2]; // Phase 1
    struct
    {
      SelectedOffsetT selected;
      CandidateOffsetT candidate;
    } global_bases; // Phase 2
  };
};

struct atomics_buffer
{};

template <typename KeyT, typename ValueT, typename SelOff, typename CandOff, int TileItems, bool KeysOnly>
struct staged_buffer_impl;

template <typename KeyT, typename ValueT, typename SelOff, typename CandOff, int TileItems>
struct staged_buffer_impl<KeyT, ValueT, SelOff, CandOff, TileItems, /*KeysOnly=*/false>
{
  union
  {
    KeyT keys[TileItems];
    ValueT values[TileItems];
  };
  partition_counters<SelOff, CandOff> cnt;
};

template <typename KeyT, typename ValueT, typename SelOff, typename CandOff, int TileItems>
struct staged_buffer_impl<KeyT, ValueT, SelOff, CandOff, TileItems, /*KeysOnly=*/true>
{
  KeyT keys[TileItems];
  partition_counters<SelOff, CandOff> cnt;
};

template <typename KeyT, typename ValueT, typename SelOff, typename CandOff, int TileItems, bool KeysOnly>
struct sharedmem_buffer_impl;

template <typename KeyT, typename ValueT, typename SelOff, typename CandOff, int TileItems>
struct sharedmem_buffer_impl<KeyT, ValueT, SelOff, CandOff, TileItems, /*KeysOnly=*/false>
{
  KeyT keys[TileItems];
  ValueT values[TileItems];
  partition_counters<SelOff, CandOff> cnt;
};

template <typename KeyT, typename ValueT, typename SelOff, typename CandOff, int TileItems>
struct sharedmem_buffer_impl<KeyT, ValueT, SelOff, CandOff, TileItems, /*KeysOnly=*/true>
{
  KeyT keys[TileItems];
  partition_counters<SelOff, CandOff> cnt;
};

//---------------------------------------------------------------------
// BlockPartition -- primary template (declaration only; uses specializations)
//---------------------------------------------------------------------

template <typename KeyT,
          typename ValueT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          int BlockThreads,
          int ItemsPerThread,
          BlockPartitionStrategy Strategy,
          bool HasCandidates   = true,
          bool HasCandidateCap = false>
class BlockPartition;

//---------------------------------------------------------------------
// Atomics specialization: fused per-item scatter via global atomicAdd.
// No smem, no cross-call state. Provides PartitionKeys (for keys-only callers)
// and PartitionPairs (for paired callers).
//---------------------------------------------------------------------

template <typename KeyT,
          typename ValueT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          int BlockThreads,
          int ItemsPerThread,
          bool HasCandidates,
          bool HasCandidateCap>
class BlockPartition<KeyT,
                     ValueT,
                     SelectedOffsetT,
                     CandidateOffsetT,
                     BlockThreads,
                     ItemsPerThread,
                     BlockPartitionStrategy::Atomics,
                     HasCandidates,
                     HasCandidateCap>
{
  static_assert(!HasCandidateCap || HasCandidates, "HasCandidateCap requires HasCandidates");

  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;

public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  using buffer_t                  = atomics_buffer;

  template <typename SelKeyOutIt, typename CandKeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void PartitionKeys(
    buffer_t& /*buffer*/,
    const KeyT (&keys)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    SelKeyOutIt selected_keys_out,
    CandKeyOutIt candidate_keys_out,
    SelectedOffsetT* selected_counter,
    CandidateOffsetT* candidate_counter,
    CandidateOffsetT max_candidate_count = ::cuda::std::numeric_limits<CandidateOffsetT>::max())
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (classes[j] == candidate_class::rejected)
      {
        continue;
      }
      if constexpr (!HasCandidates)
      {
        const auto pos       = atomicAdd(selected_counter, SelectedOffsetT{1});
        selected_keys_out[pos] = keys[j];
      }
      else
      {
        if (classes[j] == candidate_class::selected)
        {
          const auto pos         = atomicAdd(selected_counter, SelectedOffsetT{1});
          selected_keys_out[pos] = keys[j];
        }
        else // candidate
        {
          const auto pos = atomicAdd(candidate_counter, CandidateOffsetT{1});
          if constexpr (HasCandidateCap)
          {
            if (pos >= max_candidate_count)
            {
              continue;
            }
          }
          else
          {
            (void) max_candidate_count;
          }
          candidate_keys_out[pos] = keys[j];
        }
      }
    }
  }

  template <typename SelKeyOutIt, typename CandKeyOutIt, typename SelValOutIt, typename CandValOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void PartitionPairs(
    buffer_t& /*buffer*/,
    const KeyT (&keys)[ItemsPerThread],
    const ValueT (&values)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    SelKeyOutIt selected_keys_out,
    CandKeyOutIt candidate_keys_out,
    SelValOutIt selected_vals_out,
    CandValOutIt candidate_vals_out,
    SelectedOffsetT* selected_counter,
    CandidateOffsetT* candidate_counter,
    CandidateOffsetT max_candidate_count = ::cuda::std::numeric_limits<CandidateOffsetT>::max())
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (classes[j] == candidate_class::rejected)
      {
        continue;
      }
      if constexpr (!HasCandidates)
      {
        const auto pos         = atomicAdd(selected_counter, SelectedOffsetT{1});
        selected_keys_out[pos] = keys[j];
        if constexpr (!keys_only)
        {
          selected_vals_out[pos] = values[j];
        }
      }
      else
      {
        if (classes[j] == candidate_class::selected)
        {
          const auto pos         = atomicAdd(selected_counter, SelectedOffsetT{1});
          selected_keys_out[pos] = keys[j];
          if constexpr (!keys_only)
          {
            selected_vals_out[pos] = values[j];
          }
        }
        else // candidate
        {
          const auto pos = atomicAdd(candidate_counter, CandidateOffsetT{1});
          if constexpr (HasCandidateCap)
          {
            if (pos >= max_candidate_count)
            {
              continue;
            }
          }
          else
          {
            (void) max_candidate_count;
          }
          candidate_keys_out[pos] = keys[j];
          if constexpr (!keys_only)
          {
            candidate_vals_out[pos] = values[j];
          }
        }
      }
    }
  }
};

//---------------------------------------------------------------------
// Staged specialization: smem scatter + cooperative coalesced store.
// Two-phase flow:
//   Phase 1 (in smem): 32-bit smem atomicAdd to determine an intra-tile position.
//   Phase 2 (to global): thread 0 claims global bases via one atomicAdd per class
//                        per block, then the whole block flushes smem->global in
//                        coalesced STG.
// Supports the split flow (PartitionKeys; then a caller-controlled __syncthreads
// and/or BlockLoad of values; then ScatterValues) so keys and values never
// coexist in registers.
//---------------------------------------------------------------------

template <typename KeyT,
          typename ValueT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          int BlockThreads,
          int ItemsPerThread,
          bool HasCandidates,
          bool HasCandidateCap>
class BlockPartition<KeyT,
                     ValueT,
                     SelectedOffsetT,
                     CandidateOffsetT,
                     BlockThreads,
                     ItemsPerThread,
                     BlockPartitionStrategy::Staged,
                     HasCandidates,
                     HasCandidateCap>
{
  static_assert(!HasCandidateCap || HasCandidates, "HasCandidateCap requires HasCandidates");

  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;

public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  using buffer_t = staged_buffer_impl<KeyT, ValueT, SelectedOffsetT, CandidateOffsetT, tile_items, keys_only>;

private:
  // -1 = rejected/invalid; [0, selected_cnt_) = front region; [tile_items - candidate_cnt_,
  // tile_items) = back region.
  int positions_[ItemsPerThread];

  int selected_cnt_  = 0;
  int candidate_cnt_ = 0;

  SelectedOffsetT selected_base_   = SelectedOffsetT{};
  CandidateOffsetT candidate_base_ = CandidateOffsetT{};
  CandidateOffsetT max_candidate_count_ = ::cuda::std::numeric_limits<CandidateOffsetT>::max();

  template <typename SelKeyOutIt, typename CandKeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void partition_keys_common(
    buffer_t& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    SelKeyOutIt selected_keys_out,
    CandKeyOutIt candidate_keys_out,
    SelectedOffsetT* selected_counter,
    CandidateOffsetT* candidate_counter,
    CandidateOffsetT max_candidate_count)
  {
    if (threadIdx.x == 0)
    {
      buffer.cnt.counters[0] = 0;
      if constexpr (HasCandidates)
      {
        buffer.cnt.counters[1] = 0;
      }
    }
    __syncthreads();

    // Phase 1: scatter keys into smem; remember positions.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (classes[j] == candidate_class::rejected)
      {
        positions_[j] = -1;
      }
      else if constexpr (!HasCandidates)
      {
        const int pos = atomicAdd(&buffer.cnt.counters[0], 1);
        buffer.keys[pos] = keys[j];
        positions_[j]    = pos;
      }
      else
      {
        if (classes[j] == candidate_class::selected)
        {
          const int pos    = atomicAdd(&buffer.cnt.counters[0], 1);
          buffer.keys[pos] = keys[j];
          positions_[j]    = pos;
        }
        else // candidate
        {
          const int pos    = atomicAdd(&buffer.cnt.counters[1], 1);
          const int idx    = tile_items - 1 - pos;
          buffer.keys[idx] = keys[j];
          positions_[j]    = idx;
        }
      }
    }
    __syncthreads();

    // Phase 2: snapshot counts, thread 0 claims global bases.
    selected_cnt_ = buffer.cnt.counters[0];
    if constexpr (HasCandidates)
    {
      candidate_cnt_ = buffer.cnt.counters[1];
    }
    if constexpr (HasCandidateCap)
    {
      max_candidate_count_ = max_candidate_count;
    }
    else
    {
      (void) max_candidate_count;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
      buffer.cnt.global_bases.selected = atomicAdd(selected_counter, static_cast<SelectedOffsetT>(selected_cnt_));
      if constexpr (HasCandidates)
      {
        buffer.cnt.global_bases.candidate =
          atomicAdd(candidate_counter, static_cast<CandidateOffsetT>(candidate_cnt_));
      }
    }
    __syncthreads();
    selected_base_ = buffer.cnt.global_bases.selected;
    if constexpr (HasCandidates)
    {
      candidate_base_ = buffer.cnt.global_bases.candidate;
    }

    // Phase 3: cooperative coalesced store of keys.
    for (int i = static_cast<int>(threadIdx.x); i < selected_cnt_; i += BlockThreads)
    {
      selected_keys_out[selected_base_ + static_cast<SelectedOffsetT>(i)] = buffer.keys[i];
    }

    if constexpr (HasCandidates)
    {
      const int cand_to_write = candidate_write_count();
      for (int i = static_cast<int>(threadIdx.x); i < cand_to_write; i += BlockThreads)
      {
        candidate_keys_out[candidate_base_ + static_cast<CandidateOffsetT>(i)] =
          buffer.keys[tile_items - candidate_cnt_ + i];
      }
    }

    __syncthreads();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE int candidate_write_count() const
  {
    if constexpr (!HasCandidates)
    {
      return 0;
    }
    else if constexpr (!HasCandidateCap)
    {
      return candidate_cnt_;
    }
    else
    {
      const CandidateOffsetT cap      = max_candidate_count_;
      const CandidateOffsetT writable = (cap > candidate_base_) ? (cap - candidate_base_) : CandidateOffsetT{0};
      const CandidateOffsetT clamped  = (::cuda::std::min) (static_cast<CandidateOffsetT>(candidate_cnt_), writable);
      return static_cast<int>(clamped);
    }
  }

public:
  template <typename SelKeyOutIt, typename CandKeyOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void PartitionKeys(
    buffer_t& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    SelKeyOutIt selected_keys_out,
    CandKeyOutIt candidate_keys_out,
    SelectedOffsetT* selected_counter,
    CandidateOffsetT* candidate_counter,
    CandidateOffsetT max_candidate_count = ::cuda::std::numeric_limits<CandidateOffsetT>::max())
  {
    partition_keys_common(
      buffer, keys, classes, selected_keys_out, candidate_keys_out, selected_counter, candidate_counter, max_candidate_count);
  }

  template <typename SelValOutIt, typename CandValOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterValues(
    buffer_t& buffer,
    const ValueT (&values)[ItemsPerThread],
    SelValOutIt selected_vals_out,
    CandValOutIt candidate_vals_out)
  {
    if constexpr (keys_only)
    {
      (void) buffer;
      (void) values;
      (void) selected_vals_out;
      (void) candidate_vals_out;
      return;
    }
    else
    {
      // Scatter values into the same smem buffer using remembered positions.
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < ItemsPerThread; ++j)
      {
        if (positions_[j] >= 0)
        {
          buffer.values[positions_[j]] = values[j];
        }
      }
      __syncthreads();

      for (int i = static_cast<int>(threadIdx.x); i < selected_cnt_; i += BlockThreads)
      {
        selected_vals_out[selected_base_ + static_cast<SelectedOffsetT>(i)] = buffer.values[i];
      }

      if constexpr (HasCandidates)
      {
        const int cand_to_write = candidate_write_count();
        for (int i = static_cast<int>(threadIdx.x); i < cand_to_write; i += BlockThreads)
        {
          candidate_vals_out[candidate_base_ + static_cast<CandidateOffsetT>(i)] =
            buffer.values[tile_items - candidate_cnt_ + i];
        }
      }

      __syncthreads();
    }
  }

  template <typename SelKeyOutIt, typename CandKeyOutIt, typename SelValOutIt, typename CandValOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void PartitionPairs(
    buffer_t& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const ValueT (&values)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    SelKeyOutIt selected_keys_out,
    CandKeyOutIt candidate_keys_out,
    SelValOutIt selected_vals_out,
    CandValOutIt candidate_vals_out,
    SelectedOffsetT* selected_counter,
    CandidateOffsetT* candidate_counter,
    CandidateOffsetT max_candidate_count = ::cuda::std::numeric_limits<CandidateOffsetT>::max())
  {
    PartitionKeys(
      buffer, keys, classes, selected_keys_out, candidate_keys_out, selected_counter, candidate_counter, max_candidate_count);
    ScatterValues(buffer, values, selected_vals_out, candidate_vals_out);
  }
};

//---------------------------------------------------------------------
// SharedMem specialization: fused PartitionPairs only. Keys and values share one smem
// region (not a union; both live simultaneously). Single smem-atomic scatter for both,
// followed by two coalesced stores. Eliminates the `positions_[]` register array at the
// cost of a second smem buffer for values.
//---------------------------------------------------------------------

template <typename KeyT,
          typename ValueT,
          typename SelectedOffsetT,
          typename CandidateOffsetT,
          int BlockThreads,
          int ItemsPerThread,
          bool HasCandidates,
          bool HasCandidateCap>
class BlockPartition<KeyT,
                     ValueT,
                     SelectedOffsetT,
                     CandidateOffsetT,
                     BlockThreads,
                     ItemsPerThread,
                     BlockPartitionStrategy::SharedMem,
                     HasCandidates,
                     HasCandidateCap>
{
  static_assert(!HasCandidateCap || HasCandidates, "HasCandidateCap requires HasCandidates");

  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;

public:
  static constexpr int tile_items = BlockThreads * ItemsPerThread;
  using buffer_t = sharedmem_buffer_impl<KeyT, ValueT, SelectedOffsetT, CandidateOffsetT, tile_items, keys_only>;

  template <typename SelKeyOutIt, typename CandKeyOutIt, typename SelValOutIt, typename CandValOutIt>
  _CCCL_DEVICE _CCCL_FORCEINLINE void PartitionPairs(
    buffer_t& buffer,
    const KeyT (&keys)[ItemsPerThread],
    const ValueT (&values)[ItemsPerThread],
    const candidate_class (&classes)[ItemsPerThread],
    SelKeyOutIt selected_keys_out,
    CandKeyOutIt candidate_keys_out,
    SelValOutIt selected_vals_out,
    CandValOutIt candidate_vals_out,
    SelectedOffsetT* selected_counter,
    CandidateOffsetT* candidate_counter,
    CandidateOffsetT max_candidate_count = ::cuda::std::numeric_limits<CandidateOffsetT>::max())
  {
    if (threadIdx.x == 0)
    {
      buffer.cnt.counters[0] = 0;
      if constexpr (HasCandidates)
      {
        buffer.cnt.counters[1] = 0;
      }
    }
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      if (classes[j] == candidate_class::rejected)
      {
        continue;
      }
      int idx;
      if constexpr (!HasCandidates)
      {
        idx = atomicAdd(&buffer.cnt.counters[0], 1);
      }
      else
      {
        if (classes[j] == candidate_class::selected)
        {
          idx = atomicAdd(&buffer.cnt.counters[0], 1);
        }
        else
        {
          const int pos = atomicAdd(&buffer.cnt.counters[1], 1);
          idx           = tile_items - 1 - pos;
        }
      }
      buffer.keys[idx] = keys[j];
      if constexpr (!keys_only)
      {
        buffer.values[idx] = values[j];
      }
    }
    __syncthreads();

    const int sc = buffer.cnt.counters[0];
    int cc       = 0;
    if constexpr (HasCandidates)
    {
      cc = buffer.cnt.counters[1];
    }

    if (threadIdx.x == 0)
    {
      buffer.cnt.global_bases.selected = atomicAdd(selected_counter, static_cast<SelectedOffsetT>(sc));
      if constexpr (HasCandidates)
      {
        buffer.cnt.global_bases.candidate = atomicAdd(candidate_counter, static_cast<CandidateOffsetT>(cc));
      }
    }
    __syncthreads();

    const SelectedOffsetT sb = buffer.cnt.global_bases.selected;
    for (int i = static_cast<int>(threadIdx.x); i < sc; i += BlockThreads)
    {
      selected_keys_out[sb + static_cast<SelectedOffsetT>(i)] = buffer.keys[i];
      if constexpr (!keys_only)
      {
        selected_vals_out[sb + static_cast<SelectedOffsetT>(i)] = buffer.values[i];
      }
    }

    if constexpr (HasCandidates)
    {
      const CandidateOffsetT cb = buffer.cnt.global_bases.candidate;
      int cand_to_write;
      if constexpr (HasCandidateCap)
      {
        const CandidateOffsetT writable =
          (max_candidate_count > cb) ? (max_candidate_count - cb) : CandidateOffsetT{0};
        cand_to_write =
          static_cast<int>((::cuda::std::min) (static_cast<CandidateOffsetT>(cc), writable));
      }
      else
      {
        cand_to_write = cc;
        (void) max_candidate_count;
      }
      for (int i = static_cast<int>(threadIdx.x); i < cand_to_write; i += BlockThreads)
      {
        candidate_keys_out[cb + static_cast<CandidateOffsetT>(i)] = buffer.keys[tile_items - cc + i];
        if constexpr (!keys_only)
        {
          candidate_vals_out[cb + static_cast<CandidateOffsetT>(i)] = buffer.values[tile_items - cc + i];
        }
      }
    }
    __syncthreads();
  }
};

} // namespace detail::topk

CUB_NAMESPACE_END
