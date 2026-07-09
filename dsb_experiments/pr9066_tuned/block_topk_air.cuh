// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// PR #9066's composed block_topk_air with the dsb_experiments latency optimizations applied:
//   * original-value scatter: a register copy of the keys is kept across the sieve, which is
//     invoked with UntwiddleKeys=false -- no flip-back tracking and no un-twiddle pass; unranked
//     key slots are restored from the copy so the returned register contents match the original
//     behavior
//   * pair exchange for key-value selection: keys and values are scattered together and
//     gathered once, removing two barriers and one full item pass from the epilogue (exchange
//     grows from tile*max(sizeof(KeyT), sizeof(ValueT)) to tile*sizeof(pair))
//   * the rank counters live outside the sieve/exchange union and are primed before the sieve
//     runs, removing the rank stage's reset phase and barrier

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_scan.cuh>
#include <cub/block/block_topk_rank.cuh>
#include <cub/block/specializations/block_topk_rank_atomic.cuh>
#include <cub/block/specializations/block_topk_sieve_air.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @brief Block-level top-k by radix selection. (See PR #9066 for the sieve/rank composition;
//! this variant applies the dsb_experiments latency optimizations.)
template <typename KeyT, int ThreadsPerBlock, int ItemsPerThread, typename ValueT = NullType>
class block_topk_air
{
private:
  static constexpr int threads_per_block = ThreadsPerBlock;
  static constexpr int items_per_thread  = ItemsPerThread;
  static constexpr int tile_items        = threads_per_block * items_per_thread;

  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;

  using block_sieve_t         = block_topk_sieve<KeyT, threads_per_block>;
  using block_sieve_storage_t = typename block_sieve_t::TempStorage;
  using block_rank_t          = block_topk_rank<threads_per_block>;
  using block_rank_storage_t  = typename block_rank_t::TempStorage;

  static_assert(
    ::cuda::std::is_base_of_v<Uninitialized<typename block_topk_sieve_air<KeyT, threads_per_block>::TempStorage>,
                              block_sieve_storage_t>,
    "Wrong sieve specialization");
  static_assert(::cuda::std::is_base_of_v<Uninitialized<typename block_topk_rank_atomic<threads_per_block>::TempStorage>,
                                          block_rank_storage_t>,
                "Wrong rank specialization");

  struct pair_t
  {
    KeyT key;
    ValueT value;
  };
  struct keys_exchange_t
  {
    KeyT keys[tile_items];
  };
  struct pairs_exchange_t
  {
    pair_t pairs[tile_items];
  };
  using exchange_t = ::cuda::std::conditional_t<keys_only, keys_exchange_t, pairs_exchange_t>;

  struct TempStorage_
  {
    union
    {
      block_sieve_storage_t sieve_storage;
      exchange_t exchange;
    } stage;
    // outside the union: primed before the sieve stage runs
    block_rank_storage_t rank_storage;
  };

  /// Shared storage reference
  TempStorage_& storage;

  /// Linear thread index
  int linear_tid;

  template <detail::topk::select Dir, bool Full>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static block_topk_key_states<items_per_thread> sieve_select(
    block_sieve_storage_t& sieve_storage,
    KeyT (&keys)[items_per_thread],
    int k,
    int valid_items,
    int begin_bit,
    int end_bit)
  {
    // UntwiddleKeys = false: the caller keeps original keys in registers
    if constexpr (Dir == detail::topk::select::max)
    {
      return block_sieve_t(sieve_storage)
        .template select_max<Full, true, false>(keys, k, valid_items, begin_bit, end_bit);
    }
    else
    {
      return block_sieve_t(sieve_storage)
        .template select_min<Full, true, false>(keys, k, valid_items, begin_bit, end_bit);
    }
  }

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void select_topk(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    int valid_items,
    int begin_bit,
    int end_bit)
  {
    if constexpr (!IsFullTile)
    {
      _CCCL_ASSERT(valid_items > 0 && valid_items <= tile_items, "valid_items must be in [1, tile_items]");
    }

    // TODO (elstehle): Short-circuit if begin_bit is constrained to be non-negative
    begin_bit = (::cuda::std::max) (begin_bit, 0);

    // TODO (elstehle): Short-circuit if end_bit is constrained to be less than the maximum number of bits in the key
    // type
    const int max_bit = int(sizeof(KeyT) * 8);
    if (end_bit > max_bit)
    {
      end_bit = max_bit;
    }

    // TODO (elstehle): Short-circuit if k is greater than the number of items in the tile
    if ((!IsFullTile && k >= valid_items) || k >= tile_items)
    {
      return;
    }

    block_rank_t rank(storage.rank_storage);
    rank.prime(); // ordered before the ranking by the sieve's internal barriers

    // Keep the original keys; the sieve leaves keys[] bit-twiddled (UntwiddleKeys = false)
    KeyT original_keys[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      original_keys[i] = keys[i];
    }

    auto states =
      sieve_select<SelectDirection, IsFullTile>(storage.stage.sieve_storage, keys, k, valid_items, begin_bit, end_bit);
    // Make sure smem can be reused by the exchange stage
    __syncthreads();

    int scatter_indices[items_per_thread];
    rank.rank_key_states(states, scatter_indices);

    if constexpr (keys_only)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        if (scatter_indices[i] >= 0)
        {
          storage.stage.exchange.keys[scatter_indices[i]] = original_keys[i];
        }
      }
      __syncthreads();
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const int buffer_idx = linear_tid * items_per_thread + i;
        keys[i]              = (buffer_idx < k) ? storage.stage.exchange.keys[buffer_idx] : original_keys[i];
      }
    }
    else
    {
      // pair scatter: keys and values together, one gather, two barriers fewer
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        if (scatter_indices[i] >= 0)
        {
          storage.stage.exchange.pairs[scatter_indices[i]] = pair_t{original_keys[i], values[i]};
        }
      }
      __syncthreads();
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const int buffer_idx = linear_tid * items_per_thread + i;
        if (buffer_idx < k)
        {
          const pair_t p = storage.stage.exchange.pairs[buffer_idx];
          keys[i]        = p.key;
          values[i]      = p.value;
        }
        else
        {
          keys[i] = original_keys[i];
        }
      }
    }
  }

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_air(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(ThreadsPerBlock, 1, 1))
  {}

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  select_keys(KeyT (&keys)[items_per_thread], int k, int valid_items, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    NullType values[ItemsPerThread];
    select_topk<SelectDirection, IsFullTile>(keys, values, k, valid_items, begin_bit, end_bit);
  }

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void select_pairs(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    int valid_items,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    select_topk<SelectDirection, IsFullTile>(keys, values, k, valid_items, begin_bit, end_bit);
  }
};
} // namespace detail
CUB_NAMESPACE_END
