// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// N6 prototype (swarm study, idea card ideas/queue/N6_equiwidth_bucket_select.md): block top-k by selection
// on a fixed-point image of the keys instead of on their radix digits.
//
// The radix passes of block_topk_air spend their first 8 bits on the sign and the exponent, which separate
// little on logit-like data, so bf16 always takes two passes and fp32 two to four. Here every key is mapped
// onto a 23-bit fixed-point number over the bracket [lo, hi] of the tile, u = round((v - lo) * scale), with
// hi the tile maximum and lo the minimum over threads of the per-thread maximum (at least Threads items are
// at or above lo, so for k <= Threads the k-th largest is inside the bracket) or the tile minimum for larger
// k. The mapping is monotone (every operation rounds monotonically), so a radix selection over u's digits
// is an exact selection by value as long as the candidates that end up sharing a u are resolved exactly.
// The passes are block_topk_air's (atomic histogram over the candidates, fused scan and bucket choice, pass
// state broadcast), on three 8, 8 and 7 bit digits of u computed per item per pass from the fp32 value
// (one FADD, one FFMA with the 2^23 rounding constant, one signed max as the lower clamp). The exits:
//
//   * the pivot bucket holds exactly the remaining k candidates (block_topk_air's early exit),
//   * from the second pass on, the minimum and maximum of the candidates' order-preserving key bits are
//     tracked with two fire-and-forget shared atomics per candidate item, and equal min and max means the
//     candidates are one tie class, which is final (any k of them, or the smallest indices),
//   * when the three u passes are exhausted with distinct candidates (two values closer than the u step), or
//     when the first pass did not shrink the candidate set to a quarter of the tile (a heavy tail collapses
//     the bracket's resolution), the selection continues with radix passes on the key bits restricted to the
//     candidates, from the byte that holds the highest differing bit of the tracked min and max.
//
// Index-only epilogue as in block_topk_b1: selected items write their striped source index (item i of thread
// t is item i * Threads + t) to the output by ticket, ties by ticket or by the warp-redundant rank
// (tie_break::smaller_index). Full tiles only, k in [1, N), select the largest. -0.0 and +0.0 map to the
// same u, and are one tie class only when NormalizeSignedZero folds -0.0 onto +0.0 up front (needed under
// prefer_smaller_index). Infinities clamp into the end bins and are ordered exactly by the key passes if the
// pivot falls on them. NaN is not supported.

#include <cub/config.cuh>

#include <cub/block/block_scan.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/util_ptx.cuh>

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/bit>
#include <cuda/std/cstdint>

#include <cuda_bf16.h>

#include "block_topk_b1.cuh"

namespace topk_b1
{

template <typename KeyT,
          int ThreadsPerBlock,
          int ItemsPerThread,
          bool NormalizeSignedZero = false,
          tie_break TieBreak       = tie_break::none>
class block_topk_n6
{
private:
  static constexpr int threads_per_block  = ThreadsPerBlock;
  static constexpr int items_per_thread   = ItemsPerThread;
  static constexpr int tile_items         = threads_per_block * items_per_thread;
  static constexpr int radix_bits         = 8;
  static constexpr int num_buckets        = 256;
  static constexpr int buckets_per_thread = ::cuda::ceil_div(num_buckets, threads_per_block);
  static constexpr int num_warps          = threads_per_block / 32;
  static constexpr bool striped_histogram = buckets_per_thread > 1 && (num_buckets % threads_per_block == 0);
  static constexpr bool deterministic_ties = TieBreak == tie_break::smaller_index;
  static constexpr bool is_bf16            = ::cuda::std::is_same_v<KeyT, __nv_bfloat16>;
  // the lowest byte of the 32-bit order-preserving key that can differ between two distinct keys
  static constexpr int lowest_key_byte = is_bf16 ? 2 : 0;
  static_assert(is_bf16 || ::cuda::std::is_same_v<KeyT, float>, "fp32 or bf16 keys");
  static_assert(threads_per_block % 32 == 0, "whole warps");

  using u32             = ::cuda::std::uint32_t;
  using histo_counter_t = u32;
  using block_scan_t    = BlockScan<histo_counter_t, threads_per_block, BLOCK_SCAN_WARP_SCANS>;
  using tie_mask_t = ::cuda::std::conditional_t<(items_per_thread > 32), ::cuda::std::uint64_t, u32>;
  static_assert(items_per_thread <= 64 || !deterministic_ties, "the tie decision is kept in a 64-bit mask");

  // bits(2^23): the fixed-point image of the bracket occupies [u_lo, u_lo + 2^23)
  static constexpr u32 u_lo = 0x4B000000u;

  struct TempStorage_
  {
    struct
    {
      histo_counter_t histogram[2][num_buckets];
      typename block_scan_t::TempStorage scan_temp_storage;
    } passes;
    struct
    {
      histo_counter_t selected;
      histo_counter_t candidates;
      int bucket;
    } pass_state;
    // per warp: max of the items, min of the per-thread maxima, min of the items (order-preserving bits)
    u32 bracket[num_warps][3];
    // double-buffered min and max of the candidates' order-preserving key bits
    u32 key_min[2];
    u32 key_max[2];
    histo_counter_t selected_offset[2];
    histo_counter_t tie_table[deterministic_ties ? num_warps : 1][deterministic_ties ? items_per_thread : 1];
  };

  TempStorage_& storage;
  int linear_tid;

  // block-uniform selection state
  struct state_t
  {
    float lo;
    float scale;
    u32 u_mask;
    u32 u_prefix;
    u32 k_mask;
    u32 k_prefix;
    int k_remaining;
    int num_candidates;
  };

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE int item_slot(int i) const
  {
    return i * threads_per_block + linear_tid;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static constexpr int bin_slot(int bin)
  {
    if constexpr (striped_histogram)
    {
      return (bin % buckets_per_thread) * threads_per_block + bin / buckets_per_thread;
    }
    else
    {
      return bin;
    }
  }

  // order-preserving unsigned image of an fp32 value (the radix twiddle)
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static u32 twiddle(float v)
  {
    const u32 b = ::cuda::std::bit_cast<u32>(v);
    const u32 m = static_cast<u32>(static_cast<::cuda::std::int32_t>(b) >> 31) | 0x80000000u;
    return b ^ m;
  }
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static float untwiddle(u32 t)
  {
    const u32 m = (static_cast<::cuda::std::int32_t>(t) >= 0) ? 0xFFFFFFFFu : 0x80000000u;
    return ::cuda::std::bit_cast<float>(t ^ m);
  }

  // fixed-point image of a value over the bracket: bits(round((v - lo) * scale + 2^23)) clamped below
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static u32 fixed_point(float v, float lo, float scale)
  {
    const float f = fmaf(v - lo, scale, 8388608.0f);
    const auto b  = ::cuda::std::bit_cast<::cuda::std::int32_t>(f);
    return static_cast<u32>((::cuda::std::max) (b, static_cast<::cuda::std::int32_t>(u_lo)));
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void zero_histogram(histo_counter_t (&histogram)[num_buckets])
  {
    int histo_offset = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + threads_per_block <= num_buckets; histo_offset += threads_per_block)
    {
      histogram[histo_offset + linear_tid] = 0;
    }
    if ((num_buckets % threads_per_block != 0) && (histo_offset + linear_tid < num_buckets))
    {
      histogram[histo_offset + linear_tid] = 0;
    }
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  scan_and_choose_bucket(const histo_counter_t (&histogram)[num_buckets], histo_counter_t k, int pass)
  {
    histo_counter_t counts[buckets_per_thread]{};
    const int base = linear_tid * buckets_per_thread;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        counts[i] = histogram[bin_slot(bin_idx)];
      }
    }
    histo_counter_t exclusive_sums[buckets_per_thread];
    block_scan_t(storage.passes.scan_temp_storage).ExclusiveSum(counts, exclusive_sums);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        const histo_counter_t exclusive = exclusive_sums[i];
        const histo_counter_t inclusive = exclusive + counts[i];
        if (exclusive < k && inclusive >= k)
        {
          storage.pass_state.bucket     = bin_idx;
          storage.pass_state.candidates = counts[i];
          storage.pass_state.selected   = exclusive;
        }
      }
    }
    // Reset the next pass's min and max trackers (their last readers precede this pass's histogram barrier)
    if (linear_tid == 0)
    {
      storage.key_min[(pass + 1) & 1] = 0xFFFFFFFFu;
      storage.key_max[(pass + 1) & 1] = 0u;
    }
  }

  // One pass over the fixed-point digits [shift, shift + bits) of u. TrackMinMax: candidates also record
  // the min and max of their key bits. Returns true when the pivot bucket holds exactly the remaining k.
  template <bool TrackMinMax>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  run_u_pass(const float (&vals)[items_per_thread], state_t& st, int pass, int shift, int bits)
  {
    histo_counter_t(&histogram)[num_buckets] = storage.passes.histogram[pass & 1];
    const u32 digit_mask                      = (1u << bits) - 1u;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const u32 u = fixed_point(vals[i], st.lo, st.scale);
      if ((u & st.u_mask) == st.u_prefix)
      {
        const u32 digit = (u >> shift) & digit_mask;
        atomicAdd_block(&histogram[bin_slot(static_cast<int>(digit_mask - digit))], histo_counter_t{1});
        if constexpr (TrackMinMax)
        {
          const u32 t = twiddle(vals[i]);
          atomicMin_block(&storage.key_min[pass & 1], t);
          atomicMax_block(&storage.key_max[pass & 1], t);
        }
      }
    }
    zero_histogram(storage.passes.histogram[(pass + 1) & 1]);
    __syncthreads();

    scan_and_choose_bucket(histogram, static_cast<histo_counter_t>(st.k_remaining), pass);
    __syncthreads();

    st.k_remaining -= storage.pass_state.selected;
    st.num_candidates = storage.pass_state.candidates;
    const u32 kth_digit = digit_mask - static_cast<u32>(storage.pass_state.bucket);
    st.u_prefix |= kth_digit << shift;
    st.u_mask |= digit_mask << shift;
    return st.num_candidates == st.k_remaining;
  }

  // One pass over byte `byte` of the order-preserving key bits, restricted to the candidates of the u passes
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  run_key_pass(const float (&vals)[items_per_thread], state_t& st, int pass, int byte)
  {
    histo_counter_t(&histogram)[num_buckets] = storage.passes.histogram[pass & 1];
    const int shift                           = byte * 8;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const u32 u = fixed_point(vals[i], st.lo, st.scale);
      const u32 t = twiddle(vals[i]);
      if ((u & st.u_mask) == st.u_prefix && (t & st.k_mask) == st.k_prefix)
      {
        const u32 digit = (t >> shift) & 0xFFu;
        atomicAdd_block(&histogram[bin_slot(static_cast<int>(0xFFu - digit))], histo_counter_t{1});
      }
    }
    zero_histogram(storage.passes.histogram[(pass + 1) & 1]);
    __syncthreads();

    scan_and_choose_bucket(histogram, static_cast<histo_counter_t>(st.k_remaining), pass);
    __syncthreads();

    st.k_remaining -= storage.pass_state.selected;
    st.num_candidates = storage.pass_state.candidates;
    const u32 kth_digit = 0xFFu - static_cast<u32>(storage.pass_state.bucket);
    st.k_prefix |= kth_digit << shift;
    st.k_mask |= 0xFFu << shift;
    return st.num_candidates == st.k_remaining;
  }

  // Deterministic tie-break among the candidates (exact ties), see block_topk_b1::resolve_ties
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void resolve_ties(
    const bool (&cand)[items_per_thread], int take, tie_mask_t& taken, int (&tie_rank)[items_per_thread])
  {
    const int warp         = linear_tid / 32;
    const int lane         = linear_tid % 32;
    const unsigned lt_mask = (1u << lane) - 1u;
    taken                  = 0;
    unsigned ballots[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      ballots[i] = __ballot_sync(0xffffffffu, cand[i]);
      if (lane == (i % 32))
      {
        storage.tie_table[warp][i] = __popc(ballots[i]);
      }
    }
    __syncthreads();
    constexpr int rounds = ::cuda::ceil_div(items_per_thread, 32);
    int base_lane[rounds];
    int running = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int rnd = 0; rnd < rounds; ++rnd)
    {
      const int slot      = rnd * 32 + lane;
      const bool has_slot = slot < items_per_thread;
      int total           = 0;
      int wpre            = 0;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int w = 0; w < num_warps; ++w)
      {
        const int v = has_slot ? static_cast<int>(storage.tie_table[w][has_slot ? slot : 0]) : 0;
        wpre += (w < warp) ? v : 0;
        total += v;
      }
      constexpr int slots_in_round = (items_per_thread < 32) ? items_per_thread : 32;
      int incl                     = total;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int d = 1; d < slots_in_round; d <<= 1)
      {
        const int n = __shfl_up_sync(0xffffffffu, incl, d);
        if (lane >= d)
        {
          incl += n;
        }
      }
      base_lane[rnd] = running + (incl - total) + wpre;
      running += __shfl_sync(0xffffffffu, incl, slots_in_round - 1);
    }
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const int base = __shfl_sync(0xffffffffu, base_lane[i / 32], i % 32);
      tie_rank[i]    = base + __popc(ballots[i] & lt_mask);
      taken |= static_cast<tie_mask_t>(tie_rank[i] < take) << i;
    }
  }

  // Scatter: class of every item against the final state, tickets for the selected items
  template <bool WithKeyPrefix>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  scatter(const float (&vals)[items_per_thread], const state_t& st, int total_selected, int k, int* out_idx)
  {
    const bool select_all_candidates = st.num_candidates + total_selected == k;
    bool is_sel[items_per_thread];
    bool is_cand[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const u32 up = fixed_point(vals[i], st.lo, st.scale) & st.u_mask;
      if constexpr (WithKeyPrefix)
      {
        const u32 kp = twiddle(vals[i]) & st.k_mask;
        is_sel[i]    = up > st.u_prefix || (up == st.u_prefix && kp > st.k_prefix);
        is_cand[i]   = up == st.u_prefix && kp == st.k_prefix;
      }
      else
      {
        is_sel[i]  = up > st.u_prefix;
        is_cand[i] = up == st.u_prefix;
      }
    }

    [[maybe_unused]] tie_mask_t taken = 0;
    [[maybe_unused]] int tie_rank[items_per_thread];
    if constexpr (deterministic_ties)
    {
      if (!select_all_candidates)
      {
        resolve_ties(is_cand, k - total_selected, taken, tie_rank);
      }
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      if constexpr (deterministic_ties)
      {
        if (is_sel[i] || (is_cand[i] && select_all_candidates))
        {
          const auto ticket = atomicAdd_block(&storage.selected_offset[0], histo_counter_t{1});
          out_idx[ticket]   = item_slot(i);
        }
        else if (is_cand[i] && ((taken >> i) & 1))
        {
          out_idx[total_selected + tie_rank[i]] = item_slot(i);
        }
      }
      else
      {
        const int item_class = (!select_all_candidates) && is_cand[i] ? 1 : 0;
        if (is_sel[i] || is_cand[i])
        {
          const auto ticket = atomicAdd_block(&storage.selected_offset[item_class], histo_counter_t{1});
          const auto pos    = (item_class == 1) ? static_cast<histo_counter_t>(total_selected) + ticket : ticket;
          out_idx[pos]      = item_slot(i);
        }
      }
    }
  }

public:
  using TempStorage = Uninitialized<TempStorage_>;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_n6(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(ThreadsPerBlock, 1, 1))
  {}

  //! Selects the k largest keys of a full striped tile and writes their source indices (item i of thread t
  //! is item i * Threads + t) to out_idx[0, k) in unspecified order.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void select_indices_striped(const KeyT (&keys)[items_per_thread], int k, int* out_idx)
  {
    _CCCL_ASSERT(k > 0 && k < tile_items, "k must be in (0, tile_items)");
    const int warp = linear_tid / 32;
    const int lane = linear_tid % 32;

    // Prologue: fp32 values, per-thread max and min, histogram and counters, bracket partials per warp
    float vals[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      if constexpr (is_bf16)
      {
        vals[i] = __bfloat162float(keys[i]);
      }
      else
      {
        vals[i] = keys[i];
      }
      if constexpr (NormalizeSignedZero)
      {
        vals[i] = vals[i] + 0.0f; // -0.0 + 0.0 is +0.0 in round to nearest
      }
    }
    float tmax = vals[0];
    float tmin = vals[0];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 1; i < items_per_thread; ++i)
    {
      tmax = fmaxf(tmax, vals[i]);
      tmin = fminf(tmin, vals[i]);
    }
    const u32 tmax_t = twiddle(tmax);
    const u32 tmin_t = twiddle(tmin);
    const u32 w_hi   = __reduce_max_sync(0xffffffffu, tmax_t);
    const u32 w_lo   = __reduce_min_sync(0xffffffffu, tmax_t);
    const u32 w_min  = __reduce_min_sync(0xffffffffu, tmin_t);
    if (lane == 0)
    {
      storage.bracket[warp][0] = w_hi;
      storage.bracket[warp][1] = w_lo;
      storage.bracket[warp][2] = w_min;
    }
    if (linear_tid == 0)
    {
      storage.selected_offset[0] = 0;
      storage.selected_offset[1] = 0;
      storage.key_min[1]         = 0xFFFFFFFFu;
      storage.key_max[1]         = 0u;
    }
    zero_histogram(storage.passes.histogram[0]);
    __syncthreads();

    u32 hi_t = storage.bracket[0][0];
    u32 lo_t = storage.bracket[0][k <= threads_per_block ? 1 : 2];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int w = 1; w < num_warps; ++w)
    {
      hi_t = (::cuda::std::max) (hi_t, storage.bracket[w][0]);
      lo_t = (::cuda::std::min) (lo_t, storage.bracket[w][k <= threads_per_block ? 1 : 2]);
    }
    constexpr float flt_max = 3.4028234663852886e38f;
    const float hi          = fminf(untwiddle(hi_t), flt_max);
    const float lo          = fmaxf(untwiddle(lo_t), -flt_max);
    state_t st;
    st.lo             = lo;
    st.scale          = (hi > lo) ? (8388600.0f / (hi - lo)) : 0.0f;
    st.u_mask         = 0;
    st.u_prefix       = 0;
    st.k_mask         = 0;
    st.k_prefix       = 0;
    st.k_remaining    = k;
    st.num_candidates = tile_items;

    int pass        = 0;
    bool done       = run_u_pass<false>(vals, st, pass++, 15, 8);
    bool key_passes = false;
    if (!done)
    {
      int start_byte = 3;
      if (st.num_candidates <= tile_items / 4)
      {
        done = run_u_pass<true>(vals, st, pass++, 7, 8);
        if (!done)
        {
          u32 kmin = storage.key_min[(pass - 1) & 1];
          u32 kmax = storage.key_max[(pass - 1) & 1];
          done     = (kmin == kmax);
          if (!done)
          {
            done = run_u_pass<true>(vals, st, pass++, 0, 7);
            if (!done)
            {
              kmin = storage.key_min[(pass - 1) & 1];
              kmax = storage.key_max[(pass - 1) & 1];
              done = (kmin == kmax);
              if (!done)
              {
                start_byte = (31 - ::cuda::std::countl_zero(kmin ^ kmax)) / 8;
              }
            }
          }
        }
      }
      if (!done)
      {
        key_passes = true;
        for (int byte = start_byte; byte >= lowest_key_byte; --byte)
        {
          if (run_key_pass(vals, st, pass++, byte))
          {
            break;
          }
        }
      }
    }
    const int total_selected = k - st.k_remaining;
    if (key_passes)
    {
      scatter<true>(vals, st, total_selected, k, out_idx);
    }
    else
    {
      scatter<false>(vals, st, total_selected, k, out_idx);
    }
  }
};
} // namespace topk_b1
