// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// N9 prototype (idea card ideas/queue/N9_w2_finish_pivot_bin.md): approaches/B1/block_topk_n6.cuh with one
// change, the second pass is replaced by a packed redux finish on one warp when the pivot bin after the
// first pass holds at most 32 items and the pass did not end the selection. The pivot bin's members take a
// ticket and write their (key, index) word to a 32-word buffer, warp 0 loads one word per lane and runs
// k_rem rounds of redux.max (one redux per round on the packed bf16 word, two for fp32: the key, then the
// inverted index among the holders of the key), which is exact under both requirements. The items above
// the pivot bin are scattered by ticket as in N6. Everything else is N6 (B1_n6b, per-bucket tracking).
//
// N6 as copied:
// block top-k by selection on a fixed-point image of the keys instead of on their radix digits.
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

#include "../B1/block_topk_b1.cuh"

namespace topk_b1
{

// PerBucketTracking: from the second pass on, the candidates record the min and max of their key bits per
// histogram bucket (two fire-and-forget atomics per candidate item), so the tie-class exit tests the pivot
// bucket alone. Without it the min and max are tracked over all candidates of the pass, which only detects a
// tie class once it is the whole candidate set (one pass later when the pivot bucket is a tie class inside a
// larger candidate set, the bf16 case).
template <typename KeyT,
          int ThreadsPerBlock,
          int ItemsPerThread,
          bool NormalizeSignedZero = false,
          tie_break TieBreak       = tie_break::none,
          bool PerBucketTracking   = false>
class block_topk_n6w2
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
  // N9: pivot bins up to this size finish on one warp
  static constexpr int fin      = 32;
  static constexpr u32 inv_mask = 0xFFFFu; // inverted index, tile_items <= 65536
  static_assert(tile_items <= 65536, "the inverted index is kept in 16 bits");

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
    // double-buffered min and max of the candidates' order-preserving key bits, over all candidates of the
    // pass or per histogram bucket
    u32 key_min[2][PerBucketTracking ? num_buckets : 1];
    u32 key_max[2][PerBucketTracking ? num_buckets : 1];
    histo_counter_t selected_offset[2];
    histo_counter_t tie_table[deterministic_ties ? num_warps : 1][deterministic_ties ? items_per_thread : 1];
    // N9: the pivot bin's (key, index) words for the warp finish, filled by ticket
    u32 pivot_key[fin];
    u32 pivot_inv[fin];
    histo_counter_t pivot_ticket;
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

  // fixed-point image of a value over the bracket: bits(round((v - lo) * scale + 2^23)) clamped into
  // [u_lo, u_lo + 2^23). Values below the bracket (and -inf) land on u_lo, +inf on the top.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static u32 fixed_point(float v, float lo, float scale)
  {
    const float f = fmaf(v - lo, scale, 8388608.0f);
    const auto b  = ::cuda::std::bit_cast<::cuda::std::int32_t>(f);
    const auto c  = (::cuda::std::min) ((::cuda::std::max) (b, static_cast<::cuda::std::int32_t>(u_lo)),
                                        static_cast<::cuda::std::int32_t>(u_lo + 0x7FFFFFu));
    return static_cast<u32>(c);
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
    if constexpr (!PerBucketTracking)
    {
      if (linear_tid == 0)
      {
        storage.key_min[(pass + 1) & 1][0] = 0xFFFFFFFFu;
        storage.key_max[(pass + 1) & 1][0] = 0u;
      }
    }
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void reset_trackers(int buffer)
  {
    if constexpr (PerBucketTracking)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int off = 0; off < num_buckets; off += threads_per_block)
      {
        if (off + linear_tid < num_buckets)
        {
          storage.key_min[buffer][off + linear_tid] = 0xFFFFFFFFu;
          storage.key_max[buffer][off + linear_tid] = 0u;
        }
      }
    }
    else
    {
      if (linear_tid == 0)
      {
        storage.key_min[buffer][0] = 0xFFFFFFFFu;
        storage.key_max[buffer][0] = 0u;
      }
    }
  }

  // One pass over the fixed-point digits [shift, shift + bits) of u. TrackMinMax: candidates also record
  // the min and max of their key bits. Returns true when the pivot bucket holds exactly the remaining k.
  // The bracket parameters are passed through an empty asm statement per pass, so the compiler recomputes
  // the fixed-point image in every pass (three instructions per item) instead of keeping one image per item
  // live across the passes, which doubles the register footprint and spills at large items per thread.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static void opaque(float& lo, float& scale)
  {
    asm volatile("" : "+f"(lo), "+f"(scale));
  }

  template <bool TrackMinMax>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool
  run_u_pass(const float (&vals)[items_per_thread], state_t& st, int pass, int shift, int bits)
  {
    histo_counter_t(&histogram)[num_buckets] = storage.passes.histogram[pass & 1];
    const u32 digit_mask                      = (1u << bits) - 1u;
    float lo = st.lo, scale = st.scale;
    opaque(lo, scale);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const u32 u = fixed_point(vals[i], lo, scale);
      if ((u & st.u_mask) == st.u_prefix)
      {
        const u32 digit  = (u >> shift) & digit_mask;
        const int bucket = static_cast<int>(digit_mask - digit);
        atomicAdd_block(&histogram[bin_slot(bucket)], histo_counter_t{1});
        if constexpr (TrackMinMax)
        {
          const u32 t     = twiddle(vals[i]);
          const int tslot = PerBucketTracking ? bucket : 0;
          atomicMin_block(&storage.key_min[pass & 1][tslot], t);
          atomicMax_block(&storage.key_max[pass & 1][tslot], t);
        }
      }
    }
    zero_histogram(storage.passes.histogram[(pass + 1) & 1]);
    if constexpr (PerBucketTracking)
    {
      reset_trackers((pass + 1) & 1);
    }
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
    float lo = st.lo, scale = st.scale;
    opaque(lo, scale);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const u32 u = fixed_point(vals[i], lo, scale);
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

  // Scatter: class of every item against the final state, tickets for the selected items.
  // SkipCandidates (N9): the pivot bin's items are not scattered here, they go to the warp finish buffer.
  template <bool WithKeyPrefix, bool SkipCandidates = false>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  scatter(const float (&vals)[items_per_thread], const state_t& st, int total_selected, int k, int* out_idx)
  {
    const bool select_all_candidates = !SkipCandidates && (st.num_candidates + total_selected == k);
    float lo = st.lo, scale = st.scale;
    opaque(lo, scale);
    auto classify = [&](int i, bool& sel, bool& cand) {
      const u32 up = fixed_point(vals[i], lo, scale) & st.u_mask;
      if constexpr (WithKeyPrefix)
      {
        const u32 kp = twiddle(vals[i]) & st.k_mask;
        sel          = up > st.u_prefix || (up == st.u_prefix && kp > st.k_prefix);
        cand         = up == st.u_prefix && kp == st.k_prefix;
      }
      else
      {
        sel  = up > st.u_prefix;
        cand = up == st.u_prefix;
      }
    };

    [[maybe_unused]] tie_mask_t taken = 0;
    [[maybe_unused]] int tie_rank[items_per_thread];
    if constexpr (deterministic_ties && !SkipCandidates)
    {
      if (!select_all_candidates)
      {
        bool is_cand[items_per_thread];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < items_per_thread; ++i)
        {
          bool sel;
          classify(i, sel, is_cand[i]);
        }
        resolve_ties(is_cand, k - total_selected, taken, tie_rank);
      }
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      bool is_sel_i, is_cand_i;
      classify(i, is_sel_i, is_cand_i);
      if constexpr (SkipCandidates)
      {
        if (is_sel_i)
        {
          const auto ticket = atomicAdd_block(&storage.selected_offset[0], histo_counter_t{1});
          out_idx[ticket]   = item_slot(i);
        }
        else if (is_cand_i)
        {
          const auto t         = atomicAdd_block(&storage.pivot_ticket, histo_counter_t{1});
          storage.pivot_key[t] = twiddle(vals[i]);
          storage.pivot_inv[t] = inv_mask - static_cast<u32>(item_slot(i));
        }
      }
      else if constexpr (deterministic_ties)
      {
        if (is_sel_i || (is_cand_i && select_all_candidates))
        {
          // see block_topk_b1: an index the compiler cannot prove uniform keeps the plain per-item atomic
          int sel_class = 0;
          asm volatile("" : "+r"(sel_class));
          const auto ticket = atomicAdd_block(&storage.selected_offset[sel_class], histo_counter_t{1});
          out_idx[ticket]   = item_slot(i);
        }
        else if (is_cand_i && ((taken >> i) & 1))
        {
          out_idx[total_selected + tie_rank[i]] = item_slot(i);
        }
      }
      else
      {
        const int item_class = (!select_all_candidates) && is_cand_i ? 1 : 0;
        if (is_sel_i || is_cand_i)
        {
          // Ties draw tickets beyond the remaining k as well, only the first k_remaining of them are written
          const auto ticket = atomicAdd_block(&storage.selected_offset[item_class], histo_counter_t{1});
          const auto pos    = (item_class == 1) ? static_cast<histo_counter_t>(total_selected) + ticket : ticket;
          if (item_class == 0 || static_cast<int>(ticket) < k - total_selected)
          {
            out_idx[pos] = item_slot(i);
          }
        }
      }
    }
  }

  // N9: warp 0 extracts k_rem winners from the count pivot words by (key descending, index ascending)
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void warp_finish(int count, int k_rem, int* out)
  {
    const int lane = linear_tid;
    const bool has = lane < count;
    if constexpr (is_bf16)
    {
      // the bf16 key image is the top half of the fp32 image, the inverted index the low half
      u32 w = has ? ((storage.pivot_key[lane] & 0xFFFF0000u) | storage.pivot_inv[lane]) : 0u;
      for (int r = 0; r < k_rem; ++r)
      {
        const u32 m = __reduce_max_sync(0xffffffffu, w);
        if (lane == (r & 31))
        {
          out[r] = static_cast<int>(inv_mask - (m & 0xFFFFu));
        }
        w = (w == m) ? 0u : w;
      }
    }
    else
    {
      const u32 tk  = has ? storage.pivot_key[lane] : 0u;
      const u32 inv = has ? storage.pivot_inv[lane] : 0u;
      bool alive    = has;
      for (int r = 0; r < k_rem; ++r)
      {
        const u32 m   = __reduce_max_sync(0xffffffffu, alive ? tk : 0u);
        const bool hm = alive && (tk == m);
        const u32 i   = __reduce_max_sync(0xffffffffu, hm ? inv : 0u);
        if (lane == (r & 31))
        {
          out[r] = static_cast<int>(inv_mask - i);
        }
        alive = alive && !(hm && inv == i);
      }
    }
  }

public:
  using TempStorage = Uninitialized<TempStorage_>;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_n6w2(TempStorage& storage)
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
      storage.pivot_ticket       = 0;
    }
    zero_histogram(storage.passes.histogram[0]);
    reset_trackers(1);
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
    // The scale is finite and nonzero in every case (a zero scale would turn an infinite item into NaN):
    // the degenerate brackets (empty width, or a width that overflows fp32) fall back to the key passes.
    const float width = hi - lo;
    st.lo             = lo;
    st.scale          = (width > 0.0f && width < flt_max) ? (8388600.0f / width) : (width > 0.0f ? 1.0e-32f : 1.0f);
    st.u_mask         = 0;
    st.u_prefix       = 0;
    st.k_mask         = 0;
    st.k_prefix       = 0;
    st.k_remaining    = k;
    st.num_candidates = tile_items;

    int pass        = 0;
    bool done       = run_u_pass<false>(vals, st, pass++, 15, 8);
    bool key_passes = false;
    // N9: a small pivot bin finishes on one warp instead of a second pass
    const bool w2 = !done && (st.num_candidates <= fin);
    if (w2)
    {
      const int total_selected = k - st.k_remaining;
      scatter<false, true>(vals, st, total_selected, k, out_idx);
      __syncthreads();
      if (linear_tid < 32)
      {
        warp_finish(st.num_candidates, st.k_remaining, out_idx + total_selected);
      }
      return;
    }
    if (!done)
    {
      int start_byte = 3;
      if (st.num_candidates <= tile_items / 4)
      {
        done = run_u_pass<true>(vals, st, pass++, 7, 8);
        if (!done)
        {
          u32 kmin = storage.key_min[(pass - 1) & 1][PerBucketTracking ? storage.pass_state.bucket : 0];
          u32 kmax = storage.key_max[(pass - 1) & 1][PerBucketTracking ? storage.pass_state.bucket : 0];
          done     = (kmin == kmax);
          if (!done)
          {
            done = run_u_pass<true>(vals, st, pass++, 0, 7);
            if (!done)
            {
              kmin = storage.key_min[(pass - 1) & 1][PerBucketTracking ? storage.pass_state.bucket : 0];
              kmax = storage.key_max[(pass - 1) & 1][PerBucketTracking ? storage.pass_state.bucket : 0];
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
