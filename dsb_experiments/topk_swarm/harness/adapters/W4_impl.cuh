// W4 device drivers around W4_core.cuh: the block chunk merge (chunk_adapter) and the warp
// truncated merge sort (wmerge_adapter). Barriers live here only, the per-thread phases are in
// the core so the host test can run them.
//
// Contract (both forms): keys bf16 or fp32, int32 indices derived from the blocked position,
// requirements not_guaranteed and prefer_smaller_index by the same code (the word order is
// (key descending, index ascending), the output is exact and sorted: slot r holds rank r).
// NaN: twiddled bitwise, a positive NaN orders above +inf, a negative NaN below -inf, and the
// all-ones negative NaN collides with the absent-item filler word, never produced by focus-set
// inputs. Minus zero is folded onto plus zero before the twiddle.
//
// Block chunk merge (W4_chunk<M>): block of 256 threads, blocked input in registers whose M
// consecutive runs of N / M items each arrive sorted by (key descending, index ascending),
// N / M a power of two, k a power of two with k <= N / 2 and every merge level splitting evenly
// over the threads (w4::tree_valid). Temp storage: two run buffers (level parity, fp32 as key
// and index halves, the level 0 keys without an index array) plus a 256-int split exchange,
// 2 log2(M) + 1 __syncthreads per call. The
// input runs are not checked, an unsorted run gives a wrong result (the oracle sees it).
//
// Warp truncated merge sort (W4_wmerge): one warp, blocked input of N = 32 x IPT keys with IPT a
// power of two in [2, 64], k a power of two in [32, N / 2]. Per-lane Batcher sort of IPT words,
// then 5 merge-path levels through shared memory truncated at k. Temp storage two run buffers,
// 6 __syncwarp per call, one __shfl_up per level, no barriers.
#pragma once

#include <cuda_bf16.h>

#include "../topk_harness.cuh"
#include "W4_core.cuh"

namespace w4
{

template <>
struct key_word<__nv_bfloat16>
{
  using W                     = uint32_t;
  static constexpr bool split = false;
  W4_FN static uint32_t tw(__nv_bfloat16 h)
  {
    return twiddle_bf16(static_cast<uint32_t>(__bfloat16_as_ushort(h)));
  }
};

W4_HD inline constexpr bool key_ok(topk::key_type kt, int n)
{
  return (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16 && n <= 65536);
}

W4_HD inline constexpr bool
chunk_supported(topk::key_type kt, int threads, int n, int k, topk::requirement req, int m)
{
  if (!key_ok(kt, n) || threads != 256 || req == topk::requirement::prefer_larger_index)
  {
    return false;
  }
  if (n % threads != 0 || n % m != 0)
  {
    return false;
  }
  const int run_len = n / m;
  const int ipt     = n / threads;
  if (!is_pow2(run_len) || run_len < ipt || run_len % ipt != 0)
  {
    return false;
  }
  return tree_valid(threads, m, run_len, k);
}

W4_HD inline constexpr bool wmerge_supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
{
  if (!key_ok(kt, n) || threads != 32 || req == topk::requirement::prefer_larger_index)
  {
    return false;
  }
  if (n % 32 != 0)
  {
    return false;
  }
  const int ipt = n / 32;
  if (!is_pow2(ipt) || ipt < 2 || ipt > 64)
  {
    return false;
  }
  return tree_valid(32, 32, ipt, k);
}

template <typename KeyT, int Threads, int N, int K, int M>
struct chunk_adapter
{
  using core               = chunk_core<KeyT, Threads, N, K, M>;
  using tree               = typename core::tree;
  using TempStorage        = typename core::TempStorage;
  static constexpr int IPT = core::IPT;

  template <int Lv>
  W4_FN static void levels(int t, TempStorage& ts, int* out_idx)
  {
    if constexpr (Lv < tree::levels)
    {
      using lv        = typename tree::template level<Lv>;
      const int a_end = lv::split_end(t, ts);
      ts.xs[t]        = a_end;
      __syncthreads();
      const int a_beg = (t % lv::TPP == 0) ? 0 : ts.xs[t - 1];
      lv::merge(t, ts, out_idx, a_beg, a_end);
      if constexpr (Lv + 1 < tree::levels)
      {
        __syncthreads();
        levels<Lv + 1>(t, ts, out_idx);
      }
    }
  }

  W4_FN static void run(const KeyT (&keys)[IPT], TempStorage& ts, int* out_idx)
  {
    const int t = static_cast<int>(threadIdx.x);
    core::stage(keys, t, ts);
    __syncthreads();
    levels<0>(t, ts, out_idx);
  }
};

template <typename KeyT, int N, int K>
struct wmerge_adapter
{
  using core               = wmerge_core<KeyT, N, K>;
  using tree               = typename core::tree;
  using TempStorage        = typename core::TempStorage;
  static constexpr int IPT = core::IPT;

  template <int Lv>
  W4_FN static void levels(int lane, TempStorage& ts, int* out_idx)
  {
    if constexpr (Lv < tree::levels)
    {
      using lv        = typename tree::template level<Lv>;
      const int a_end = lv::split_end(lane, ts);
      const int up    = __shfl_up_sync(0xffffffffu, a_end, 1);
      const int a_beg = (lane % lv::TPP == 0) ? 0 : up;
      lv::merge(lane, ts, out_idx, a_beg, a_end);
      if constexpr (Lv + 1 < tree::levels)
      {
        __syncwarp();
        levels<Lv + 1>(lane, ts, out_idx);
      }
    }
  }

  W4_FN static void run(const KeyT (&keys)[IPT], TempStorage& ts, int* out_idx)
  {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    core::prepare(keys, lane, ts);
    __syncwarp();
    levels<0>(lane, ts, out_idx);
  }
};

} // namespace w4
