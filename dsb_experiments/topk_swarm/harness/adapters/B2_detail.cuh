// Shared pieces of the B2 adapters (atomics based iterative extraction at block scope): the
// order-preserving twiddles, the per-thread sorting network and the branchless extraction round.
// Not an adapter by itself, included by the B2_<variant>.cuh headers.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

#include "../topk_harness.cuh"

namespace b2b
{

// Order-preserving twiddles in the negate form: a positive key gets its sign bit set, a negative
// key is negated as an unsigned integer. -0.0 and +0.0 both map to 0x8000...0, so they compare
// equal as the oracle demands, and no non-NaN key maps to 0, which keeps the empty-slot sentinel
// 0 of the atomic rounds free. A NaN with all payload bits set is the one key that maps to 0 (32
// bit) or 0 (16 bit, for 0xffff), it is never produced by focus-set inputs.
__device__ __forceinline__ unsigned twiddle32(float f)
{
  const unsigned u = __float_as_uint(f);
  return (static_cast<int>(u) < 0) ? (0u - u) : (u | 0x80000000u);
}

__device__ __forceinline__ unsigned twiddle16(__nv_bfloat16 h)
{
  const unsigned u = static_cast<unsigned>(__bfloat16_as_ushort(h));
  return (u & 0x8000u) ? (0x10000u - u) : (u | 0x8000u);
}

__host__ __device__ inline constexpr int log2_int(int x)
{
  int r = 0;
  while ((1 << r) < x)
  {
    ++r;
  }
  return r;
}

// Descending compare-exchange on unsigned words (two VIMNMX).
__device__ __forceinline__ void cswap_desc(unsigned& a, unsigned& b)
{
  const unsigned hi = a > b ? a : b;
  const unsigned lo = a > b ? b : a;
  a                 = hi;
  b                 = lo;
}

// Batcher odd-even merge sort over a register array, static indices only, N a power of two.
template <int IPT, int Lo, int N, int R>
__device__ __forceinline__ void oe_merge(unsigned (&v)[IPT])
{
  constexpr int M = R * 2;
  if constexpr (M < N)
  {
    oe_merge<IPT, Lo, N, M>(v);
    oe_merge<IPT, Lo + R, N, M>(v);
#pragma unroll
    for (int i = Lo + R; i + R < Lo + N; i += M)
    {
      cswap_desc(v[i], v[i + R]);
    }
  }
  else
  {
    cswap_desc(v[Lo], v[Lo + R]);
  }
}

template <int IPT, int Lo, int N>
__device__ __forceinline__ void oe_sort(unsigned (&v)[IPT])
{
  if constexpr (N > 1)
  {
    constexpr int M = N / 2;
    oe_sort<IPT, Lo, M>(v);
    oe_sort<IPT, Lo + M, M>(v);
    oe_merge<IPT, Lo, N, 1>(v);
  }
}

template <int IPT>
__device__ __forceinline__ void sort_desc(unsigned (&v)[IPT])
{
  oe_sort<IPT, 0, IPT>(v);
}

// One extraction round on a per-thread list sorted descending: post the head, barrier, read the
// block maximum, and shift the list by one where the head was the maximum (predicated selects,
// never a branch on the winner). The sentinel 0 fills from the back. Returns the round maximum.
template <int IPT>
__device__ __forceinline__ unsigned round_shift1(unsigned* slot, unsigned (&p)[IPT])
{
  atomicMax(slot, p[0]);
  __syncthreads();
  const unsigned m = *slot;
  const bool w     = (p[0] == m);
#pragma unroll
  for (int i = 0; i + 1 < IPT; ++i)
  {
    p[i] = w ? p[i + 1] : p[i];
  }
  p[IPT - 1] = w ? 0u : p[IPT - 1];
  return m;
}

// Rounds are fully unrolled up to 64 so slot addresses are immediates, partially above.
template <int K>
struct round_unroll
{
  static constexpr int value = (K <= 64) ? K : 8;
};

// ---- two-level per-thread list: a sorted head of H words and an unsorted rest of IPT - H words
// Partial sort: groups of H are sorted with the Batcher network, then merged pairwise in a tree
// where each merge keeps the H largest of two sorted groups (bitonic max, then a half-cleaner
// sort) in the lower group and leaves the H smallest unsorted in the upper group. Afterwards
// v[0 .. H-1] is the sorted head and v[H .. IPT-1] the rest, every rest word at most the head's
// last word.
template <int IPT, int H, int A, int Bg>
__device__ __forceinline__ void merge_keep_top(unsigned (&v)[IPT])
{
#pragma unroll
  for (int i = 0; i < H; ++i)
  {
    const unsigned a = v[A + i];
    const unsigned b = v[Bg + H - 1 - i];
    v[A + i]         = a > b ? a : b;
    v[Bg + H - 1 - i] = a > b ? b : a;
  }
#pragma unroll
  for (int h = H / 2; h >= 1; h /= 2)
  {
#pragma unroll
    for (int i = 0; i < H; ++i)
    {
      if ((i & h) == 0)
      {
        cswap_desc(v[A + i], v[A + i + h]);
      }
    }
  }
}

template <int IPT, int H, int G, int S>
__device__ __forceinline__ void merge_tree(unsigned (&v)[IPT])
{
  if constexpr (G + S < IPT / H)
  {
    merge_keep_top<IPT, H, G * H, (G + S) * H>(v);
    merge_tree<IPT, H, G + 2 * S, S>(v);
  }
}

template <int IPT, int H, int S>
__device__ __forceinline__ void merge_levels(unsigned (&v)[IPT])
{
  if constexpr (S < IPT / H)
  {
    merge_tree<IPT, H, 0, S>(v);
    merge_levels<IPT, H, 2 * S>(v);
  }
}

template <int IPT, int H, int G>
__device__ __forceinline__ void sort_groups(unsigned (&v)[IPT])
{
  if constexpr (G < IPT / H)
  {
    oe_sort<IPT, G * H, H>(v);
    sort_groups<IPT, H, G + 1>(v);
  }
}

template <int IPT, int H>
__device__ __forceinline__ void partial_sort_top(unsigned (&v)[IPT])
{
  static_assert(IPT % H == 0 && (IPT / H) >= 1, "H must divide IPT");
  sort_groups<IPT, H, 0>(v);
  merge_levels<IPT, H, 1>(v);
}

// Extraction round on the two-level list: post the head, barrier, read, shift the sorted head by
// one where it won (H predicated selects). When the head runs empty and the rest still holds
// words, the winning thread alone moves the largest rest word into the head (a divergent branch
// taken by at most one thread per round, and only after H wins of the same thread).
template <int IPT, int H>
__device__ __forceinline__ unsigned round_shift_twolevel(unsigned* slot, unsigned (&p)[IPT])
{
  atomicMax(slot, p[0]);
  __syncthreads();
  const unsigned m = *slot;
  const bool w     = (p[0] == m);
#pragma unroll
  for (int i = 0; i + 1 < H; ++i)
  {
    p[i] = w ? p[i + 1] : p[i];
  }
  p[H - 1] = w ? 0u : p[H - 1];
  if constexpr (IPT > H)
  {
    if (p[0] == 0u)
    {
      unsigned mx = p[H];
#pragma unroll
      for (int i = H + 1; i < IPT; ++i)
      {
        mx = mx > p[i] ? mx : p[i];
      }
#pragma unroll
      for (int i = H; i < IPT; ++i)
      {
        p[i] = (p[i] == mx) ? 0u : p[i];
      }
      p[0] = mx;
    }
  }
  return m;
}

// Round of the two-level list with a warp pre-reduction: one redux.sync.max per warp and one
// atomic per warp (8 same-address atomics per round instead of 256), the rest as above.
template <int IPT, int H>
__device__ __forceinline__ unsigned round_shift_twolevel_redux(unsigned* slot, unsigned (&p)[IPT])
{
  const unsigned wmax = __reduce_max_sync(0xffffffffu, p[0]);
  if ((threadIdx.x & 31u) == 0u)
  {
    atomicMax(slot, wmax);
  }
  __syncthreads();
  const unsigned m = *slot;
  const bool w     = (p[0] == m);
#pragma unroll
  for (int i = 0; i + 1 < H; ++i)
  {
    p[i] = w ? p[i + 1] : p[i];
  }
  p[H - 1] = w ? 0u : p[H - 1];
  if constexpr (IPT > H)
  {
    if (p[0] == 0u)
    {
      unsigned mx = p[H];
#pragma unroll
      for (int i = H + 1; i < IPT; ++i)
      {
        mx = mx > p[i] ? mx : p[i];
      }
#pragma unroll
      for (int i = H; i < IPT; ++i)
      {
        p[i] = (p[i] == mx) ? 0u : p[i];
      }
      p[0] = mx;
    }
  }
  return m;
}

} // namespace b2b
