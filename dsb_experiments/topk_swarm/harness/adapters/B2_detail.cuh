// Shared pieces of the B2 adapters (atomics based iterative extraction at block scope): the
// order-preserving twiddles, the per-thread sorting network and the branchless extraction round.
// Not an adapter by itself, included by the B2_<variant>.cuh headers.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

#include "../topk_harness.cuh"

namespace b2
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

} // namespace b2
