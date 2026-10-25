// Shared helpers for the W1 adapters (approach W1, bitonic network top-k).
//
// Packing: a key and its source index go into one unsigned word whose integer order is the order
// (key descending, index ascending) that the prefer_smaller_index requirement asks for. The
// floating key is made order preserving by the usual sign flip (positive keys get the sign bit
// set, negative keys are complemented), -0.0 is mapped to +0.0 first so that the two zeros compare
// equal, and the index enters inverted so that among equal keys the smaller index gives the larger
// word. A keys-only network that keeps the largest words then selects the right set with the
// tie-break for free, and the output index is read back from the low bits.
//
// NaN: a NaN key twiddles to a word above every finite key (positive NaN) or below (negative
// NaN), so a NaN would be selected or dropped deterministically by its sign bit. Focus-set inputs
// have no NaN.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

namespace w1
{
template <typename KeyT>
struct packer;

// bf16 key in the high half, inverted 16-bit index in the low half, N at most 65536.
template <>
struct packer<__nv_bfloat16>
{
  using word_t                   = uint32_t;
  static constexpr int max_items = 65536;

  __device__ __forceinline__ static word_t pack(__nv_bfloat16 key, int idx)
  {
    uint16_t b;
    memcpy(&b, &key, sizeof(b));
    uint32_t u          = b;
    u                   = (u == 0x8000u) ? 0u : u; // -0.0 equals +0.0
    const uint32_t sign = 0u - (u >> 15); // all ones for a negative key
    const uint32_t t    = (u ^ (sign | 0x8000u)) & 0xffffu;
    return (t << 16) | (0xffffu - static_cast<uint32_t>(idx));
  }
  __device__ __forceinline__ static int index(word_t w)
  {
    return static_cast<int>(0xffffu - (w & 0xffffu));
  }
};

// fp32 key in the high 32 bits, inverted 32-bit index in the low 32 bits.
template <>
struct packer<float>
{
  using word_t                   = uint64_t;
  static constexpr int max_items = 1 << 30;

  __device__ __forceinline__ static word_t pack(float key, int idx)
  {
    uint32_t u;
    memcpy(&u, &key, sizeof(u));
    u                   = (u == 0x80000000u) ? 0u : u; // -0.0 equals +0.0
    const uint32_t sign = 0u - (u >> 31);
    const uint32_t t    = u ^ (sign | 0x80000000u);
    return (static_cast<uint64_t>(t) << 32) | (0xffffffffu - static_cast<uint32_t>(idx));
  }
  __device__ __forceinline__ static int index(word_t w)
  {
    return static_cast<int>(0xffffffffu - static_cast<uint32_t>(w));
  }
};

// Integer order helpers for the network. 32-bit words go through min and max (one VIMNMX each),
// 64-bit words through a compare and selects since there is no 64-bit min max instruction.
template <typename W>
struct word_ops;

template <>
struct word_ops<uint32_t>
{
  __device__ __forceinline__ static uint32_t max(uint32_t a, uint32_t b)
  {
    return ::max(a, b);
  }
  // keep the larger or the smaller of the own and the partner word
  __device__ __forceinline__ static void exchange(uint32_t& k, uint32_t other, bool keep_max)
  {
    k = keep_max ? ::max(k, other) : ::min(k, other);
  }
  // two words of one lane: a gets the smaller one unless Desc
  template <bool Desc>
  __device__ __forceinline__ static void exchange_regs(uint32_t& a, uint32_t& b)
  {
    const uint32_t lo = ::min(a, b);
    const uint32_t hi = ::max(a, b);
    a                 = Desc ? hi : lo;
    b                 = Desc ? lo : hi;
  }
};

template <>
struct word_ops<uint64_t>
{
  __device__ __forceinline__ static uint64_t max(uint64_t a, uint64_t b)
  {
    return a > b ? a : b;
  }
  __device__ __forceinline__ static void exchange(uint64_t& k, uint64_t other, bool keep_max)
  {
    const bool gt = other > k;
    k             = (gt == keep_max) ? other : k;
  }
  template <bool Desc>
  __device__ __forceinline__ static void exchange_regs(uint64_t& a, uint64_t& b)
  {
    const bool sw     = (a > b) != Desc;
    const uint64_t na = sw ? b : a;
    const uint64_t nb = sw ? a : b;
    a                 = na;
    b                 = nb;
  }
};

template <typename W>
__device__ __forceinline__ W shfl_xor(W v, int stride, unsigned mask)
{
  return __shfl_xor_sync(mask, v, stride);
}

__host__ __device__ constexpr int log2_int(int x)
{
  int r = 0;
  while ((1 << r) < x)
  {
    ++r;
  }
  return r;
}
__host__ __device__ constexpr bool is_pow2(int x)
{
  return x > 0 && (x & (x - 1)) == 0;
}
} // namespace w1
