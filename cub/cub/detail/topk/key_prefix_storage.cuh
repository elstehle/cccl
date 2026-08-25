// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Radix key-prefix helpers shared by the single-problem and batched top-k implementations:
//!   - `key_prefix_storage_t` -- storage for the already-resolved high bits of the running k-th
//!     key (it backs each problem's / segment's per-pass `kth_key_bits` counter field); two
//!     shapes, selected by whether the key type twiddles into a single unsigned word.
//!   - `calc_start_bit` -- the start bit (LSB-indexed) processed by a given radix pass.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
template <typename KeyT, bool CanTwiddle = detail::radix::can_twiddle<KeyT>>
struct key_prefix_storage_t;

template <typename KeyT>
struct key_prefix_storage_t<KeyT, true>
{
  using bits_t = typename Traits<KeyT>::UnsignedBits;
  bits_t bits;
};

// Bit-vector for accumulating prefix digits via funnel shift. Each pass shifts the existing
// contents left by BitsPerPass and ORs the new bucket at the bottom. Sized to hold all
// decomposed bits of KeyT plus headroom for the shift padding of the last pass.
template <typename KeyT>
struct key_prefix_storage_t<KeyT, false>
{
  static constexpr int num_words = ::cuda::ceil_div<int>(sizeof(KeyT) * 8 + 31, 32);
  unsigned int words[num_words];

  // Funnel-shifts the entire bit-vector left by `shift` positions and inserts `value` into the
  // vacated low bits. Each word receives carry bits from its lower neighbor (high-to-low order
  // so each word reads its neighbor's original value). The final word is filled from `value`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void shift_or(int shift, unsigned int value)
  {
    _CCCL_ASSERT(shift > 0 && shift < 32, "shift_or requires 0 < shift < 32");
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = num_words - 1; i > 0; --i)
    {
      words[i] = __funnelshift_l(words[i - 1], words[i], shift);
    }
    words[0] = (words[0] << shift) | value;
  }
};

// Calculates the starting bit for a given pass (bit 0 is the least significant (rightmost) bit).
// We process the input from the most to the least significant bit. This way, we can skip some passes in the end.
template <typename T, int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int calc_start_bit(const int pass)
{
  int start_bit = int{sizeof(T)} * 8 - (pass + 1) * BitsPerPass;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}

template <int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_start_bit(const int total_bits, const int pass)
{
  int start_bit = total_bits - (pass + 1) * BitsPerPass;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}
} // namespace detail::topk

CUB_NAMESPACE_END
