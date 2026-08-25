// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! `identify_candidates_op_t` -- classifies an input key against the running k-th-key boundary
//! (selected / candidate / rejected) for a given pass. Shared by the single-problem and batched
//! top-k dispatch paths.

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
#include <cub/detail/topk/candidate_class.cuh>
#include <cub/detail/topk/key_prefix_storage.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
// Check if the input element is still a candidate for the target pass.
template <typename T,
          select SelectDirection,
          int BitsPerPass,
          typename DecomposerT,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct identify_candidates_op_t;

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT>
struct identify_candidates_op_t<T, SelectDirection, BitsPerPass, DecomposerT, true>
{
  using unsigned_bits_t = typename Traits<T>::UnsignedBits;
  using key_prefix_t    = key_prefix_storage_t<T>;
  unsigned_bits_t* kth_key_bits;
  int start_bit;
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  identify_candidates_op_t(key_prefix_t* kth_key_bits, int pass, int /*total_bits*/, DecomposerT /*decomposer*/)
      : kth_key_bits(&kth_key_bits->bits)
  {
    start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE candidate_class operator()(T key) const
  {
    auto bits = reinterpret_cast<unsigned_bits_t&>(key);
    bits      = Traits<T>::TwiddleIn(bits);

    if constexpr (SelectDirection != select::min)
    {
      bits = ~bits;
    }

    bits = (bits >> start_bit) << start_bit;

    return (bits < *kth_key_bits) ? candidate_class::selected
         : (bits == *kth_key_bits)
           ? candidate_class::candidate
           : candidate_class::rejected;
  }
};

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT>
struct identify_candidates_op_t<T, SelectDirection, BitsPerPass, DecomposerT, false>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using radix_traits_t                = detail::radix::traits_t<T>;
  using bit_ordered_type              = typename radix_traits_t::bit_ordered_type;
  using key_prefix_t                  = key_prefix_storage_t<T>;

  key_prefix_t* kth_key_bits{};
  int pass{};
  int total_bits{};
  DecomposerT decomposer{};

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  identify_candidates_op_t(key_prefix_t* kth_key_bits, int pass, int total_bits, DecomposerT decomposer)
      : kth_key_bits(kth_key_bits)
      , pass(pass)
      , total_bits(total_bits)
      , decomposer(decomposer)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE candidate_class operator()(T key) const
  {
    if (pass <= 0)
    {
      return candidate_class::candidate;
    }

    bit_ordered_type ordered = key;
    ordered                  = RadixSortTwiddle<is_descending, T>::In(ordered, decomposer);

    // Build the key's prefix using the same funnel shift as set_kth_key_bits
    key_prefix_t key_prefix{};
    for (int prefix_pass = 0; prefix_pass < pass; ++prefix_pass)
    {
      const int start_bit = calc_start_bit<BitsPerPass>(total_bits, prefix_pass);
      const int num_bits =
        calc_start_bit<BitsPerPass>(total_bits, prefix_pass - 1) - calc_start_bit<BitsPerPass>(total_bits, prefix_pass);
      auto extractor =
        radix_traits_t::template digit_extractor<ShiftDigitExtractor<T>>(start_bit, num_bits, decomposer);
      key_prefix.shift_or(BitsPerPass, static_cast<unsigned int>(extractor.Digit(ordered)));
    }

    // Compare word-by-word from MSB to LSB
    const int total_prefix_bits = pass * BitsPerPass;
    const int top_word_idx      = (total_prefix_bits - 1) / 32;
    const int bits_in_top_word  = ((total_prefix_bits - 1) % 32) + 1;

    // Top word may be partially filled
    {
      unsigned int key_w = key_prefix.words[top_word_idx];
      unsigned int kth_w = kth_key_bits->words[top_word_idx];
      if (bits_in_top_word < 32)
      {
        const unsigned int mask = (1u << bits_in_top_word) - 1u;
        key_w &= mask;
        kth_w &= mask;
      }
      if (key_w < kth_w)
      {
        return candidate_class::selected;
      }
      if (key_w > kth_w)
      {
        return candidate_class::rejected;
      }
    }

    // Remaining words are fully populated
    for (int w = top_word_idx - 1; w >= 0; --w)
    {
      if (key_prefix.words[w] < kth_key_bits->words[w])
      {
        return candidate_class::selected;
      }
      if (key_prefix.words[w] > kth_key_bits->words[w])
      {
        return candidate_class::rejected;
      }
    }

    return candidate_class::candidate;
  }
};
} // namespace detail::topk

CUB_NAMESPACE_END
