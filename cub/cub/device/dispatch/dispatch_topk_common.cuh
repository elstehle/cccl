// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_topk_common.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/detail/topk/block_partition.cuh>
#include <cub/device/dispatch/dispatch_topk_identify_candidates.cuh>
#include <cub/device/dispatch/tuning/tuning_topk.cuh>
#include <cub/util_type.cuh>

#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_output_iterator.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// Bring the generic `detail::topk` symbols (`select`, `candidate_class`, ...) into scope so the
// batched shared-layer ops below can use them unqualified.
using namespace detail::topk;
//---------------------------------------------------------------------
// Indexed Top-K helpers
//---------------------------------------------------------------------
// Gathers a value from the user's input iterator by index. On the indexed value path the agent
// sees a counting_iterator as the value input and stores `OffsetT` indices in the candidate buffer;
// at write-out a transform_iterator maps them back: `values_out[pos] = values_in[idx]`.
template <typename ValueInputIteratorT>
struct topk_index_gather_op
{
  ValueInputIteratorT user_values_in;

  template <typename IndexT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto operator()(IndexT index) const -> it_value_t<ValueInputIteratorT>
  {
    return user_values_in[index];
  }
};

//---------------------------------------------------------------------
// Helpers for extracting bins (histogram computation) and classifying items (selected, candidate, rejected)
//---------------------------------------------------------------------
// Used in the bin ID calculation to exclude bits unrelated to the current pass
template <typename T, int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr unsigned calc_mask(const int pass)
{
  int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
  return (1 << num_bits) - 1;
}

// Get the bin ID from the value of element
template <typename T,
          select SelectDirection,
          int BitsPerPass,
          typename DecomposerT,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct extract_bin_op_t;

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT>
struct extract_bin_op_t<T, SelectDirection, BitsPerPass, DecomposerT, true>
{
  int start_bit{};
  unsigned mask{};

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE extract_bin_op_t(int pass, int /*total_bits*/, DecomposerT /*decomposer*/)
      : start_bit(calc_start_bit<T, BitsPerPass>(pass))
      , mask(calc_mask<T, BitsPerPass>(pass))
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    bits      = Traits<T>::TwiddleIn(bits);
    if constexpr (SelectDirection != select::min)
    {
      bits = ~bits;
    }
    int bucket = (bits >> start_bit) & mask;
    return bucket;
  }
};

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT>
struct extract_bin_op_t<T, SelectDirection, BitsPerPass, DecomposerT, false>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using radix_traits_t                = detail::radix::traits_t<T>;
  using bit_ordered_type              = typename radix_traits_t::bit_ordered_type;
  using digit_extractor_t = typename radix_traits_t::template digit_extractor_t<ShiftDigitExtractor<T>, DecomposerT>;

  DecomposerT decomposer{};
  digit_extractor_t digit_extractor;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE extract_bin_op_t(int pass, int total_bits, DecomposerT decomposer)
      : decomposer(decomposer)
      , digit_extractor(radix_traits_t::template digit_extractor<ShiftDigitExtractor<T>>(
          calc_start_bit<BitsPerPass>(total_bits, pass),
          calc_start_bit<BitsPerPass>(total_bits, pass - 1) - calc_start_bit<BitsPerPass>(total_bits, pass),
          decomposer))
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    bit_ordered_type ordered = key;
    ordered                  = RadixSortTwiddle<is_descending, T>::In(ordered, decomposer);
    return static_cast<int>(digit_extractor.Digit(ordered));
  }
};

} // namespace detail::batched_topk

CUB_NAMESPACE_END
