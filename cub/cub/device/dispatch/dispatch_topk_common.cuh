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
#include <cub/device/dispatch/tuning/tuning_topk.cuh>
#include <cub/util_type.cuh>

#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_output_iterator.h>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
//---------------------------------------------------------------------
// Indexed Top-K helpers
//---------------------------------------------------------------------
// Gathers a value from the user's input iterator using an index. Used on the indexed value path: the agent will only
// see a counting_iterator as the value input, maintaining indexes of `OffsetT` indices in the values' candidate buffer. 
// During write-out, the agent will use a transform_iterator as the output iterator, writing `user_d_values_out[pos] = user_d_values_in[idx]`.
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

// Picks the value-channel iterator types passed to the kernels based on the resolved `value_materialization_mode`.
template <bool Indexed, typename ValueInputIteratorT, typename ValueOutputIteratorT, typename OffsetT>
struct effective_value_iterators
{
  // Type alias for the input iterator
  using in_t    = ValueInputIteratorT;
  // Type alias for the output iterator
  using out_t   = ValueOutputIteratorT;
  // Type alias for value type
  using value_t = it_value_t<ValueInputIteratorT>;
};

template <typename ValueInputIteratorT, typename ValueOutputIteratorT, typename OffsetT>
struct effective_value_iterators<true, ValueInputIteratorT, ValueOutputIteratorT, OffsetT>
{
  // Type alias for the input iterator (indexed top-k)
  using in_t    = ::cuda::counting_iterator<OffsetT>;
  // Type alias for the output iterator  (translating indices back to values)
  using out_t   = ::cuda::transform_output_iterator<topk_index_gather_op<ValueInputIteratorT>, ValueOutputIteratorT>;
  // Type alias for value type (i.e., indices for the idnexed top-k)
  using value_t = OffsetT;
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

// ----------------------------------------------------------------------------
// Static-pass sibling ops (`*_static_t`).
//
// The dynamic variants (`extract_bin_op_t`, `identify_candidates_op_t`) below
// store `pass` / `start_bit` / `mask` as runtime members initialised by their
// ctors from the kernel-arg `pass`. The static siblings take `Pass` as a
// template parameter so that all of those per-pass scalars become `constexpr`,
// and the ops carry zero per-pass runtime state. For `identify_candidates_op`
// the only surviving runtime member is the segment-dependent
// `kth_key_bits` pointer; for `extract_bin_op` the structures are stateless.
//
// Selecting a sibling at dispatch time turns each filter-pass launch into a
// kernel instantiation specialised on `Pass`, which lets ptxas constant-fold
// `start_bit` / `mask` into shift/mask immediates (or BFE) and drops a few
// persistent registers on the agent's hot path.
//
// The ctor of each sibling accepts the same arguments as the dynamic version
// so that existing call sites in `agent_batched_topk` work as a drop-in.
// ----------------------------------------------------------------------------

// Get the bin ID from the value of element
template <typename T,
          select SelectDirection,
          int BitsPerPass,
          typename DecomposerT,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct extract_bin_op_t;

template <typename T,
          select SelectDirection,
          int BitsPerPass,
          int Pass,
          typename DecomposerT,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct extract_bin_op_static_t;

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT>
struct extract_bin_op_t<T, SelectDirection, BitsPerPass, DecomposerT, true>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using bit_ordered_type              = typename Traits<T>::UnsignedBits;

  int pass{};
  int start_bit{};
  unsigned mask{};

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE extract_bin_op_t(int pass, int /*total_bits*/, DecomposerT /*decomposer*/)
      : pass(pass)
      , start_bit(calc_start_bit<T, BitsPerPass>(pass))
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

// Static-pass sibling of `extract_bin_op_t<..., true>`. `start_bit` and
// `mask` are derived from the compile-time `Pass` template parameter; the
// struct has no per-pass runtime state. Drop-in for the dynamic ctor.
template <typename T, select SelectDirection, int BitsPerPass, int Pass, typename DecomposerT>
struct extract_bin_op_static_t<T, SelectDirection, BitsPerPass, Pass, DecomposerT, true>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using bit_ordered_type              = typename Traits<T>::UnsignedBits;
  static constexpr int start_bit      = calc_start_bit<T, BitsPerPass>(Pass);
  static constexpr unsigned int mask  = calc_mask<T, BitsPerPass>(Pass);

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  extract_bin_op_static_t(int /*pass*/ = 0, int /*total_bits*/ = 0, DecomposerT /*decomposer*/ = {})
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    bits      = Traits<T>::TwiddleIn(bits);
    if constexpr (SelectDirection != select::min)
    {
      bits = ~bits;
    }
    return static_cast<int>((bits >> start_bit) & mask);
  }
};

// Check if the input element is still a candidate for the target pass.
template <typename T,
          select SelectDirection,
          int BitsPerPass,
          typename DecomposerT,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct identify_candidates_op_t;

template <typename T,
          select SelectDirection,
          int BitsPerPass,
          int Pass,
          typename DecomposerT,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct identify_candidates_op_static_t;

template <typename T,
          select SelectDirection,
          int BitsPerPass,
          int Pass,
          typename DecomposerT,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct identify_candidates_op_static_value_t;

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

// Static-pass sibling of `identify_candidates_op_t<..., true>`. `start_bit`
// is derived from the compile-time `Pass` template parameter and the
// high-bit-mask used in `(bits >> start_bit) << start_bit` becomes a
// `constexpr` immediate. `kth_key_bits` stays as a runtime pointer (per
// segment per pass). Drop-in ctor for `identify_candidates_op_t`.
template <typename T, select SelectDirection, int BitsPerPass, int Pass, typename DecomposerT>
struct identify_candidates_op_static_t<T, SelectDirection, BitsPerPass, Pass, DecomposerT, true>
{
  using unsigned_bits_t              = typename Traits<T>::UnsignedBits;
  using key_prefix_t                 = key_prefix_storage_t<T>;
  static constexpr int start_bit     = calc_start_bit<T, BitsPerPass>(Pass - 1);
  static constexpr bool holds_value  = false;

  unsigned_bits_t* kth_key_bits;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE identify_candidates_op_static_t(
    key_prefix_t* kth_key_bits, int /*pass*/ = 0, int /*total_bits*/ = 0, DecomposerT /*decomposer*/ = {})
      : kth_key_bits(&kth_key_bits->bits)
  {}

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
         : (bits == *kth_key_bits) ? candidate_class::candidate
                                   : candidate_class::rejected;
  }
};

// Value-holding sibling of `identify_candidates_op_static_t<..., true>`.
// Carries the dereferenced `kth_key_bits` *value* (loaded once at op
// construction time from the per-segment counter) rather than the pointer
// to it, so the per-item `operator()` does not hit the LSU. Comes at the
// cost of one extra persistent register (the value) and a slightly heavier
// ctor (one global load instead of just a pointer copy).
template <typename T, select SelectDirection, int BitsPerPass, int Pass, typename DecomposerT>
struct identify_candidates_op_static_value_t<T, SelectDirection, BitsPerPass, Pass, DecomposerT, true>
{
  using unsigned_bits_t              = typename Traits<T>::UnsignedBits;
  using key_prefix_t                 = key_prefix_storage_t<T>;
  static constexpr int start_bit     = calc_start_bit<T, BitsPerPass>(Pass - 1);
  static constexpr bool holds_value  = true;

  unsigned_bits_t kth_key_value;

  // Construct from the per-segment counter's `kth_key_bits` field. Reads
  // the value at ctor time so subsequent `operator()` calls do not hit
  // the LSU. Drop-in for the pointer-based ctor signature except that
  // pass / total_bits / decomposer are unused (they were only needed to
  // derive `start_bit`, which is now compile-time).
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE identify_candidates_op_static_value_t(
    key_prefix_t* kth_key_bits, int /*pass*/ = 0, int /*total_bits*/ = 0, DecomposerT /*decomposer*/ = {})
      : kth_key_value(kth_key_bits->bits)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE candidate_class operator()(T key) const
  {
    auto bits = reinterpret_cast<unsigned_bits_t&>(key);
    bits      = Traits<T>::TwiddleIn(bits);
    if constexpr (SelectDirection != select::min)
    {
      bits = ~bits;
    }
    bits = (bits >> start_bit) << start_bit;
    return (bits < kth_key_value) ? candidate_class::selected
         : (bits == kth_key_value) ? candidate_class::candidate
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
