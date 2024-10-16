/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in device-wide select.
 */

#pragma once

#include <cub/config.cuh>

#include <cub/block/block_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <iterator>

CUB_NAMESPACE_BEGIN
/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentTopK
 *
 * @tparam _BLOCK_THREADS
 *   Threads per thread block
 *
 * @tparam _ITEMS_PER_THREAD
 *   Items per thread (per tile of input)
 *
 * @tparam _LOAD_ALGORITHM
 *   The BlockLoad algorithm to use
 *
 * @tparam _HISTOGRAM_ALGORITHM
 *   The BlockHistogram algorithm to use
 *
 * @tparam _SCAN_ALGORITHM
 *   The BlockScan algorithm to use
 */

template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD,
          int _BITS_PER_PASS,
          int _NUM_PASSES,
          int _NUM_BUCKETS,
          int _COFFICIENT_FOR_BUFFER,
          BlockLoadAlgorithm _LOAD_ALGORITHM,
          BlockHistogramAlgorithm _HISTOGRAM_ALGORITHM,
          BlockScanAlgorithm _SCAN_ALGORITHM>
struct AgentTopKPolicy
{
  enum
  {
    /// Threads per thread block
    BLOCK_THREADS = _BLOCK_THREADS,

    /// Items per thread (per tile of input)
    ITEMS_PER_THREAD = _ITEMS_PER_THREAD,

    /// BITS Processed per pass
    BITS_PER_PASS = _BITS_PER_PASS,

    /// Num passes
    NUM_PASSES = _NUM_PASSES,

    /// Buckets per pass
    NUM_BUCKETS = _NUM_BUCKETS,

    /// Cofficient for reducing memry
    COFFICIENT_FOR_BUFFER = _COFFICIENT_FOR_BUFFER,
  };

  /// The BlockLoad algorithm to use
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;

  /// The BlockHistogram algorithm to use
  static constexpr cub::BlockHistogramAlgorithm HISTOGRAM_ALGORITHM = _HISTOGRAM_ALGORITHM;

  /// The BlockScan algorithm to use
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

template <typename KeyInT, typename NumItemsT>
struct alignas(128) Counter
{
  // We are processing the values in multiple passes, from most significant to least
  // significant. In each pass, we keep the length of input (`len`) and the `k` of
  // current pass, and update them at the end of the pass.
  NumItemsT k;
  NumItemsT len;

  //  `previous_len` is the length of input in previous pass. Note that `previous_len`
  //  rather than `len` is used for the filtering step because filtering is indeed for
  //  previous pass (see comments before `radix_kernel`).
  NumItemsT previous_len;

  // We determine the bits of the k_th value inside the mask processed by the pass. The
  // already known bits are stored in `kth_value_bits`. It's used to discriminate a
  // element is a result (written to `out`), a candidate for next pass (written to
  // `out_buf`), or not useful (discarded). The bits that are not yet processed do not
  // matter for this purpose.
  typename cub::Traits<KeyInT>::UnsignedBits kth_value_bits;

  // Record how many elements have passed filtering. It's used to determine the position
  // in the `out_buf` where an element should be written.
  alignas(128) NumItemsT filter_cnt;

  // For a row inside a batch, we may launch multiple thread blocks. This counter is
  // used to determine if the current block is the last running block. If so, this block
  // will execute scan() and choose_bucket().
  alignas(128) unsigned int finished_block_cnt;

  // Record how many elements have been written to the front of `out`. Elements less (if
  // select_min==true) than the k-th value are written from front to back.
  alignas(128) NumItemsT out_cnt;

  // Record how many elements have been written to the back of `out`. Elements equal to
  // the k-th value are written from back to front. We need to keep count of them
  // separately because the number of elements that <= the k-th value might exceed k.
  alignas(128) NumItemsT out_back_cnt;
};

/**
 * @brief AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in
 * device-wide topK
 *
 * Performs functor-based selection if SelectOpT functor type != NullType
 * Otherwise performs flag-based selection if FlagsInputIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 *
 * @tparam AgentTopKPolicyT
 *   Parameterized AgentTopKPolicy tuning policy type
 *
 * @tparam KeyInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input keys @iterator
 *
 * @tparam KeyOutputIteratorT
 *   **[inferred]** Random-access output iterator type for writing output keys @iterator
 *
 * @tparam ValueInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input values @iterator
 *
 * @tparam ValueOutputIteratorT
 *   **[inferred]** Random-access input iterator type for writing output values @iterator
 *
 * @tparam NumItemsT
 * Type of variable num_items and k
 *
 * @tparam ExtractBinOpT
 *   Operations to extract the bin from the input key value
 *
 */
//
template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename ExtractBinOpT,
          typename NumItemsT>
struct AgentTopK
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // Indicates whether the BlockLoad algorithm uses shared memory to load or exchange the data
  static constexpr bool loads_via_smem =
    !(AgentTopKPolicyT::LOAD_ALGORITHM == cub::BLOCK_LOAD_DIRECT
      || AgentTopKPolicyT::LOAD_ALGORITHM == cub::BLOCK_LOAD_STRIPED
      || AgentTopKPolicyT::LOAD_ALGORITHM == cub::BLOCK_LOAD_VECTORIZE);

  // The key and value type
  using KeyInT    = cub::detail::value_t<KeyInputIteratorT>;
  using KeyOutT   = cub::detail::value_t<KeyOutputIteratorT>;
  using ValueInT  = cub::detail::value_t<ValueInputIteratorT>;
  using ValueOutT = cub::detail::value_t<ValueOutputIteratorT>;

  static constexpr ::cuda::std::int32_t BLOCK_THREADS    = AgentTopKPolicyT::BLOCK_THREADS;
  static constexpr ::cuda::std::int32_t ITEMS_PER_THREAD = AgentTopKPolicyT::ITEMS_PER_THREAD;
  static constexpr ::cuda::std::int32_t TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Parameterized BlockLoad type for input data
  using BlockLoadT = cub::BlockLoad<KeyInT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentTopKPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockScan type
  using BlockScanT = cub::BlockScan<NumItemsT, BLOCK_THREADS, AgentTopKPolicyT::SCAN_ALGORITHM>;

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------
  //_TempStorage& temp_storage; ///< Reference to temp_storage
  KeyInT d_keys_in; ///< Input keys
  KeyOutT d_keys_out; ///< Output keys
  ValueInT d_values_in; ///< Input values
  ValueOutT d_values_out; ///< Output values
  NumItemsT num_items; ///< Total number of input items
  NumItemsT k; ///< Total number of output items
  ExtractBinOpT extract_bin_op; /// The operation for bin
  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------
  /**
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param d_keys_in
   *   Input data, keys
   *
   * @param d_keys_out
   *   Output data, keys
   *
   * @param d_values_in
   *   Input data, values
   *
   * @param d_values_in
   *   Output data, values
   *
   * @param extract_bin_op
   *   Extract bin operator
   *
   * @param num_items
   *   Total number of input items
   *
   * @param k
   *   The K value. Will find K elements from num_items elements
   *
   */
  //    TempStorage& temp_storage,       : temp_storage(temp_storage.Alias())
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentTopK(
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumItemsT k,
    ExtractBinOpT extract_bin_op)
      : d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , num_items(num_items)
      , k(k)
      , extract_bin_op(extract_bin_op)
  {}

  //---------------------------------------------------------------------
  // Utility methods for initializing the selections
  //---------------------------------------------------------------------
  /**
   * Initialize selections (specialized for selection operator)
   */
};

CUB_NAMESPACE_END
