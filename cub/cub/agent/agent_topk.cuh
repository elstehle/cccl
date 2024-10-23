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
#include <cub/block/block_store.cuh>
#include <cub/util_type.cuh>

#include <cuda/atomic>

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

    /// Cofficient for reducing memry
    COFFICIENT_FOR_BUFFER = _COFFICIENT_FOR_BUFFER,
  };
  // /// Threads per thread block
  // static constexpr int BLOCK_THREADS = _BLOCK_THREADS;

  // /// Items per thread (per tile of input)
  // static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;

  // /// BITS Processed per pass
  // static constexpr int BITS_PER_PASS = _BITS_PER_PASS;

  // /// Cofficient for reducing memry
  // static constexpr int COFFICIENT_FOR_BUFFER = _COFFICIENT_FOR_BUFFER;

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

  // We determine the bits of the k_th key inside the mask processed by the pass. The
  // already known bits are stored in `kth_key_bits`. It's used to discriminate a
  // element is a result (written to `out`), a candidate for next pass (written to
  // `out_buf`), or not useful (discarded). The bits that are not yet processed do not
  // matter for this purpose.
  typename cub::Traits<KeyInT>::UnsignedBits kth_key_bits;

  // Record how many elements have passed filtering. It's used to determine the position
  // in the `out_buf` where an element should be written.
  alignas(128) NumItemsT filter_cnt;

  // For a row inside a batch, we may launch multiple thread blocks. This counter is
  // used to determine if the current block is the last running block. If so, this block
  // will execute scan() and choose_bucket().
  alignas(128) unsigned int finished_block_cnt;

  // Record how many elements have been written to the front of `out`. Elements less (if
  // SELECT_MIN==true) than the k-th key are written from front to back.
  alignas(128) NumItemsT out_cnt;

  // Record how many elements have been written to the back of `out`. Elements equal to
  // the k-th key are written from back to front. We need to keep count of them
  // separately because the number of elements that <= the k-th key might exceed k.
  alignas(128) NumItemsT out_back_cnt;
};

/**
 * @brief Operations for calculating the bin index based on the input
 */
template <typename T, bool SELECT_MIN>
struct ExtractBinOp
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE typename cub::Traits<T>::UnsignedBits operator()(T key)
  {
    auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
    bits      = cub::Traits<T>::TwiddleIn(bits);
    if (!SELECT_MIN)
    {
      bits = ~bits;
    }
    return bits;
  }
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
          typename IdxInputIteratorT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename ExtractBinOpT,
          typename NumItemsT,
          bool SELECT_MIN,
          bool FusedLastFilter>
struct AgentTopK
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------
  // Indicates whether the BlockLoad algorithm uses shared memory to load or exchange the data
  static constexpr bool loads_via_smem =
    !(AgentTopKPolicyT::TopKPolicyT::LOAD_ALGORITHM == cub::BLOCK_LOAD_DIRECT
      || AgentTopKPolicyT::TopKPolicyT::LOAD_ALGORITHM == cub::BLOCK_LOAD_STRIPED
      || AgentTopKPolicyT::TopKPolicyT::LOAD_ALGORITHM == cub::BLOCK_LOAD_VECTORIZE);

  // The key and value type
  using KeyInT = cub::detail::value_t<KeyInputIteratorT>;

  static constexpr ::cuda::std::int32_t BLOCK_THREADS         = AgentTopKPolicyT::TopKPolicyT::BLOCK_THREADS;
  static constexpr ::cuda::std::int32_t ITEMS_PER_THREAD      = AgentTopKPolicyT::TopKPolicyT::ITEMS_PER_THREAD;
  static constexpr ::cuda::std::int32_t BITS_PER_PASS         = AgentTopKPolicyT::TopKPolicyT::BITS_PER_PASS;
  static constexpr ::cuda::std::int32_t COFFICIENT_FOR_BUFFER = AgentTopKPolicyT::TopKPolicyT::COFFICIENT_FOR_BUFFER;
  static constexpr ::cuda::std::int32_t TILE_ITEMS            = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr int num_buckets                            = 1 << BITS_PER_PASS;

  // Parameterized BlockLoad type for input data
  using BlockLoadT =
    cub::BlockLoad<KeyInT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentTopKPolicyT::TopKPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockScan type
  using BlockScanT = cub::BlockScan<NumItemsT, BLOCK_THREADS, AgentTopKPolicyT::TopKPolicyT::SCAN_ALGORITHM>;

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------
  KeyInputIteratorT d_keys_in; ///< Input keys
  KeyOutputIteratorT d_keys_out; ///< Output keys
  ValueInputIteratorT d_values_in; ///< Input values
  ValueOutputIteratorT d_values_out; ///< Output values
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
  // Utility methods for device topK
  //---------------------------------------------------------------------

  _CCCL_DEVICE typename cub::Traits<KeyInT>::UnsignedBits twiddle_in(KeyInT key, bool select_min)
  {
    auto bits = reinterpret_cast<typename cub::Traits<KeyInT>::UnsignedBits&>(key);
    bits      = cub::Traits<KeyInT>::TwiddleIn(bits);
    if (!select_min)
    {
      bits = ~bits;
    }
    return bits;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_num_passes()
  {
    return ::cuda::ceil_div<int>(sizeof(KeyInT) * 8, BITS_PER_PASS);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_num_buckets()
  {
    return 1 << BITS_PER_PASS;
  }

  /**
   * Bit 0 is the least significant (rightmost);
   * this implementation processes input from the most to the least significant bit.
   * This way, we can skip some passes in the end at the cost of having an unsorted output.
   *
   * NB: Use pass=-1 for calc_mask().
   */

  _CCCL_DEVICE constexpr int calc_start_bit(int pass)
  {
    int start_bit = static_cast<int>(sizeof(KeyInT) * 8) - (pass + 1) * BITS_PER_PASS;
    if (start_bit < 0)
    {
      start_bit = 0;
    }
    return start_bit;
  }

  _CCCL_DEVICE constexpr unsigned calc_mask(int pass)
  {
    static_assert(BITS_PER_PASS <= 31);
    int num_bits = calc_start_bit(pass - 1) - calc_start_bit(pass);
    return (1 << num_bits) - 1;
  }
  // sync_width should >= warp_size
  template <typename Func>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  vectorized_process(size_t thread_rank, size_t num_threads, KeyInputIteratorT in, NumItemsT len, Func f)
  {
    using WideT             = float4;
    constexpr int WARP_SIZE = 32;
    if (sizeof(KeyInT) >= sizeof(WideT))
    {
      for (NumItemsT i = thread_rank; i < len; i += num_threads)
      {
        f(in[i], i);
      }
    }
    else
    {
      static_assert(sizeof(WideT) % sizeof(KeyInT) == 0);
      constexpr int items_per_scalar = sizeof(WideT) / sizeof(KeyInT);
      // TODO: it's UB
      union
      {
        WideT scalar;
        KeyInT array[items_per_scalar];
      } wide;

      int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
                     ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(KeyInT))
                     : 0;
      if (skip_cnt > len)
      {
        skip_cnt = len;
      }
      const WideT* in_cast     = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
      const NumItemsT len_cast = (len - skip_cnt) / items_per_scalar;

      for (NumItemsT i = thread_rank; i < len_cast; i += num_threads)
      {
        wide.scalar            = in_cast[i];
        const NumItemsT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
        for (int j = 0; j < items_per_scalar; ++j)
        {
          f(wide.array[j], real_i + j);
        }
      }

      static_assert(WARP_SIZE >= items_per_scalar);
      // and because items_per_scalar > skip_cnt, WARP_SIZE > skip_cnt
      // no need to use loop
      if (thread_rank < skip_cnt)
      {
        f(in[thread_rank], thread_rank);
      }
      // because len_cast = (len - skip_cnt) / items_per_scalar,
      // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
      // and so
      // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <= WARP_SIZE
      // no need to use loop
      const NumItemsT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
      if (remain_i < len)
      {
        f(in[remain_i], remain_i);
      }
    }
  }

  /**
   * Fused filtering of the current pass and building histogram for the next pass
   * (see steps 4 & 1 in `radix_kernel` description).
   */

  _CCCL_DEVICE _CCCL_FORCEINLINE void filterAndHistogram(
    KeyInputIteratorT in_buf,
    IdxInputIteratorT in_idx_buf,
    KeyOutputIteratorT out_buf,
    IdxInputIteratorT out_idx_buf,
    NumItemsT previous_len,
    Counter<KeyInT, NumItemsT>* counter,
    NumItemsT* histogram,
    int pass,
    bool early_stop)
  {
    __shared__ NumItemsT histogram_smem[num_buckets];
    for (NumItemsT i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
      histogram_smem[i] = 0;
    }
    __syncthreads();

    int start_bit       = calc_start_bit(pass);
    const unsigned mask = calc_mask(pass);

    if (pass == 0)
    {
      // Passed to vectorized_process, this function executes in all blocks in parallel,
      // i.e. the work is split along the input (both, in batches and chunks of a single
      // row). Later, the histograms are merged using atomicAdd.
      auto f = [start_bit, mask, this](KeyInT key, NumItemsT) {
        int bucket = (twiddle_in(key, SELECT_MIN) >> start_bit) & mask; // calc_bucket(key, start_bit, mask);
        atomicAdd(histogram_smem + bucket, static_cast<NumItemsT>(1));
      };
      vectorized_process(
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
        static_cast<size_t>(blockDim.x) * gridDim.x,
        in_buf,
        previous_len,
        f);
    }
    else
    {
      NumItemsT* p_filter_cnt = &counter->filter_cnt;
      NumItemsT* p_out_cnt    = &counter->out_cnt;
      const auto kth_key_bits = counter->kth_key_bits;
      int previous_start_bit  = calc_start_bit(pass - 1);

      // See the remark above on the distributed execution of `f` using
      // vectorized_process.
      auto f =
        [in_idx_buf,
         out_buf,
         out_idx_buf,
         start_bit,
         mask,
         previous_start_bit,
         kth_key_bits,
         counter,
         p_filter_cnt,
         p_out_cnt,
         this,
         early_stop](KeyInT key, NumItemsT i) {
          const auto previous_bits = (twiddle_in(key, SELECT_MIN) >> previous_start_bit) << previous_start_bit;
          if (previous_bits == kth_key_bits)
          {
            if (early_stop)
            {
              NumItemsT pos     = atomicAdd(p_out_cnt, static_cast<NumItemsT>(1));
              NumItemsT index   = in_idx_buf ? in_idx_buf[i] : i;
              d_keys_out[pos]   = key;
              d_values_out[pos] = d_values_in[index];
            }
            else
            {
              if (out_buf)
              {
                NumItemsT pos    = atomicAdd(p_filter_cnt, static_cast<NumItemsT>(1));
                out_buf[pos]     = key;
                out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;
              }

              int bucket = (twiddle_in(key, SELECT_MIN) >> start_bit) & mask; // calc_bucket(key, start_bit, mask);
              atomicAdd(histogram_smem + bucket, static_cast<NumItemsT>(1));
            }
          }
          // the condition `(out_buf || early_stop)` is a little tricky:
          // If we skip writing to `out_buf` (when `out_buf` is nullptr), we should skip
          // writing to `out` too. So we won't write the same key to `out` multiple
          // times in different passes. And if we keep skipping the writing, keys will
          // be written in `last_filter_kernel()` at last. But when `early_stop` is
          // true, we need to write to `out` since it's the last chance.
          else if ((out_buf || early_stop) && previous_bits < kth_key_bits)
          {
            NumItemsT pos     = atomicAdd(p_out_cnt, static_cast<NumItemsT>(1));
            NumItemsT index   = in_idx_buf ? in_idx_buf[i] : i;
            d_keys_out[pos]   = key;
            d_values_out[pos] = d_values_in[index];
          }
        };
      vectorized_process(
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
        static_cast<size_t>(blockDim.x) * gridDim.x,
        in_buf,
        previous_len,
        f);
    }
    if (early_stop)
    {
      return;
    }
    __syncthreads();

    // merge histograms produced by individual blocks
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
      if (histogram_smem[i] != 0)
      {
#ifdef USE_ATOMIC_REF_SMEM
        auto atomic_ref = cuda::atomic_ref<NumItemsT, cuda::thread_scope::thread_scope_system>{histogram[i]};
        atomic_ref.fetch_add(histogram_smem[i], cuda::memory_order_relaxed);
#else
        atomicAdd(histogram + i, histogram_smem[i]);
#endif
      }
    }
  }

  /**
   * Replace histogram with its own prefix sum
   * (step 2 in `radix_kernel` description)
   */

  _CCCL_DEVICE _CCCL_FORCEINLINE void scan(volatile NumItemsT* histogram)
  {
    if (num_buckets >= BLOCK_THREADS)
    {
      static_assert(num_buckets % BLOCK_THREADS == 0);
      constexpr int items_per_thread = num_buckets / BLOCK_THREADS;
      typedef cub::BlockLoad<NumItemsT, BLOCK_THREADS, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
      typedef cub::BlockStore<NumItemsT, BLOCK_THREADS, items_per_thread, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
      typedef cub::BlockScan<NumItemsT, BLOCK_THREADS> BlockScan;

      __shared__ union
      {
        typename BlockLoad::TempStorage load;
        typename BlockScan::TempStorage scan;
        typename BlockStore::TempStorage store;
      } temp_storage;
      NumItemsT thread_data[items_per_thread];

      BlockLoad(temp_storage.load).Load(histogram, thread_data);
      __syncthreads();

      BlockScan(temp_storage.scan).InclusiveSum(thread_data, thread_data);
      __syncthreads();

      BlockStore(temp_storage.store).Store(histogram, thread_data);
    }
    else
    {
      typedef cub::BlockScan<NumItemsT, BLOCK_THREADS> BlockScan;
      __shared__ typename BlockScan::TempStorage temp_storage;

      NumItemsT thread_data = 0;
      if (threadIdx.x < num_buckets)
      {
        thread_data = histogram[threadIdx.x];
      }

      BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
      __syncthreads();

      if (threadIdx.x < num_buckets)
      {
        histogram[threadIdx.x] = thread_data;
      }
    }
  }

  /**
   * Calculate in which bucket the k-th value will fall
   *  (steps 3 in `radix_kernel` description)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  choose_bucket(Counter<KeyInT, NumItemsT>* counter, const NumItemsT* histogram, const NumItemsT k, const int pass)
  {
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
      NumItemsT prev = (i == 0) ? 0 : histogram[i - 1];
      NumItemsT cur  = histogram[i];

      // one and only one thread will satisfy this condition, so counter is written by
      // only one thread
      if (prev < k && cur >= k)
      {
        counter->k                                        = k - prev; // how many values still are there to find
        counter->len                                      = cur - prev; // number of values in next pass
        typename cub::Traits<KeyInT>::UnsignedBits bucket = i;
        int start_bit                                     = calc_start_bit(pass);
        counter->kth_key_bits |= bucket << start_bit;
      }
    }
  }

  // For one-block version, last_filter() could be called when pass < num_passes - 1.
  // So `pass` could not be constexpr
  _CCCL_DEVICE _CCCL_FORCEINLINE void last_filter(
    const KeyInT* in_buf,
    const NumItemsT* in_idx_buf,
    NumItemsT current_len,
    NumItemsT k,
    Counter<KeyInT, NumItemsT>* counter,
    const int pass)
  {
    const auto kth_key_bits = counter->kth_key_bits;
    int start_bit           = calc_start_bit(pass);

    // changed in choose_bucket(); need to reload
    NumItemsT num_of_kth_needed = counter->k;
    NumItemsT* p_out_cnt        = &counter->out_cnt;
    NumItemsT* p_out_back_cnt   = &counter->out_back_cnt;
    for (NumItemsT i = threadIdx.x; i < current_len; i += blockDim.x)
    {
      const KeyInT key = in_buf[i];
      const auto bits  = (twiddle_in(key, SELECT_MIN) >> start_bit) << start_bit;
      if (bits < kth_key_bits)
      {
        NumItemsT pos   = atomicAdd(p_out_cnt, static_cast<NumItemsT>(1));
        NumItemsT index = in_idx_buf ? in_idx_buf[i] : i;
        d_keys_out[pos] = key;
        // For one-block version, `in_idx_buf` could be nullptr at pass 0.
        // For non one-block version, if writing has been skipped, `in_idx_buf` could
        // be nullptr if `in_buf` is `in`
        d_values_out[pos] = d_values_in[index];
      }
      else if (bits == kth_key_bits)
      {
        NumItemsT new_idx  = in_idx_buf ? in_idx_buf[i] : i;
        NumItemsT back_pos = atomicAdd(p_out_back_cnt, static_cast<NumItemsT>(1));

        if (back_pos < num_of_kth_needed)
        {
          NumItemsT pos     = k - 1 - back_pos;
          d_keys_out[pos]   = key;
          d_values_out[pos] = d_values_in[new_idx];
        }
      }
    }
  }

  /**
   * @brief One sweep topK (specialized for topK operator)
   *
   * @param in_buf
   *   Buffer address for input data
   *
   * @param in_idx_buf
   *   Buffer address for index of the input data
   *
   * @param out_buf
   *   Buffer address for output data
   *
   * @param out_idx_buf
   *   Buffer address for index of the output data
   *
   * @param counter
   *   Record the meta data for different passes
   *
   * @param histogram
   *   Record the element number of each bucket
   *
   * @param pass
   *   Indicate which pass are processed currently
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void InvokeOneSweep(
    KeyInputIteratorT in_buf,
    IdxInputIteratorT in_idx_buf,
    KeyInputIteratorT out_buf,
    IdxInputIteratorT out_idx_buf,
    Counter<KeyInT, NumItemsT>* counter,
    NumItemsT* histogram,
    int pass)
  {
    NumItemsT current_k;
    NumItemsT previous_len;
    NumItemsT current_len;

    if (pass == 0)
    {
      current_k    = k;
      previous_len = num_items;
      current_len  = num_items;
    }
    else
    {
      current_k    = counter->k;
      current_len  = counter->len;
      previous_len = counter->previous_len;
    }

    if (current_len == 0)
    {
      return;
    }

    const bool early_stop   = (current_len == current_k);
    const NumItemsT buf_len = max(256, num_items / COFFICIENT_FOR_BUFFER);

    if (previous_len > buf_len)
    {
      in_buf       = d_keys_in;
      in_idx_buf   = nullptr;
      previous_len = num_items;
    }

    // "current_len > buf_len" means current pass will skip writing buffer
    if (current_len > buf_len)
    {
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    }

    filterAndHistogram(in_buf, in_idx_buf, out_buf, out_idx_buf, previous_len, counter, histogram, pass, early_stop);
    __threadfence();

    bool is_last_block = false;
    if (threadIdx.x == 0)
    {
      unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
      is_last_block         = (finished == (gridDim.x - 1));
    }

    if (__syncthreads_or(is_last_block))
    {
      if (early_stop)
      {
        if (threadIdx.x == 0)
        {
          // `last_filter_kernel()` requires setting previous_len
          counter->previous_len = 0;
          counter->len          = 0;
        }
        return;
      }

      scan(histogram);
      __syncthreads();
      choose_bucket(counter, histogram, current_k, pass);
      __syncthreads();

      int num_passes = calc_num_passes();
      // reset for next pass
      if (pass != num_passes - 1)
      {
        for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
        {
          histogram[i] = 0;
        }
      }
      if (threadIdx.x == 0)
      {
        // `last_filter_kernel()` requires setting previous_len even in the last pass
        counter->previous_len = current_len;
        // not necessary for the last pass, but put it here anyway
        counter->filter_cnt = 0;
      }

      if (pass == num_passes - 1)
      {
        volatile const NumItemsT num_of_kth_needed = counter->k;
        __syncthreads();
        if (FusedLastFilter)
        {
          last_filter(out_buf ? out_buf : in_buf,
                      out_idx_buf ? out_idx_buf : in_idx_buf,
                      out_buf ? current_len : num_items,
                      k,
                      counter,
                      pass);
        }
      }
    }
  }
};

CUB_NAMESPACE_END
