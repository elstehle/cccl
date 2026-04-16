// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in device-wide topK.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
//! @brief Parameterizable tuning policy type for AgentTopK
//!
//! @tparam ThreadsPerBlock
//!   Threads per thread block
//!
//! @tparam ItemsPerThread
//!   Items per thread (per tile of input)
//!
//! @tparam BitsPerPass
//!   Number of bits processed per pass
//!
//! @tparam LoadAlgorithm
//!   The BlockLoad algorithm to use
//!
//! @tparam ScanAlgorithm
//!   The BlockScan algorithm to use
//!
template <int ThreadsPerBlock,
          int ItemsPerThread,
          int BitsPerPass,
          BlockLoadAlgorithm LoadAlgorithm,
          BlockScanAlgorithm ScanAlgorithm>
struct AgentTopKPolicy
{
  static constexpr int threads_per_block             = ThreadsPerBlock;
  static constexpr int items_per_thread              = ItemsPerThread;
  static constexpr int bits_per_pass                 = BitsPerPass;
  static constexpr BlockLoadAlgorithm load_algorithm = LoadAlgorithm;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = ScanAlgorithm;
};

template <typename KeyT, bool CanTwiddle = detail::radix::can_twiddle<KeyT>>
struct key_prefix_storage_t;

template <typename KeyT>
struct key_prefix_storage_t<KeyT, true>
{
  using bits_t = typename Traits<KeyT>::UnsignedBits;
  bits_t bits;
};

// Calculates the number of passes needed for a type T with BitsPerPass bits processed per pass.
template <typename T>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int calc_num_passes(int bits_per_pass)
{
  return ::cuda::ceil_div<int>(sizeof(T) * 8, bits_per_pass);
}

template <int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int calc_num_passes(const int total_bits)
{
  return ::cuda::ceil_div<int>(total_bits, BitsPerPass);
}

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

template <typename KeyT, int BitsPerPass>
_CCCL_DEVICE _CCCL_FORCEINLINE void
set_kth_key_bits(key_prefix_storage_t<KeyT>& prefix, const int pass, const int bin_index)
{
  if constexpr (detail::radix::can_twiddle<KeyT>)
  {
    using bits_t        = typename Traits<KeyT>::UnsignedBits;
    const int start_bit = calc_start_bit<KeyT, BitsPerPass>(pass);
    bits_t bucket       = bin_index;
    prefix.bits |= static_cast<bits_t>(bucket) << start_bit;
  }
  else
  {
    prefix.shift_or(BitsPerPass, bin_index);
  }
}

template <typename KeyInT, typename OffsetT, typename OutOffsetT>
struct alignas(128) Counter
{
  // We are processing the items in multiple passes, from most-significant to least-significant bits. In each pass, we
  // keep the length of input (`len`) and the `k` of current pass, and update them at the end of the pass.
  OutOffsetT k;
  OffsetT len;

  // `previous_len` is the length of the input in the previous pass. Note that `previous_len` rather than `len` is used
  // for the filtering step because filtering is indeed for previous pass.
  OffsetT previous_len;

  // We determine the bits of the k_th key inside the mask processed by the pass. The
  // already known bits are stored in `kth_key_bits`. It's used to discriminate a
  // element is a result (written to `out`), a candidate for next pass (written to
  // `out_buf`), or not useful (discarded). The bits that are not yet processed do not
  // matter for this purpose.
  key_prefix_storage_t<KeyInT> kth_key_bits;

  // Record how many elements have passed filtering. It's used to determine the position
  // in the `out_buf` where an element should be written.
  alignas(128) OffsetT filter_cnt;

  // For a row inside a batch, we may launch multiple thread blocks. This counter is
  // used to determine if the current block is the last running block. If so, this block
  // will execute compute_bin_offsets() and choose_bucket().
  alignas(128) unsigned int finished_block_cnt;

  // Record how many elements have been written to the front of `out`. Elements less (if
  // SelectMin==true) than the k-th key are written from front to back.
  alignas(128) OutOffsetT out_cnt;

  // Record how many elements have been written to the back of `out`. Elements equal to
  // the k-th key are written from back to front. We need to keep count of them
  // separately because the number of elements that <= the k-th key might exceed k.
  alignas(128) OutOffsetT out_back_cnt;
  // The 'alignas' is necessary to improve the performance of global memory accessing by isolating the request,
  // especially for the segment version.
};

enum class candidate_class
{
  // The given candidate is definitely amongst the top-k items
  selected,
  // The given candidate may or may not be amongst the top-k items
  candidate,
  // The given candidate is definitely not amongst the top-k items
  rejected
};

//! @brief AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in
//! device-wide topK
//!
//! @tparam AgentTopKPolicyT
//!   Parameterized AgentTopKPolicy tuning policy type
//!
//! @tparam KeyInputIteratorT
//!   **[inferred]** Random-access input iterator type for reading input keys @iterator
//!
//! @tparam KeyOutputIteratorT
//!   **[inferred]** Random-access output iterator type for writing output keys @iterator
//!
//! @tparam ValueInputIteratorT
//!   **[inferred]** Random-access input iterator type for reading input values @iterator
//!
//! @tparam ValueOutputIteratorT
//!   **[inferred]** Random-access output iterator type for writing output values @iterator
//!
//! @tparam ExtractBinOpT
//!   Operations to extract the bin from the input key values
//!
//! @tparam IdentifyCandidatesOpT
//!    Operations to filter the input key values
//!
//! @tparam OffsetT
//!   Type of variable num_items
//!
//! @tparam OutOffsetT
//!   Type of variable k
//!
template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          typename OffsetT,
          typename OutOffsetT>
struct AgentTopK
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------
  // The key and value type
  using key_in_t   = it_value_t<KeyInputIteratorT>;
  using value_in_t = it_value_t<ValueInputIteratorT>;

  static constexpr int threads_per_block = AgentTopKPolicyT::threads_per_block;
  static constexpr int items_per_thread  = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass     = AgentTopKPolicyT::bits_per_pass;
  static constexpr int tile_items        = threads_per_block * items_per_thread;
  static constexpr int num_buckets       = 1 << bits_per_pass;

  static constexpr bool keys_only      = ::cuda::std::is_same_v<value_in_t, NullType>;
  static constexpr int bins_per_thread = ::cuda::ceil_div(num_buckets, threads_per_block);

  // Parameterized BlockLoad type for input data
  using block_load_input_t = BlockLoad<key_in_t, threads_per_block, items_per_thread, AgentTopKPolicyT::load_algorithm>;
  using block_load_trans_t = BlockLoad<OffsetT, threads_per_block, bins_per_thread, BLOCK_LOAD_TRANSPOSE>;
  // Parameterized BlockScan type
  using block_scan_t = BlockScan<OffsetT, threads_per_block, AgentTopKPolicyT::SCAN_ALGORITHM>;
  // Parameterized BlockStore type
  using block_store_trans_t = BlockStore<OffsetT, threads_per_block, bins_per_thread, BLOCK_STORE_TRANSPOSE>;

  // Shared memory
  struct _TempStorage
  {
    union
    {
      // Smem needed for loading
      typename block_load_input_t::TempStorage load_input;
      typename block_load_trans_t::TempStorage load_trans;
      // Smem needed for scan
      typename block_scan_t::TempStorage scan;
      // Smem needed for storing
      typename block_store_trans_t::TempStorage store_trans;
    };
    OffsetT histogram[num_buckets];
  };
  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------
  _TempStorage& temp_storage; // Reference to temp_storage
  KeyInputIteratorT d_keys_in; // Input keys
  KeyOutputIteratorT d_keys_out; // Output keys
  ValueInputIteratorT d_values_in; // Input values
  ValueOutputIteratorT d_values_out; // Output values
  OffsetT num_items; // Total number of input items
  OutOffsetT k; // Total number of output items
  OffsetT buffer_length; // Size of the buffer for storing intermediate candidates
  ExtractBinOpT extract_bin_op; // The operation for bin
  IdentifyCandidatesOpT identify_candidates_op; // The operation for filtering

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------
  //! @param temp_storage
  //!   Reference to temp_storage
  //!
  //! @param d_keys_in
  //!   Input data, keys
  //!
  //! @param d_keys_out
  //!   Output data, keys
  //!
  //! @param d_values_in
  //!   Input data, values
  //!
  //! @param d_values_out
  //!   Output data, values
  //!
  //! @param num_items
  //!   Total number of input items
  //!
  //! @param k
  //!   The K value. Will find K elements from num_items elements
  //!
  //! @param buffer_length
  //!   The size of the buffer for storing intermediate candidates
  //!
  //! @param extract_bin_op
  //!   Extract bin operator
  //!
  //! @param identify_candidates_op
  //!   Filter operator
  //!
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentTopK(
    TempStorage& temp_storage,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    OffsetT num_items,
    OutOffsetT k,
    OffsetT buffer_length,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op)
      : temp_storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , num_items(num_items)
      , k(k)
      , buffer_length(buffer_length)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
  {}

  //---------------------------------------------------------------------
  // Utility methods for device topK
  //---------------------------------------------------------------------

  // Process a range of input data in tiles, calling f(key, index) for each element
  template <typename InputItT, typename FuncT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void process_range(InputItT in, const OffsetT num_items, FuncT f)
  {
    key_in_t thread_data[items_per_thread];

    const OffsetT items_per_pass   = tile_items * gridDim.x;
    const OffsetT total_num_blocks = ::cuda::ceil_div(num_items, tile_items);

    const OffsetT num_remaining_elements = num_items % tile_items;
    const OffsetT last_block_id          = (total_num_blocks - 1) % gridDim.x;

    OffsetT tile_base = blockIdx.x * tile_items;
    OffsetT offset    = threadIdx.x * items_per_thread + tile_base;

    for (int i_block = blockIdx.x; i_block < total_num_blocks - 1; i_block += gridDim.x)
    {
      // Ensure that the temporary storage from previous iteration can be reused
      __syncthreads();

      block_load_input_t(temp_storage.load_input).Load(in + tile_base, thread_data);
      for (int j = 0; j < items_per_thread; ++j)
      {
        f(thread_data[j], offset + j);
      }
      tile_base += items_per_pass;
      offset += items_per_pass;
    }

    // Last tile specialized code-path
    if (blockIdx.x == last_block_id)
    {
      // Ensure that the temporary storage from the previous loop can be reused
      __syncthreads();

      if (num_remaining_elements == 0)
      {
        block_load_input_t(temp_storage.load_input).Load(in + tile_base, thread_data);
      }
      else
      {
        block_load_input_t(temp_storage.load_input).Load(in + tile_base, thread_data, num_remaining_elements);
      }

      for (int j = 0; j < items_per_thread; ++j)
      {
        if ((offset + j) < num_items)
        {
          f(thread_data[j], offset + j);
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void init_histograms(OffsetT* histogram)
  {
    // Initialize histogram bin counts to zeros
    int histo_offset = 0;

    // Loop unrolling is beneficial for performance here
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + threads_per_block <= num_buckets; histo_offset += threads_per_block)
    {
      histogram[histo_offset + threadIdx.x] = 0;
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % threads_per_block != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      histogram[histo_offset + threadIdx.x] = 0;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void merge_histograms(OffsetT* global_histogram)
  {
    int histo_offset = 0;

    // Loop unrolling is beneficial for performance here
    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + threads_per_block <= num_buckets; histo_offset += threads_per_block)
    {
      if (temp_storage.histogram[histo_offset + threadIdx.x] != 0)
      {
        atomicAdd(global_histogram + (histo_offset + threadIdx.x), temp_storage.histogram[histo_offset + threadIdx.x]);
      }
    }

    // Finish up with guarded merging if necessary
    if ((num_buckets % threads_per_block != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      atomicAdd(global_histogram + (histo_offset + threadIdx.x), temp_storage.histogram[histo_offset + threadIdx.x]);
    }
  }

  // Fused filtering of the current pass and building histogram for the next pass
  _CCCL_DEVICE _CCCL_FORCEINLINE void filter_and_histogram(
    key_in_t* in_buf,
    OffsetT* in_idx_buf,
    key_in_t* out_buf,
    OffsetT* out_idx_buf,
    OffsetT previous_len,
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OffsetT* histogram,
    bool early_stop,
    bool load_from_original_input)
  {
    // Initialize shared memory histogram
    init_histograms(temp_storage.histogram);

    // Make sure the histogram was initialized
    __syncthreads();

    OffsetT* p_filter_cnt = &counter->filter_cnt;
    OutOffsetT* p_out_cnt = &counter->out_cnt;

    // Lambda for early_stop = true (i.e., we have identified the exact "splitter" key):
    // Select all items that fall into the bin of the k-th item (i.e., the 'candidates') and the ones that fall into
    // bins preceding the k-th item bin (i.e., 'selected' items), write them to output.
    // We can skip histogram computation because we don't need to further passes to refine the candidates.
    auto f_early_stop = [load_from_original_input, in_idx_buf, p_out_cnt, this](key_in_t key, OffsetT i) {
      const candidate_class pre_res = identify_candidates_op(key);
      if (pre_res == candidate_class::candidate || pre_res == candidate_class::selected)
      {
        const OutOffsetT pos = atomicAdd(p_out_cnt, OutOffsetT{1});
        d_keys_out[pos]      = key;
        if constexpr (!keys_only)
        {
          const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
          d_values_out[pos]   = d_values_in[index];
        }
      }
    };

    // Lambda for early_stop = false, out_buf != nullptr (i.e., we need to further refine the candidates in the next
    // pass): Write out selected items to output, write candidates to out_buf, and build histogram for candidates.
    auto f_with_out_buf = [load_from_original_input, in_idx_buf, out_buf, out_idx_buf, p_filter_cnt, p_out_cnt, this](
                            key_in_t key, OffsetT i) {
      const candidate_class pre_res = identify_candidates_op(key);
      if (pre_res == candidate_class::candidate)
      {
        const OffsetT pos = atomicAdd(p_filter_cnt, OffsetT{1});
        out_buf[pos]      = key;
        if constexpr (!keys_only)
        {
          const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
          out_idx_buf[pos]    = index;
        }

        const int bucket = extract_bin_op(key);
        atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
      }
      else if (pre_res == candidate_class::selected)
      {
        const OutOffsetT pos = atomicAdd(p_out_cnt, OutOffsetT{1});
        d_keys_out[pos]      = key;
        if constexpr (!keys_only)
        {
          const OffsetT index = in_idx_buf ? in_idx_buf[i] : i;
          d_values_out[pos]   = d_values_in[index];
        }
      }
    };

    // Lambda for early_stop = false, out_buf = nullptr (i.e., we need to further refine the candidates in the next
    // pass, but we skip writing candidates to out_buf):
    // Just build histogram for candidates.
    // Note: We will only begin writing to d_keys_out starting from the pass in which the number of output-candidates
    // is small enough to fit into the output buffer (otherwise, we would be writing the same items to d_keys_out
    // multiple times).
    auto f_no_out_buf = [this](key_in_t key, OffsetT i) {
      const candidate_class pre_res = identify_candidates_op(key);
      if (pre_res == candidate_class::candidate)
      {
        const int bucket = extract_bin_op(key);
        atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
      }
    };

    // Choose and invoke the appropriate lambda with the correct input source
    // If the input size exceeds the allocated buffer size, we know for sure we haven't started writing candidates to
    // the output buffer yet
    if (load_from_original_input)
    {
      if (early_stop)
      {
        process_range(d_keys_in, previous_len, f_early_stop);
      }
      else if (out_buf)
      {
        process_range(d_keys_in, previous_len, f_with_out_buf);
      }
      else
      {
        process_range(d_keys_in, previous_len, f_no_out_buf);
      }
    }
    else
    {
      if (early_stop)
      {
        process_range(in_buf, previous_len, f_early_stop);
      }
      else if (out_buf)
      {
        process_range(in_buf, previous_len, f_with_out_buf);
      }
      else
      {
        process_range(in_buf, previous_len, f_no_out_buf);
      }
    }

    // Early stop means that subsequent passes are not needed
    if (early_stop)
    {
      return;
    }

    // Ensure all threads have contributed to the histogram before accumulating in the global memory
    __syncthreads();

    // Merge the locally aggregated histogram into the global histogram
    merge_histograms(histogram);
  }

  // Replace histogram with its own prefix sum
  _CCCL_DEVICE _CCCL_FORCEINLINE void compute_bin_offsets(volatile OffsetT* histogram)
  {
    OffsetT thread_data[bins_per_thread]{};

    // Load global histogram (we can skip initializing oob-items to zero because they won't be stored back)
    block_load_trans_t(temp_storage.load_trans).Load(histogram, thread_data, num_buckets);
    __syncthreads();

    block_scan_t(temp_storage.scan).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    block_store_trans_t(temp_storage.store_trans).Store(temp_storage.histogram, thread_data, num_buckets);
  }

  // Identify the bucket that the k-th value falls into
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  choose_bucket(Counter<key_in_t, OffsetT, OutOffsetT>* counter, const OutOffsetT k, const int pass)
  {
    // Initialize histogram bin counts to zeros
    int histo_offset = 0;

    auto body = [&] {
      const int bin_idx  = histo_offset + threadIdx.x;
      const OffsetT prev = (bin_idx == 0) ? 0 : temp_storage.histogram[bin_idx - 1];
      const OffsetT cur  = temp_storage.histogram[bin_idx];

      // Identify the bin that the k-th item falls into. One and only one thread will satisfy this condition, so counter
      // is written by only one thread
      if (prev < k && cur >= k)
      {
        // The number of items that are yet to be identified
        counter->k = k - prev;

        // The number of candidates in the next pass
        counter->len              = cur - prev;
        const unsigned int bucket = static_cast<unsigned int>(bin_idx);
        // Update the "splitter" key by adding the radix digit of the k-th item bin of this pass
        set_kth_key_bits<key_in_t, bits_per_pass>(counter->kth_key_bits, pass, bucket);
      }
    };

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + threads_per_block <= num_buckets; histo_offset += threads_per_block)
    {
      body();
    }
    // Finish up with guarded initialization if necessary
    if ((num_buckets % threads_per_block != 0) && (histo_offset + threadIdx.x < num_buckets))
    {
      body();
    }
  }

  // Performs the last-block coordination after histogram accumulation: ensures global visibility,
  // detects the last finishing block, runs the prefix sum, identifies the k-th bucket, and resets
  // the histogram for the next pass. The caller-supplied counter_update_fn runs on thread 0 of the
  // last block to update pass-specific counter state.
  template <typename CounterUpdateFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void finalize_pass(
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OffsetT* histogram,
    OutOffsetT current_k,
    int pass,
    bool is_last_pass,
    CounterUpdateFn counter_update_fn)
  {
    // Ensure all writes to the global memory-histogram are visible to all threads before
    // proceeding to compute the prefix sum over the histogram.
    __threadfence();

    // Identify the last block in the grid to perform the prefix sum over the histogram
    bool is_last_block = false;
    if (threadIdx.x == 0)
    {
      unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
      is_last_block         = (finished == (gridDim.x - 1));
    }

    // syncthreads ensures that the BlockLoad for loading the global histogram can reuse the temporary storage
    if (__syncthreads_or(is_last_block))
    {
      if (threadIdx.x == 0)
      {
        counter_update_fn();
      }

      // Compute prefix sum over the histogram's bin counts
      compute_bin_offsets(histogram);

      // Make sure the prefix sum has been written to shared memory before choose_bucket()
      __syncthreads();

      // Identify the bucket that the k-th item falls into
      choose_bucket(counter, current_k, pass);

      if (!is_last_pass)
      {
        init_histograms(histogram);
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke_last_filter(
    key_in_t* in_buf, OffsetT* in_idx_buf, Counter<key_in_t, OffsetT, OutOffsetT>* counter, OutOffsetT k, int pass)
  {
    const bool load_from_original_input = (pass <= 1) || counter->previous_len > buffer_length;
    const OffsetT current_len           = load_from_original_input ? num_items : counter->previous_len;
    in_idx_buf = load_from_original_input ? nullptr : in_idx_buf; // ? out_idx_buf : in_idx_buf;

    if (current_len == 0)
    {
      return;
    }

    // changed in choose_bucket(); need to reload
    OffsetT num_of_kth_needed  = counter->k;
    OutOffsetT* p_out_cnt      = &counter->out_cnt;
    OutOffsetT* p_out_back_cnt = &counter->out_back_cnt;

    auto f = [this, p_out_cnt, in_idx_buf, p_out_back_cnt, num_of_kth_needed, k, load_from_original_input](
               key_in_t key, OffsetT i) {
      const candidate_class res = identify_candidates_op(key);
      if (res == candidate_class::selected)
      {
        const OutOffsetT pos = atomicAdd(p_out_cnt, OffsetT{1});
        d_keys_out[pos]      = key;
        if constexpr (!keys_only)
        {
          // If writing has been skipped up to this point, `in_idx_buf` is nullptr
          const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
          d_values_out[pos]   = d_values_in[index];
        }
      }
      else if (res == candidate_class::candidate)
      {
        const OutOffsetT back_pos = atomicAdd(p_out_back_cnt, OffsetT{1});

        if (back_pos < num_of_kth_needed)
        {
          const OutOffsetT pos = k - 1 - back_pos;
          d_keys_out[pos]      = key;
          if constexpr (!keys_only)
          {
            const OffsetT new_idx = load_from_original_input ? i : in_idx_buf[i];
            d_values_out[pos]     = d_values_in[new_idx];
          }
        }
      }
    };

    if (load_from_original_input)
    {
      process_range(d_keys_in, current_len, f);
    }
    else
    {
      process_range(in_buf, current_len, f);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke_filter_and_histogram(
    key_in_t* in_buf,
    OffsetT* in_idx_buf,
    key_in_t* out_buf,
    OffsetT* out_idx_buf,
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OffsetT* histogram,
    int pass,
    bool is_last_pass)
  {
    const OutOffsetT current_k = counter->k;
    const OffsetT current_len  = counter->len;
    OffsetT previous_len       = counter->previous_len;

    // If current_len is 0, it means all the candidates have been found in previous passes.
    if (current_len == 0)
    {
      return;
    }

    // Early stop means that the bin containing the k-th element has been identified, and all
    // the elements in this bin are exactly the remaining k items we need to find. So we can
    // stop the process after this filtering pass.
    const bool early_stop = (current_len == static_cast<OffsetT>(current_k));

    // If previous_len > buffer_length, it means we haven't started writing candidates to out_buf yet,
    // so have to make sure to load input directly from the original input.
    // Also, unless we've had the chance to do at least one filtering pass, our input is definitely the original input
    // (this is to guard against edge cases, e.g., buffer_length=num_items=1).
    const bool load_from_original_input = (pass <= 1) || previous_len > buffer_length;

    if (load_from_original_input)
    {
      in_idx_buf   = nullptr;
      previous_len = num_items;
    }

    // "current_len > buffer_length" means current pass will skip writing buffer
    if (current_len > buffer_length)
    {
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    }

    // Fused filtering of candidates and histogram computation over the output-candidates
    filter_and_histogram(
      in_buf, in_idx_buf, out_buf, out_idx_buf, previous_len, counter, histogram, early_stop, load_from_original_input);

    finalize_pass(counter, histogram, current_k, pass, is_last_pass, [counter, current_len, early_stop] {
      if (early_stop)
      {
        // TODO (elstehle): Why do we need to reset previous_len here? I think we can skip it.
        counter->previous_len = 0;
        counter->len          = 0;
      }
      else
      {
        counter->previous_len = current_len;
        counter->filter_cnt   = 0;
      }
    });
  }

  // Histogram-only pass: computes the histogram over the full input without filtering.
  // Used for the first radix pass before any candidates have been identified.
  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke_histogram_only(
    Counter<key_in_t, OffsetT, OutOffsetT>* counter, OffsetT* histogram, int pass, bool is_last_pass)
  {
    // Initialize shared memory histogram
    init_histograms(temp_storage.histogram);
    __syncthreads();

    // Compute per-thread block histograms over the full input
    auto f = [this](key_in_t key, OffsetT /*index*/) {
      const int bucket = extract_bin_op(key);
      atomicAdd(temp_storage.histogram + bucket, OffsetT{1});
    };
    process_range(d_keys_in, num_items, f);

    // Ensure all threads have contributed to the histogram before accumulating in global memory
    __syncthreads();

    // Merge the locally aggregated histogram into the global histogram
    merge_histograms(histogram);

    finalize_pass(counter, histogram, k, pass, is_last_pass, [counter, this] {
      counter->previous_len = num_items;
      counter->filter_cnt   = 0;
    });
  }
};

//---------------------------------------------------------------------
// Free functions for common histogram operations
//---------------------------------------------------------------------

// Zero-initialize a histogram in shared or global memory
template <int BlockThreads, int NumBuckets, typename CounterT>
_CCCL_DEVICE _CCCL_FORCEINLINE void init_histogram(CounterT* histogram)
{
  int histo_offset = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (; histo_offset + BlockThreads <= NumBuckets; histo_offset += BlockThreads)
  {
    histogram[histo_offset + threadIdx.x] = 0;
  }
  if ((NumBuckets % BlockThreads != 0) && (histo_offset + static_cast<int>(threadIdx.x) < NumBuckets))
  {
    histogram[histo_offset + threadIdx.x] = 0;
  }
}

// Atomically merge a block-local histogram into a global histogram
template <int BlockThreads, int NumBuckets, typename CounterT>
_CCCL_DEVICE _CCCL_FORCEINLINE void merge_histogram(const CounterT* local_histogram, CounterT* global_histogram)
{
  int histo_offset = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (; histo_offset + BlockThreads <= NumBuckets; histo_offset += BlockThreads)
  {
    if (local_histogram[histo_offset + threadIdx.x] != 0)
    {
      atomicAdd(global_histogram + (histo_offset + threadIdx.x), local_histogram[histo_offset + threadIdx.x]);
    }
  }
  if ((NumBuckets % BlockThreads != 0) && (histo_offset + static_cast<int>(threadIdx.x) < NumBuckets))
  {
    if (local_histogram[histo_offset + threadIdx.x] != 0)
    {
      atomicAdd(global_histogram + (histo_offset + threadIdx.x), local_histogram[histo_offset + threadIdx.x]);
    }
  }
}

//---------------------------------------------------------------------
// Prefix sum over histogram bins: temp storage and compute function
//---------------------------------------------------------------------

template <int BlockThreads, int BitsPerPass, BlockScanAlgorithm ScanAlgorithm, typename OffsetT>
struct BinPrefixSumTempStorage
{
  static constexpr int num_buckets     = 1 << BitsPerPass;
  static constexpr int bins_per_thread = ::cuda::ceil_div(num_buckets, BlockThreads);

  using block_load_t  = BlockLoad<OffsetT, BlockThreads, bins_per_thread, BLOCK_LOAD_TRANSPOSE>;
  using block_scan_t  = BlockScan<OffsetT, BlockThreads, ScanAlgorithm>;
  using block_store_t = BlockStore<OffsetT, BlockThreads, bins_per_thread, BLOCK_STORE_TRANSPOSE>;

  union
  {
    typename block_load_t::TempStorage load;
    typename block_scan_t::TempStorage scan;
    typename block_store_t::TempStorage store;
  };
};

// Compute inclusive prefix sum over histogram bin counts.
// Reads from input_histogram (global), writes prefix sums to output_histogram (shared).
template <int BlockThreads, int BitsPerPass, BlockScanAlgorithm ScanAlgorithm, typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE void compute_bin_offsets(
  BinPrefixSumTempStorage<BlockThreads, BitsPerPass, ScanAlgorithm, OffsetT>& temp,
  volatile OffsetT* input_histogram,
  OffsetT* output_histogram)
{
  using storage_t                        = BinPrefixSumTempStorage<BlockThreads, BitsPerPass, ScanAlgorithm, OffsetT>;
  static constexpr int num_buckets_local = storage_t::num_buckets;
  static constexpr int bins_per_thread   = storage_t::bins_per_thread;

  OffsetT thread_data[bins_per_thread]{};
  typename storage_t::block_load_t(temp.load).Load(input_histogram, thread_data, num_buckets_local);
  __syncthreads();
  typename storage_t::block_scan_t(temp.scan).InclusiveSum(thread_data, thread_data);
  __syncthreads();
  typename storage_t::block_store_t(temp.store).Store(output_histogram, thread_data, num_buckets_local);
}

//---------------------------------------------------------------------
// choose_bucket: identify the bucket containing the k-th element
//---------------------------------------------------------------------

template <int BlockThreads, int BitsPerPass, typename KeyInT, typename OffsetT, typename OutOffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE void choose_bucket(
  const OffsetT* prefix_sum_histogram,
  Counter<KeyInT, OffsetT, OutOffsetT>* counter,
  const OutOffsetT k,
  const int pass)
{
  static constexpr int num_buckets = 1 << BitsPerPass;
  int histo_offset                 = 0;

  auto body = [&] {
    const int bin_idx  = histo_offset + threadIdx.x;
    const OffsetT prev = (bin_idx == 0) ? 0 : prefix_sum_histogram[bin_idx - 1];
    const OffsetT cur  = prefix_sum_histogram[bin_idx];

    if (prev < k && cur >= k)
    {
      counter->k   = k - prev;
      counter->len = cur - prev;
      set_kth_key_bits<KeyInT, BitsPerPass>(
        counter->kth_key_bits, pass, static_cast<unsigned int>(bin_idx));
    }
  };

  _CCCL_PRAGMA_UNROLL_FULL()
  for (; histo_offset + BlockThreads <= num_buckets; histo_offset += BlockThreads)
  {
    body();
  }
  if ((num_buckets % BlockThreads != 0) && (histo_offset + static_cast<int>(threadIdx.x) < num_buckets))
  {
    body();
  }
}

//---------------------------------------------------------------------
// finalize_pass: last-block coordination after histogram accumulation
//---------------------------------------------------------------------

// Ensures global visibility of histogram writes, detects the last finishing block,
// runs the prefix sum, identifies the k-th bucket, and resets the histogram for
// the next pass. counter_update_fn runs on thread 0 of the last block.
template <int BlockThreads,
          int BitsPerPass,
          BlockScanAlgorithm ScanAlgorithm,
          typename KeyInT,
          typename OffsetT,
          typename OutOffsetT,
          typename CounterUpdateFn>
_CCCL_DEVICE _CCCL_FORCEINLINE void finalize_pass(
  BinPrefixSumTempStorage<BlockThreads, BitsPerPass, ScanAlgorithm, OffsetT>& prefix_sum_temp,
  OffsetT* histogram_smem,
  Counter<KeyInT, OffsetT, OutOffsetT>* counter,
  OffsetT* global_histogram,
  OutOffsetT current_k,
  int pass,
  bool is_last_pass,
  CounterUpdateFn counter_update_fn)
{
  static constexpr int num_buckets = 1 << BitsPerPass;

  __threadfence();

  bool is_last_block = false;
  if (threadIdx.x == 0)
  {
    unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
    is_last_block         = (finished == (gridDim.x - 1));
  }

  if (__syncthreads_or(is_last_block))
  {
    if (threadIdx.x == 0)
    {
      counter_update_fn();
    }

    compute_bin_offsets(prefix_sum_temp, global_histogram, histogram_smem);
    __syncthreads();

    choose_bucket<BlockThreads, BitsPerPass>(histogram_smem, counter, current_k, pass);

    if (!is_last_pass)
    {
      init_histogram<BlockThreads, num_buckets>(global_histogram);
    }
  }
}

//---------------------------------------------------------------------
// process_histogram_only: tile processor that builds a radix histogram
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT>
struct process_histogram_only
{
  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;

  using prefix_sum_temp_t =
    BinPrefixSumTempStorage<block_threads, bits_per_pass, AgentTopKPolicyT::SCAN_ALGORITHM, OffsetT>;

  struct _TempStorage
  {
    prefix_sum_temp_t prefix_sum;
    OffsetT histogram[num_buckets];
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  prefix_sum_temp_t& prefix_sum;
  OffsetT* histogram;
  ExtractBinOpT extract_bin_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE
  process_histogram_only(TempStorage& ts, ExtractBinOpT extract_bin_op)
      : prefix_sum(ts.Alias().prefix_sum)
      , histogram(ts.Alias().histogram)
      , extract_bin_op(extract_bin_op)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE
  process_histogram_only(prefix_sum_temp_t& prefix_sum, OffsetT* histogram, ExtractBinOpT extract_bin_op)
      : prefix_sum(prefix_sum)
      , histogram(histogram)
      , extract_bin_op(extract_bin_op)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void segment_prologue()
  {
    init_histogram<block_threads, num_buckets>(histogram);
    __syncthreads();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void tile_prologue() {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile(
    const KeyInT (&items)[items_per_thread], OffsetT /*thread_offset*/, int num_thread_items)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      if (j < num_thread_items)
      {
        const int bucket = extract_bin_op(items[j]);
        atomicAdd(histogram + bucket, OffsetT{1});
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void tile_epilogue() {}

  template <typename CounterUpdateFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void segment_epilogue(
    Counter<KeyInT, OffsetT, OutOffsetT>* counter,
    OffsetT* global_histogram,
    OutOffsetT current_k,
    int pass,
    bool is_last_pass,
    CounterUpdateFn counter_update_fn)
  {
    __syncthreads();

    merge_histogram<block_threads, num_buckets>(histogram, global_histogram);

    finalize_pass(
      prefix_sum,
      histogram,
      counter,
      global_histogram,
      current_k,
      pass,
      is_last_pass,
      counter_update_fn);
  }
};

//---------------------------------------------------------------------
// Sink callables for process_filter_and_histogram
//
// Each sink defines the I/O routing for classified items. Two static
// constexpr flags control which phases process_filter_and_histogram
// executes:
//   accumulate_histogram – build local histogram for candidates
//   needs_finalize       – run last-block prefix-sum / choose_bucket
//---------------------------------------------------------------------

// filter_buffered: selected → output, candidate → temp buffer.
// Histogram accumulation for candidates is handled by process_filter_and_histogram.
template <typename KeyInT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT>
struct sink_filter_buffered
{
  static constexpr bool accumulate_histogram = true;
  static constexpr bool needs_finalize       = true;
  static constexpr bool keys_only            = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  KeyOutputIteratorT d_keys_out;
  ValueInputIteratorT d_values_in;
  ValueOutputIteratorT d_values_out;
  KeyInT* out_buf;
  OffsetT* out_idx_buf;
  OffsetT* in_idx_buf;
  OffsetT* p_filter_cnt;
  OutOffsetT* p_out_cnt;
  bool load_from_original_input;

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(KeyInT key, OffsetT i, candidate_class result)
  {
    if (result == candidate_class::candidate)
    {
      const OffsetT pos = atomicAdd(p_filter_cnt, OffsetT{1});
      out_buf[pos]      = key;
      if constexpr (!keys_only)
      {
        const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
        out_idx_buf[pos]    = index;
      }
    }
    else // selected
    {
      const OutOffsetT pos = atomicAdd(p_out_cnt, OutOffsetT{1});
      d_keys_out[pos]      = key;
      if constexpr (!keys_only)
      {
        const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
        d_values_out[pos]   = d_values_in[index];
      }
    }
  }
};

// filter_unbuffered: no writes at all. Candidates contribute to the
// histogram (handled by process_filter_and_histogram) but nothing is
// written to output or temp buffers.
struct sink_filter_unbuffered
{
  static constexpr bool accumulate_histogram = true;
  static constexpr bool needs_finalize       = true;

  template <typename KeyInT, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(KeyInT, OffsetT, candidate_class) {}
};

// filter_early_stop: selected | candidate → output. No histogram
// needed because this is the final filtering pass.
template <typename KeyInT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT>
struct sink_filter_early_stop
{
  static constexpr bool accumulate_histogram = false;
  static constexpr bool needs_finalize       = true;
  static constexpr bool keys_only            = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  KeyOutputIteratorT d_keys_out;
  ValueInputIteratorT d_values_in;
  ValueOutputIteratorT d_values_out;
  OutOffsetT* p_out_cnt;
  OffsetT* in_idx_buf;
  bool load_from_original_input;

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(KeyInT key, OffsetT i, candidate_class /*result*/)
  {
    const OutOffsetT pos = atomicAdd(p_out_cnt, OutOffsetT{1});
    d_keys_out[pos]      = key;
    if constexpr (!keys_only)
    {
      const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
      d_values_out[pos]   = d_values_in[index];
    }
  }
};

// last_filter: selected → output front, candidate → output back
// (capped at remaining k). No histogram, no finalize.
template <typename KeyInT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT>
struct sink_last_filter
{
  static constexpr bool accumulate_histogram = false;
  static constexpr bool needs_finalize       = false;
  static constexpr bool keys_only            = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  KeyOutputIteratorT d_keys_out;
  ValueInputIteratorT d_values_in;
  ValueOutputIteratorT d_values_out;
  OutOffsetT* p_out_cnt;
  OutOffsetT* p_out_back_cnt;
  OutOffsetT num_of_kth_needed;
  OutOffsetT k;
  OffsetT* in_idx_buf;
  bool load_from_original_input;

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(KeyInT key, OffsetT i, candidate_class result)
  {
    if (result == candidate_class::selected)
    {
      const OutOffsetT pos = atomicAdd(p_out_cnt, OutOffsetT{1});
      d_keys_out[pos]      = key;
      if constexpr (!keys_only)
      {
        const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
        d_values_out[pos]   = d_values_in[index];
      }
    }
    else // candidate
    {
      const OutOffsetT back_pos = atomicAdd(p_out_back_cnt, OutOffsetT{1});
      if (back_pos < num_of_kth_needed)
      {
        const OutOffsetT pos = k - 1 - back_pos;
        d_keys_out[pos]      = key;
        if constexpr (!keys_only)
        {
          const OffsetT index = load_from_original_input ? i : in_idx_buf[i];
          d_values_out[pos]   = d_values_in[index];
        }
      }
    }
  }
};

//---------------------------------------------------------------------
// process_filter_and_histogram: tile processor that classifies items
// and routes them through a caller-supplied sink. Optionally builds a
// radix histogram over candidates (controlled by SinkT::accumulate_histogram).
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          typename SinkT,
          typename OffsetT,
          typename OutOffsetT>
struct process_filter_and_histogram
{
  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;

  using prefix_sum_temp_t =
    BinPrefixSumTempStorage<block_threads, bits_per_pass, AgentTopKPolicyT::SCAN_ALGORITHM, OffsetT>;

  struct _TempStorage
  {
    prefix_sum_temp_t prefix_sum;
    OffsetT histogram[num_buckets];
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  prefix_sum_temp_t& prefix_sum;
  OffsetT* histogram;
  ExtractBinOpT extract_bin_op;
  IdentifyCandidatesOpT identify_candidates_op;
  SinkT sink;

  _CCCL_DEVICE _CCCL_FORCEINLINE
  process_filter_and_histogram(
    TempStorage& ts,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op,
    SinkT sink)
      : prefix_sum(ts.Alias().prefix_sum)
      , histogram(ts.Alias().histogram)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
      , sink(sink)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE
  process_filter_and_histogram(
    prefix_sum_temp_t& prefix_sum,
    OffsetT* histogram,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op,
    SinkT sink)
      : prefix_sum(prefix_sum)
      , histogram(histogram)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
      , sink(sink)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void segment_prologue()
  {
    if constexpr (SinkT::accumulate_histogram)
    {
      init_histogram<block_threads, num_buckets>(histogram);
      __syncthreads();
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void tile_prologue() {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void process_tile(
    const KeyInT (&items)[items_per_thread],
    OffsetT thread_offset,
    int num_thread_items)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; ++j)
    {
      if (j < num_thread_items)
      {
        const candidate_class result = identify_candidates_op(items[j]);
        if constexpr (SinkT::accumulate_histogram)
        {
          if (result == candidate_class::candidate)
          {
            const int bucket = extract_bin_op(items[j]);
            atomicAdd(histogram + bucket, OffsetT{1});
          }
        }
        if (result != candidate_class::rejected)
        {
          sink(items[j], thread_offset + j, result);
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void tile_epilogue() {}

  template <typename CounterUpdateFn>
  _CCCL_DEVICE _CCCL_FORCEINLINE void segment_epilogue(
    Counter<KeyInT, OffsetT, OutOffsetT>* counter,
    OffsetT* global_histogram,
    OutOffsetT current_k,
    int pass,
    bool is_last_pass,
    CounterUpdateFn counter_update_fn)
  {
    if constexpr (SinkT::accumulate_histogram)
    {
      __syncthreads();
      merge_histogram<block_threads, num_buckets>(histogram, global_histogram);
    }
    if constexpr (SinkT::needs_finalize)
    {
      finalize_pass(
        prefix_sum,
        histogram,
        counter,
        global_histogram,
        current_k,
        pass,
        is_last_pass,
        counter_update_fn);
    }
  }
};

//---------------------------------------------------------------------
// input_channel: wraps multiple iterators of the same value type with
// a runtime selector. Allows BlockTileLoader::consume to accept a
// single argument instead of branching at the call site.
//---------------------------------------------------------------------

template <typename... IteratorTs>
struct input_channel
{
  ::cuda::std::tuple<IteratorTs...> iterators;
  int active;
};

template <typename... IteratorTs>
_CCCL_DEVICE _CCCL_FORCEINLINE auto make_input_channel(int active, IteratorTs... its)
  -> input_channel<IteratorTs...>
{
  return {::cuda::std::tuple<IteratorTs...>{its...}, active};
}

//---------------------------------------------------------------------
// BlockTileLoader: standalone building block that iterates over a
// range in grid-stride tiles, loads each tile via BlockLoad, and
// delegates per-tile processing to a caller-supplied processor.
//
// ProcessorT must provide:
//   void tile_prologue();
//   void process_tile(const KeyT (&items)[items_per_thread],
//                     OffsetT thread_offset, int num_thread_items);
//   void tile_epilogue();
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT, typename KeyT, typename OffsetT>
struct BlockTileLoader
{
  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int tile_items       = block_threads * items_per_thread;

  using block_load_t = BlockLoad<KeyT, block_threads, items_per_thread, AgentTopKPolicyT::load_algorithm>;

  struct _TempStorage
  {
    typename block_load_t::TempStorage load;
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& temp_storage;

  _CCCL_DEVICE _CCCL_FORCEINLINE
  BlockTileLoader(_TempStorage& temp_storage)
      : temp_storage(temp_storage)
  {}

  // Plain iterator overload
  template <typename ProcessorT, typename InputItT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume(ProcessorT& proc, InputItT in, OffsetT num_items)
  {
    consume_impl(proc, in, num_items);
  }

  // input_channel overload — same tiling loop, branch at the Load site
  template <typename ProcessorT, typename... IteratorTs>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume(
    ProcessorT& proc, input_channel<IteratorTs...> channel, OffsetT num_items)
  {
    consume_impl(proc, channel, num_items);
  }

private:
  // Load a full tile from a plain iterator
  template <typename InputItT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(InputItT in, OffsetT tile_base, KeyT (&thread_data)[items_per_thread])
  {
    block_load_t(temp_storage.load).Load(in + tile_base, thread_data);
  }

  // Load a partial tile from a plain iterator
  template <typename InputItT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(
    InputItT in, OffsetT tile_base, KeyT (&thread_data)[items_per_thread], OffsetT valid_items)
  {
    block_load_t(temp_storage.load).Load(in + tile_base, thread_data, valid_items);
  }

  // Load a full tile from an input_channel — dispatch to the active iterator
  template <typename... IteratorTs>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(
    input_channel<IteratorTs...>& channel, OffsetT tile_base, KeyT (&thread_data)[items_per_thread])
  {
    channel_dispatch(channel, [&](auto& it) {
      block_load_t(temp_storage.load).Load(it + tile_base, thread_data);
    });
  }

  // Load a partial tile from an input_channel — dispatch to the active iterator
  template <typename... IteratorTs>
  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(
    input_channel<IteratorTs...>& channel,
    OffsetT tile_base,
    KeyT (&thread_data)[items_per_thread],
    OffsetT valid_items)
  {
    channel_dispatch(channel, [&](auto& it) {
      block_load_t(temp_storage.load).Load(it + tile_base, thread_data, valid_items);
    });
  }

  // Runtime dispatch over input_channel entries via fold expression
  template <typename... IteratorTs, typename Fn>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void channel_dispatch(input_channel<IteratorTs...>& channel, Fn fn)
  {
    channel_dispatch_impl(channel, fn, ::cuda::std::index_sequence_for<IteratorTs...>{});
  }

  template <typename... IteratorTs, typename Fn, ::cuda::std::size_t... Is>
  _CCCL_DEVICE _CCCL_FORCEINLINE static void channel_dispatch_impl(
    input_channel<IteratorTs...>& channel, Fn fn, ::cuda::std::index_sequence<Is...>)
  {
    ((channel.active == static_cast<int>(Is)
        ? (fn(::cuda::std::get<Is>(channel.iterators)), false)
        : true) && ...);
  }

  // Single tiling loop parameterized on the input source (plain iterator or input_channel).
  // load_tile overloads handle the dispatch.
  template <typename ProcessorT, typename InputSourceT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_impl(ProcessorT& proc, InputSourceT& in, OffsetT num_items)
  {
    KeyT thread_data[items_per_thread];

    const OffsetT items_per_pass         = static_cast<OffsetT>(tile_items) * gridDim.x;
    const OffsetT total_num_blocks       = ::cuda::ceil_div(num_items, static_cast<OffsetT>(tile_items));
    const OffsetT num_remaining_elements = num_items % tile_items;
    const OffsetT last_block_id          = (total_num_blocks - 1) % gridDim.x;

    OffsetT tile_base = static_cast<OffsetT>(blockIdx.x) * tile_items;

    for (OffsetT i_block = blockIdx.x; i_block < total_num_blocks - 1; i_block += gridDim.x)
    {
      __syncthreads();

      proc.tile_prologue();
      load_tile(in, tile_base, thread_data);

      const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
      proc.process_tile(thread_data, thread_offset, items_per_thread);
      proc.tile_epilogue();

      tile_base += items_per_pass;
    }

    if (static_cast<OffsetT>(blockIdx.x) == last_block_id)
    {
      __syncthreads();

      proc.tile_prologue();

      if (num_remaining_elements == 0)
      {
        load_tile(in, tile_base, thread_data);
      }
      else
      {
        load_tile(in, tile_base, thread_data, num_remaining_elements);
      }

      const OffsetT thread_offset = tile_base + static_cast<OffsetT>(threadIdx.x) * items_per_thread;
      const int num_thread_items =
        (num_remaining_elements == 0)
          ? items_per_thread
          : static_cast<int>(::cuda::std::max(
              OffsetT{0},
              ::cuda::std::min(static_cast<OffsetT>(items_per_thread),
                               num_items - thread_offset)));

      proc.process_tile(thread_data, thread_offset, num_thread_items);
      proc.tile_epilogue();
    }
  }
};

//---------------------------------------------------------------------
// AgentTopKHistogram: dedicated agent for the histogram-only pass
// (pass 0). Only needs the input iterator, extract_bin_op, and
// histogram infrastructure — no output iterators, no sinks.
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename ExtractBinOpT,
          typename OffsetT,
          typename OutOffsetT>
struct AgentTopKHistogram
{
  using key_in_t = it_value_t<KeyInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int num_buckets      = 1 << bits_per_pass;

  using tile_loader_t   = BlockTileLoader<AgentTopKPolicyT, key_in_t, OffsetT>;
  using prefix_sum_temp_t =
    BinPrefixSumTempStorage<block_threads, bits_per_pass, AgentTopKPolicyT::SCAN_ALGORITHM, OffsetT>;

  struct _TempStorage
  {
    union
    {
      typename tile_loader_t::_TempStorage loader;
      prefix_sum_temp_t prefix_sum;
    };
    OffsetT histogram[num_buckets];
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& temp_storage;
  KeyInputIteratorT d_keys_in;
  OffsetT num_items;
  OutOffsetT k;
  ExtractBinOpT extract_bin_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentTopKHistogram(
    TempStorage& temp_storage,
    const KeyInputIteratorT d_keys_in,
    OffsetT num_items,
    OutOffsetT k,
    ExtractBinOpT extract_bin_op)
      : temp_storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , num_items(num_items)
      , k(k)
      , extract_bin_op(extract_bin_op)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke(
    Counter<key_in_t, OffsetT, OutOffsetT>* counter, OffsetT* global_histogram, int pass, bool is_last_pass)
  {
    process_histogram_only<AgentTopKPolicyT, key_in_t, ExtractBinOpT, OffsetT, OutOffsetT> proc(
      temp_storage.prefix_sum, temp_storage.histogram, extract_bin_op);

    tile_loader_t loader(temp_storage.loader);

    proc.segment_prologue();
    loader.consume(proc, d_keys_in, num_items);
    proc.segment_epilogue(counter, global_histogram, k, pass, is_last_pass, [counter, this] {
      counter->previous_len = num_items;
      counter->filter_cnt   = 0;
    });
  }
};

//---------------------------------------------------------------------
// AgentTopKRefactored: composes the extracted processors, sinks,
// and free functions into a complete agent for the filter+histogram
// and last-filter passes.
//---------------------------------------------------------------------

template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          typename OffsetT,
          typename OutOffsetT>
struct AgentTopKRefactored
{
  using key_in_t = it_value_t<KeyInputIteratorT>;

  static constexpr int block_threads    = AgentTopKPolicyT::block_threads;
  static constexpr int items_per_thread = AgentTopKPolicyT::items_per_thread;
  static constexpr int bits_per_pass    = AgentTopKPolicyT::bits_per_pass;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr int num_buckets      = 1 << bits_per_pass;
  static constexpr bool keys_only       = ::cuda::std::is_same_v<ValueInputIteratorT, NullType*>;

  using tile_loader_t   = BlockTileLoader<AgentTopKPolicyT, key_in_t, OffsetT>;
  using prefix_sum_temp_t =
    BinPrefixSumTempStorage<block_threads, bits_per_pass, AgentTopKPolicyT::SCAN_ALGORITHM, OffsetT>;

  // BlockTileLoader and prefix_sum are used at different phases (tile
  // loading vs segment_epilogue) so they safely share storage via a union.
  struct _TempStorage
  {
    union
    {
      typename tile_loader_t::_TempStorage loader;
      prefix_sum_temp_t prefix_sum;
    };
    OffsetT histogram[num_buckets];
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& temp_storage;
  KeyInputIteratorT d_keys_in;
  KeyOutputIteratorT d_keys_out;
  ValueInputIteratorT d_values_in;
  ValueOutputIteratorT d_values_out;
  OffsetT num_items;
  OutOffsetT k;
  OffsetT buffer_length;
  ExtractBinOpT extract_bin_op;
  IdentifyCandidatesOpT identify_candidates_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentTopKRefactored(
    TempStorage& temp_storage,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    OffsetT num_items,
    OutOffsetT k,
    OffsetT buffer_length,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op)
      : temp_storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , num_items(num_items)
      , k(k)
      , buffer_length(buffer_length)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
  {}

  //---------------------------------------------------------------------
  // Public interface — filter+histogram and last-filter entry points
  //---------------------------------------------------------------------

  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke_filter_and_histogram(
    key_in_t* in_buf,
    OffsetT* in_idx_buf,
    key_in_t* out_buf,
    OffsetT* out_idx_buf,
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OffsetT* global_histogram,
    int pass,
    bool is_last_pass)
  {
    const OutOffsetT current_k = counter->k;
    const OffsetT current_len  = counter->len;
    OffsetT previous_len       = counter->previous_len;

    if (current_len == 0)
    {
      return;
    }

    const bool early_stop               = (current_len == static_cast<OffsetT>(current_k));
    const bool load_from_original_input = (pass <= 1) || previous_len > buffer_length;

    if (load_from_original_input)
    {
      in_idx_buf   = nullptr;
      previous_len = num_items;
    }
    if (current_len > buffer_length)
    {
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    }

    auto counter_update_fn = [counter, current_len, early_stop] {
      if (early_stop)
      {
        counter->previous_len = 0;
        counter->len          = 0;
      }
      else
      {
        counter->previous_len = current_len;
        counter->filter_cnt   = 0;
      }
    };

    tile_loader_t loader(temp_storage.loader);

    auto run = [&](auto sink) {
      using sink_t = decltype(sink);
      using proc_t = process_filter_and_histogram<
        AgentTopKPolicyT, key_in_t, ExtractBinOpT, IdentifyCandidatesOpT, sink_t, OffsetT, OutOffsetT>;

      proc_t proc(temp_storage.prefix_sum, temp_storage.histogram,
                  extract_bin_op, identify_candidates_op, sink);

      proc.segment_prologue();
      loader.consume(proc,
                     make_input_channel(load_from_original_input ? 0 : 1, d_keys_in, in_buf),
                     previous_len);
      proc.segment_epilogue(counter, global_histogram, current_k, pass, is_last_pass, counter_update_fn);
    };

    using sink_buffered_t =
      sink_filter_buffered<key_in_t, KeyOutputIteratorT, ValueInputIteratorT, ValueOutputIteratorT, OffsetT, OutOffsetT>;
    using sink_early_stop_t =
      sink_filter_early_stop<key_in_t, KeyOutputIteratorT, ValueInputIteratorT, ValueOutputIteratorT, OffsetT, OutOffsetT>;

    if (early_stop)
    {
      run(sink_early_stop_t{
        d_keys_out, d_values_in, d_values_out, &counter->out_cnt, in_idx_buf, load_from_original_input});
    }
    else if (out_buf)
    {
      run(sink_buffered_t{
        d_keys_out, d_values_in, d_values_out, out_buf, out_idx_buf, in_idx_buf,
        &counter->filter_cnt, &counter->out_cnt, load_from_original_input});
    }
    else
    {
      run(sink_filter_unbuffered{});
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void invoke_last_filter(
    key_in_t* in_buf,
    OffsetT* in_idx_buf,
    Counter<key_in_t, OffsetT, OutOffsetT>* counter,
    OutOffsetT k_val,
    int pass)
  {
    const bool load_from_original_input = (pass <= 1) || counter->previous_len > buffer_length;
    const OffsetT current_len           = load_from_original_input ? num_items : counter->previous_len;
    in_idx_buf                          = load_from_original_input ? nullptr : in_idx_buf;

    if (current_len == 0)
    {
      return;
    }

    using sink_t =
      sink_last_filter<key_in_t, KeyOutputIteratorT, ValueInputIteratorT, ValueOutputIteratorT, OffsetT, OutOffsetT>;

    sink_t sink{
      d_keys_out, d_values_in, d_values_out,
      &counter->out_cnt, &counter->out_back_cnt,
      static_cast<OutOffsetT>(counter->k), k_val,
      in_idx_buf, load_from_original_input};

    using proc_t = process_filter_and_histogram<
      AgentTopKPolicyT, key_in_t, ExtractBinOpT, IdentifyCandidatesOpT, sink_t, OffsetT, OutOffsetT>;

    proc_t proc(temp_storage.prefix_sum, temp_storage.histogram,
                extract_bin_op, identify_candidates_op, sink);

    tile_loader_t loader(temp_storage.loader);

    proc.segment_prologue();
    loader.consume(proc,
                   make_input_channel(load_from_original_input ? 0 : 1, d_keys_in, in_buf),
                   current_len);
    proc.segment_epilogue(counter, nullptr, OutOffsetT{0}, pass, false, [] {});
  }
};

} // namespace detail::topk
CUB_NAMESPACE_END
