/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding K largest (or smallest) items from sequences
//! of unordered data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/nvtx.cuh>
#include <cub/device/dispatch/dispatch_topk.cuh>

#include <iterator>

#include <stdio.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! @brief DeviceTopK provides device-wide, parallel operations for
//!        parallel operations for finding largest (or smallest) K items from sequences of unordered data
//!        items residing within device-accessible memory.
//!
//! @par Overview
//! TopK problem tries to find largest (or smallest) K items in a list. A relative problem is called
//! [*K selection problem*](https://en.wikipedia.org/wiki/Selection_algorithm), which find the Kth largest
//! (or smallest) value in a list. DeviceTopK will return K items as results (ordered or unordered). It is
//! based on an algorithm called [*AIR TopK*](https://dl.acm.org/doi/10.1145/3581784.3607062).
//!
//! @par Supported Types
//! DeviceTopK can process all of the built-in C++ numeric primitive types
//! (`unsigned char`, `int`, `double`, etc.) as well as CUDA's `__half`
//! and `__nv_bfloat16` 16-bit floating-point types.
//!
//! @par Stability
//! DeviceTopK provide stable and unstable version.
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceTopK}
//!
//! Performance
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @linear_performance{top-k}
//!
//! @endrst

struct DeviceTopK
{
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
  //!   **[inferred]** Random-access input iterator type for writing output values @iterator
  //!
  //! @tparam NumItemsT
  //! Type of variable num_items and k
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Pointer to the input data of key data to find top K
  //!
  //! @param[out] d_keys_out
  //!   Pointer to the K output sequence of key data
  //!
  //! @param[in] d_values_in
  //!   Pointer to the corresponding input sequence of associated value items
  //!
  //! @param[out] d_values_out
  //!   Pointer to the correspondingly output sequence of associated
  //!   value items
  //!
  //! @param[in] num_items
  //!   Number of items to be processed
  //!
  //! @param[in] k
  //!   The K value. Will find K elements from num_items elements
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueInputIteratorT,
            typename ValueOutputIteratorT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t TopKPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumItemsT k,
    cudaStream_t stream = 0)
  {
    static constexpr bool SelectMin = false;
    return DispatchTopK<KeyInputIteratorT,
                        KeyOutputIteratorT,
                        ValueInputIteratorT,
                        ValueOutputIteratorT,
                        NumItemsT,
                        SelectMin>::
      Dispatch(
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, k, stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename KeyInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueInputIteratorT,
            typename ValueOutputIteratorT,
            typename NumItemsT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t TopKPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumItemsT k,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return TopKPairs<KeyInputIteratorT, KeyOutputIteratorT, ValueInputIteratorT, ValueOutputIteratorT, NumItemsT>(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, k, stream);
  }
#endif
};

CUB_NAMESPACE_END
