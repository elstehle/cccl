/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
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

//! @file Operations for reading linear tiles of data into the CUDA thread block.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_exchange.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

//! @name Blocked arrangement I/O (direct)
//! @{

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block.
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputIteratorT
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base input iterator for loading from
//!
//! @param[out] items
//!   Data to load
template <typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectBlocked(int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
{
// Load directly in thread-blocked order
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    items[ITEM] = block_itr[(linear_tid * ITEMS_PER_THREAD) + ITEM];
  }
}

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block, guarded by range.
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputIteratorT
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base input iterator for loading from
//!
//! @param[out] items
//!   Data to load
//!
//! @param[in] valid_items
//!   Number of valid items to load
template <typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectBlocked(int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
{
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    if ((linear_tid * ITEMS_PER_THREAD) + ITEM < valid_items)
    {
      items[ITEM] = block_itr[(linear_tid * ITEMS_PER_THREAD) + ITEM];
    }
  }
}

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block, guarded
//! by range, with a fall-back assignment of out-of-bound elements.
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputIteratorT
//!   **[inferred]** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base input iterator for loading from
//!
//! @param[out] items
//!   Data to load
//!
//! @param[in] valid_items
//!   Number of valid items to load
//!
//! @param[in] oob_default
//!   Default value to assign out-of-bound items
template <typename InputT, typename DefaultT, int ITEMS_PER_THREAD, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectBlocked(
  int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
{
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    items[ITEM] = oob_default;
  }

  LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

//! @brief Internal implementation for load vectorization
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_ptr
//!   Input pointer for loading from
//!
//! @param[out] items
//!   Data to load
template <CacheLoadModifier MODIFIER, typename T, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
InternalLoadDirectBlockedVectorized(int linear_tid, T* block_ptr, T (&items)[ITEMS_PER_THREAD])
{
  // Biggest memory access word that T is a whole multiple of
  using DeviceWord = typename UnitWord<T>::DeviceWord;

  enum
  {
    TOTAL_WORDS = sizeof(items) / sizeof(DeviceWord),

    VECTOR_SIZE = (TOTAL_WORDS % 4 == 0) ? 4
                : (TOTAL_WORDS % 2 == 0) ? 2
                                         : 1,

    VECTORS_PER_THREAD = TOTAL_WORDS / VECTOR_SIZE,
  };

  // Vector type
  using Vector = typename CubVector<DeviceWord, VECTOR_SIZE>::Type;

  // Vector items
  Vector vec_items[VECTORS_PER_THREAD];

  // Aliased input ptr
  Vector* vec_ptr = reinterpret_cast<Vector*>(block_ptr) + (linear_tid * VECTORS_PER_THREAD);

// Load directly in thread-blocked order
#  pragma unroll
  for (int ITEM = 0; ITEM < VECTORS_PER_THREAD; ITEM++)
  {
    vec_items[ITEM] = ThreadLoad<MODIFIER>(vec_ptr + ITEM);
  }

// Copy
#  pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    items[ITEM] = *(reinterpret_cast<T*>(vec_items) + ITEM);
  }
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block.
//!
//! @blocked
//!
//! The input offset (``block_ptr + block_offset``) must be quad-item aligned
//!
//! The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
//!
//! - ``ITEMS_PER_THREAD`` is odd
//! - The data type ``T`` is not a built-in primitive or CUDA vector type
//!   (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_ptr
//!   Input pointer for loading from
//!
//! @param[out] items
//!   Data to load
template <typename T, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectBlockedVectorized(int linear_tid, T* block_ptr, T (&items)[ITEMS_PER_THREAD])
{
  InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, block_ptr, items);
}

//! @} end member group
//! @name Striped arrangement I/O (direct)
//! @{

//! @rst
//! Load a linear segment of items into a striped arrangement across the thread block.
//!
//! @striped
//!
//! @endrst
//!
//! @tparam BLOCK_THREADS
//!   The thread block size in threads
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputIteratorT
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base input iterator for loading from
//!
//! @param[out] items
//!   Data to load
template <int BLOCK_THREADS, typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectStriped(int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
{
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    items[ITEM] = block_itr[linear_tid + ITEM * BLOCK_THREADS];
  }
}

namespace detail
{

template <int BLOCK_THREADS, typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT, typename TransformOpT>
_CCCL_DEVICE _CCCL_FORCEINLINE void load_transform_direct_striped(
  int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], TransformOpT transform_op)
{
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    items[ITEM] = transform_op(block_itr[linear_tid + ITEM * BLOCK_THREADS]);
  }
}

} // namespace detail

//! @rst
//! Load a linear segment of items into a striped arrangement across the thread block, guarded by range
//!
//! @striped
//!
//! @endrst
//!
//! @tparam BLOCK_THREADS
//!   The thread block size in threads
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputIteratorT
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base input iterator for loading from
//!
//! @param[out] items
//!   Data to load
//!
//! @param[in] valid_items
//!   Number of valid items to load
//!
template <int BLOCK_THREADS, typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectStriped(int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
{
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    if (linear_tid + (ITEM * BLOCK_THREADS) < valid_items)
    {
      items[ITEM] = block_itr[linear_tid + ITEM * BLOCK_THREADS];
    }
  }
}

//! @rst
//! Load a linear segment of items into a striped arrangement across the thread block, guarded
//! by range, with a fall-back assignment of out-of-bound elements.
//!
//! @striped
//!
//! @endrst
//!
//! @tparam BLOCK_THREADS
//!   The thread block size in threads
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputIteratorT
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base input iterator for loading from
//!
//! @param[out] items
//!   Data to load
//!
//! @param[in] valid_items
//!   Number of valid items to load
//!
//! @param[in] oob_default
//!   Default value to assign out-of-bound items
template <int BLOCK_THREADS, typename InputT, typename DefaultT, int ITEMS_PER_THREAD, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectStriped(
  int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
{
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    items[ITEM] = oob_default;
  }

  LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, valid_items);
}

//! @} end member group
//! @name Warp-striped arrangement I/O (direct)
//! @{

//! @rst
//! Load a linear segment of items into a warp-striped arrangement across the thread block.
//!
//! @warpstriped
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//!
//! @endrst
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputIteratorT
//!   **inferred** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base input iterator for loading from
//!
//! @param[out] items
//!   Data to load
template <typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectWarpStriped(int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
{
  int tid         = linear_tid & (CUB_PTX_WARP_THREADS - 1);
  int wid         = linear_tid >> CUB_PTX_LOG_WARP_THREADS;
  int warp_offset = wid * CUB_PTX_WARP_THREADS * ITEMS_PER_THREAD;

// Load directly in warp-striped order
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    new (&items[ITEM]) InputT(block_itr[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]);
  }
}

//! @rst
//! Load a linear segment of items into a warp-striped arrangement across the thread block, guarded by range
//!
//! @warpstriped
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//!
//! @endrst
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputIteratorT
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base input iterator for loading from
//!
//! @param[out] items
//!   Data to load
//!
//! @param[in] valid_items
//!   Number of valid items to load
template <typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectWarpStriped(int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
{
  int tid         = linear_tid & (CUB_PTX_WARP_THREADS - 1);
  int wid         = linear_tid >> CUB_PTX_LOG_WARP_THREADS;
  int warp_offset = wid * CUB_PTX_WARP_THREADS * ITEMS_PER_THREAD;

// Load directly in warp-striped order
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    if (warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS) < valid_items)
    {
      new (&items[ITEM]) InputT(block_itr[warp_offset + tid + (ITEM * CUB_PTX_WARP_THREADS)]);
    }
  }
}

//! @rst
//! Load a linear segment of items into a warp-striped arrangement across the thread block,
//! guarded by range, with a fall-back assignment of out-of-bound elements.
//!
//! @warpstriped
//!
//! @endrst
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputIteratorT
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base input iterator for loading from
//!
//! @param[out] items
//!   Data to load
//!
//! @param[in] valid_items
//!   Number of valid items to load
//!
//! @param[in] oob_default
//!   Default value to assign out-of-bound items
template <typename InputT, typename DefaultT, int ITEMS_PER_THREAD, typename InputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectWarpStriped(
  int linear_tid, InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
{
// Load directly in warp-striped order
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    items[ITEM] = oob_default;
  }

  LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items);
}

//! @} end member group

//! @brief cub::BlockLoadAlgorithm enumerates alternative algorithms for cub::BlockLoad to read a
//!        linear segment of data from memory into a blocked arrangement across a CUDA thread block.
enum BlockLoadAlgorithm
{
  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) decreases as the
  //! access stride between threads increases (i.e., the number items per thread).
  //! @endrst
  BLOCK_LOAD_DIRECT,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) doesn't depend on
  //! the number of items per thread.
  //!
  //! @endrst
  BLOCK_LOAD_STRIPED,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read
  //! from memory using CUDA's built-in vectorized loads as a coalescing optimization.
  //! For example, ``ld.global.v4.s32`` instructions will be generated
  //! when ``T = int`` and ``ITEMS_PER_THREAD % 4 == 0``.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high until the the
  //!   access stride between threads (i.e., the number items per thread) exceeds the
  //!   maximum vector load width (typically 4 items or 64B, whichever is lower).
  //! - The following conditions will prevent vectorization and loading will fall
  //!   back to cub::BLOCK_LOAD_DIRECT:
  //!
  //!   - ``ITEMS_PER_THREAD`` is odd
  //!   - The ``InputIteratorT`` is not a simple pointer type
  //!   - The block input offset is not quadword-aligned
  //!   - The data type ``T`` is not a built-in primitive or CUDA vector type
  //!     (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
  //!
  //! @endrst
  BLOCK_LOAD_VECTORIZE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is read efficiently from memory and then
  //! locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless
  //!   of items loaded per thread.
  //! - The local reordering incurs slightly longer latencies and throughput than the
  //!   direct cub::BLOCK_LOAD_DIRECT and cub::BLOCK_LOAD_VECTORIZE alternatives.
  //!
  //! @endrst
  BLOCK_LOAD_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read efficiently from memory and then
  //! locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - BLOCK_THREADS must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless of items loaded per thread.
  //! - The local reordering incurs slightly larger latencies than the
  //!   direct cub::BLOCK_LOAD_DIRECT and cub::BLOCK_LOAD_VECTORIZE alternatives.
  //! - Provisions more shared storage, but incurs smaller latencies than the
  //!   BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED alternative.
  //!
  //! @endrst
  BLOCK_LOAD_WARP_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! Like ``BLOCK_LOAD_WARP_TRANSPOSE``, a :ref:`warp-striped arrangement <flexible-data-arrangement>`
  //! of data is read directly from memory and then is locally transposed into a
  //! :ref:`blocked arrangement <flexible-data-arrangement>`. To reduce the shared memory requirement, only one
  //! warp's worth of shared memory is provisioned and is subsequently time-sliced among warps.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - BLOCK_THREADS must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless
  //!   of items loaded per thread.
  //! - Provisions less shared memory temporary storage, but incurs larger
  //!   latencies than the BLOCK_LOAD_WARP_TRANSPOSE alternative.
  //!
  //! @endrst
  BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
};

//! @rst
//! The BlockLoad class provides :ref:`collective <collective-primitives>` data movement methods for loading a linear
//! segment of items from memory into a :ref:`blocked arrangement <flexible-data-arrangement>` across a
//! CUDA thread block.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - The BlockLoad class provides a single data movement abstraction that can be specialized
//!   to implement different cub::BlockLoadAlgorithm strategies.  This facilitates different
//!   performance policies for different architectures, data types, granularity sizes, etc.
//! - BlockLoad can be optionally specialized by different data movement strategies:
//!
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_DIRECT`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from memory.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_STRIPED`:
//!      A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_VECTORIZE`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from memory
//!      using CUDA's built-in vectorized loads as a coalescing optimization.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_TRANSPOSE`:
//!      A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory and is then
//!      locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_WARP_TRANSPOSE`:
//!      A :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read directly from memory and is then
//!      locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED`:
//!      A :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read directly from memory and is then
//!      locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>` one warp at a time.
//!
//! - @rowmajor
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockLoad}
//!
//! The code snippet below illustrates the loading of a linear
//! segment of 512 integers into a "blocked" arrangement across 128 threads where each
//! thread owns 4 consecutive items. The load is specialized for ``BLOCK_LOAD_WARP_TRANSPOSE``,
//! meaning memory references are efficiently coalesced using a warp-striped access
//! pattern (after which items are locally reordered among threads).
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
//!
//!    __global__ void ExampleKernel(int *d_data, ...)
//!    {
//!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
//!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
//!
//!        // Allocate shared memory for BlockLoad
//!        __shared__ typename BlockLoad::TempStorage temp_storage;
//!
//!        // Load a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        BlockLoad(temp_storage).Load(d_data, thread_data);
//!
//! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...``.
//! The set of ``thread_data`` across the block of threads in those threads will be
//! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of
//! dynamically shared memory with BlockReduce and how to re-purpose the same memory region.
//! This example can be easily adapted to the storage required by BlockLoad.
//!
//! @endrst
//!
//! @tparam InputT
//!   The data type to read into (which must be convertible from the input iterator's value type).
//!
//! @tparam BLOCK_DIM_X
//!   The thread block length in threads along the X dimension
//!
//! @tparam ITEMS_PER_THREAD
//!   The number of consecutive items partitioned onto each thread.
//!
//! @tparam ALGORITHM
//!   **[optional]** cub::BlockLoadAlgorithm tuning policy. default: ``cub::BLOCK_LOAD_DIRECT``.
//!
//! @tparam WARP_TIME_SLICING
//!   **[optional]** Whether or not only one warp's worth of shared memory should be
//!   allocated and time-sliced among block-warps during any load-related data transpositions
//!   (versus each warp having its own storage). (default: false)
//!
//! @tparam BLOCK_DIM_Y
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1)
//!
//! @tparam BLOCK_DIM_Z
//!  **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
//! @tparam LEGACY_PTX_ARCH
//!  **[optional]** Unused.
template <typename InputT,
          int BLOCK_DIM_X,
          int ITEMS_PER_THREAD,
          BlockLoadAlgorithm ALGORITHM = BLOCK_LOAD_DIRECT,
          int BLOCK_DIM_Y              = 1,
          int BLOCK_DIM_Z              = 1,
          int LEGACY_PTX_ARCH          = 0>
class BlockLoad
{
private:
  /// Constants
  enum
  {
    /// The thread block size in threads
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
  };

  /// Load helper
  template <BlockLoadAlgorithm _POLICY, int DUMMY>
  struct LoadInternal;

  /**
   * BLOCK_LOAD_DIRECT specialization of load helper
   */
  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_DIRECT, DUMMY>
  {
    /// Shared memory storage layout type
    using TempStorage = NullType;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    /**
     * @brief Load a linear segment of items from memory
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectBlocked(linear_tid, block_itr, items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range, with a fall-back
     *        assignment of out-of-bound elements
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     *
     * @param[in] oob_default
     *   Default value to assign out-of-bound items
     */
    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default);
    }
  };

  /**
   * BLOCK_LOAD_STRIPED specialization of load helper
   */
  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_STRIPED, DUMMY>
  {
    /// Shared memory storage layout type
    using TempStorage = NullType;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    /**
     * @brief Load a linear segment of items from memory
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, valid_items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range, with a fall-back
     *        assignment of out-of-bound elements
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     *
     * @param[in] oob_default
     *   Default value to assign out-of-bound items
     */
    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, valid_items, oob_default);
    }
  };

  /**
   * BLOCK_LOAD_VECTORIZE specialization of load helper
   */
  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_VECTORIZE, DUMMY>
  {
    /// Shared memory storage layout type
    using TempStorage = NullType;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    /**
     * @brief Load a linear segment of items from memory, specialized for native pointer types
     * (attempts vectorization)
     *
     * @param[in] block_ptr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputT* block_ptr, InputT (&items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, block_ptr, items);
    }

    /**
     * @brief Load a linear segment of items from memory, specialized for native pointer types
     * (attempts vectorization)
     *
     * @param[in] block_ptr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(const InputT* block_ptr, InputT (&items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, block_ptr, items);
    }

    /**
     * @brief Load a linear segment of items from memory, specialized for native pointer types
     *        (attempts vectorization)
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     */
    template <CacheLoadModifier MODIFIER, typename ValueType, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(CacheModifiedInputIterator<MODIFIER, ValueType, OffsetT> block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<MODIFIER>(linear_tid, block_itr.ptr, items);
    }

    /**
     * @brief Load a linear segment of items from memory, specialized for opaque input iterators
     *        (skips vectorization)
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     */
    template <typename _InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(_InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectBlocked(linear_tid, block_itr, items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range (skips vectorization)
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range, with a fall-back
     *        assignment of out-of-bound elements (skips vectorization)
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     *
     * @param[in] oob_default
     *   Default value to assign out-of-bound items
     */
    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default);
    }
  };

  /**
   * BLOCK_LOAD_TRANSPOSE specialization of load helper
   */
  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_TRANSPOSE, DUMMY>
  {
    // BlockExchange utility type for keys
    using BlockExchange = BlockExchange<InputT, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z>;

    /// Shared memory storage layout type
    struct _TempStorage : BlockExchange::TempStorage
    {};

    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage>
    {};

    /// Thread reference to shared storage
    _TempStorage& temp_storage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    /**
     * @brief Load a linear segment of items from memory
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items);
      BlockExchange(temp_storage).StripedToBlocked(items, items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, valid_items);
      BlockExchange(temp_storage).StripedToBlocked(items, items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range, with a fall-back
     * assignment of out-of-bound elements
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     *
     * @param[in] oob_default
     *   Default value to assign out-of-bound items
     */
    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, valid_items, oob_default);
      BlockExchange(temp_storage).StripedToBlocked(items, items);
    }
  };

  /**
   * BLOCK_LOAD_WARP_TRANSPOSE specialization of load helper
   */
  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_WARP_TRANSPOSE, DUMMY>
  {
    enum
    {
      WARP_THREADS = CUB_WARP_THREADS(0)
    };

    // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
    static_assert(int(BLOCK_THREADS) % int(WARP_THREADS) == 0, "BLOCK_THREADS must be a multiple of WARP_THREADS");

    // BlockExchange utility type for keys
    using BlockExchange = BlockExchange<InputT, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z>;

    /// Shared memory storage layout type
    struct _TempStorage : BlockExchange::TempStorage
    {};

    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage>
    {};

    /// Thread reference to shared storage
    _TempStorage& temp_storage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    /**
     * @brief Load a linear segment of items from memory
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectWarpStriped(linear_tid, block_itr, items);
      BlockExchange(temp_storage).WarpStripedToBlocked(items, items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items);
      BlockExchange(temp_storage).WarpStripedToBlocked(items, items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range, with a fall-back
     *        assignment of out-of-bound elements
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     *
     * @param[in] oob_default
     *   Default value to assign out-of-bound items
     */
    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items, oob_default);
      BlockExchange(temp_storage).WarpStripedToBlocked(items, items);
    }
  };

  /**
   * BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED specialization of load helper
   */
  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED, DUMMY>
  {
    enum
    {
      WARP_THREADS = CUB_WARP_THREADS(0)
    };

    // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
    static_assert(int(BLOCK_THREADS) % int(WARP_THREADS) == 0, "BLOCK_THREADS must be a multiple of WARP_THREADS");

    // BlockExchange utility type for keys
    using BlockExchange = BlockExchange<InputT, BLOCK_DIM_X, ITEMS_PER_THREAD, true, BLOCK_DIM_Y, BLOCK_DIM_Z>;

    /// Shared memory storage layout type
    struct _TempStorage : BlockExchange::TempStorage
    {};

    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage>
    {};

    /// Thread reference to shared storage
    _TempStorage& temp_storage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    /**
     * @brief Load a linear segment of items from memory
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectWarpStriped(linear_tid, block_itr, items);
      BlockExchange(temp_storage).WarpStripedToBlocked(items, items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     */
    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items);
      BlockExchange(temp_storage).WarpStripedToBlocked(items, items);
    }

    /**
     * @brief Load a linear segment of items from memory, guarded by range, with a fall-back
     *        assignment of out-of-bound elements
     *
     * @param[in] block_itr
     *   The thread block's base input iterator for loading from
     *
     * @param[out] items
     *   Data to load
     *
     * @param[in] valid_items
     *   Number of valid items to load
     *
     * @param[in] oob_default
     *   Default value to assign out-of-bound items
     */
    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items, oob_default);
      BlockExchange(temp_storage).WarpStripedToBlocked(items, items);
    }
  };

  /// Internal load implementation to use
  using InternalLoad = LoadInternal<ALGORITHM, 0>;

  /// Shared memory storage layout type
  using _TempStorage = typename InternalLoad::TempStorage;

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  /// Thread reference to shared storage
  _TempStorage& temp_storage;

  /// Linear thread-id
  int linear_tid;

public:
  /// @smemstorage{BlockLoad}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  /**
   * @brief Collective constructor using a private static allocation of shared memory as temporary
   *        storage.
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoad()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation as temporary storage.
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoad(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @} end member group
  //! @name Data movement
  //! @{

  //! @rst
  //! Load a linear segment of items from memory.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the loading of a linear
  //! segment of 512 integers into a "blocked" arrangement across 128 threads where each
  //! thread owns 4 consecutive items. The load is specialized for ``BLOCK_LOAD_WARP_TRANSPOSE``,
  //! meaning memory references are efficiently coalesced using a warp-striped access
  //! pattern (after which items are locally reordered among threads).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, ...)
  //!    {
  //!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockLoad
  //!        __shared__ typename BlockLoad::TempStorage temp_storage;
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        BlockLoad(temp_storage).Load(d_data, thread_data);
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...``.
  //! The set of ``thread_data`` across the block of threads in those threads will be
  //! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
  //!
  //! @endrst
  //!
  //! @param[in] block_itr
  //!   The thread block's base input iterator for loading from
  //!
  //! @param[out] items
  //!   Data to load
  template <typename InputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
  {
    InternalLoad(temp_storage, linear_tid).Load(block_itr, items);
  }

  //! @rst
  //!
  //! Load a linear segment of items from memory, guarded by range.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the guarded loading of a linear
  //! segment of 512 integers into a "blocked" arrangement across 128 threads where each
  //! thread owns 4 consecutive items. The load is specialized for ``BLOCK_LOAD_WARP_TRANSPOSE``,
  //! meaning memory references are efficiently coalesced using a warp-striped access
  //! pattern (after which items are locally reordered among threads).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int valid_items, ...)
  //!    {
  //!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockLoad
  //!        __shared__ typename BlockLoad::TempStorage temp_storage;
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        BlockLoad(temp_storage).Load(d_data, thread_data, valid_items);
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, 6...`` and ``valid_items`` is ``5``.
  //! The set of ``thread_data`` across the block of threads in those threads will be
  //! ``{ [0,1,2,3], [4,?,?,?], ..., [?,?,?,?] }``, with only the first two threads
  //! being unmasked to load portions of valid data (and other items remaining unassigned).
  //!
  //! @endrst
  //!
  //! @param[in] block_itr
  //!   The thread block's base input iterator for loading from
  //!
  //! @param[out] items
  //!   Data to load
  //!
  //! @param[in] valid_items
  //!   Number of valid items to load
  template <typename InputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
  {
    InternalLoad(temp_storage, linear_tid).Load(block_itr, items, valid_items);
  }

  //! @rst
  //! Load a linear segment of items from memory, guarded by range, with a fall-back
  //! assignment of out-of-bound elements
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the guarded loading of a linear
  //! segment of 512 integers into a "blocked" arrangement across 128 threads where each
  //! thread owns 4 consecutive items. The load is specialized for ``BLOCK_LOAD_WARP_TRANSPOSE``,
  //! meaning memory references are efficiently coalesced using a warp-striped access
  //! pattern (after which items are locally reordered among threads).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int valid_items, ...)
  //!    {
  //!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockLoad
  //!        __shared__ typename BlockLoad::TempStorage temp_storage;
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        BlockLoad(temp_storage).Load(d_data, thread_data, valid_items, -1);
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, 6...``
  //! ``valid_items`` is ``5``, and the out-of-bounds default is ``-1``.
  //! The set of ``thread_data`` across the block of threads in those threads will be
  //! ``{ [0,1,2,3], [4,-1,-1,-1], ..., [-1,-1,-1,-1] }``, with only the first two threads
  //! being unmasked to load portions of valid data (and other items are assigned ``-1``)
  //!
  //! @endrst
  //!
  //! @param[in] block_itr
  //!   The thread block's base input iterator for loading from
  //!
  //! @param[out] items
  //!   Data to load
  //!
  //! @param[in] valid_items
  //!   Number of valid items to load
  //!
  //! @param[in] oob_default
  //!   Default value to assign out-of-bound items
  template <typename InputIteratorT, typename DefaultT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
  {
    InternalLoad(temp_storage, linear_tid).Load(block_itr, items, valid_items, oob_default);
  }

  //@}  end member group
};

template <class Policy, class It, class T = cub::detail::value_t<It>>
struct BlockLoadType
{
  using type = cub::BlockLoad<T, Policy::BLOCK_THREADS, Policy::ITEMS_PER_THREAD, Policy::LOAD_ALGORITHM>;
};

CUB_NAMESPACE_END
