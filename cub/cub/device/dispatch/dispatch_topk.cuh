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
 * @file
 *   cub::DeviceTopK provides device-wide, parallel operations for finding K largest (or smallest) items
 * from sequences of unordered data items residing within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#include <cub/agent/agent_topk.cuh>
#include <cub/block/block_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/cmath>

#include <cstdio>
#include <iterator>

#include <nv/target>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace topk
{

template <class KeyT>
constexpr int calc_bits_per_pass()
{
  return sizeof(KeyT) == 1 ? 8 : sizeof(KeyT) == 2 ? 11 : sizeof(KeyT) == 4 ? 11 : sizeof(KeyT) == 8 ? 11 : 8;
}

template <typename T, int BitsPerPass>
__host__ __device__ constexpr int calc_num_passes()
{
  return ::cuda::ceil_div<int>(sizeof(T) * 8, BitsPerPass);
}

template <int BitsPerPass>
__host__ __device__ constexpr int calc_num_buckets()
{
  return 1 << BitsPerPass;
}

template <class KeyInT>
struct sm90_tuning
{
  static constexpr int threads = 256; // Number of threads per block

  using WideT                                      = float4;
  static constexpr int nominal_4b_items_per_thread = 16;
  static constexpr int items_per_scaler            = CUB_MAX(sizeof(WideT) / sizeof(KeyInT), 1);
  static constexpr int items                       = items_per_scaler * nominal_4b_items_per_thread;

  static constexpr int bits_per_pass          = detail::topk::calc_bits_per_pass<KeyInT>();
  static constexpr int num_passes             = detail::topk::calc_num_passes<KeyInT, bits_per_pass>();
  static constexpr int num_buckets            = detail::topk::calc_num_buckets<bits_per_pass>();
  static constexpr int coefficient_for_buffer = 128;

  static constexpr cub::BlockLoadAlgorithm load_algorithm = cub::BLOCK_LOAD_DIRECT;
};

} // namespace topk
} // namespace detail

template <class KeyInT, class KeyOutT, class ValueInT, class ValueOutT, class NumItemsT>
struct device_topk_policy_hub
{
  struct DefaultTuning
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 10;

    static constexpr int ITEMS_PER_THREAD =
      CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(KeyInT))));

    static constexpr int bits_per_pass          = detail::topk::calc_bits_per_pass<KeyInT>();
    static constexpr int num_passes             = detail::topk::calc_num_passes<KeyInT, bits_per_pass>();
    static constexpr int num_buckets            = detail::topk::calc_num_buckets<bits_per_pass>();
    static constexpr int coefficient_for_buffer = 128;

    using TopKPolicyT =
      AgentTopKPolicy<128,
                      ITEMS_PER_THREAD,
                      bits_per_pass,
                      num_passes,
                      num_buckets,
                      coefficient_for_buffer,
                      cub::BLOCK_LOAD_DIRECT,
                      cub::BLOCK_HISTO_ATOMIC,
                      cub::BLOCK_SCAN_WARP_SCANS>;
  };

  struct Policy350
      : DefaultTuning
      , cub::ChainedPolicy<350, Policy350, Policy350>
  {};

  struct Policy900 : cub::ChainedPolicy<900, Policy900, Policy350>
  {
    using tuning = detail::topk::sm90_tuning<KeyInT>;

    using TopKPolicyT =
      AgentTopKPolicy<tuning::threads,
                      tuning::items,
                      tuning::bits_per_pass,
                      tuning::num_passes,
                      tuning::num_buckets,
                      tuning::coefficient_for_buffer,
                      tuning::load_algorithm,
                      cub::BLOCK_HISTO_ATOMIC,
                      cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = Policy900;
};

/*
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
 * @param[in] d_temp_storage
 *   Device-accessible allocation of temporary storage. When `nullptr`, the
 *   required allocation size is written to `temp_storage_bytes` and no work is done.
 *
 * @param[in,out] temp_storage_bytes
 *   Reference to size in bytes of `d_temp_storage` allocation
 *
 * @param[in] d_keys_in
 *   Pointer to the input data of key data to find top K
 *
 * @param[out] d_keys_out
 *   Pointer to the K output sequence of key data
 *
 * @param[in] d_values_in
 *   Pointer to the corresponding input sequence of associated value items
 *
 * @param[out] d_values_out
 *   Pointer to the correspondingly output sequence of associated
 *   value items
 *
 * @param[in] num_items
 *   Number of items to be processed
 *
 * @param[in] k
 *   The K value. Will find K elements from num_items elements
 *
 * @param[in] stream
 *   @rst
 *   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
 *   @endrst
 */
template <typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename NumItemsT,
          typename SelectedPolicy = device_topk_policy_hub<cub::detail::value_t<KeyInputIteratorT>,
                                                           cub::detail::value_t<KeyOutputIteratorT>,
                                                           cub::detail::value_t<ValueInputIteratorT>,
                                                           cub::detail::value_t<ValueOutputIteratorT>,
                                                           NumItemsT>>
struct DispatchTopK : SelectedPolicy
{
  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to `temp_storage_bytes`
  /// and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  KeyInputIteratorT d_keys_in;

  /// Pointer to the K output sequence of key data
  KeyOutputIteratorT d_keys_out;

  /// Pointer to the corresponding input sequence of associated value items
  ValueInputIteratorT d_values_in;

  /// Pointer to the correspondingly output sequence of associated value items
  ValueOutputIteratorT d_values_out;

  /// Number of items to be processed
  NumItemsT num_items;

  /// The K value. Will find K elements from num_items elements
  NumItemsT k;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  /*
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input data of key data to find top K
   *
   * @param[out] d_keys_out
   *   Pointer to the K output sequence of key data
   *
   * @param[in] d_values_in
   *   Pointer to the corresponding input sequence of associated value items
   *
   * @param[out] d_values_out
   *   Pointer to the correspondingly output sequence of associated
   *   value items
   *
   * @param[in] num_items
   *   Number of items to be processed
   *
   * @param[in] k
   *   The K value. Will find K elements from num_items elements
   *
   * @param[in] stream
   *   @rst
   *   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
   *   @endrst
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchTopK(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumItemsT k,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , num_items(num_items)
      , k(k)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;
    using Policy     = typename ActivePolicyT::TopKPolicyT;

    using KeyInT   = cub::detail::value_t<KeyInputIteratorT>;
    using ValueInT = cub::detail::value_t<ValueInputIteratorT>;

    cudaError error = cudaSuccess;

    constexpr int block_threads    = Policy::BLOCK_THREADS; // Threads per block
    constexpr int items_per_thread = Policy::ITEMS_PER_THREAD; // Items per thread
    constexpr int tile_size        = block_threads * items_per_thread; // Items per block
    int num_tiles                  = static_cast<int>(::cuda::ceil_div(num_items, tile_size)); // Num of blocks

    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Specify temporary storage allocation requirements
      size_t size_counter        = sizeof(Counter<KeyInT, NumItemsT>);
      size_t size_histogram      = Policy::NUM_BUCKETS * sizeof(NumItemsT);
      size_t num_candidates      = Policy::COFFICIENT_FOR_BUFFER;
      size_t allocation_sizes[6] = {
        size_counter,
        size_histogram,
        num_candidates * sizeof(KeyInT),
        num_candidates * sizeof(ValueInT),
        num_candidates * sizeof(KeyInT),
        num_candidates * sizeof(ValueInT)};

      // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
      void* allocations[6] = {};

      error = CubDebug(cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      // Init the buffer for descriptor and histogram
      error = CubDebug(cudaMemsetAsync(
        allocations[0], 0, static_cast<char*>(allocations[2]) - static_cast<char*>(allocations[0]), stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get grid size for scanning tiles
      int max_dim_x;
      error = CubDebug(cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal));

      dim3 topk_grid_size;
      topk_grid_size.z = 1;
      topk_grid_size.y = ::cuda::ceil_div(num_tiles, max_dim_x);
      topk_grid_size.x = CUB_MIN(num_tiles, max_dim_x);

      for (int pass = 0; pass < Policy::NUM_PASSES; pass++)
      {
// Log select_if_kernel configuration @todo check the kernel launch
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        {
          // Get SM occupancy for select_if_kernel
          int range_select_sm_occupancy;
          error = CubDebug(MaxSmOccupancy(range_select_sm_occupancy, // out
                                          select_if_kernel,
                                          block_threads));
          if (cudaSuccess != error)
          {
            break;
          }

          _CubLog("Invoking select_if_kernel<<<{%d,%d,%d}, %d, 0, "
                  "%lld>>>(), %d items per thread, %d SM occupancy\n",
                  scan_grid_size.x,
                  scan_grid_size.y,
                  scan_grid_size.z,
                  block_threads,
                  (long long) stream,
                  items_per_thread,
                  range_select_sm_occupancy);
        }
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG
      }
      // set address for
    } while (0);
    return error;
  }

  /*
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input data of key data to find top K
   *
   * @param[out] d_keys_out
   *   Pointer to the K output sequence of key data
   *
   * @param[in] d_values_in
   *   Pointer to the corresponding input sequence of associated value items
   *
   * @param[out] d_values_out
   *   Pointer to the correspondingly output sequence of associated
   *   value items
   *
   * @param[in] num_items
   *   Number of items to be processed
   *
   * @param[in] k
   *   The K value. Will find K elements from num_items elements
   *
   * @param[in] stream
   *   @rst
   *   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
   *   @endrst
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumItemsT k,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;

    int ptx_version = 0;
    if (cudaError_t error = CubDebug(cub::PtxVersion(ptx_version)))
    {
      return error;
    }

    DispatchTopK dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      k,
      stream,
      ptx_version);

    return CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
  }
};

CUB_NAMESPACE_END
