//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__STREAM_STREAM_REF
#define _CUDAX__STREAM_STREAM_REF

#include <cuda/std/detail/__config>
#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__device/all_devices.cuh>
#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__event/timed_event.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

namespace cuda::experimental
{

namespace detail
{
// 0 is a valid stream in CUDA, so we need some other invalid stream representation
// Can't make it constexpr, because cudaStream_t is a pointer type
static const ::cudaStream_t __invalid_stream = reinterpret_cast<cudaStream_t>(~0ULL);
} // namespace detail

//! @brief A non-owning wrapper for cudaStream_t.
struct stream_ref : ::cuda::stream_ref
{
  stream_ref() = delete;

  //! @brief Wrap a native \c ::cudaStream_t in a \c stream_ref
  //!
  //! @post `this->get() == __stream`
  _CUDAX_HOST_API constexpr stream_ref(value_type __stream) noexcept
      : ::cuda::stream_ref{__stream}
  {}

  //! @brief Converting constructor from \c ::cuda::stream_ref
  //!
  //! @post `*this == __other`
  _CUDAX_HOST_API constexpr stream_ref(const ::cuda::stream_ref& __other) noexcept
      : ::cuda::stream_ref(__other)
  {}

  /// Disallow construction from an `int`, e.g., `0`.
  stream_ref(int) = delete;

  /// Disallow construction from `nullptr`.
  stream_ref(_CUDA_VSTD::nullptr_t) = delete;

  /**
   * \brief Queries if all operations on the stream have completed.
   *
   * \throws cuda::cuda_error if the query fails.
   *
   * \return `true` if all operations have completed, or `false` if not.
   */
  _CCCL_NODISCARD bool is_done() const
  {
    return ready();
  }

  //! @brief Create a new event and record it into this stream
  //!
  //! @return A new event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  _CCCL_NODISCARD _CUDAX_HOST_API event record_event(event::flags __flags = event::flags::none) const
  {
    return event(*this, __flags);
  }

  //! @brief Create a new timed event and record it into this stream
  //!
  //! @return A new timed event that was recorded into this stream
  //!
  //! @throws cuda_error if event creation or record failed
  _CCCL_NODISCARD _CUDAX_HOST_API timed_event record_timed_event(event::flags __flags = event::flags::none) const
  {
    return timed_event(*this, __flags);
  }

  //! \brief Synchronizes the wrapped stream.
  //!
  //! \throws cuda::cuda_error if synchronization fails.
  //!
  _CUDAX_TRIVIAL_HOST_API void wait() const
  {
    this->::cuda::stream_ref::wait();
  }

  //! @brief Make all future work submitted into this stream depend on completion of the specified event
  //!
  //! @param __ev Event that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  _CUDAX_HOST_API void wait(event_ref __ev) const
  {
    _CCCL_ASSERT(__ev.get() != nullptr, "cuda::experimental::stream_ref::wait invalid event passed");
    // Need to use driver API, cudaStreamWaitEvent would push dev 0 if stack was empty
    detail::driver::streamWaitEvent(get(), __ev.get());
  }

  //! @brief Make all future work submitted into this stream depend on completion of all work from the specified
  //! stream
  //!
  //! @param __other Stream that this stream should wait for
  //!
  //! @throws cuda_error if inserting the dependency fails
  _CUDAX_HOST_API void wait(stream_ref __other) const
  {
    // TODO consider an optimization to not create an event every time and instead have one persistent event or one
    // per stream
    _CCCL_ASSERT(__stream != detail::__invalid_stream, "cuda::experimental::stream_ref::wait invalid stream passed");
    if (*this != __other)
    {
      event __tmp(__other);
      wait(__tmp);
    }
  }

  //! @brief Get the logical device under which this stream was created.
  //!
  //! Compared to `device()` member function the returned \c logical_device will
  //! hold a green context for streams created under one.
  _CUDAX_HOST_API logical_device get_logical_device() const
  {
    CUcontext __stream_ctx;
    ::cuda::experimental::logical_device::kinds __ctx_kind = ::cuda::experimental::logical_device::kinds::device;
#if CUDART_VERSION >= 12050
    if (detail::driver::getVersion() >= 12050)
    {
      auto __ctx = detail::driver::streamGetCtx_v2(__stream);
      if (__ctx.__ctx_kind == detail::driver::__ctx_from_stream::__kind::__green)
      {
        __stream_ctx = detail::driver::ctxFromGreenCtx(__ctx.__ctx_ptr.__green);
        __ctx_kind   = ::cuda::experimental::logical_device::kinds::green_context;
      }
      else
      {
        __stream_ctx = __ctx.__ctx_ptr.__device;
        __ctx_kind   = ::cuda::experimental::logical_device::kinds::device;
      }
    }
    else
#endif // CUDART_VERSION >= 12050
    {
      __stream_ctx = detail::driver::streamGetCtx(__stream);
      __ctx_kind   = ::cuda::experimental::logical_device::kinds::device;
    }
    // Because the stream can come from_native_handle, we can't just loop over devices comparing contexts,
    // lower to CUDART for this instead
    __ensure_current_device __setter(__stream_ctx);
    int __id;
    _CCCL_TRY_CUDA_API(cudaGetDevice, "Could not get device from a stream", &__id);
    return __logical_device_access::make_logical_device(__id, __stream_ctx, __ctx_kind);
  }

  //! @brief Get device under which this stream was created.
  //!
  //! Note: In case of a stream created under a `green_context` the device on which that `green_context` was created is
  //! returned
  //!
  //! @throws cuda_error if device check fails
  _CUDAX_HOST_API device_ref get_device() const
  {
    return get_logical_device().get_underlying_device();
  }
};

} // namespace cuda::experimental

#endif // _CUDAX__STREAM_STREAM_REF
