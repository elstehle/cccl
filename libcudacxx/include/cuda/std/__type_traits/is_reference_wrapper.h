//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/reference_wrapper.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cv.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_reference_wrapper_impl : public false_type
{};
template <class _Tp>
struct __is_reference_wrapper_impl<reference_wrapper<_Tp>> : public true_type
{};
template <class _Tp>
struct __is_reference_wrapper : public __is_reference_wrapper_impl<remove_cv_t<_Tp>>
{};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_ENABLE_IF_H
