/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/adl/transform.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/transform.h>
#include <thrust/transform.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename UnaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform");
  using thrust::system::detail::generic::transform;
  return transform(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, op);
} // end transform()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryFunction op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform");
  using thrust::system::detail::generic::transform;
  return transform(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, result, op);
} // end transform()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  ForwardIterator result,
  UnaryFunction op,
  Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_if");
  using thrust::system::detail::generic::transform_if;
  return transform_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, op, pred);
} // end transform_if()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  ForwardIterator result,
  UnaryFunction op,
  Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_if");
  using thrust::system::detail::generic::transform_if;
  return transform_if(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, stencil, result, op, pred);
} // end transform_if()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator transform_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator3 stencil,
  ForwardIterator result,
  BinaryFunction binary_op,
  Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_if");
  using thrust::system::detail::generic::transform_if;
  return transform_if(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    first1,
    last1,
    first2,
    stencil,
    result,
    binary_op,
    pred);
} // end transform_if()

template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
OutputIterator transform(InputIterator first, InputIterator last, OutputIterator result, UnaryFunction op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::transform(select_system(system1, system2), first, last, result, op);
} // end transform()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
OutputIterator
transform(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, BinaryFunction op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::transform(select_system(system1, system2, system3), first1, last1, first2, result, op);
} // end transform()

template <typename InputIterator, typename ForwardIterator, typename UnaryFunction, typename Predicate>
ForwardIterator
transform_if(InputIterator first, InputIterator last, ForwardIterator result, UnaryFunction unary_op, Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_if");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<ForwardIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::transform_if(select_system(system1, system2), first, last, result, unary_op, pred);
} // end transform_if()

template <typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
ForwardIterator transform_if(
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  ForwardIterator result,
  UnaryFunction unary_op,
  Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_if");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<ForwardIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::transform_if(select_system(system1, system2, system3), first, last, stencil, result, unary_op, pred);
} // end transform_if()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
ForwardIterator transform_if(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator3 stencil,
  ForwardIterator result,
  BinaryFunction binary_op,
  Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_if");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<ForwardIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::transform_if(
    select_system(system1, system2, system3, system4), first1, last1, first2, stencil, result, binary_op, pred);
} // end transform_if()

THRUST_NAMESPACE_END
