// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Unit tests for the top-k-private foundation building blocks in
//! `cub/detail/topk/tile_data_source.cuh`:
//!   - storage helpers (`phase_union`, `phase_aggregate`, `at<I>`)
//!   - reserve callbacks (`atomic_reserve_range_op`, `back_grow_capped_reserve_op`)
//!   - generative-iterator trait (`is_generative_iterator`, `is_generative_iterator_v`)
//!   - sync `TileDataSource` specializations (direct / sync_block_load / multi_source).
//! Async `async_to_shared_data_source` cases are gated on SM90 and live in their own
//! test cases below.

#include <cub/detail/topk/tile_data_source.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <algorithm>
#include <vector>

#include <c2h/catch2_test_helper.h>

namespace tds = cub::detail::topk;
namespace cd  = cub::detail;

//---------------------------------------------------------------------
// 1. Storage helpers
//---------------------------------------------------------------------

// File-static (rather than anonymous-namespace) helpers: nvcc generates
// `__device_stub__` symbols for __global__ kernels that conflict with anonymous
// namespaces in some configurations.
struct A8
{
  cuda::std::uint64_t v;
};
struct A16
{
  cuda::std::uint64_t a;
  cuda::std::uint64_t b;
};
struct A32
{
  cuda::std::uint64_t x[4];
};

// Compile-time invariants: phase_union sizes to max-of-tenants and phase_aggregate sizes
// to at-least sum-of-tenants (+ alignment slack).
static_assert(sizeof(cd::phase_union<cuda::std::tuple<A8, A16>>) >= sizeof(A16),
              "phase_union must be at least max(sizeof(Ts)...)");
static_assert(sizeof(cd::phase_union<cuda::std::tuple<A8, A16>>) <= sizeof(A16) + alignof(A16),
              "phase_union must not exceed max(sizeof(Ts)...) + alignment slack");
static_assert(sizeof(cd::phase_union<cuda::std::tuple<A16, A8, A32>>) >= sizeof(A32),
              "phase_union must be at least max(sizeof(Ts)...) over arbitrary order");
static_assert(sizeof(cd::phase_aggregate<cuda::std::tuple<A8, A16>>) >= sizeof(A8) + sizeof(A16),
              "phase_aggregate must be at least the sum of its tenants");

struct U_t
{
  cuda::std::int32_t a;
  cuda::std::int32_t b;
};
struct V_t
{
  cuda::std::int64_t v;
};

__global__ void phase_union_kernel(cuda::std::int32_t* out_a, cuda::std::int64_t* out_v)
{
  __shared__ cd::phase_union<cuda::std::tuple<U_t, V_t>> arena;

  if (threadIdx.x == 0)
  {
    auto& u  = cd::at<0>(arena);
    u.a      = 7;
    u.b      = 11;
    out_a[0] = u.a;
    out_a[1] = u.b;

    // Sequential phase: write to slot 1 after slot 0 is fully read out (we ourselves
    // bracket via the 'reads happened above' ordering on a single thread).
    auto& v  = cd::at<1>(arena);
    v.v      = 0x12345678abcdef01LL;
    out_v[0] = v.v;
  }
}

C2H_TEST("topk phase_union allows typed at<I> access in two distinct phases", "[block][topk][foundation]")
{
  thrust::device_vector<cuda::std::int32_t> out_a(2, 0);
  thrust::device_vector<cuda::std::int64_t> out_v(1, 0);

  phase_union_kernel<<<1, 32>>>(thrust::raw_pointer_cast(out_a.data()), thrust::raw_pointer_cast(out_v.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(out_a[0] == 7);
  REQUIRE(out_a[1] == 11);
  REQUIRE(out_v[0] == 0x12345678abcdef01LL);
}

__global__ void phase_aggregate_kernel(cuda::std::int32_t* out_uba, cuda::std::int64_t* out_v)
{
  __shared__ cd::phase_aggregate<cuda::std::tuple<U_t, V_t>> arena;

  if (threadIdx.x == 0)
  {
    auto& u    = cd::at<0>(arena);
    auto& v    = cd::at<1>(arena);
    u.a        = 1;
    u.b        = 2;
    v.v        = 3;
    out_uba[0] = u.a;
    out_uba[1] = u.b;
    out_v[0]   = v.v;
  }
}

C2H_TEST("topk phase_aggregate keeps coexisting tenants distinct", "[block][topk][foundation]")
{
  thrust::device_vector<cuda::std::int32_t> out_uba(2, 0);
  thrust::device_vector<cuda::std::int64_t> out_v(1, 0);

  phase_aggregate_kernel<<<1, 32>>>(thrust::raw_pointer_cast(out_uba.data()), thrust::raw_pointer_cast(out_v.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(out_uba[0] == 1);
  REQUIRE(out_uba[1] == 2);
  REQUIRE(out_v[0] == 3);
}

//---------------------------------------------------------------------
// 2. Reserve callbacks
//---------------------------------------------------------------------

static_assert(tds::atomic_reserve_range_op<unsigned>::may_grant_less == false,
              "atomic_reserve_range_op grants exactly n");
static_assert(tds::back_grow_capped_reserve_op<unsigned>::may_grant_less == true,
              "back_grow_capped_reserve_op may grant less");

template <typename Op>
__global__ void
run_reserve_kernel(Op op, const unsigned* d_requests, int num_requests, unsigned* d_bases, unsigned* d_grants)
{
  // Single-threaded: walk the request sequence sequentially and record (base, granted).
  // The op already does the atomicAdd against its `counter` member; sequencing on
  // thread 0 gives a deterministic order that the host can compare against.
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int i = 0; i < num_requests; ++i)
    {
      const auto r = op(d_requests[i]);
      d_bases[i]   = r.first;
      d_grants[i]  = r.second;
    }
  }
}

C2H_TEST("topk atomic_reserve_range_op returns (prev, n)", "[block][topk][foundation]")
{
  std::vector<unsigned> req = {3u, 5u, 1u};
  thrust::device_vector<unsigned> d_req(req.begin(), req.end());
  thrust::device_vector<unsigned> d_counter(1, 0);
  thrust::device_vector<unsigned> d_bases(req.size(), 0);
  thrust::device_vector<unsigned> d_grants(req.size(), 0);

  tds::atomic_reserve_range_op<unsigned> op{thrust::raw_pointer_cast(d_counter.data())};

  run_reserve_kernel<<<1, 32>>>(
    op,
    thrust::raw_pointer_cast(d_req.data()),
    static_cast<int>(req.size()),
    thrust::raw_pointer_cast(d_bases.data()),
    thrust::raw_pointer_cast(d_grants.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  std::vector<unsigned> bases(d_bases.begin(), d_bases.end());
  std::vector<unsigned> grants(d_grants.begin(), d_grants.end());
  REQUIRE(bases == std::vector<unsigned>{0u, 3u, 8u});
  REQUIRE(grants == std::vector<unsigned>{3u, 5u, 1u});
  REQUIRE(d_counter[0] == 9u);
}

C2H_TEST("topk back_grow_capped_reserve_op clamps grants and stacks bases backwards", "[block][topk][foundation]")
{
  // Cap at 7 items, back_anchor at 100. Three requests of 4, 4, 4:
  //   req 0: prev=0, writable=7,  granted=min(4,7)=4, base=100-0-4=96.
  //   req 1: prev=4, writable=3,  granted=min(4,3)=3, base=100-4-3=93.
  //   req 2: prev=8, writable=0,  granted=0,          base=100-8-0=92.
  // Counter is bumped by the unclamped n on every call, so the final value is 4+4+4=12.
  std::vector<unsigned> req = {4u, 4u, 4u};
  thrust::device_vector<unsigned> d_req(req.begin(), req.end());
  thrust::device_vector<unsigned> d_counter(1, 0);
  thrust::device_vector<unsigned> d_bases(req.size(), 0);
  thrust::device_vector<unsigned> d_grants(req.size(), 0);

  tds::back_grow_capped_reserve_op<unsigned> op{
    thrust::raw_pointer_cast(d_counter.data()), /*back_anchor=*/100u, /*cap=*/7u};

  run_reserve_kernel<<<1, 32>>>(
    op,
    thrust::raw_pointer_cast(d_req.data()),
    static_cast<int>(req.size()),
    thrust::raw_pointer_cast(d_bases.data()),
    thrust::raw_pointer_cast(d_grants.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  std::vector<unsigned> bases(d_bases.begin(), d_bases.end());
  std::vector<unsigned> grants(d_grants.begin(), d_grants.end());
  REQUIRE(bases == std::vector<unsigned>{96u, 93u, 92u});
  REQUIRE(grants == std::vector<unsigned>{4u, 3u, 0u});
  REQUIRE(d_counter[0] == 12u);
}

C2H_TEST("topk back_grow_capped_reserve_op handles cap=0 and exact-fit", "[block][topk][foundation]")
{
  {
    // cap=0: every grant is 0; bases follow `back_anchor - prev` (with `prev` growing
    // on every call by the unclamped n). Bases don't matter when granted=0 (nothing is
    // written), but we lock the formula so accidental sign / arithmetic changes get
    // caught.
    std::vector<unsigned> req = {1u, 2u, 3u};
    thrust::device_vector<unsigned> d_req(req.begin(), req.end());
    thrust::device_vector<unsigned> d_counter(1, 0);
    thrust::device_vector<unsigned> d_bases(req.size(), 99);
    thrust::device_vector<unsigned> d_grants(req.size(), 99);

    tds::back_grow_capped_reserve_op<unsigned> op{
      thrust::raw_pointer_cast(d_counter.data()), /*back_anchor=*/50u, /*cap=*/0u};
    run_reserve_kernel<<<1, 32>>>(
      op,
      thrust::raw_pointer_cast(d_req.data()),
      static_cast<int>(req.size()),
      thrust::raw_pointer_cast(d_bases.data()),
      thrust::raw_pointer_cast(d_grants.data()));
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    std::vector<unsigned> bases(d_bases.begin(), d_bases.end());
    std::vector<unsigned> grants(d_grants.begin(), d_grants.end());
    // base[i] = back_anchor - prev_i - granted_i = 50 - prev_i - 0.
    REQUIRE(bases == std::vector<unsigned>{50u, 49u, 47u});
    REQUIRE(grants == std::vector<unsigned>{0u, 0u, 0u});
    REQUIRE(d_counter[0] == 6u);
  }

  {
    // exact-fit: cap == sum-of-requests, every grant equals its request.
    std::vector<unsigned> req = {2u, 3u, 5u};
    thrust::device_vector<unsigned> d_req(req.begin(), req.end());
    thrust::device_vector<unsigned> d_counter(1, 0);
    thrust::device_vector<unsigned> d_bases(req.size(), 0);
    thrust::device_vector<unsigned> d_grants(req.size(), 0);

    tds::back_grow_capped_reserve_op<unsigned> op{
      thrust::raw_pointer_cast(d_counter.data()), /*back_anchor=*/100u, /*cap=*/10u};
    run_reserve_kernel<<<1, 32>>>(
      op,
      thrust::raw_pointer_cast(d_req.data()),
      static_cast<int>(req.size()),
      thrust::raw_pointer_cast(d_bases.data()),
      thrust::raw_pointer_cast(d_grants.data()));
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    std::vector<unsigned> bases(d_bases.begin(), d_bases.end());
    std::vector<unsigned> grants(d_grants.begin(), d_grants.end());
    REQUIRE(bases == std::vector<unsigned>{98u, 95u, 90u});
    REQUIRE(grants == std::vector<unsigned>{2u, 3u, 5u});
  }
}

//---------------------------------------------------------------------
// 3. Generative-iterator trait + factory selection
//---------------------------------------------------------------------

static_assert(cd::is_generative_iterator_v<cuda::counting_iterator<int>>,
              "counting_iterator must be detected as generative");
static_assert(cd::is_generative_iterator_v<cuda::counting_iterator<long long>>,
              "counting_iterator<long long> must be detected as generative");
static_assert(cd::is_generative_iterator_v<const cuda::counting_iterator<int>>,
              "const-qualified counting_iterator must be detected as generative");
static_assert(!cd::is_generative_iterator_v<int*>, "raw pointer is not generative");
static_assert(!cd::is_generative_iterator_v<const int*>, "raw const pointer is not generative");

// Factory must downgrade counting_iterator to direct_data_source regardless of the
// requested tile_load_kind.
static_assert(
  cuda::std::is_same_v<
    tds::tile_data_source_t<cuda::counting_iterator<int>, tds::tile_load_kind::block_load_vectorize, 128, 4>,
    tds::direct_data_source<cuda::counting_iterator<int>, 128, 4>>,
  "factory must downgrade counting_iterator to direct_data_source");

static_assert(
  cuda::std::is_same_v<
    tds::tile_data_source_t<cuda::counting_iterator<int>, tds::tile_load_kind::block_load_to_shared_async, 128, 4>,
    tds::direct_data_source<cuda::counting_iterator<int>, 128, 4>>,
  "factory must downgrade counting_iterator to direct_data_source even when "
  "the requested kind is async TMA");

// For a raw pointer the factory must honor the configured kind.
static_assert(cuda::std::is_same_v<tds::tile_data_source_t<int*, tds::tile_load_kind::direct, 128, 4>,
                                   tds::direct_data_source<int*, 128, 4>>,
              "factory honors `direct` for raw pointers");

static_assert(cuda::std::is_same_v<tds::tile_data_source_t<int*, tds::tile_load_kind::block_load_vectorize, 128, 4>,
                                   tds::sync_block_load_data_source<int*, 128, 4, cub::BLOCK_LOAD_VECTORIZE>>,
              "factory honors `block_load_vectorize` for raw pointers");

//---------------------------------------------------------------------
// 4. TileDataSource specializations (sync)
//---------------------------------------------------------------------

template <typename ValueT, int BlockThreads, int ItemsPerThread, typename OffsetT>
__global__ void direct_kernel(const ValueT* in, ValueT* out, OffsetT tile_base, OffsetT num_items, bool is_full)
{
  using ds_t = tds::direct_data_source<const ValueT*, BlockThreads, ItemsPerThread, OffsetT>;
  __shared__ typename ds_t::TempStorage state;
  __shared__ typename ds_t::ScratchStorage scratch;

  ds_t ds{in, state};
  ds.set_tile_base(tile_base);

  ValueT items[ItemsPerThread]{};
  if (is_full)
  {
    auto h = ds.submit_load(scratch);
    h.complete_load(items);
  }
  else
  {
    auto h = ds.submit_load(scratch, num_items);
    h.complete_load(items);
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int j = 0; j < ItemsPerThread; ++j)
  {
    out[static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + j] = items[j];
  }
}

template <typename ValueT, int BlockThreads, int ItemsPerThread, typename OffsetT>
__global__ void sync_block_load_kernel(const ValueT* in, ValueT* out, OffsetT tile_base, OffsetT num_items, bool is_full)
{
  using ds_t =
    tds::sync_block_load_data_source<const ValueT*, BlockThreads, ItemsPerThread, cub::BLOCK_LOAD_DIRECT, OffsetT>;
  __shared__ typename ds_t::TempStorage state;
  __shared__ typename ds_t::ScratchStorage scratch;

  ds_t ds{in, state};
  ds.set_tile_base(tile_base);

  ValueT items[ItemsPerThread]{};
  if (is_full)
  {
    auto h = ds.submit_load(scratch);
    h.complete_load(items);
  }
  else
  {
    auto h = ds.submit_load(scratch, num_items);
    h.complete_load(items);
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int j = 0; j < ItemsPerThread; ++j)
  {
    out[static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + j] = items[j];
  }
}

template <typename ValueT, int BlockThreads, int ItemsPerThread, typename OffsetT>
__global__ void multi_source_kernel(
  const ValueT* in_a, const ValueT* in_b, ValueT* out, OffsetT tile_base, OffsetT num_items, bool is_full, bool pick_b)
{
  using src_a_t = tds::direct_data_source<const ValueT*, BlockThreads, ItemsPerThread, OffsetT>;
  using src_b_t =
    tds::sync_block_load_data_source<const ValueT*, BlockThreads, ItemsPerThread, cub::BLOCK_LOAD_DIRECT, OffsetT>;
  using ds_t = tds::multi_source_data_source<src_a_t, src_b_t, OffsetT>;

  __shared__ typename ds_t::TempStorage state;
  __shared__ typename ds_t::ScratchStorage scratch;

  src_a_t a{in_a, state.a};
  src_b_t b{in_b, state.b};
  ds_t ds{a, b, pick_b};
  ds.set_tile_base(tile_base);

  ValueT items[ItemsPerThread]{};
  if (is_full)
  {
    auto h = ds.submit_load(scratch);
    h.complete_load(items);
  }
  else
  {
    auto h = ds.submit_load(scratch, num_items);
    h.complete_load(items);
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int j = 0; j < ItemsPerThread; ++j)
  {
    out[static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + j] = items[j];
  }
}

template <typename ValueT>
std::vector<ValueT> make_input(int n_total)
{
  std::vector<ValueT> v(n_total);
  for (int i = 0; i < n_total; ++i)
  {
    v[i] = static_cast<ValueT>(i * 7 + 3);
  }
  return v;
}

template <typename ValueT, int BlockThreads, int ItemsPerThread>
std::vector<ValueT> build_expected_blocked(const std::vector<ValueT>& input, int tile_base, int num_valid)
{
  constexpr int tile_items = BlockThreads * ItemsPerThread;
  std::vector<ValueT> expected(tile_items, ValueT{});
  for (int t = 0; t < BlockThreads; ++t)
  {
    for (int j = 0; j < ItemsPerThread; ++j)
    {
      const int idx = t * ItemsPerThread + j;
      if (idx < num_valid)
      {
        expected[idx] = input[tile_base + idx];
      }
    }
  }
  return expected;
}

enum class sync_kind
{
  direct,
  block_load
};

template <typename ValueT, int BlockThreads, int ItemsPerThread>
void sweep_sync(sync_kind kind, int num_items)
{
  using OffsetT            = cuda::std::int64_t;
  constexpr int tile_items = BlockThreads * ItemsPerThread;

  auto h_input = make_input<ValueT>(num_items);
  thrust::device_vector<ValueT> d_input(h_input.begin(), h_input.end());

  // Walk every tile (full or partial). The data source's contract requires the agent
  // to call set_tile_base + the matching submit overload for each tile; we exercise
  // both overloads by re-launching with the per-tile `is_full` and `num_valid`.
  const int num_tiles = (num_items + tile_items - 1) / tile_items;
  for (int t = 0; t < num_tiles; ++t)
  {
    const int tile_base = t * tile_items;
    const int num_valid = std::min(tile_items, num_items - tile_base);
    const bool is_full  = (num_valid == tile_items);

    thrust::device_vector<ValueT> d_out(tile_items, ValueT{});
    if (kind == sync_kind::direct)
    {
      direct_kernel<ValueT, BlockThreads, ItemsPerThread, OffsetT><<<1, BlockThreads>>>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_out.data()),
        static_cast<OffsetT>(tile_base),
        static_cast<OffsetT>(num_valid),
        is_full);
    }
    else
    {
      sync_block_load_kernel<ValueT, BlockThreads, ItemsPerThread, OffsetT><<<1, BlockThreads>>>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_out.data()),
        static_cast<OffsetT>(tile_base),
        static_cast<OffsetT>(num_valid),
        is_full);
    }
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<ValueT> got(d_out);
    auto expected = build_expected_blocked<ValueT, BlockThreads, ItemsPerThread>(h_input, tile_base, num_valid);
    REQUIRE(std::equal(got.begin(), got.end(), expected.begin()));
  }
}

C2H_TEST("topk direct_data_source delivers BLOCKED items across full and partial tiles", "[block][topk][foundation]")
{
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;

  for (int n : {tile_items, tile_items * 3, tile_items * 2 + 17, 7})
  {
    sweep_sync<int, BlockThreads, ItemsPerThread>(sync_kind::direct, n);
  }
}

C2H_TEST("topk sync_block_load_data_source delivers BLOCKED items across full and partial tiles",
         "[block][topk][foundation]")
{
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;

  for (int n : {tile_items, tile_items * 3, tile_items * 2 + 17, 7})
  {
    sweep_sync<int, BlockThreads, ItemsPerThread>(sync_kind::block_load, n);
  }
}

C2H_TEST("topk multi_source_data_source forwards to the active source on both branches", "[block][topk][foundation]")
{
  using OffsetT                = cuda::std::int64_t;
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;

  for (int num_items : {tile_items, tile_items * 2 + 13})
  {
    auto h_a = make_input<int>(num_items);
    // Distinct sequence so the two branches' outputs never accidentally coincide.
    std::vector<int> h_b(num_items);
    for (int i = 0; i < num_items; ++i)
    {
      h_b[i] = -1 - i;
    }
    thrust::device_vector<int> d_a(h_a.begin(), h_a.end());
    thrust::device_vector<int> d_b(h_b.begin(), h_b.end());

    const int num_tiles = (num_items + tile_items - 1) / tile_items;
    for (int t = 0; t < num_tiles; ++t)
    {
      const int tile_base = t * tile_items;
      const int num_valid = std::min(tile_items, num_items - tile_base);
      const bool is_full  = (num_valid == tile_items);

      for (bool pick_b : {false, true})
      {
        thrust::device_vector<int> d_out(tile_items, 0);
        multi_source_kernel<int, BlockThreads, ItemsPerThread, OffsetT><<<1, BlockThreads>>>(
          thrust::raw_pointer_cast(d_a.data()),
          thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_out.data()),
          static_cast<OffsetT>(tile_base),
          static_cast<OffsetT>(num_valid),
          is_full,
          pick_b);
        REQUIRE(cudaSuccess == cudaPeekAtLastError());
        REQUIRE(cudaSuccess == cudaDeviceSynchronize());

        thrust::host_vector<int> got(d_out);
        auto expected =
          build_expected_blocked<int, BlockThreads, ItemsPerThread>(pick_b ? h_b : h_a, tile_base, num_valid);
        REQUIRE(std::equal(got.begin(), got.end(), expected.begin()));
      }
    }
  }
}

template <int BlockThreads, int ItemsPerThread, typename OffsetT>
__global__ void counting_iterator_factory_kernel(int* out, OffsetT tile_base)
{
  using counting_t = cuda::counting_iterator<int>;
  using ds_t =
    tds::tile_data_source_t<counting_t, tds::tile_load_kind::block_load_vectorize, BlockThreads, ItemsPerThread, OffsetT>;
  __shared__ typename ds_t::TempStorage state;
  __shared__ typename ds_t::ScratchStorage scratch;

  auto ds = tds::
    make_tile_data_source<counting_t, tds::tile_load_kind::block_load_vectorize, BlockThreads, ItemsPerThread, OffsetT>(
      counting_t{0}, state);
  ds.set_tile_base(tile_base);

  auto h = ds.submit_load(scratch);
  int items[ItemsPerThread]{};
  h.complete_load(items);

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int j = 0; j < ItemsPerThread; ++j)
  {
    out[static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + j] = items[j];
  }
}

C2H_TEST("topk make_tile_data_source over counting_iterator yields per-thread arithmetic", "[block][topk][foundation]")
{
  using OffsetT                = cuda::std::int64_t;
  constexpr int BlockThreads   = 32;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;
  constexpr int tile_base      = tile_items * 7;

  thrust::device_vector<int> d_out(tile_items, -1);
  counting_iterator_factory_kernel<BlockThreads, ItemsPerThread, OffsetT>
    <<<1, BlockThreads>>>(thrust::raw_pointer_cast(d_out.data()), static_cast<OffsetT>(tile_base));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  thrust::host_vector<int> got(d_out);
  std::vector<int> expected(tile_items);
  for (int i = 0; i < tile_items; ++i)
  {
    expected[i] = tile_base + i;
  }
  REQUIRE(std::equal(got.begin(), got.end(), expected.begin()));
}

//---------------------------------------------------------------------
// 5. async_to_shared_data_source (TMA / cp.async / fallback). Gated on SM90 because
//    the TMA path requires Hopper+. On older arches the underlying primitive falls
//    back to scalar copy and the data source is still functionally correct, but the
//    architecture document only commits to the TMA path; we follow the convention in
//    `catch2_test_block_load_to_shared.cu` and skip on < SM90.
//---------------------------------------------------------------------

template <typename ValueT, int BlockThreads, int ItemsPerThread, typename OffsetT>
__global__ void async_to_shared_kernel(const ValueT* in, ValueT* out, OffsetT tile_base, OffsetT num_items, bool is_full)
{
  using ds_t = tds::async_to_shared_data_source<const ValueT*, BlockThreads, ItemsPerThread, alignof(ValueT), OffsetT>;
  __shared__ typename ds_t::TempStorage state;
  __shared__ typename ds_t::ScratchStorage scratch;

  ds_t ds{in, state};
  ds.set_tile_base(tile_base);

  ValueT items[ItemsPerThread]{};
  __syncthreads();
  if (is_full)
  {
    auto h = ds.submit_load(scratch);
    h.complete_load(items);
  }
  else
  {
    auto h = ds.submit_load(scratch, num_items);
    h.complete_load(items);
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int j = 0; j < ItemsPerThread; ++j)
  {
    out[static_cast<OffsetT>(threadIdx.x) * ItemsPerThread + j] = items[j];
  }

  __syncthreads();
  ds.invalidate();
}

template <typename ValueT, int BlockThreads, int ItemsPerThread>
void sweep_async_tma(int num_items)
{
  using OffsetT            = cuda::std::int64_t;
  constexpr int tile_items = BlockThreads * ItemsPerThread;

  auto h_input = make_input<ValueT>(num_items);
  thrust::device_vector<ValueT> d_input(h_input.begin(), h_input.end());

  const int num_tiles = (num_items + tile_items - 1) / tile_items;
  for (int t = 0; t < num_tiles; ++t)
  {
    const int tile_base = t * tile_items;
    const int num_valid = std::min(tile_items, num_items - tile_base);
    const bool is_full  = (num_valid == tile_items);

    thrust::device_vector<ValueT> d_out(tile_items, ValueT{});
    async_to_shared_kernel<ValueT, BlockThreads, ItemsPerThread, OffsetT><<<1, BlockThreads>>>(
      thrust::raw_pointer_cast(d_input.data()),
      thrust::raw_pointer_cast(d_out.data()),
      static_cast<OffsetT>(tile_base),
      static_cast<OffsetT>(num_valid),
      is_full);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    thrust::host_vector<ValueT> got(d_out);
    auto expected = build_expected_blocked<ValueT, BlockThreads, ItemsPerThread>(h_input, tile_base, num_valid);
    REQUIRE(std::equal(got.begin(), got.end(), expected.begin()));
  }
}

C2H_TEST("topk async_to_shared_data_source delivers BLOCKED items across full and partial tiles (SM90+)",
         "[block][topk][foundation]")
{
  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));
  cudaDeviceProp props{};
  REQUIRE(cudaSuccess == cudaGetDeviceProperties(&props, current_device));
  if (props.major < 9)
  {
    SUCCEED("Skipping async TMA TileDataSource test on pre-SM90 device.");
    return;
  }

  // BlockLoadToShared requires block_threads >= 2 * (bulk_copy_min_align - 1) = 30.
  // Use 64 threads x 4 IPT / 256 items per tile so the staging buffer comfortably fits
  // and the partial-tile path is meaningful.
  constexpr int BlockThreads   = 64;
  constexpr int ItemsPerThread = 4;
  constexpr int tile_items     = BlockThreads * ItemsPerThread;

  for (int n : {tile_items, tile_items * 3, tile_items * 2 + 17, 5})
  {
    sweep_async_tma<int, BlockThreads, ItemsPerThread>(n);
  }
}
