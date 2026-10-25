// Adapter B6_pairs: the pairs form of B6. cub::BlockRadixSort sorts the twiddled key (16-bit for
// bf16, 32-bit for fp32) as the key and carries the source index as the value, descending,
// stable, over all key bits. Against B6_radix this measures what the separate value exchange
// per pass costs (one more barrier and one more buffer trip per pass) against the wider packed
// word (a 64-bit word for fp32, no extra exchange). Everything else, including the striped
// output and the tie order, is as in B6_radix.
//
// Contract: as B6_radix (block scope, 128 to 512 threads, blocked registers in, N <= 32768,
// k in [1, N), bf16 or fp32, both unsorted requirements, sorted top-k indices to smem out).
// NaN and -0.0 as in B6_radix.
#pragma once

#include <cub/block/block_radix_sort.cuh>

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

#include "../topk_harness.cuh"
#include "B6_radix.cuh"

namespace b6
{

template <typename KeyT>
struct pair_key;

template <>
struct pair_key<__nv_bfloat16>
{
  using type = uint16_t;
  __device__ __forceinline__ static type make(__nv_bfloat16 k)
  {
    return static_cast<uint16_t>(twiddle_bf16(k));
  }
};

template <>
struct pair_key<float>
{
  using type = uint32_t;
  __device__ __forceinline__ static type make(float k)
  {
    return twiddle_f32(k);
  }
};

} // namespace b6

struct B6_pairs_info
{
  static constexpr const char* id       = "B6_pairs";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "cub::BlockRadixSort descending pairs sort (twiddled key, index value), blocked registers in, sorted top-k "
    "indices to smem out, both unsorted requirements at the same cost";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return B6_radix_info::supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B6_pairs
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[IPT];

  using pk          = b6::pair_key<KeyT>;
  using skey_t      = typename pk::type;
  using sort_t      = cub::BlockRadixSort<skey_t, Threads, IPT, int>;
  using TempStorage = typename sort_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[IPT], TempStorage& ts, int* out_idx, KeyT*)
  {
    skey_t sk[IPT];
    int v[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      sk[i] = pk::make(keys[i]);
      v[i]  = idx[i];
    }
    sort_t(ts).SortDescendingBlockedToStriped(sk, v);
    const int t = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      if constexpr ((i + 1) * Threads <= K)
      {
        out_idx[i * Threads + t] = v[i];
      }
      else if constexpr (i * Threads < K)
      {
        if (i * Threads + t < K)
        {
          out_idx[i * Threads + t] = v[i];
        }
      }
    }
  }
};
