// Adapter B6_radix: full block radix sort and truncation instead of selection (approach B6).
// Every item becomes one unsigned word that orders like (key descending, index ascending):
//   bf16: 32-bit word, twiddled 16-bit key in the high half, the source index in the low half.
//   fp32: 64-bit word, twiddled 32-bit key in the high half, the source index in the low half.
// cub::BlockRadixSort sorts the words descending, keys only, over the key bits alone (begin_bit
// is the first key bit): the LSD sort is stable and the input is in index order, so equal keys
// come out in ascending index order without touching the index bits (4 passes of 4 bits for
// bf16, 8 for fp32, with the stock 4-bit digit and the stock BlockRadixRank). The last pass
// leaves the words striped, so rank r sits in item r / Threads of thread r mod Threads and the
// top k are stored to the k output slots with one predicated conflict-free store per item.
//
// Contract:
//   scope block, 128, 256 or 512 threads, blocked input in registers, N = Threads x IPT with
//   IPT >= 1 and N <= 32768 (BlockRadixRank's 16-bit counters), k in [1, N), keys bf16 or fp32,
//   requirements not_guaranteed and prefer_smaller_index (the same code, the order is inherent).
//   Output: k indices in the harness output array, sorted by key descending then index ascending
//   (slot r holds rank r). Temp storage: BlockRadixSort's (ranking counters or the exchange
//   buffer of N words, whichever is larger). Sync: __syncthreads inside, about five per pass, the
//   caller separates consecutive calls by a block barrier (the harness does). The input keys are
//   left intact. NaN: twiddled bitwise, a positive NaN orders above +inf and a negative NaN below
//   -inf, never produced by focus-set inputs. -0.0 is normalized to +0.0 before twiddling so the
//   two zeros form one tie class ordered by index.
#pragma once

#include <cub/block/block_radix_sort.cuh>

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

#include "../topk_harness.cuh"

namespace b6
{

// Order preserving twiddle of a bf16 key with -0.0 normalized to +0.0: ascending unsigned order
// is ascending key order.
__device__ __forceinline__ uint32_t twiddle_bf16(__nv_bfloat16 h)
{
  uint16_t b;
  memcpy(&b, &h, sizeof(b));
  const uint32_t u = (b == 0x8000u) ? 0u : static_cast<uint32_t>(b);
  return u ^ 0x8000u ^ ((u >> 15) * 0x7fffu);
}

// Order preserving twiddle of an fp32 key with -0.0 normalized to +0.0 (x + 0.0 leaves every
// other value unchanged, including denormals, and turns -0.0 into +0.0 under round to nearest).
__device__ __forceinline__ uint32_t twiddle_f32(float f)
{
  const uint32_t u = __float_as_uint(__fadd_rn(f, 0.0f));
  return u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
}

template <typename KeyT>
struct word;

template <>
struct word<__nv_bfloat16>
{
  using type                    = uint32_t;
  static constexpr int key_bits = 16;
  __device__ __forceinline__ static type make(__nv_bfloat16 k, int idx)
  {
    return (twiddle_bf16(k) << 16) | static_cast<uint32_t>(idx);
  }
};

template <>
struct word<float>
{
  using type                    = uint64_t;
  static constexpr int key_bits = 32;
  __device__ __forceinline__ static type make(float k, int idx)
  {
    return (static_cast<uint64_t>(twiddle_f32(k)) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(idx));
  }
};

// Store the top K ranks of a striped tile of sorted words: item i of thread t is rank
// i x Threads + t. Items whose whole slice lies beyond K are dropped at compile time.
template <int Threads, int IPT, int K, typename WordT>
__device__ __forceinline__ void store_top_k_striped(const WordT (&w)[IPT], WordT index_mask, int* out_idx)
{
  const int t = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    if constexpr ((i + 1) * Threads <= K)
    {
      out_idx[i * Threads + t] = static_cast<int>(w[i] & index_mask);
    }
    else if constexpr (i * Threads < K)
    {
      if (i * Threads + t < K)
      {
        out_idx[i * Threads + t] = static_cast<int>(w[i] & index_mask);
      }
    }
  }
}

} // namespace b6

struct B6_radix_info
{
  static constexpr const char* id       = "B6_radix";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "cub::BlockRadixSort descending on the packed key-index word over the key bits only, blocked registers in, "
    "sorted top-k indices to smem out, both unsorted requirements at the same cost";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (n <= 32768) && (k >= 1) && (k < n);
    const bool req_ok = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B6_radix
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[IPT];

  using wt                         = b6::word<KeyT>;
  using word_t                     = typename wt::type;
  static constexpr int word_bits   = 8 * static_cast<int>(sizeof(word_t));
  static constexpr int begin_bit   = word_bits - wt::key_bits;
  static constexpr word_t idx_mask = (static_cast<word_t>(1) << begin_bit) - 1;

  using sort_t      = cub::BlockRadixSort<word_t, Threads, IPT>;
  using TempStorage = typename sort_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[IPT], TempStorage& ts, int* out_idx, KeyT*)
  {
    word_t w[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      w[i] = wt::make(keys[i], idx[i]);
    }
    sort_t(ts).SortDescendingBlockedToStriped(w, begin_bit, word_bits);
    b6::store_top_k_striped<Threads, IPT, K>(w, idx_mask, out_idx);
  }
};
