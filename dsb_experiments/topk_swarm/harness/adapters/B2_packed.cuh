// Adapter B2_packed: atomics based iterative extraction (approach B2) on packed key-index words.
// Every thread keeps its items as 32-bit words that order by (twiddled key, smaller index first),
// sorted descending in registers. A round posts every thread's head into one shared slot with a
// fire-and-forget atomicMax, one barrier, one read back, and a predicated shift of the winner's
// list. The k slots are the sorted answer and the indices are unpacked from them at the end.
//
// bf16: the word is (twiddled 16-bit key << 16) | (0xffff ^ index), exact, no tie machinery.
// fp32 (idea N4): the word keeps the top 32 - log2(N) bits of the twiddled key and the inverted
//   index in the low log2(N) bits. The rounds select the exact top-k by (truncated key, index). A
//   counting pass compares the size of the tie class of the k-th truncated key with the number of
//   its members among the slots. When they differ (rare on random data), the class members are
//   re-ranked by their remaining key bits with the same rounds (second-level packing), which makes
//   the result exact by (full key, index).
//
// Contract:
//   scope block, Threads 128, 256 or 512, blocked input in registers, N = Threads x IPT with IPT
//   a power of two, IPT <= 128 for bf16 and <= 64 for fp32, N <= 65536, k in [1, min(N/2, 256)],
//   keys bf16 or fp32, both requirements (prefer_smaller_index is inherent in the word order, the
//   same code runs for both). Output: k indices in the harness output array, sorted by key
//   descending then index ascending. Temp storage: k + 1 words. Sync: __syncthreads inside (1 init
//   plus k rounds, fp32 adds 1 for the tie count and 1 plus the re-ranked rounds on the tie path),
//   assumes the caller separates consecutive calls by a block barrier. NaN: twiddled bitwise, a
//   positive NaN ranks above +inf and a negative NaN below -inf, never in focus-set inputs. A
//   negative NaN with all mantissa bits set at the last index packs to the empty-slot word 0 and
//   is never selected. -0.0 is normalized to +0.0 before packing, so the two zeros tie by index.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "../topk_harness.cuh"

namespace b2
{

__host__ __device__ constexpr int log2_int(int n)
{
  int r = 0;
  while ((1 << r) < n)
  {
    ++r;
  }
  return r;
}

// Batcher odd-even merge sort as a compile-time table of compare-exchange pairs, so that the
// register indices are constants after the unrolled loop below (no local memory).
__host__ __device__ constexpr int oem_count(int n)
{
  int c = 0;
  for (int p = 1; p < n; p *= 2)
  {
    for (int k = p; k >= 1; k /= 2)
    {
      for (int j = k % p; j + k < n; j += 2 * k)
      {
        for (int i = 0; i < k; ++i)
        {
          if ((i + j) / (2 * p) == (i + j + k) / (2 * p))
          {
            ++c;
          }
        }
      }
    }
  }
  return c;
}

template <int N>
struct oem_network
{
  static constexpr int count = oem_count(N);
  int a[count > 0 ? count : 1];
  int b[count > 0 ? count : 1];
  __host__ __device__ constexpr oem_network()
      : a{}
      , b{}
  {
    int c = 0;
    for (int p = 1; p < N; p *= 2)
    {
      for (int k = p; k >= 1; k /= 2)
      {
        for (int j = k % p; j + k < N; j += 2 * k)
        {
          for (int i = 0; i < k; ++i)
          {
            if ((i + j) / (2 * p) == (i + j + k) / (2 * p))
            {
              a[c] = i + j;
              b[c] = i + j + k;
              ++c;
            }
          }
        }
      }
    }
  }
};

template <int N>
__device__ __forceinline__ void sort_desc(uint32_t (&v)[N])
{
  constexpr oem_network<N> net{};
#pragma unroll
  for (int c = 0; c < net.count; ++c)
  {
    const uint32_t x = v[net.a[c]];
    const uint32_t y = v[net.b[c]];
    v[net.a[c]]      = max(x, y);
    v[net.b[c]]      = min(x, y);
  }
}

// Order preserving twiddle of fp32 with -0.0 normalized to +0.0 (x + 0.0 leaves every other value
// unchanged, including denormals, and turns -0.0 into +0.0 under round to nearest).
__device__ __forceinline__ uint32_t twiddle32(float f)
{
  const uint32_t u = __float_as_uint(__fadd_rn(f, 0.0f));
  return u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
}

// Order preserving twiddle of two bf16 keys held in one word, -0.0 normalized the same way.
__device__ __forceinline__ uint32_t twiddle_bf16x2(uint32_t w)
{
  __nv_bfloat162 h;
  memcpy(&h, &w, sizeof(h));
  h = __hadd2(h, __float2bfloat162_rn(0.0f));
  memcpy(&w, &h, sizeof(w));
  const uint32_t s = (w >> 15) & 0x00010001u;
  return w ^ 0x80008000u ^ (s * 0x7fffu);
}

// One extraction round: post the head, barrier, read the round maximum, shift the winner's list.
template <int IPT>
__device__ __forceinline__ uint32_t round(uint32_t* slot, uint32_t (&p)[IPT])
{
  atomicMax(slot, p[0]);
  __syncthreads();
  const uint32_t M = *slot;
  const bool w     = (p[0] == M);
#pragma unroll
  for (int i = 0; i + 1 < IPT; ++i)
  {
    p[i] = w ? p[i + 1] : p[i];
  }
  p[IPT - 1] = w ? 0u : p[IPT - 1];
  return M;
}

} // namespace b2

struct B2_packed_info
{
  static constexpr const char* id       = "B2_packed";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "packed key-index words, k fire-and-forget atomicMax rounds, blocked registers in, sorted indices to smem out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const int ipt        = (threads > 0) ? n / threads : 0;
    const int ipt_max    = (kt == topk::key_type::fp32) ? 64 : 128;
    // the sort network and the shift list are written for a power of two items per thread, the
    // index must fit in 16 bits next to a 16-bit key, k rounds above 256 cost more than any radix
    // path (and the slot array is k words)
    const bool shape_ok = (n % threads == 0) && topk::is_pow2(ipt) && (ipt <= ipt_max) && (n <= 65536) && (k >= 1)
                       && (k <= 256) && (2 * k <= n);
    const bool req_ok = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B2_packed
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[IPT];
  static constexpr bool is_fp32                        = std::is_same_v<KeyT, float>;
  static constexpr int index_bits                      = b2::log2_int(N);
  static constexpr uint32_t index_mask                 = static_cast<uint32_t>(N - 1);
  static constexpr bool unroll_rounds                  = (K * IPT <= 4096);

  struct TempStorage
  {
    uint32_t slot[K];
    uint32_t cnt;
  };

  // fp32 tie path: re-rank the members of the tie class of the k-th truncated key by their
  // remaining key bits, with the same rounds, into slots j .. K-1.
  __device__ __forceinline__ static void
  finish(const uint32_t (&tw)[IPT], uint32_t tk, uint32_t invbase, int j, TempStorage& ts, int* out_idx)
  {
    const int tid = static_cast<int>(threadIdx.x);
    uint32_t q[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      const bool member = ((tw[i] & ~index_mask) == tk);
      q[i] = member ? ((tw[i] << (32 - index_bits)) | (invbase & ~static_cast<uint32_t>(i))) : 0u;
    }
    b2::sort_desc(q);
    for (int t = tid; t < K; t += Threads)
    {
      if (t >= j)
      {
        ts.slot[t] = 0;
      }
    }
    __syncthreads();
#pragma unroll 1
    for (int r = j; r < K; ++r)
    {
      b2::round(&ts.slot[r], q);
    }
    for (int t = tid; t < K; t += Threads)
    {
      if (t >= j)
      {
        out_idx[t] = static_cast<int>(index_mask & ~ts.slot[t]);
      }
    }
  }

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[IPT], TempStorage& ts, int* out_idx, KeyT*)
  {
    const int tid       = static_cast<int>(threadIdx.x);
    const uint32_t base = static_cast<uint32_t>(tid) * IPT;

    for (int t = tid; t < K; t += Threads)
    {
      ts.slot[t] = 0;
    }
    if constexpr (is_fp32)
    {
      if (tid == 0)
      {
        ts.cnt = 0;
      }
    }

    uint32_t p[IPT];
    uint32_t tw[is_fp32 ? IPT : 1];
    uint32_t invbase = 0;
    if constexpr (is_fp32)
    {
      invbase = index_mask & ~base;
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        tw[i] = b2::twiddle32(keys[i]);
        p[i]  = (tw[i] & ~index_mask) | (invbase & ~static_cast<uint32_t>(i));
      }
    }
    else
    {
      invbase = 0xffffu ^ base;
#pragma unroll
      for (int j = 0; j + 1 < IPT; j += 2)
      {
        uint32_t w;
        memcpy(&w, &keys[j], sizeof(w));
        const uint32_t t2 = b2::twiddle_bf16x2(w);
        p[j]              = (t2 << 16) | (invbase ^ static_cast<uint32_t>(j));
        p[j + 1]          = (t2 & 0xffff0000u) | (invbase ^ static_cast<uint32_t>(j + 1));
      }
      if constexpr (IPT % 2 == 1)
      {
        uint16_t h;
        memcpy(&h, &keys[IPT - 1], sizeof(h));
        const uint32_t t2 = b2::twiddle_bf16x2(static_cast<uint32_t>(h));
        p[IPT - 1]        = (t2 << 16) | (invbase ^ static_cast<uint32_t>(IPT - 1));
      }
    }
    b2::sort_desc(p);
    __syncthreads();

    uint32_t M = 0;
    if constexpr (unroll_rounds)
    {
#pragma unroll
      for (int r = 0; r < K; ++r)
      {
        M = b2::round(&ts.slot[r], p);
      }
    }
    else
    {
#pragma unroll 1
      for (int r = 0; r < K; ++r)
      {
        M = b2::round(&ts.slot[r], p);
      }
    }

    if constexpr (!is_fp32)
    {
      for (int t = tid; t < K; t += Threads)
      {
        out_idx[t] = static_cast<int>(0xffffu & ~ts.slot[t]);
      }
    }
    else
    {
      // tie class of the k-th truncated key: members among the slots against members anywhere
      const uint32_t tk = M & ~index_mask;
      uint32_t in       = 0;
      for (int t = tid; t < K; t += Threads)
      {
        const uint32_t s = ts.slot[t];
        out_idx[t]       = static_cast<int>(index_mask & ~s);
        in += ((s & ~index_mask) == tk) ? 1u : 0u;
      }
      uint32_t all = 0;
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        all += ((tw[i] & ~index_mask) == tk) ? 1u : 0u;
      }
      atomicAdd(&ts.cnt, all | (in << 20));
      __syncthreads();
      const uint32_t c = ts.cnt;
      if ((c & 0xfffffu) != (c >> 20))
      {
        finish(tw, tk, invbase, K - static_cast<int>(c >> 20), ts, out_idx);
      }
    }
  }
};
