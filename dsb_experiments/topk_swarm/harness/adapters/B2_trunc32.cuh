// Adapter B2_trunc32: B2 with the N4 truncated-key packing for fp32 (idea card N4). The bf16 path
// is the one of B2_baseline. For fp32 the top 32 - log2(N) bits of the twiddled key and the
// inverted log2(N)-bit index are packed into one word, the k packed rounds select the exact top-k
// by (truncated key, smaller index first). The result equals the top-k by full key except inside
// the tie class of the k-th truncated key. One block-wide OR (__syncthreads_or) over "my largest
// remaining word is in that class" decides whether a finish is needed, which on random fp32 input
// is rare at N <= 4096. The finish recomputes the first slot of the class, then runs k - a packed
// rounds on (low key bits, inverted index) over the class members only, replacing slots a to k-1.
// The finish orders by full key then index, so prefer_smaller_index holds for both key types.
//
// Contract:
//   scope block, 256 threads (128 and 512 accepted), blocked input in registers, N = Threads x IPT
//   with IPT a power of two, k in [1, N/2], N <= 65536, keys bf16 or fp32, both requirements.
//   Output: k indices in the harness's shared output array, sorted descending by key except inside
//   the finished tie class for fp32. Temp storage: k slot words (fp32: 2k words and one int).
//   Sync: k + 1 __syncthreads for bf16, k + 1 plus one __syncthreads_or for fp32, plus 1 + (k - a)
//   on the finish path. Assumes the caller separates consecutive calls that reuse the temp storage
//   by a barrier. NaN: a NaN twiddles above +inf and is selected first, never produced by focus-set
//   inputs. -0.0 equals +0.0 (negate twiddle). Keys are clobbered.
#pragma once

#include "../topk_harness.cuh"
#include "B2_detail.cuh"

struct B2_trunc32_info
{
  static constexpr const char* id            = "B2_trunc32";
  static constexpr topk::scope scope         = topk::scope::block;
  static constexpr bool data_dependent       = true;
  static constexpr const char* contract      =
    "packed key-index rounds for bf16 and truncated-key packed rounds with an exact tie finish for "
    "fp32, both requirements, blocked registers in, indices to smem out";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && topk::is_pow2(n / threads) && (k >= 1) && (k <= n / 2)
                       && (n <= 65536);
    const bool req_ok = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B2_trunc32
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  static constexpr bool is_bf16                        = std::is_same_v<KeyT, __nv_bfloat16>;

  // fp32 packing: B index bits (N a power of two, so N - 1 fits), 32 - B key bits
  static constexpr int B             = b2b::log2_int(N);
  static constexpr unsigned lo_mask  = (1u << B) - 1u; // inverted index in the packed word
  static constexpr unsigned hi_mask  = ~lo_mask; // truncated key
  static constexpr int B2            = 32 - B; // finish word: low key bits above, inverted index below
  static constexpr unsigned lo_mask2 = (B2 >= 32) ? 0xffffffffu : ((1u << B2) - 1u);

  struct TempStorage16
  {
    unsigned slot[K];
  };
  struct TempStorage32
  {
    unsigned slot[K];
    unsigned lo[K];
    int a;
  };
  using TempStorage = std::conditional_t<is_bf16, TempStorage16, TempStorage32>;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    if constexpr (is_bf16)
    {
      run16(keys, ts, out_idx);
    }
    else
    {
      run32(keys, ts, out_idx);
    }
  }

  __device__ __forceinline__ static void run16(key_storage& keys, TempStorage& ts, int* out_idx)
  {
    const int tid = static_cast<int>(threadIdx.x);
    for (int t = tid; t < K; t += Threads)
    {
      ts.slot[t] = 0u;
    }
    unsigned p[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      p[i] = (b2b::twiddle16(keys[i]) << 16) | (0xffffu - static_cast<unsigned>(tid * IPT + i));
    }
    b2b::sort_desc<IPT>(p);
    __syncthreads();
#pragma unroll b2b::round_unroll<K>::value
    for (int r = 0; r < K; ++r)
    {
      b2b::round_shift1<IPT>(&ts.slot[r], p);
    }
    for (int t = tid; t < K; t += Threads)
    {
      out_idx[t] = static_cast<int>(0xffffu - (ts.slot[t] & 0xffffu));
    }
  }

  __device__ __forceinline__ static void run32(key_storage& keys, TempStorage& ts, int* out_idx)
  {
    const int tid = static_cast<int>(threadIdx.x);
    for (int t = tid; t < K; t += Threads)
    {
      ts.slot[t] = 0u;
      ts.lo[t]   = 0u;
    }
    unsigned tw[IPT];
    unsigned p[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      tw[i] = b2b::twiddle32(keys[i]);
      p[i]  = (tw[i] & hi_mask) | (lo_mask - static_cast<unsigned>(tid * IPT + i));
    }
    b2b::sort_desc<IPT>(p);
    __syncthreads();
    unsigned last = 0u;
#pragma unroll b2b::round_unroll<K>::value
    for (int r = 0; r < K; ++r)
    {
      last = b2b::round_shift1<IPT>(&ts.slot[r], p);
    }
    // Tie class of the k-th truncated key. My largest remaining word is the only one that can be
    // in it (the list is sorted and every remaining word is at most the k-th slot).
    const unsigned tkk = last & hi_mask;
    const bool pending = ((p[0] & hi_mask) == tkk) && (p[0] != 0u);
    int a              = K;
    if (__syncthreads_or(pending))
    {
      // a = first slot of the class (the class occupies slots a to K - 1)
      for (int t = tid; t < K; t += Threads)
      {
        const bool tie_t = ((ts.slot[t] & hi_mask) == tkk);
        const bool tie_p = (t > 0) && ((ts.slot[t - 1] & hi_mask) == tkk);
        if (tie_t && !tie_p)
        {
          ts.a = t;
        }
      }
      __syncthreads();
      a = ts.a;
      // finish words: low key bits above, inverted index below, 0 outside the class
      unsigned w[IPT];
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        const bool in = ((tw[i] & hi_mask) == tkk);
        w[i]          = in ? ((tw[i] << B2) | (lo_mask2 - static_cast<unsigned>(tid * IPT + i))) : 0u;
      }
      for (int r = a; r < K; ++r)
      {
        unsigned m = w[0];
#pragma unroll
        for (int i = 1; i < IPT; ++i)
        {
          m = m > w[i] ? m : w[i];
        }
        atomicMax(&ts.lo[r], m);
        __syncthreads();
        const unsigned mm = ts.lo[r];
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          w[i] = (w[i] == mm) ? 0u : w[i];
        }
      }
    }
    for (int t = tid; t < K; t += Threads)
    {
      out_idx[t] = (t < a) ? static_cast<int>(lo_mask - (ts.slot[t] & lo_mask))
                           : static_cast<int>(lo_mask2 - (ts.lo[t] & lo_mask2));
    }
  }
};
