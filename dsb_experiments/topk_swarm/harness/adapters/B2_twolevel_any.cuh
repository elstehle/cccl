// Adapter B2_twolevel_any: B2_twolevel with the refill of the per-thread head behind a warp-uniform
// branch. B2_twolevel tests "my head ran empty" per thread and refills behind a divergent branch,
// which puts a reconvergence point into every round even when no thread takes it. Here the round
// asks the warp once (__any_sync) and the whole warp runs the predicated refill only when some lane
// needs it, so the common round is the head shift plus one VOTE. Everything else, including the
// contract, is B2_twolevel (B2_trunc32's two key paths, both requirements, N <= 65536).
#pragma once

#include "../topk_harness.cuh"
#include "B2_detail.cuh"

struct B2_twolevel_any_info
{
  static constexpr const char* id            = "B2_twolevel_any";
  static constexpr topk::scope scope         = topk::scope::block;
  static constexpr bool data_dependent       = true;
  static constexpr const char* contract      =
    "B2_twolevel with a warp-uniform refill branch (__any_sync), both key types, both requirements, "
    "blocked registers in, indices to smem out";

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
struct B2_twolevel_any
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr int H                               = (IPT < 4) ? IPT : 4;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  static constexpr bool is_bf16                        = std::is_same_v<KeyT, __nv_bfloat16>;

  static constexpr int B             = b2b::log2_int(N);
  static constexpr unsigned lo_mask  = (1u << B) - 1u;
  static constexpr unsigned hi_mask  = ~lo_mask;
  static constexpr int B2            = 32 - B;
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
    b2b::partial_sort_top<IPT, H>(p);
    __syncthreads();
#pragma unroll b2b::round_unroll<K>::value
    for (int r = 0; r < K; ++r)
    {
      b2b::round_shift_twolevel_any<IPT, H>(&ts.slot[r], p);
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
    b2b::partial_sort_top<IPT, H>(p);
    __syncthreads();
    unsigned last = 0u;
#pragma unroll b2b::round_unroll<K>::value
    for (int r = 0; r < K; ++r)
    {
      last = b2b::round_shift_twolevel_any<IPT, H>(&ts.slot[r], p);
    }
    const unsigned tkk = last & hi_mask;
    const bool pending = ((p[0] & hi_mask) == tkk) && (p[0] != 0u);
    int a              = K;
    if (__syncthreads_or(pending))
    {
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
