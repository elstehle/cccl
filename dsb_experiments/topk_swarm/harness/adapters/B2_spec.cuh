// Adapter B2_spec: B2_packed with the warp pre-reduction taken off the round's dependent chain.
// ptxas aggregates a warp's fire-and-forget atomicMax into one redux.sync plus one predicated
// atomic, and in B2_packed that redux sits between the read back of round r and the atomic of
// round r + 1. Here every warp carries its two largest heads (wmax, wsec) from the previous round's
// state. The lanes holding them post directly, together with the lane that won the previous round
// (its head changed), so the atomic of round r + 1 depends on the read back of round r through one
// compare and one select only. The two redux that refresh wmax and wsec for the following round run
// in the shadow of the barrier, and the tail of the winner's shift (items 1 and up) is deferred by
// one round so it runs in the shadow of the read back. Posting two or three lanes per warp to the
// same address costs a few cycles of serialization in the shared memory unit. The result is the
// same as B2_packed in every cell, the contract is that of B2_packed (see B2_packed.cuh).
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "../topk_harness.cuh"
#include "B2_packed.cuh"

struct B2_spec_info
{
  static constexpr const char* id       = "B2_spec";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "B2_packed rounds with stale warp max and second max posting, redux off the dependent chain, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return B2_packed_info::supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B2_spec
{
  using base                                           = B2_packed<KeyT, Threads, N, K, Req>;
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[IPT];
  static constexpr bool is_fp32                        = std::is_same_v<KeyT, float>;
  static constexpr int index_bits                      = b2::log2_int(N);
  static constexpr uint32_t index_mask                 = static_cast<uint32_t>(N - 1);
  static constexpr bool unroll_rounds                  = (K * IPT <= 4096);
  static constexpr uint32_t full_mask                  = 0xffffffffu;
  using TempStorage                                    = typename base::TempStorage;

  struct round_state
  {
    uint32_t wmax, wsec;
    bool post, wprev;
  };

  __device__ __forceinline__ static void refresh(uint32_t head, round_state& st, bool w)
  {
    st.wmax = __reduce_max_sync(full_mask, head);
    st.wsec = __reduce_max_sync(full_mask, (head == st.wmax) ? 0u : head);
    st.post = w | (head == st.wmax) | (head == st.wsec);
    st.wprev = w;
  }

  __device__ __forceinline__ static uint32_t round(uint32_t* slot, uint32_t (&p)[IPT], round_state& st)
  {
    if (st.post)
    {
      atomicMax(slot, p[0]);
    }
    __syncthreads();
    const uint32_t M = *slot;
    // deferred tail of the previous round's shift, independent of M
    if constexpr (IPT > 1)
    {
#pragma unroll
      for (int i = 1; i + 1 < IPT; ++i)
      {
        p[i] = st.wprev ? p[i + 1] : p[i];
      }
      p[IPT - 1] = st.wprev ? 0u : p[IPT - 1];
    }
    const bool w = (p[0] == M);
    if constexpr (IPT > 1)
    {
      p[0] = w ? p[1] : p[0];
    }
    else
    {
      p[0] = w ? 0u : p[0];
    }
    // the post predicate of the next round uses the stale wmax and wsec of this round's heads:
    // a lane that did not win keeps its head, the winner posts because of w
    const bool post_next = w | (p[0] == st.wmax) | (p[0] == st.wsec);
    refresh(p[0], st, w);
    st.post = post_next | st.post;
    return M;
  }

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[IPT], TempStorage& ts, int* out_idx, KeyT*)
  {
    const int tid       = static_cast<int>(threadIdx.x);
    const uint32_t base_idx = static_cast<uint32_t>(tid) * IPT;

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
      invbase = index_mask & ~base_idx;
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        tw[i] = b2::twiddle32(keys[i]);
        p[i]  = (tw[i] & ~index_mask) | (invbase & ~static_cast<uint32_t>(i));
      }
    }
    else
    {
      invbase = 0xffffu ^ base_idx;
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

    round_state st;
    refresh(p[0], st, false);
    __syncthreads();

    uint32_t M = 0;
    if constexpr (unroll_rounds)
    {
#pragma unroll
      for (int r = 0; r < K; ++r)
      {
        M = round(&ts.slot[r], p, st);
      }
    }
    else
    {
#pragma unroll 1
      for (int r = 0; r < K; ++r)
      {
        M = round(&ts.slot[r], p, st);
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
        base::finish(tw, tk, invbase, K - static_cast<int>(c >> 20), ts, out_idx);
      }
    }
  }
};
