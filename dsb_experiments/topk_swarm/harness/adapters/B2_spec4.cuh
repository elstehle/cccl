// Adapter B2_spec4: B2_spec2 with a single-level refresh. The three redux that refresh the warp
// state (third head, the b0 lane's item after next, the b1 lane's next item) do not depend on the
// round maximum, so ptxas issues them before the read back and the warp never waits for a redux
// between the read back and its next post. One predicated PTX red per warp as in B2_spec2. ptxas aggregates a warp's fire-and-forget atomicMax into redux.sync plus one elected atomic,
// which is the cheap form (B2_spec showed that 256 un-aggregated shared atomics per round cost about
// 55 cycles more), but in B2_packed that redux depends on the previous round's read back. Here every
// warp refreshes, in the shadow of the barrier, its maximum head b0, the maximum of the other heads
// b1 and the next item x of the lane that holds b0. After the read back M the warp's new maximum is
// b0 when it did not win (b0 != M) and max(b1, x) when it did, so lane 0 posts a uniform value after
// one compare and one select. The post is a PTX red with a per-lane address offset of zero that the
// compiler cannot fold, which keeps ptxas from inserting its own redux. The winner's tail shift is
// deferred by one round. Result and contract are those of B2_packed (see B2_packed.cuh).
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "../topk_harness.cuh"
#include "B2_packed.cuh"

struct B2_spec4_info
{
  static constexpr const char* id       = "B2_spec4";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "B2_packed rounds, one uniform atomic per warp from speculative warp max, second max and next item, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return B2_packed_info::supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B2_spec4
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
    uint32_t b0, b1, x;  // warp maximum head, maximum of the other heads, next item of the b0 lane
    uint32_t c1, c2, c3; // redux results in flight: third head, b0 lane's item after next, b1 lane's next item
    uint32_t z;          // per-lane runtime zero for the post address
    bool won;            // this warp held the round maximum
  };

  __device__ __forceinline__ static uint32_t redux_max(uint32_t v)
  {
    uint32_t r;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    return r;
  }

  template <int I>
  __device__ __forceinline__ static uint32_t item(const uint32_t (&p)[IPT])
  {
    if constexpr (IPT > I)
    {
      return p[I];
    }
    else
    {
      return 0u;
    }
  }

  __device__ __forceinline__ static void post_max(uint32_t* slot, uint32_t z, uint32_t v, bool pred)
  {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(slot)) + z;
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t@p red.shared.max.u32 [%0], %1;\n\t}"
                 :
                 : "r"(addr), "r"(v), "r"(static_cast<uint32_t>(pred))
                 : "memory");
  }

  // Round r: fold the previous redux results, issue the next three, post, barrier, read back, shift.
  __device__ __forceinline__ static uint32_t
  round(uint32_t* slot, uint32_t (&p)[IPT], round_state& st, uint32_t& post_val, bool lane0)
  {
    post_max(slot, st.z, post_val, lane0);
    {
      const bool up      = (st.x > st.b1);
      const uint32_t nb0 = up ? st.x : st.b1;
      const uint32_t nb1 = up ? st.b1 : max(st.c1, st.x);
      const uint32_t nx  = up ? st.c2 : st.c3;
      st.b0              = st.won ? nb0 : st.b0;
      st.b1              = st.won ? nb1 : st.b1;
      st.x               = st.won ? nx : st.x;
    }
    const bool top     = (p[0] == st.b0);
    const bool sec     = (p[0] == st.b1);
    const uint32_t alt = max(st.b1, st.x);
    st.c1 = redux_max((top | sec) ? 0u : p[0]);
    st.c2 = redux_max(top ? item<2>(p) : 0u);
    st.c3 = redux_max(sec ? item<1>(p) : 0u);
    __syncthreads();
    const uint32_t M = *slot;
    st.won           = (st.b0 == M);
    post_val         = st.won ? alt : st.b0;
    const bool w     = (p[0] == M);
#pragma unroll
    for (int i = 0; i + 1 < IPT; ++i)
    {
      p[i] = w ? p[i + 1] : p[i];
    }
    p[IPT - 1] = w ? 0u : p[IPT - 1];
    return M;
  }

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[IPT], TempStorage& ts, int* out_idx, KeyT*)
  {
    const int tid           = static_cast<int>(threadIdx.x);
    const uint32_t base_idx = static_cast<uint32_t>(tid) * IPT;
    const bool lane0        = ((tid & 31) == 0);

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

    // initial state: two-level redux once per call
    round_state st;
    st.b0          = __reduce_max_sync(full_mask, p[0]);
    const bool top = (p[0] == st.b0);
    st.x           = __reduce_max_sync(full_mask, top ? item<1>(p) : 0u);
    st.b1          = __reduce_max_sync(full_mask, top ? 0u : p[0]);
    st.c1 = st.c2 = st.c3 = 0;
    st.won = false;
    __syncthreads();
    // runtime zero per lane: slot[0] is zero here and unknown to the compiler
    st.z = ts.slot[0] & static_cast<uint32_t>(tid & 31);

    uint32_t M        = 0;
    uint32_t post_val = st.b0;
    if constexpr (unroll_rounds)
    {
#pragma unroll
      for (int r = 0; r < K; ++r)
      {
        M = round(&ts.slot[r], p, st, post_val, lane0);
      }
    }
    else
    {
#pragma unroll 1
      for (int r = 0; r < K; ++r)
      {
        M = round(&ts.slot[r], p, st, post_val, lane0);
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
