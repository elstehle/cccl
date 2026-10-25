// Adapter B2_spec3: B2_packed rounds with the warp pre-reduction off the round's dependent chain and
// no shared atomics. Every warp keeps three uniform values from the previous round: b0, its largest
// head, b1, the largest head of its other lanes, and x, the next item of the b0 lane. After the read
// back M the warp's new maximum is b0 when it did not win (b0 != M) and max(b1, x) when it did, so
// the value for the next round follows from one compare and one select, and every lane stores it to
// the warp's word of the round (same word, same value, one shared memory transaction, no predicate,
// no atomic unit, no branch). After the barrier every thread reads the 8 words and takes their max.
// The refresh of b0, b1 and x needs three independent redux.sync whose inputs do not depend on the
// round maximum (the third head, the b0 lane's item after next, the b1 lane's next item), so ptxas
// issues them before the read back and they complete under its latency. Result and contract are those of B2_packed (see B2_packed.cuh),
// temp storage is Threads / 32 words per round plus k slots for the fp32 tie path.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "../topk_harness.cuh"
#include "B2_packed.cuh"

struct B2_spec3_info
{
  static constexpr const char* id       = "B2_spec3";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "B2_packed rounds, per-warp store of the speculative warp max, 8-way max after the barrier, no atomics, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return B2_packed_info::supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B2_spec3
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
  static constexpr int warps                           = Threads / 32;

  struct TempStorage
  {
    __align__(16) uint32_t post[K][warps]; // one word per warp and round
    uint32_t slot[K];                      // round maxima, written by thread 0 (fp32 tie path)
    uint32_t cnt;
  };

  struct state
  {
    uint32_t b0, b1, x;   // warp maximum head, maximum of the other heads, next item of the b0 lane
    uint32_t c1, c2, c3;  // redux results in flight: third head, winner's next item, b1 lane's next item
    bool won;             // this warp held the round maximum
  };

  // redux.sync as volatile PTX: the intrinsic form is sunk by the compiler across the barrier to its
  // first use, which puts the redux back on the dependent chain. The volatile asm stays where it is
  // written, before the barrier, so its latency overlaps the barrier and the read back.
  __device__ __forceinline__ static uint32_t redux_max(uint32_t v)
  {
    uint32_t r;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    return r;
  }

  __device__ __forceinline__ static uint32_t item1(const uint32_t (&p)[IPT])
  {
    if constexpr (IPT > 1)
    {
      return p[1];
    }
    else
    {
      return 0u;
    }
  }

  __device__ __forceinline__ static uint32_t item2(const uint32_t (&p)[IPT])
  {
    if constexpr (IPT > 2)
    {
      return p[2];
    }
    else
    {
      return 0u;
    }
  }

  // One round. post_val is the warp maximum for this round, computed from the state of the round
  // before. Returns the round maximum M and leaves the next post value in post_val.
  __device__ __forceinline__ static uint32_t
  round(uint32_t (&words)[warps], uint32_t (&p)[IPT], state& st, uint32_t& post_val, int warp)
  {
    words[warp] = post_val;
    // fold the redux results of the previous round into the state (branchless, all inputs ready)
    {
      const bool up      = (st.x > st.b1); // the winner's next item is the new warp maximum
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
    // three independent redux for the next fold, none depends on this round's maximum
    st.c1 = redux_max((top | sec) ? 0u : p[0]);
    st.c2 = redux_max(top ? item2(p) : 0u);
    st.c3 = redux_max(sec ? item1(p) : 0u);
    __syncthreads();
    uint32_t M;
    if constexpr (warps == 8)
    {
      const uint32_t m1 = max(max(words[0], words[1]), words[2]);
      const uint32_t m2 = max(max(words[3], words[4]), words[5]);
      const uint32_t m3 = max(words[6], words[7]);
      M                 = max(max(m1, m2), m3);
    }
    else
    {
      M = words[0];
#pragma unroll
      for (int i = 1; i < warps; ++i)
      {
        M = max(M, words[i]);
      }
    }
    st.won   = (st.b0 == M);
    post_val = st.won ? alt : st.b0;
    const bool w = (p[0] == M);
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
    const int warp          = tid >> 5;

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
    state st;
    st.b0          = __reduce_max_sync(full_mask, p[0]);
    const bool top = (p[0] == st.b0);
    st.x           = __reduce_max_sync(full_mask, top ? item1(p) : 0u);
    st.b1          = __reduce_max_sync(full_mask, top ? 0u : p[0]);
    st.c1 = st.c2 = st.c3 = 0;
    st.won            = false;
    uint32_t post_val = st.b0;
    uint32_t mine     = 0; // thread t keeps the maximum of round t for the readout
    uint32_t M        = 0;

    if constexpr (unroll_rounds)
    {
#pragma unroll
      for (int r = 0; r < K; ++r)
      {
        M    = round(ts.post[r], p, st, post_val, warp);
        mine = (tid == r) ? M : mine;
        if constexpr (is_fp32)
        {
          if (tid == 0)
          {
            ts.slot[r] = M;
          }
        }
      }
    }
    else
    {
#pragma unroll 1
      for (int r = 0; r < K; ++r)
      {
        M    = round(ts.post[r], p, st, post_val, warp);
        mine = (tid == r) ? M : mine;
        if constexpr (is_fp32)
        {
          if (tid == 0)
          {
            ts.slot[r] = M;
          }
        }
      }
    }

    if constexpr (!is_fp32)
    {
      if (tid < K)
      {
        out_idx[tid] = static_cast<int>(0xffffu & ~mine);
      }
    }
    else
    {
      const uint32_t tk = M & ~index_mask;
      uint32_t in       = 0;
      if (tid < K)
      {
        out_idx[tid] = static_cast<int>(index_mask & ~mine);
        in           = ((mine & ~index_mask) == tk) ? 1u : 0u;
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
        // the finish works on ts.slot (written by thread 0 every round, visible after the barrier)
        typename base::TempStorage& bts = *reinterpret_cast<typename base::TempStorage*>(&ts.slot[0]);
        base::finish(tw, tk, invbase, K - static_cast<int>(c >> 20), bts, out_idx);
      }
    }
  }
};
