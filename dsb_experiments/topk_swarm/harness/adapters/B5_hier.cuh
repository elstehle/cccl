// Adapter B5_hier: hierarchical warp then block composition (idea B5) built from the W2 packed
// round. Every warp runs the packed W2 extraction on its share of the row and writes its k words
// to shared memory, one barrier, warp 0 runs the packed extraction once more over the Warps x k
// candidate words and writes the k indices. fp32 goes through the N4 truncated-key words and a
// block-scope tie pass after the merge. See W2_impl.cuh for the word and the round.
//
// Contract:
//   scope block, Threads in {128, 256, 512} (a multiple of 32), blocked input in registers
//   (thread t holds items t*IPT .. t*IPT+IPT-1), N = Threads x IPT with IPT >= 1 and N <= 65536
//   (bf16) or 32768 (fp32), k in [1, N/2] with Warps x k <= 2048 (the merge list of warp 0), keys
//   bf16 or fp32, int32 indices derived from the position, both tie-break requirements by
//   construction. Output: k indices in the harness's shared output array in descending key order
//   (the fp32 tie pass may rewrite the last slots of the tie class). Temp storage: Warps x k words
//   plus a few counters. Sync: performs __syncthreads inside (1 for bf16, 3 for fp32 on the fast
//   path, 2 more per slow tie round), assumes the caller separates consecutive calls that reuse
//   the temp storage by a block barrier (the harness does). NaN: see W2_impl.cuh.
#pragma once

#include "W2_impl.cuh"

struct B5_hier_info
{
  static constexpr const char* id       = "B5_hier";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "per-warp packed W2 rounds, smem exchange of Warps x k words, one barrier, warp 0 packed W2 merge, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return w2::supported_shape(kt, threads, n, k, req, 128, 512) && (threads % 32 == 0) && ((threads / 32) * k <= 2048);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B5_hier
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];

  static constexpr int Warps   = Threads / 32;
  static constexpr int IdxBits = w2::ceil_log2(N);
  static constexpr int L1      = w2::next_pow2(w2::cmin(K, items_per_thread));
  static constexpr int P1      = topk::round_up(items_per_thread, L1);
  static constexpr int Cands   = Warps * K;
  static constexpr int IPT2    = topk::round_up(Cands, 32) / 32;
  static constexpr int L2      = w2::next_pow2(w2::cmin(K, IPT2));
  static constexpr int P2      = topk::round_up(IPT2, L2);
  static constexpr bool is_f32 = std::is_same_v<KeyT, float>;

  struct TempStorage
  {
    uint32_t cand[Cands];
    uint32_t last;
    uint32_t cnt[Warps];
    uint32_t val[Warps];
    uint32_t idxv[Warps];
  };

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    constexpr int IPT  = items_per_thread;
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid / 32;
    const int lane     = tid % 32;
    const int idx_base = tid * IPT;

    // Phase 1: every warp extracts its k largest words into shared memory
    {
      uint32_t w[P1];
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        w[i] = w2::pack<KeyT, IdxBits>(keys[i], static_cast<uint32_t>(idx_base + i));
      }
#pragma unroll
      for (int i = IPT; i < P1; ++i)
      {
        w[i] = 0u;
      }
      uint32_t list[L1];
      w2::select_top<L1, P1>(w, list);
      w2::extract_rounds<32, K, L1>(list, lane, 0xffffffffu, w2::emit_word{ts.cand + warp * K});
    }
    __syncthreads();

    // Phase 2: warp 0 extracts the k largest of the Warps x k candidates
    if (warp == 0)
    {
      uint32_t w[P2];
#pragma unroll
      for (int j = 0; j < IPT2; ++j)
      {
        const int pos = j * 32 + lane;
        w[j]          = (pos < Cands) ? ts.cand[pos] : 0u;
      }
#pragma unroll
      for (int j = IPT2; j < P2; ++j)
      {
        w[j] = 0u;
      }
      uint32_t list[L2];
      w2::select_top<L2, P2>(w, list);
      const uint32_t m_last =
        w2::extract_rounds<32, K, L2>(list, lane, 0xffffffffu, w2::emit_index<IdxBits>{out_idx});
      if constexpr (is_f32)
      {
        if (lane == 0)
        {
          ts.last = m_last;
        }
      }
    }

    if constexpr (is_f32)
    {
      // Block-scope N4 tie pass, see W2_impl.cuh for the warp form
      __syncthreads();
      const uint32_t m_last       = ts.last;
      constexpr uint32_t idx_mask = (1u << IdxBits) - 1u;
      uint32_t cnt                = 0;
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        const uint32_t w  = w2::pack<float, IdxBits>(keys[i], static_cast<uint32_t>(idx_base + i));
        const bool member = ((w ^ m_last) <= idx_mask);
        cnt += member ? (w >= m_last ? 0x10001u : 1u) : 0u;
      }
      cnt = __reduce_add_sync(0xffffffffu, cnt);
      if (lane == 0)
      {
        ts.cnt[warp] = cnt;
      }
      __syncthreads();
      uint32_t tot = 0;
#pragma unroll
      for (int q = 0; q < Warps; ++q)
      {
        tot += ts.cnt[q];
      }
      if ((tot & 0xffffu) == (tot >> 16))
      {
        return;
      }
      const int e = static_cast<int>(tot >> 16);
      uint32_t tw[IPT];
      bool mem[IPT];
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        tw[i]            = w2::twiddle_f32(keys[i]);
        const uint32_t w = (tw[i] & ~idx_mask) | (idx_mask ^ static_cast<uint32_t>(idx_base + i));
        mem[i]           = ((w ^ m_last) <= idx_mask);
      }
#pragma unroll 1
      for (int j = 0; j < e; ++j)
      {
        uint32_t cand = 0;
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          cand = mem[i] ? ::max(cand, tw[i]) : cand;
        }
        cand = __reduce_max_sync(0xffffffffu, cand);
        if (lane == 0)
        {
          ts.val[warp] = cand;
        }
        __syncthreads();
        uint32_t m = 0;
#pragma unroll
        for (int q = 0; q < Warps; ++q)
        {
          m = ::max(m, ts.val[q]);
        }
        uint32_t myidx = 0xffffffffu;
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          myidx = (mem[i] && tw[i] == m) ? ::min(myidx, static_cast<uint32_t>(idx_base + i)) : myidx;
        }
        myidx = __reduce_min_sync(0xffffffffu, myidx);
        if (lane == 0)
        {
          ts.idxv[warp] = myidx;
        }
        __syncthreads();
        uint32_t widx = 0xffffffffu;
#pragma unroll
        for (int q = 0; q < Warps; ++q)
        {
          widx = ::min(widx, ts.idxv[q]);
        }
        if (tid == 0)
        {
          out_idx[K - e + j] = static_cast<int>(widx);
        }
#pragma unroll
        for (int i = 0; i < IPT; ++i)
        {
          mem[i] = mem[i] && (static_cast<uint32_t>(idx_base + i) != widx);
        }
      }
    }
  }
};
