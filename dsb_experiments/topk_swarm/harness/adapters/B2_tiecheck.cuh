// Adapter B2_tiecheck: B2_packed (instance A's baseline, commit 8212d9e0d2) with one change in the
// fp32 path. The tie check after the k rounds asks whether any thread's largest remaining packed
// word lies in the tie class of the k-th truncated key, with one __syncthreads_or (a barrier with
// a block-wide OR), instead of counting class members everywhere and among the slots with an
// atomicAdd, a barrier and a read back. The remaining head is the only remaining word of a thread
// that can be in the class, since the list is sorted and every remaining word is at most the k-th
// slot. When the OR is set, the first slot of the class is found by the thread that holds it (one
// store and one barrier, on the finish path only) and B2_packed's finish runs unchanged. The bf16
// path and everything else are B2_packed's. Contract as B2_packed.
#pragma once

#include "../topk_harness.cuh"
#include "B2_packed.cuh"

struct B2_tiecheck_info
{
  static constexpr const char* id       = "B2_tiecheck";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "B2_packed with the fp32 tie check as one __syncthreads_or over the remaining heads, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return B2_packed_info::supported(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B2_tiecheck
{
  using base                                           = B2_packed<KeyT, Threads, N, K, Req>;
  static constexpr int items_per_thread                = base::items_per_thread;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = base::input_arrangement;
  static constexpr topk::output_form output            = base::output;
  using key_storage                                    = typename base::key_storage;
  static constexpr bool is_fp32                        = base::is_fp32;
  static constexpr int index_bits                      = base::index_bits;
  static constexpr uint32_t index_mask                 = base::index_mask;
  static constexpr bool unroll_rounds                  = base::unroll_rounds;
  using TempStorage                                    = typename base::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[IPT], TempStorage& ts, int* out_idx, KeyT* out_keys)
  {
    if constexpr (!is_fp32)
    {
      base::run(keys, idx, ts, out_idx, out_keys);
      return;
    }
    const int tid       = static_cast<int>(threadIdx.x);
    const uint32_t base_i = static_cast<uint32_t>(tid) * IPT;
    for (int t = tid; t < K; t += Threads)
    {
      ts.slot[t] = 0;
    }
    uint32_t p[IPT];
    uint32_t tw[IPT];
    const uint32_t invbase = index_mask & ~base_i;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      tw[i] = b2::twiddle32(keys[i]);
      p[i]  = (tw[i] & ~index_mask) | (invbase & ~static_cast<uint32_t>(i));
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
    const uint32_t tk = M & ~index_mask;
    for (int t = tid; t < K; t += Threads)
    {
      out_idx[t] = static_cast<int>(index_mask & ~ts.slot[t]);
    }
    // the one change: a block-wide OR over "my largest remaining word is in the class"
    const bool pending = ((p[0] & ~index_mask) == tk) && (p[0] != 0u);
    if (__syncthreads_or(pending))
    {
      for (int t = tid; t < K; t += Threads)
      {
        const bool tie_t = ((ts.slot[t] & ~index_mask) == tk);
        const bool tie_p = (t > 0) && ((ts.slot[t - 1] & ~index_mask) == tk);
        if (tie_t && !tie_p)
        {
          ts.cnt = static_cast<uint32_t>(t);
        }
      }
      __syncthreads();
      const int j = static_cast<int>(ts.cnt);
      base::finish(tw, tk, invbase, j, ts, out_idx);
    }
  }
};
