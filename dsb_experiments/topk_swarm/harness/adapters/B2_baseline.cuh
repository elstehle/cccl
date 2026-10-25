// Adapter B2_baseline: atomics based iterative extraction (approach B2), the simplest correct form.
//
//   bf16 keys: the twiddled 16-bit key and the inverted 16-bit index are packed into one 32-bit
//   word, so every word is unique. k rounds of a fire-and-forget shared atomicMax of each thread's
//   current head into slot[r], one barrier, one read, a predicated shift of the per-thread sorted
//   list. The k slots are the answer sorted descending by (key, smaller index first), the output
//   is the index half of each slot. Both requirements, one code path.
//
//   fp32 keys: value-only rounds on the twiddled 32-bit key (the atomic_adaptive design of
//   reference/BLOCK_TOPK_RESULTS.md without the early flood check), then a value based index
//   resolution: count the items at or above the k-th slot, a ticket scatter when the count is
//   exactly k, otherwise a per-round histogram, a prefix scan to the boundary value and a two-tier
//   scatter that drops surplus boundary ties. not_guaranteed only.
//
// Contract:
//   scope block, 256 threads (128 and 512 accepted), blocked input in registers, N = Threads x IPT
//   with IPT a power of two, k in [1, N/2], N <= 65536 (16-bit index half for bf16), keys bf16 or
//   fp32, requirement not_guaranteed for both key types and prefer_smaller_index for bf16.
//   Output: k indices in the harness's shared output array, sorted descending by key for bf16 and
//   unsorted for fp32. Temp storage: k words of slots (plus k histogram bins and 5 counters for
//   fp32). Sync: k + 1 __syncthreads for bf16, k + 2 (plus 2 on the fp32 flood path), assumes the
//   caller separates consecutive calls that reuse the temp storage by a barrier (the harness does).
//   NaN: a NaN twiddles above +inf and is selected first, never produced by focus-set inputs.
//   -0.0 equals +0.0 (negate twiddle). Keys are clobbered.
#pragma once

#include "../topk_harness.cuh"
#include "B2_detail.cuh"

struct B2_baseline_info
{
  static constexpr const char* id            = "B2_baseline";
  static constexpr topk::scope scope         = topk::scope::block;
  static constexpr bool data_dependent       = true;
  static constexpr const char* contract      =
    "packed bf16 key-index rounds (both requirements), fp32 value rounds with a ticket or histogram "
    "epilogue (not_guaranteed), blocked registers in, indices to smem out";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && topk::is_pow2(n / threads) && (k >= 1) && (k <= n / 2)
                       && (n <= 65536);
    const bool req_ok = (req == topk::requirement::not_guaranteed)
                     || (req == topk::requirement::prefer_smaller_index && kt == topk::key_type::bf16);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B2_baseline
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  static constexpr bool is_bf16                        = std::is_same_v<KeyT, __nv_bfloat16>;

  struct TempStorage16
  {
    unsigned slot[K];
  };
  struct TempStorage32
  {
    unsigned slot[K];
    unsigned rh[K];
    unsigned cnt, cntA, cntB, mstar, nA;
  };
  using TempStorage = std::conditional_t<is_bf16, TempStorage16, TempStorage32>;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    if constexpr (is_bf16)
    {
      run16(keys, ts, out_idx);
    }
    else
    {
      run32(keys, idx, ts, out_idx);
    }
  }

  // ---- bf16: packed word, the slots are the answer
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
      p[i] = (b2::twiddle16(keys[i]) << 16) | (0xffffu - static_cast<unsigned>(tid * IPT + i));
    }
    b2::sort_desc<IPT>(p);
    __syncthreads();
#pragma unroll b2::round_unroll<K>::value
    for (int r = 0; r < K; ++r)
    {
      b2::round_shift1<IPT>(&ts.slot[r], p);
    }
    for (int t = tid; t < K; t += Threads)
    {
      out_idx[t] = static_cast<int>(0xffffu - (ts.slot[t] & 0xffffu));
    }
  }

  // ---- fp32: value rounds, then index resolution by value
  __device__ __forceinline__ static void
  run32(key_storage& keys, int (&idx)[items_per_thread], TempStorage& ts, int* out_idx)
  {
    const int tid = static_cast<int>(threadIdx.x);
    for (int t = tid; t < K; t += Threads)
    {
      ts.slot[t] = 0u;
      ts.rh[t]   = 0u;
    }
    if (tid == Threads - 1)
    {
      ts.cnt  = 0u;
      ts.cntA = 0u;
      ts.cntB = 0u;
    }
    unsigned tw[IPT];
    unsigned s[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      tw[i] = b2::twiddle32(keys[i]);
      s[i]  = tw[i];
    }
    b2::sort_desc<IPT>(s);
    __syncthreads();
    unsigned T = 0u;
#pragma unroll b2::round_unroll<K>::value
    for (int r = 0; r < K; ++r)
    {
      T = b2::round_shift1<IPT>(&ts.slot[r], s);
    }
    // T is slot[K - 1]: at least K items are at or above it, every value above it was extracted
    unsigned myc = 0u;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      myc += (tw[i] >= T);
    }
    atomicAdd(&ts.cnt, myc);
    __syncthreads();
    const unsigned c = ts.cnt;
    if (c == static_cast<unsigned>(K))
    {
      // the items at or above T are exactly the top-k: one ticket each
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        if (tw[i] >= T)
        {
          const unsigned t = atomicAdd(&ts.cntA, 1u);
          out_idx[t]       = idx[i];
        }
      }
      return;
    }
    // Flood path: boundary multiplicity. Items at or above T vote into the bin of the first slot
    // holding their value (the slots are non-increasing, binary search), the inclusive prefix over
    // the bins is the count of items at or above each slot, the first crossing of K gives the
    // boundary value M* and the number of items strictly above it.
    if (myc)
    {
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        if (tw[i] >= T)
        {
          int lo = 0, hi = K; // first slot index with slot <= tw[i]
          while (lo < hi)
          {
            const int mid = (lo + hi) >> 1;
            if (ts.slot[mid] > tw[i])
            {
              lo = mid + 1;
            }
            else
            {
              hi = mid;
            }
          }
          atomicAdd(&ts.rh[lo], 1u);
        }
      }
    }
    __syncthreads();
    if (tid < 32)
    {
      unsigned carry = 0u;
      for (int base = 0; base < K; base += 32)
      {
        const int j      = base + tid;
        const unsigned r = (j < K) ? ts.rh[j] : 0u;
        unsigned cc      = r;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1)
        {
          const unsigned o = __shfl_up_sync(0xffffffffu, cc, d);
          if (tid >= d)
          {
            cc += o;
          }
        }
        cc += carry;
        if (j < K && cc >= static_cast<unsigned>(K) && (cc - r) < static_cast<unsigned>(K))
        {
          ts.mstar = ts.slot[j];
          ts.nA    = cc - r;
        }
        carry = __shfl_sync(0xffffffffu, cc, 31);
      }
    }
    __syncthreads();
    const unsigned mstar = ts.mstar;
    const unsigned nA    = ts.nA;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      if (tw[i] > mstar)
      {
        const unsigned t = atomicAdd(&ts.cntA, 1u);
        out_idx[t]       = idx[i];
      }
      else if (tw[i] == mstar)
      {
        const unsigned t   = atomicAdd(&ts.cntB, 1u);
        const unsigned pos = nA + t;
        if (pos < static_cast<unsigned>(K))
        {
          out_idx[pos] = idx[i];
        }
      }
    }
  }
};
