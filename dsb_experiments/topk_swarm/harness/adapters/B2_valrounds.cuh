// Adapter B2_valrounds: the July 2026 `atomic_adaptive` design (reference/proto_topk.cu) as a
// harness adapter, generalized to IPT items per thread and k rounds, index-only output. Rounds
// carry twiddled fp32 values only (fire-and-forget shared atomicMax, barrier, read back, predicated
// shift of the per-thread sorted list). Membership in the top-k is then a value predicate: items
// above the k-th round value are certain, items equal to it are boundary ties. Early flood check
// after round 2, fast path when exactly k items sit on or above the k-th value, else a per-round
// histogram with a 32-lane prefix scan finds the boundary value and the certain count, and a single
// ticket scatter emits indices. This is the control of the N4 kill experiment (idea card
// ideas/queue/N4_truncated_key_packing.md), not a variant meant to win.
//
// Contract:
//   scope block, Threads 128, 256 or 512, blocked input in registers, N = Threads x IPT with IPT a
//   power of two up to 64, k in [2, min(N/2, 32)] (the histogram scan is one warp), keys fp32,
//   requirement not_guaranteed only (boundary ties are emitted in shared-memory ticket order).
//   Output: k indices in the harness output array, unsorted. Temp storage: 2k + 6 words. Sync:
//   __syncthreads inside (1 init, k rounds, 1 early check, 1 count, 2 on the flood path), assumes
//   the caller separates consecutive calls by a block barrier. NaN: twiddled bitwise (positive NaN
//   above +inf, negative NaN below -inf), never in focus-set inputs. -0.0 is normalized to +0.0.
#pragma once

#include <cstdint>
#include <type_traits>

#include "../topk_harness.cuh"
#include "B2_packed.cuh"

struct B2_valrounds_info
{
  static constexpr const char* id       = "B2_valrounds";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "atomic_adaptive port: value-only atomicMax rounds plus value-based index resolution, fp32, not_guaranteed only";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const int ipt        = (threads > 0) ? n / threads : 0;
    const bool shape_ok  = (n % threads == 0) && topk::is_pow2(ipt) && (ipt <= 64) && (k >= 2) && (k <= 32) && (2 * k <= n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B2_valrounds
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[IPT];

  struct TempStorage
  {
    uint32_t slot[K];
    uint32_t rh[K];
    uint32_t cnt, cnt2;
    uint32_t mstar, nAstar;
    uint32_t cntA, cntB;
  };

  // Resolve the output once the count c of items >= slX (= slot[R-1]) is known, c >= K.
  template <int R>
  __device__ __forceinline__ static void
  resolve(const uint32_t (&tw)[IPT], int base, TempStorage& sh, uint32_t slX, uint32_t c, uint32_t myc, int* out_idx)
  {
    const int tid = static_cast<int>(threadIdx.x);
    if (c == K)
    {
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        if (tw[i] >= slX)
        {
          const uint32_t t = atomicAdd(&sh.cntA, 1u);
          out_idx[t]       = base + i;
        }
      }
      return;
    }
    if (myc)
    {
      uint32_t sl[R];
#pragma unroll
      for (int j = 0; j < R; ++j)
      {
        sl[j] = sh.slot[j];
      }
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        if (tw[i] >= slX)
        {
          uint32_t jj = 0;
#pragma unroll
          for (int j = 0; j < R; ++j)
          {
            jj += (sl[j] > tw[i]) ? 1u : 0u;
          }
          atomicAdd(&sh.rh[jj], 1u);
        }
      }
    }
    __syncthreads();
    if (tid < 32)
    {
      const int lane = tid;
      const uint32_t r = (lane < R) ? sh.rh[lane] : 0u;
      uint32_t cc      = r;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1)
      {
        const uint32_t o = __shfl_up_sync(0xffffffffu, cc, d);
        if (lane >= d)
        {
          cc += o;
        }
      }
      if (lane < R && cc >= K && (cc - r) < K)
      {
        sh.mstar  = sh.slot[lane];
        sh.nAstar = cc - r;
      }
    }
    __syncthreads();
    const uint32_t Mstar = sh.mstar;
    const uint32_t nA    = sh.nAstar;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      if (tw[i] > Mstar)
      {
        const uint32_t t = atomicAdd(&sh.cntA, 1u);
        out_idx[t]       = base + i;
      }
      else if (tw[i] == Mstar)
      {
        const uint32_t t   = atomicAdd(&sh.cntB, 1u);
        const uint32_t pos = nA + t;
        if (pos < K)
        {
          out_idx[pos] = base + i;
        }
      }
    }
  }

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[IPT], TempStorage& sh, int* out_idx, KeyT*)
  {
    const int tid  = static_cast<int>(threadIdx.x);
    const int base = tid * IPT;
    for (int t = tid; t < K; t += Threads)
    {
      sh.slot[t] = 0;
      sh.rh[t]   = 0;
    }
    if (tid == Threads - 1)
    {
      sh.cnt  = 0;
      sh.cnt2 = 0;
      sh.cntA = 0;
      sh.cntB = 0;
    }
    uint32_t tw[IPT];
    uint32_t s[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      tw[i] = b2::twiddle32(keys[i]);
      s[i]  = tw[i];
    }
    b2::sort_desc(s);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r)
    {
      b2::round(&sh.slot[r], s);
    }
    // early flood check: if >= K items sit on or above slot[1], the boundary is already known
    const uint32_t sl1 = sh.slot[1];
    uint32_t myc2      = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      myc2 += (tw[i] >= sl1) ? 1u : 0u;
    }
    atomicAdd(&sh.cnt2, myc2);
    __syncthreads();
    const uint32_t c2 = sh.cnt2;
    if (c2 >= K)
    {
      resolve<2>(tw, base, sh, sl1, c2, myc2, out_idx);
      return;
    }
#pragma unroll
    for (int r = 2; r < K; ++r)
    {
      b2::round(&sh.slot[r], s);
    }
    const uint32_t slk = sh.slot[K - 1];
    uint32_t myc       = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      myc += (tw[i] >= slk) ? 1u : 0u;
    }
    atomicAdd(&sh.cnt, myc);
    __syncthreads();
    resolve<K>(tw, base, sh, slk, sh.cnt, myc, out_idx);
  }
};
