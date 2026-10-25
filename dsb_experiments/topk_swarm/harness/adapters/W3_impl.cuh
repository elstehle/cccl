// Implementation of approach W3 (warp radix select with ballot-count histograms), shared by the
// W3 adapters. Not an adapter by itself. Machinery in N6w_impl.cuh.
//
// Passes over the key image from the most significant bit: every pass takes the next D bits as the
// digit (fewer in the last pass), counts the 2^D buckets with one ballot per digit bit (plus one
// ballot of the candidate flag after the first pass), finds the bucket that holds the r-th largest
// candidate with a suffix scan over the lanes, narrows the candidate range to that bucket and
// subtracts the candidates above it from r. The pass ends early when the pivot bucket holds exactly
// the remaining r. With COMPACT the candidates move to shared memory (index order) once they fit
// into N / COMPACT_DIV slots and the later passes run over IPT / COMPACT_DIV slots per lane.
// bf16 keys have 16 image bits (4 passes at D 4, 4 at D 5 with a 1-bit last pass), fp32 32 bits
// (8 at D 4, 7 at D 5).
#pragma once

#include "N6w_impl.cuh"

namespace w3
{

template <typename KeyT, int N, int K, int D, bool COMPACT, int COMPACT_DIV = 4>
struct primitive
{
  static constexpr int IPT  = N / 32;
  static constexpr int KB   = n6w::key_bits<KeyT>;
  static constexpr int CIPT = COMPACT ? n6w::cmax(IPT / COMPACT_DIV, 1) : 1;
  static constexpr int CAPN = 32 * CIPT;

  struct TempStorage
  {
    uint32_t tw[COMPACT ? CAPN : 1];
    uint32_t idx[COMPACT ? CAPN : 1];
  };

  __device__ __forceinline__ static bool open(const n6w::range& st)
  {
    return st.n != st.r && __clz(st.tmin ^ st.tmax) < KB;
  }

  __device__ __forceinline__ static void run(const KeyT (&keys)[IPT], TempStorage& ts, int* out_idx)
  {
    using namespace n6w;
    const int lane     = topk::group_lane<32>();
    const int idx_base = lane * IPT;
    const uint32_t lt  = lanemask_lt();

    uint32_t tw[IPT];
    load_images<KeyT, IPT>(keys, tw);
    range st{0u, FULL, K, N};
    digit_pass<KeyT, IPT, D, true>(tw, st, lane);

    if constexpr (COMPACT)
    {
#pragma unroll 1
      while (open(st) && st.n > CAPN)
      {
        digit_pass<KeyT, IPT, D, false>(tw, st, lane);
      }
      if (open(st))
      {
        uint32_t ctw[CIPT], cidx[CIPT];
        compact<IPT, CIPT>(tw, idx_blocked{idx_base}, st, ts.tw, ts.idx, ctw, cidx, lane, lt);
        const int nv = st.n;
#pragma unroll 1
        while (open(st))
        {
          digit_pass<KeyT, CIPT, D, false, true>(ctw, st, lane, nv);
        }
        emit_range<CIPT, false>(ctw, st, out_idx, idx_array<CIPT>{cidx}, lt);
        return;
      }
    }
    else
    {
#pragma unroll 1
      while (open(st))
      {
        digit_pass<KeyT, IPT, D, false>(tw, st, lane);
      }
    }
    emit_range<IPT, true>(tw, st, out_idx, idx_blocked{idx_base}, lt);
  }
};

} // namespace w3
