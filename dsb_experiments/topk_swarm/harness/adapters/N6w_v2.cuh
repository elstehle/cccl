// Adapter N6w_v2: N6w_ballot with the ballots of a chunk of items issued before any consumer and
// the per-item bin images packed six per word (N6w_impl.cuh). Same algorithm, same finish.
//
// One bracket pass of 32 equal width bins over [lo, hi], hi the warp maximum and lo the minimum
// over lanes of the lane's m-th largest key (m = ceil(k / 32) rounded up to a power of two, at most
// 8), an exact lower bound on the k-th largest. The pivot bin's candidates are then finished by the
// packed W2 rounds (at most 64 or 128 compacted candidates), taken whole when their count equals
// the remaining k, or ranked by index when they are one tie class. Larger pivot bins (k above 256
// uses the full value range as the bracket, heavy tails, bf16 tie classes that span the bin) get
// further bracket passes over the candidates' exact value range, at most four, then radix digit
// passes on the key bits as the last resort. See N6w_impl.cuh.
//
// Contract:
//   scope warp, 32 lanes, blocked input in registers (lane t holds items t*IPT .. t*IPT+IPT-1),
//   N a multiple of 32 up to 32768, k in [1, N/2], keys bf16 or fp32 (widened to fp32 in
//   registers, minus zero folded onto plus zero), int32 indices derived from the position. Both
//   requirements run the same code: the W2 finish orders ties by index and the tie-class emission
//   takes the smallest indices. Output: k indices in the harness's shared output array, unsorted
//   (items above the pivot bin in index order first, then the finish's order). Temp storage 256 or
//   512 bytes per warp (the compaction buffer of the finish). Sync: ballots, redux.sync, shuffles
//   and __syncwarp inside, no barriers. NaN: unsupported, focus-set inputs contain no NaN.
#pragma once

#include "N6w_impl.cuh"

struct N6w_v2_info
{
  static constexpr const char* id       = "N6w_v2";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "one 32-bin bracket pass by ballot counts over an exact lane bound, packed W2 finish on the pivot bin, "
    "bracket refinement and radix digits as fallbacks, blocked registers in, indices to smem out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n6w::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N6w_v2
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[IPT];

  static constexpr int IdxBits = n6w::ceil_log2(N);
  static constexpr int FIPT    = (IPT <= 16) ? 2 : 4; // finish list words per lane
  static constexpr int FIN_CAP = 32 * FIPT; // largest pivot bin the W2 finish takes
  static constexpr int MAXLEV  = 4; // bracket passes before the radix digit fallback
  static constexpr int M_raw   = n6w::next_pow2((K + 31) / 32);
  static constexpr int M       = (M_raw <= 8 && IPT % M_raw == 0) ? M_raw : 0; // 0: full range bracket

  struct TempStorage
  {
    uint32_t words[FIN_CAP];
  };

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    using namespace n6w;
    const int lane     = topk::group_lane<32>();
    const int idx_base = lane * IPT;
    const uint32_t lt  = lanemask_lt();

    float v[IPT];
    load_values<KeyT, IPT>(keys, v);

    // bracket: exact bounds on the k-th largest from the per-lane order statistics
    float mx, mth;
    lane_top<IPT, M>(v, mx, mth);
    range st;
    st.tmax = __reduce_max_sync(FULL, tw_of<KeyT>(mx));
    st.tmin = __reduce_min_sync(FULL, tw_of<KeyT>(mth));
    st.r    = K;
    st.n    = N;
    bracket_pass<KeyT, IPT, true>(v, st, lane);

#pragma unroll 1
    for (int level = 1;; ++level)
    {
      if (st.n == st.r || st.tmin == st.tmax)
      {
        break;
      }
      if (st.n <= FIN_CAP && packing_fits<KeyT, IdxBits>(st.tmin, st.tmax))
      {
        uint32_t tw[IPT];
        to_images<KeyT, IPT>(v, tw);
        w2_finish<KeyT, IPT, FIPT, IdxBits>(tw, st, K - st.r, ts.words, out_idx, lane, idx_base, lt);
        return;
      }
      if (level == MAXLEV)
      {
        break;
      }
      bracket_pass<KeyT, IPT, false>(v, st, lane);
    }

    uint32_t tw[IPT];
    to_images<KeyT, IPT>(v, tw);
    // last resort: radix digit passes over the candidates until the range is one key or exact
#pragma unroll 1
    while (st.n != st.r && __clz(st.tmin ^ st.tmax) < key_bits<KeyT>)
    {
      digit_pass<KeyT, IPT, 4, false>(tw, st, lane);
    }
    emit_range<IPT, true>(tw, st, out_idx, idx_blocked{idx_base}, lt);
  }
};
