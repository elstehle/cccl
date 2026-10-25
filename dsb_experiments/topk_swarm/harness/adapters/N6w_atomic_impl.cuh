// N6 at warp scope with a shared-memory atomic histogram instead of ballot counts. Not an adapter
// by itself, the N6w_atom<BINS> adapters instantiate it. Machinery in N6w_impl.cuh.
//
// The bracket pass computes the equal width bin of every item as in the ballot variant but counts
// with one fire-and-forget shared atomic per item into a per-warp histogram of BINS words (bin
// addresses are data dependent, so ptxas cannot aggregate them, T10 and T27, and the shared
// memory unit serializes lanes that hit the same word). Items outside the candidate range add to
// a per-lane dummy word so that the atomic stays unconditional (T32). BINS / 32 contiguous bins
// per lane feed the suffix scan, the pivot bin is located inside the pivot lane. Everything after
// the pass (exit, tie class, packed W2 finish, bracket refinement, radix digit fallback) is the
// ballot variant's code.
#pragma once

#include "N6w_impl.cuh"

namespace n6w
{

template <int BINS>
__device__ __forceinline__ uint32_t bin_image_n(float v, float lo, float scale)
{
  const float d   = v - lo;
  const int32_t x = __float_as_int(fmaf(d, scale, TWO23));
  const int32_t y = ::max(x, static_cast<int32_t>(IMG0));
  return ::min(static_cast<uint32_t>(y), IMG0 + (BINS - 1u));
}

// hist holds BINS counts (lane l owns bins l*BPL .. l*BPL+BPL-1), returns the pivot bin, its
// count and the candidates in the bins above it
template <int BINS>
__device__ __forceinline__ void locate_bins(const uint32_t* hist, int r, int lane, int& pbin, int& cnt_p, int& above)
{
  constexpr int BPL = BINS / 32;
  int h[BPL];
  int local = 0;
#pragma unroll
  for (int b = 0; b < BPL; ++b)
  {
    h[b] = static_cast<int>(hist[lane * BPL + b]);
    local += h[b];
  }
  const int s      = suffix_scan(local, lane);
  const unsigned bal = __ballot_sync(FULL, s >= r);
  const int p        = __popc(bal) - 1;
  // every lane resolves its own bins as if it were the pivot lane, lane p's result is taken
  int acc = s - local;
  int mybin = 0, myabove = acc, mycnt = 0;
#pragma unroll
  for (int b = BPL - 1; b >= 0; --b)
  {
    const int nxt  = acc + h[b];
    const bool hit = (nxt >= r) & (acc < r);
    mybin          = hit ? b : mybin;
    myabove        = hit ? acc : myabove;
    mycnt          = hit ? h[b] : mycnt;
    acc            = nxt;
  }
  const uint32_t w = __shfl_sync(FULL, (static_cast<uint32_t>(myabove) << 16) | static_cast<uint32_t>(mycnt), p);
  const int bsel   = __shfl_sync(FULL, mybin, p);
  pbin             = p * BPL + bsel;
  above            = static_cast<int>(w >> 16);
  cnt_p            = static_cast<int>(w & 0xffffu);
}

template <typename KeyT, int IPT, int BINS, bool ALL>
__device__ __forceinline__ void bracket_pass_atomic(const float (&v)[IPT], range& st, int lane, uint32_t* hist, uint32_t* dummy)
{
  constexpr int BPL = BINS / 32;
  static_assert(BPL >= 1 && BINS % 32 == 0, "bins per lane");
  const float lo    = fmaxf(untw(st.tmin), -FLT_MAX);
  const float hi    = fminf(untw(st.tmax), FLT_MAX);
  const float width = hi - lo;
  float scale       = static_cast<float>(BINS) / width;
  scale             = (scale > 0.0f && scale < FLT_MAX) ? scale : FLT_MIN; // see bracket_pass

#pragma unroll
  for (int b = 0; b < BPL; ++b)
  {
    hist[lane + 32 * b] = 0u;
  }
  __syncwarp();

  static_assert(BINS <= 256, "bin index packed into 8 bits");
  uint32_t ub[(IPT + 3) / 4]; // 8-bit bin per item, four per word
#pragma unroll
  for (int j = 0; j < (IPT + 3) / 4; ++j)
  {
    ub[j] = 0u;
  }
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const uint32_t x = bin_image_n<BINS>(v[i], lo, scale);
    const uint32_t b = x & (BINS - 1u);
    uint32_t* addr   = hist + b;
    if constexpr (!ALL)
    {
      const uint32_t t = tw_of<KeyT>(v[i]);
      const bool c     = (t >= st.tmin) & (t <= st.tmax);
      addr             = c ? addr : dummy + lane;
    }
    ub[i / 4] |= b << (8 * (i % 4));
    atomicAdd(addr, 1u);
  }
  __syncwarp();
  int pbin, np, above;
  locate_bins<BINS>(hist, st.r, lane, pbin, np, above);

  const uint32_t pb = static_cast<uint32_t>(pbin);
  uint32_t tmin = FULL, tmax = 0u;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const uint32_t t = tw_of<KeyT>(v[i]);
    bool c           = ((ub[i / 4] >> (8 * (i % 4))) & 0xffu) == pb;
    if constexpr (!ALL)
    {
      c = c & (t >= st.tmin) & (t <= st.tmax);
    }
    tmin = c ? ::min(tmin, t) : tmin;
    tmax = c ? ::max(tmax, t) : tmax;
  }
  st.tmin = __reduce_min_sync(FULL, tmin);
  st.tmax = __reduce_max_sync(FULL, tmax);
  st.r -= above;
  st.n = np;
  __syncwarp(); // the next pass zeroes the histogram
}

template <typename KeyT, int N, int K, int BINS>
struct atomic_primitive
{
  static constexpr int IPT     = N / 32;
  static constexpr int IdxBits = ceil_log2(N);
  static constexpr int FIPT    = (IPT <= 16) ? 2 : 4;
  static constexpr int FIN_CAP = 32 * FIPT;
  static constexpr int MAXLEV  = 4;
  static constexpr int M_raw   = next_pow2((K + 31) / 32);
  static constexpr int M       = (M_raw <= 8 && IPT % M_raw == 0) ? M_raw : 0;

  struct TempStorage
  {
    uint32_t hist[BINS];
    uint32_t dummy[32];
    uint32_t words[FIN_CAP];
  };

  __device__ __forceinline__ static void run(const KeyT (&keys)[IPT], TempStorage& ts, int* out_idx)
  {
    const int lane     = topk::group_lane<32>();
    const int idx_base = lane * IPT;
    const uint32_t lt  = lanemask_lt();

    float v[IPT];
    load_values<KeyT, IPT>(keys, v);
    float mx, mth;
    lane_top<IPT, M>(v, mx, mth);
    range st;
    st.tmax = __reduce_max_sync(FULL, tw_of<KeyT>(mx));
    st.tmin = __reduce_min_sync(FULL, tw_of<KeyT>(mth));
    st.r    = K;
    st.n    = N;
    bracket_pass_atomic<KeyT, IPT, BINS, true>(v, st, lane, ts.hist, ts.dummy);

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
      bracket_pass_atomic<KeyT, IPT, BINS, false>(v, st, lane, ts.hist, ts.dummy);
    }

    uint32_t tw[IPT];
    to_images<KeyT, IPT>(v, tw);
#pragma unroll 1
    while (st.n != st.r && __clz(st.tmin ^ st.tmax) < key_bits<KeyT>)
    {
      digit_pass<KeyT, IPT, 4, false>(tw, st, lane);
    }
    emit_range<IPT, true>(tw, st, out_idx, idx_blocked{idx_base}, lt);
  }
};

} // namespace n6w
