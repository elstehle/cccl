// B4: exact-bound prefilter, compaction and a small finish at block scope, index-only output.
//
// The minimum over the threads of the per-thread maximum is an exact lower bound on the k-th
// largest key for every k at most Threads (Threads items are at or above it), and it costs one
// max per item plus one warp reduction. Items at or above the bound are a small fraction of a
// large tile (about 1500 of 32768 on random input), so the first pass of a radix or bucket
// selection, which sweeps every item with a shared memory atomic, is replaced by a compare sweep
// that compacts the candidates into shared memory. The finish then runs over a few candidates per
// thread: one pass over the fixed-point image of the candidates (TECHNIQUES.md T25, N6's digit)
// leaves a pivot bin of a handful of items, which one warp resolves exactly by redux rounds on
// (key, index) words (the W2 round, T3 and T19). Positions come from packed block scans (T29),
// never from tickets.
//
// For k above Threads the per-thread items are split into Groups groups and the bound is the
// minimum over all Threads x Groups group maxima (Threads x Groups items are at or above it), one
// max per item as before. The bound is accepted at run time only when the candidate count is at
// least k and fits the candidate buffer, otherwise the selection runs over the register tile with
// the same pass machinery on the key bits (the exact fallback).
//
// Exactness: the pivot set after any pass is narrowed by further digits until one of three exits
// holds. (1) The pivot set is exactly the remaining k. (2) The pivot set is a tie class (its
// tracked key minimum equals its maximum): under not_guaranteed any k_rem of them are taken, under
// prefer_smaller_index the selection continues on the inverted index bits, which ends with exit 1.
// (3) The pivot set has at most 32 items: one warp extracts the remaining k by (key descending,
// index ascending), exact under both requirements. The digits are the three image digits (8, 8, 7
// bits), then the key bytes from the highest differing byte of the pivot set's tracked key min and
// max, then the index bits.
//
// Contract: block scope, Threads a multiple of 32 (128, 256, 512 tested in the harness), full
// striped tile of N = Threads x ItemsPerThread keys in registers (item i of thread t is input
// item i x Threads + t), N at most 32768 (16-bit count fields in the packed scans), k in [1, N), keys fp32 or bf16,
// output k source indices in the caller's shared array in unspecified order, requirements
// not_guaranteed and prefer_smaller_index (one code path, the tie-class exit differs). -0.0 is
// folded onto +0.0 only under prefer_smaller_index (the two zeros are one tie class there). NaN:
// a NaN with the sign bit clear orders above +inf and is selected first, a NaN with the sign bit
// set orders below -inf, the all-ones bit pattern collides with the empty-slot sentinel of the
// warp finish and is unsupported. Focus-set inputs contain no NaN. Sync: __syncthreads inside,
// consecutive calls that reuse the temp storage must be separated by a barrier (the harness does).
#pragma once

#include <cub/block/block_scan.cuh>

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cuda_bf16.h>

// Optional phase profile: with B4_PROFILE defined, thread 0 writes clock64 stamps to b4::g_stamps
// (16 slots per call), used by approaches/B4/profile_phases.cu only. Compiled out otherwise.
#ifdef B4_PROFILE
#  define B4_STAMP(i)                                   \
    do                                                  \
    {                                                   \
      if (threadIdx.x == 0)                             \
      {                                                 \
        ::b4::g_stamps[(i)] = clock64();                \
      }                                                 \
    } while (0)
#else
#  define B4_STAMP(i) \
    do                \
    {                 \
    } while (0)
#endif

namespace b4
{

#ifdef B4_PROFILE
__device__ long long g_stamps[16];
#endif

using u32 = ::cuda::std::uint32_t;

enum class tie_break
{
  any,
  smaller_index
};

__host__ __device__ constexpr int ceil_div(int a, int b)
{
  return (a + b - 1) / b;
}
__host__ __device__ constexpr int ceil_log2(int x)
{
  int r = 0;
  while ((1 << r) < x)
  {
    ++r;
  }
  return r;
}

// Order-preserving unsigned images of the key bits (ascending image is ascending key)
template <typename KeyT>
struct key_ops;

template <>
struct key_ops<float>
{
  static constexpr int bits = 32;
  template <bool NormalizeZero>
  __device__ __forceinline__ static u32 twiddle(float f)
  {
    if constexpr (NormalizeZero)
    {
      f = __fadd_rn(f, 0.0f); // -0.0 + 0.0 is +0.0 under round to nearest
    }
    const u32 b = __float_as_uint(f);
    const u32 m = static_cast<u32>(static_cast<int>(b) >> 31) | 0x80000000u;
    return b ^ m;
  }
  __device__ __forceinline__ static float value(u32 t)
  {
    const u32 m = (static_cast<int>(t) >= 0) ? 0xFFFFFFFFu : 0x80000000u;
    return __uint_as_float(t ^ m);
  }
};

template <>
struct key_ops<__nv_bfloat16>
{
  static constexpr int bits = 16;
  template <bool NormalizeZero>
  __device__ __forceinline__ static u32 twiddle(__nv_bfloat16 h)
  {
    u32 b = __bfloat16_as_ushort(h);
    if constexpr (NormalizeZero)
    {
      b = (b == 0x8000u) ? 0u : b;
    }
    const u32 m = (b & 0x8000u) ? 0xFFFFu : 0x8000u;
    return b ^ m;
  }
  // the fp32 value of a 16-bit image (exact, bf16 is the top half of fp32)
  __device__ __forceinline__ static float value(u32 t16)
  {
    return key_ops<float>::value(t16 << 16);
  }
};

template <typename KeyT,
          int Threads,
          int ItemsPerThread,
          int Cap,
          int Groups,
          tie_break TB = tie_break::any>
class block_topk_bound
{
public:
  static constexpr int threads    = Threads;
  static constexpr int ipt        = ItemsPerThread;
  static constexpr int tile       = Threads * ItemsPerThread;
  static constexpr int warps      = Threads / 32;
  static constexpr int cap        = (Cap < tile) ? Cap : tile; // candidates the buffer holds
  static constexpr int smax       = ceil_div(cap, Threads); // candidate slots per thread
  static constexpr int groups     = (Groups < ipt) ? Groups : ipt;
  static constexpr int per_group  = ipt / groups;
  static constexpr bool normalize = (TB == tie_break::smaller_index);
  using ko                        = key_ops<KeyT>;
  static constexpr int key_bits   = ko::bits;
  static constexpr bool packed    = (key_bits == 16); // key and inverted index in one word
  static constexpr int idx_bits   = ceil_log2(tile);
  static constexpr u32 inv_mask   = (1u << idx_bits) - 1u; // inv = inv_mask - idx
  static constexpr int num_bins   = 256;
  static constexpr int bpt        = ceil_div(num_bins, Threads);
  static constexpr int fin        = 32; // pivot sets up to this size finish on one warp
  static constexpr u32 u_lo       = 0x4B000000u; // bits(2^23), the image occupies [u_lo, u_lo + 2^23)
  using word_t                    = ::cuda::std::conditional_t<packed, u32, uint2>;
  using scan_t                    = cub::BlockScan<u32, Threads, cub::BLOCK_SCAN_WARP_SCANS>;

  static_assert(Threads % 32 == 0, "whole warps");
  static_assert(ipt % groups == 0, "the group split must be even");
  static_assert(tile <= 32768, "16-bit count fields in the packed scans");
  static_assert(!packed || idx_bits <= 16, "bf16 packs the inverted index into 16 bits");

  struct TempStorage
  {
    u32 hist[2][num_bins];
    u32 kmin[2][num_bins];
    u32 kmax[2][num_bins];
    typename scan_t::TempStorage scan;
    u32 bracket[warps][2]; // per warp: min of the group maxima, max of the items
    u32 state[5]; // selected above, pivot bucket, pivot count, pivot key min, pivot key max
    word_t pivot[fin];
    word_t cands[cap];
  };

  __device__ __forceinline__ block_topk_bound(TempStorage& s)
      : storage(s)
      , tid(static_cast<int>(threadIdx.x))
  {}

  // Writes the k source indices of the k largest keys of the striped tile to out[0, k).
  __device__ __forceinline__ void select(const KeyT (&keys)[ipt], int k, int* out)
  {
    const int warp = tid / 32;
    const int lane = tid % 32;
    B4_STAMP(0);

    // Prologue: key images, group maxima, warp partials of the bracket
    u32 tk[ipt];
#pragma unroll
    for (int i = 0; i < ipt; ++i)
    {
      tk[i] = ko::template twiddle<normalize>(keys[i]);
    }
    u32 gmax[groups];
#pragma unroll
    for (int g = 0; g < groups; ++g)
    {
      gmax[g] = tk[g * per_group];
#pragma unroll
      for (int i = 1; i < per_group; ++i)
      {
        gmax[g] = max(gmax[g], tk[g * per_group + i]);
      }
    }
    u32 gmin = gmax[0];
    u32 tmax = gmax[0];
#pragma unroll
    for (int g = 1; g < groups; ++g)
    {
      gmin = min(gmin, gmax[g]);
      tmax = max(tmax, gmax[g]);
    }
    const u32 w_lo = __reduce_min_sync(0xffffffffu, gmin);
    const u32 w_hi = __reduce_max_sync(0xffffffffu, tmax);
    if (lane == 0)
    {
      storage.bracket[warp][0] = w_lo;
      storage.bracket[warp][1] = w_hi;
    }
    reset_bins(0);
    __syncthreads();
    B4_STAMP(1);

    u32 lo = storage.bracket[0][0];
    u32 hi = storage.bracket[0][1];
#pragma unroll
    for (int w = 1; w < warps; ++w)
    {
      lo = min(lo, storage.bracket[w][0]);
      hi = max(hi, storage.bracket[w][1]);
    }

    // Candidate count and positions
    u32 cnt = 0;
#pragma unroll
    for (int i = 0; i < ipt; ++i)
    {
      cnt += (tk[i] >= lo) ? 1u : 0u;
    }
    u32 pos, total;
    scan_t(storage.scan).ExclusiveSum(cnt, pos, total);
    const int c = static_cast<int>(total);
    B4_STAMP(2);

    if (c <= cap && c >= k)
    {
      // Compaction in thread order, then every thread takes the striped slots j x Threads + tid
#pragma unroll
      for (int i = 0; i < ipt; ++i)
      {
        if (tk[i] >= lo)
        {
          storage.cands[pos] = make_word(tk[i], inv_of(i));
          ++pos;
        }
      }
      __syncthreads();
      B4_STAMP(3);
      const int slots = ceil_div_rt(c, Threads);
      u32 ctk[smax];
      u32 cinv[smax];
#pragma unroll
      for (int j = 0; j < smax; ++j)
      {
        const int q = j * Threads + tid;
        word_t w    = make_word(0u, 0u);
        if (j < slots && q < c)
        {
          w = storage.cands[q];
        }
        ctk[j]  = word_key(w);
        cinv[j] = word_inv(w);
      }
      cand_slots<smax> sl{ctk, cinv, c, slots};
      B4_STAMP(4);
      finish<smax, true>(sl, lo, hi, k, out);
    }
    else
    {
      tile_slots sl{tk, tid};
      finish<ipt, false>(sl, lo, hi, k, out);
    }
  }

private:
  TempStorage& storage;
  int tid;

  struct state_t
  {
    u32 um, up; // image prefix (mask, value)
    u32 km, kp; // key prefix
    u32 im, ip; // inverted index prefix
    int k_rem; // still to select inside the pivot set
    int sel_total; // selected above the pivot set so far (output slots [0, sel_total))
    int count; // items in the pivot set
    u32 kmin, kmax; // tracked key image min and max of the pivot set (not valid after an index pass)
    int pass;
    float lo_v, scale; // image parameters
  };

  enum class source
  {
    image,
    key,
    index
  };

  enum class emit_mode
  {
    all, // the pivot set is selected entirely
    tie_any, // the pivot set is a tie class, the first k_rem in scan order are taken
    w2 // the pivot set goes to the warp finish
  };

  // Register-resident sets the passes run over. tk(j), inv(j), valid(j); n is the uniform number
  // of live slots (the loops are unrolled over M and skip slots at or beyond n uniformly).
  template <int M>
  struct cand_slots
  {
    const u32 (&ctk)[M];
    const u32 (&cinv)[M];
    int count;
    int n;
    __device__ __forceinline__ u32 tk(int j) const
    {
      return ctk[j];
    }
    __device__ __forceinline__ u32 inv(int j) const
    {
      return cinv[j];
    }
    __device__ __forceinline__ bool valid(int j) const
    {
      return j * Threads + static_cast<int>(threadIdx.x) < count;
    }
  };
  struct tile_slots
  {
    const u32 (&ttk)[ipt];
    int tid_;
    static constexpr int n = ipt;
    __device__ __forceinline__ u32 tk(int j) const
    {
      return ttk[j];
    }
    __device__ __forceinline__ u32 inv(int j) const
    {
      return inv_mask - static_cast<u32>(j * Threads + tid_);
    }
    __device__ __forceinline__ bool valid(int) const
    {
      return true;
    }
  };

  __device__ __forceinline__ static int ceil_div_rt(int a, int b)
  {
    return (a + b - 1) / b;
  }
  __device__ __forceinline__ u32 inv_of(int i) const
  {
    return inv_mask - static_cast<u32>(i * Threads + tid);
  }
  __device__ __forceinline__ static word_t make_word(u32 tk, u32 inv)
  {
    if constexpr (packed)
    {
      return (tk << 16) | inv;
    }
    else
    {
      return make_uint2(tk, inv);
    }
  }
  __device__ __forceinline__ static u32 word_key(word_t w)
  {
    if constexpr (packed)
    {
      return w >> 16;
    }
    else
    {
      return w.x;
    }
  }
  __device__ __forceinline__ static u32 word_inv(word_t w)
  {
    if constexpr (packed)
    {
      return w & 0xFFFFu;
    }
    else
    {
      return w.y;
    }
  }

  // fixed-point image of a key over the bracket (T25): bits(round((v - lo) scale + 2^23)) clamped
  // into [u_lo, u_lo + 2^23), monotone in v
  __device__ __forceinline__ static u32 image(u32 tk, float lo_v, float scale)
  {
    const float f = fmaf(ko::value(tk) - lo_v, scale, 8388608.0f);
    const int b   = __float_as_int(f);
    const int c   = min(max(b, static_cast<int>(u_lo)), static_cast<int>(u_lo + 0x7FFFFFu));
    return static_cast<u32>(c);
  }
  // the compiler must not keep one image per slot live across passes (T24)
  __device__ __forceinline__ static void opaque(float& lo, float& scale)
  {
    asm volatile("" : "+f"(lo), "+f"(scale));
  }

  __device__ __forceinline__ void reset_bins(int buf)
  {
#pragma unroll
    for (int b = tid; b < num_bins; b += Threads)
    {
      storage.hist[buf][b] = 0u;
      storage.kmin[buf][b] = 0xFFFFFFFFu;
      storage.kmax[buf][b] = 0u;
    }
  }

  // Exclusive scan over the bins (bin 0 holds the largest digit), the thread whose bin contains
  // the k_rem-th item records the pass state, exactly one does.
  __device__ __forceinline__ void scan_and_choose(int buf, int k_rem)
  {
    u32 counts[bpt];
    u32 excl[bpt];
#pragma unroll
    for (int i = 0; i < bpt; ++i)
    {
      const int b = tid * bpt + i;
      counts[i]   = (b < num_bins) ? storage.hist[buf][b] : 0u;
    }
    scan_t(storage.scan).ExclusiveSum(counts, excl);
#pragma unroll
    for (int i = 0; i < bpt; ++i)
    {
      const int b   = tid * bpt + i;
      const u32 e   = excl[i];
      const u32 inc = e + counts[i];
      if (b < num_bins && e < static_cast<u32>(k_rem) && inc >= static_cast<u32>(k_rem))
      {
        storage.state[0] = e;
        storage.state[1] = static_cast<u32>(b);
        storage.state[2] = counts[i];
        storage.state[3] = storage.kmin[buf][b];
        storage.state[4] = storage.kmax[buf][b];
      }
    }
  }

  template <bool HasImage, class Slots>
  __device__ __forceinline__ bool in_pivot(const Slots& sl, int j, const state_t& st, float lo_v, float scale) const
  {
    bool p = ((sl.tk(j) & st.km) == st.kp) & ((sl.inv(j) & st.im) == st.ip);
    if constexpr (HasImage)
    {
      p = p & ((image(sl.tk(j), lo_v, scale) & st.um) == st.up);
    }
    return p;
  }

  // One pass: histogram of digit [shift, shift + width) of the source word over the valid pivot
  // items, key min and max per bin, scan and pivot choice, state update.
  template <int M, bool HasImage, source Src, class Slots>
  __device__ __forceinline__ void run_pass(const Slots& sl, int shift, int width, state_t& st)
  {
    const int buf        = st.pass & 1;
    const u32 digit_mask = (1u << width) - 1u;
    float lo_v = st.lo_v, scale = st.scale;
    opaque(lo_v, scale);
#pragma unroll
    for (int j = 0; j < M; ++j)
    {
      if (j < sl.n)
      {
        if (sl.valid(j) && in_pivot<HasImage>(sl, j, st, lo_v, scale))
        {
          u32 w;
          if constexpr (Src == source::image)
          {
            w = image(sl.tk(j), lo_v, scale);
          }
          else if constexpr (Src == source::key)
          {
            w = sl.tk(j);
          }
          else
          {
            w = sl.inv(j);
          }
          const u32 d = (w >> shift) & digit_mask;
          const int b = static_cast<int>(digit_mask - d);
          atomicAdd_block(&storage.hist[buf][b], 1u);
          atomicMin_block(&storage.kmin[buf][b], sl.tk(j));
          atomicMax_block(&storage.kmax[buf][b], sl.tk(j));
        }
      }
    }
    reset_bins(buf ^ 1);
    __syncthreads();
    scan_and_choose(buf, st.k_rem);
    __syncthreads();
    const int sel   = static_cast<int>(storage.state[0]);
    const u32 b     = storage.state[1];
    st.count        = static_cast<int>(storage.state[2]);
    st.kmin         = storage.state[3];
    st.kmax         = storage.state[4];
    st.k_rem -= sel;
    st.sel_total += sel;
    const u32 d = digit_mask - b;
    if constexpr (Src == source::image)
    {
      st.up |= d << shift;
      st.um |= digit_mask << shift;
    }
    else if constexpr (Src == source::key)
    {
      st.kp |= d << shift;
      st.km |= digit_mask << shift;
    }
    else
    {
      st.ip |= d << shift;
      st.im |= digit_mask << shift;
    }
    ++st.pass;
    B4_STAMP(5 + st.pass);
  }

  // Emission over the set: items lexicographically above the pivot prefix go to out[0, sel_total)
  // in scan order, the pivot items go where the mode says. Then the warp finish when asked.
  template <int M, bool HasImage, class Slots>
  __device__ __forceinline__ void emit(const Slots& sl, const state_t& st, emit_mode mode, int* out)
  {
    const bool all = (mode == emit_mode::all);
    float lo_v = st.lo_v, scale = st.scale;
    opaque(lo_v, scale);
    u32 n_sel = 0;
    u32 n_piv = 0;
#pragma unroll
    for (int j = 0; j < M; ++j)
    {
      if (j < sl.n)
      {
        bool above, piv;
        classify<HasImage>(sl, j, st, lo_v, scale, above, piv);
        const bool v = sl.valid(j);
        n_sel += (v && (above || (piv && all))) ? 1u : 0u;
        n_piv += (v && piv && !all) ? 1u : 0u;
      }
    }
    u32 excl;
    B4_STAMP(10);
    scan_t(storage.scan).ExclusiveSum((n_sel << 16) | n_piv, excl);
    B4_STAMP(11);
    u32 pos_sel = excl >> 16;
    u32 pos_piv = excl & 0xFFFFu;
    const u32 need = static_cast<u32>(st.k_rem);
#pragma unroll
    for (int j = 0; j < M; ++j)
    {
      if (j < sl.n)
      {
        bool above, piv;
        classify<HasImage>(sl, j, st, lo_v, scale, above, piv);
        const bool v   = sl.valid(j);
        const bool s   = v && (above || (piv && all));
        const bool p   = v && piv && !all;
        const int idxv = static_cast<int>(inv_mask - sl.inv(j));
        if (s)
        {
          out[pos_sel] = idxv;
        }
        if (p)
        {
          if (mode == emit_mode::w2)
          {
            storage.pivot[pos_piv] = make_word(sl.tk(j), sl.inv(j));
          }
          else if (pos_piv < need)
          {
            out[st.sel_total + static_cast<int>(pos_piv)] = idxv;
          }
        }
        pos_sel += s ? 1u : 0u;
        pos_piv += p ? 1u : 0u;
      }
    }
    B4_STAMP(12);
    if (mode == emit_mode::w2)
    {
      __syncthreads();
      B4_STAMP(13);
      if (tid < 32)
      {
        warp_finish(st.count, st.k_rem, out + st.sel_total);
      }
    }
    B4_STAMP(14);
  }

  template <bool HasImage, class Slots>
  __device__ __forceinline__ static void
  classify(const Slots& sl, int j, const state_t& st, float lo_v, float scale, bool& above, bool& piv)
  {
    const u32 kk  = sl.tk(j) & st.km;
    const u32 ii  = sl.inv(j) & st.im;
    const bool gk = kk > st.kp;
    const bool ek = kk == st.kp;
    const bool gi = ii > st.ip;
    const bool ei = ii == st.ip;
    bool a        = gk | (ek & gi);
    bool p        = ek & ei;
    if constexpr (HasImage)
    {
      const u32 uu  = image(sl.tk(j), lo_v, scale) & st.um;
      const bool gu = uu > st.up;
      const bool eu = uu == st.up;
      a             = gu | (eu & a);
      p             = eu & p;
    }
    above = a;
    piv   = p;
  }

  // Warp 0: count pivot words in storage.pivot (at most 32), k_rem rounds of redux extraction
  // by (key descending, index ascending), the r-th winner's index to out[r].
  __device__ __forceinline__ void warp_finish(int count, int k_rem, int* out)
  {
    const int lane = tid;
    const bool has = lane < count;
    if constexpr (packed)
    {
      u32 w = has ? storage.pivot[lane] : 0u;
      for (int r = 0; r < k_rem; ++r)
      {
        const u32 m = __reduce_max_sync(0xffffffffu, w);
        if (lane == (r & 31))
        {
          out[r] = static_cast<int>(inv_mask - (m & 0xFFFFu));
        }
        w = (w == m) ? 0u : w;
      }
    }
    else
    {
      const uint2 pw = has ? storage.pivot[lane] : make_uint2(0u, 0u);
      u32 tk         = pw.x;
      const u32 inv  = pw.y;
      bool alive     = has;
      for (int r = 0; r < k_rem; ++r)
      {
        const u32 m   = __reduce_max_sync(0xffffffffu, alive ? tk : 0u);
        const bool hm = alive && (tk == m);
        const u32 i   = __reduce_max_sync(0xffffffffu, hm ? inv : 0u);
        if (lane == (r & 31))
        {
          out[r] = static_cast<int>(inv_mask - i);
        }
        alive = alive && !(hm && inv == i);
      }
    }
  }

  // The pass sequence with its exits, over a register-resident set. HasImage: start with the
  // three image digits over the bracket [lo, hi] (the candidates), else with the key bytes (the
  // tile fallback).
  template <int M, bool HasImage, class Slots>
  __device__ __forceinline__ void finish(const Slots& sl, u32 lo, u32 hi, int k, int* out)
  {
    state_t st;
    st.um = st.up = st.km = st.kp = st.im = st.ip = 0u;
    st.k_rem     = k;
    st.sel_total = 0;
    st.count     = 0;
    st.kmin      = 0u;
    st.kmax      = 0xFFFFFFFFu;
    st.pass      = 0;
    constexpr float flt_max = 3.4028234663852886e38f;
    const float hi_v        = fminf(ko::value(hi), flt_max);
    const float lo_v        = fmaxf(ko::value(lo), -flt_max);
    const float width       = hi_v - lo_v;
    st.lo_v                 = lo_v;
    st.scale = (width > 0.0f && width < flt_max) ? (8388600.0f / width) : (width > 0.0f ? 1.0e-32f : 1.0f);

    bool done        = false;
    bool index_phase = false;
    emit_mode mode   = emit_mode::all;
    // The exits after a pass. The emission runs once, after the pass sequence, in the mode chosen.
    auto check = [&]() {
      if (st.count == st.k_rem)
      {
        mode = emit_mode::all;
        done = true;
      }
      else if (!index_phase && st.kmin == st.kmax)
      {
        if constexpr (TB == tie_break::any)
        {
          mode = emit_mode::tie_any;
          done = true;
        }
        else
        {
          index_phase = true;
        }
      }
      else if (st.count <= fin)
      {
        mode = emit_mode::w2;
        done = true;
      }
    };

    if constexpr (HasImage)
    {
      run_pass<M, HasImage, source::image>(sl, 15, 8, st);
      check();
      if (!done && !index_phase)
      {
        run_pass<M, HasImage, source::image>(sl, 7, 8, st);
        check();
      }
      if (!done && !index_phase)
      {
        run_pass<M, HasImage, source::image>(sl, 0, 7, st);
        check();
      }
    }
    if (!done && !index_phase)
    {
      // key bytes from the highest differing byte of the pivot set's tracked min and max
      int byte = HasImage ? (31 - ::cuda::std::countl_zero(st.kmin ^ st.kmax)) / 8 : (key_bits / 8 - 1);
      for (; byte >= 0 && !done && !index_phase; --byte)
      {
        run_pass<M, HasImage, source::key>(sl, byte * 8, 8, st);
        check();
      }
    }
    if constexpr (TB == tie_break::smaller_index)
    {
      if (!done)
      {
        // index digits: a tie class larger than the warp finish is ranked by index
        index_phase = true;
        int shift   = idx_bits;
        while (!done && shift > 0)
        {
          const int w = (shift >= 8) ? 8 : shift;
          shift -= w;
          run_pass<M, HasImage, source::index>(sl, shift, w, st);
          check();
        }
      }
    }
    emit<M, HasImage>(sl, st, mode, out);
  }
};

} // namespace b4
