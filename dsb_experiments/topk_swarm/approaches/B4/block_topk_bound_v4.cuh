// B4 v4: exact-bound prefilter, compaction and a small finish at block scope, index-only output.
// Same algorithm and contract as block_topk_bound.cuh (B4_base). Changes of v4 over v3, after the
// phase profile of v3:
//   - the pass histogram and the emission tickets are branch-free: every slot issues its shared
//     atomic, to the bin when the slot takes part and to a private dummy word of the thread
//     otherwise (T32's proposal), so the slots of a group pipeline instead of waiting on each
//     other's reconvergence region,
//   - the candidate loads are guarded per group of four slots (uniform), so the slots above the
//     live count are neither loaded nor imaged.
// Changes of v3 over v2:
//   - the register sweeps (thread maxima, candidate count, compaction stores) run on the fp32
//     value of the key, no twiddle sweep over the tile; the candidate word holds the raw key bits
//     and the inverted index, the candidates are twiddled (and -0.0 folded under
//     prefer_smaller_index) when they are loaded, a few per thread,
//   - the candidate loads are unconditional from a clamped address, no branch per slot,
//   - when one pass decided the selection (the common case) the emission takes the position of a
//     selected item from the exclusive bin prefix of that pass plus a per-bin ticket (one shared
//     atomic with return on a data-dependent address), no classify sweep, no scan and no barrier;
//     the bitmask and scan emission of v2 remains for the multi-pass states.
// Changes of v2 over the first build:
//   - the fixed-point image of a candidate is computed once when the candidate is loaded from
//     the buffer, the passes and the emission read a register, no float arithmetic per slot and
//     pass,
//   - the passes issue one histogram atomic per pivot item, no key min and max tracking; the
//     tie-class test runs only when a pivot set is too large for the warp finish and not exactly
//     the remaining k, as a block min and max over the pivot items (two redux per warp, one
//     barrier),
//   - the slot loops run in uniform groups of four slots instead of one uniform branch per slot,
//     so the slots of a group overlap and the dead groups are skipped,
//   - the emission classifies each slot once into two bitmasks and stores from the bits,
//   - the candidate count sweep uses four accumulators.
// The exits after a pass are, in this order: the pivot set is exactly the remaining k, the pivot
// set has at most 32 items (warp finish, exact under both requirements), the pivot set is a tie
// class (any k_rem under not_guaranteed, index passes under prefer_smaller_index), else the next
// digit. See block_topk_bound.cuh for the contract and the NaN and signed-zero behavior.
#pragma once

#include <cub/block/block_scan.cuh>

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cuda_bf16.h>

#ifdef B4_PROFILE
#  define B4V4_STAMP(i)                                 \
    do                                                  \
    {                                                   \
      if (threadIdx.x == 0)                             \
      {                                                 \
        ::b4v4::g_stamps[(i)] = clock64();              \
      }                                                 \
    } while (0)
#else
#  define B4V4_STAMP(i) \
    do                  \
    {                   \
    } while (0)
#endif

namespace b4v4
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
      f = __fadd_rn(f, 0.0f);
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
  static constexpr int cap        = (Cap < tile) ? Cap : tile;
  static constexpr int smax       = ceil_div(cap, Threads);
  static constexpr int groups     = (Groups < ipt) ? Groups : ipt;
  static constexpr int per_group  = ipt / groups;
  static constexpr bool normalize = (TB == tie_break::smaller_index);
  using ko                        = key_ops<KeyT>;
  static constexpr int key_bits   = ko::bits;
  static constexpr bool packed    = (key_bits == 16);
  static constexpr int idx_bits   = ceil_log2(tile);
  static constexpr u32 inv_mask   = (1u << idx_bits) - 1u;
  static constexpr int num_bins   = 256;
  static constexpr int bpt        = ceil_div(num_bins, Threads);
  static constexpr int fin        = 32;
  static constexpr int sg         = 4; // slots per uniform group
  static constexpr u32 u_lo       = 0x4B000000u;
  using word_t                    = ::cuda::std::conditional_t<packed, u32, uint2>;
  using scan_t                    = cub::BlockScan<u32, Threads, cub::BLOCK_SCAN_WARP_SCANS>;

  static_assert(Threads % 32 == 0, "whole warps");
  static_assert(ipt % groups == 0, "the group split must be even");
  static_assert(tile <= 32768, "16-bit count fields in the packed scans");
  static_assert(!packed || idx_bits <= 16, "bf16 packs the inverted index into 16 bits");

  struct TempStorage
  {
    u32 hist[2][num_bins];
    typename scan_t::TempStorage scan;
    u32 bracket[warps][2];
    u32 red[warps][2]; // lazy pivot min and max partials
    u32 state[3]; // selected above, pivot bucket, pivot count
    u32 binpos[num_bins]; // exclusive bin prefix of the last pass
    u32 dummy[Threads]; // private target of the atomics of slots that take no part
    word_t pivot[fin];
    word_t cands[cap];
  };

  __device__ __forceinline__ block_topk_bound(TempStorage& s)
      : storage(s)
      , tid(static_cast<int>(threadIdx.x))
  {}

  __device__ __forceinline__ void select(const KeyT (&keys)[ipt], int k, int* out)
  {
    const int warp = tid / 32;
    const int lane = tid % 32;
    B4V4_STAMP(0);

    // Prologue on the fp32 value of the key (bf16 widened exactly), group maxima, warp partials
    float v[ipt];
#pragma unroll
    for (int i = 0; i < ipt; ++i)
    {
      v[i] = key_value(keys[i]);
    }
    float gmax[groups];
#pragma unroll
    for (int g = 0; g < groups; ++g)
    {
      gmax[g] = v[g * per_group];
#pragma unroll
      for (int i = 1; i < per_group; ++i)
      {
        gmax[g] = fmaxf(gmax[g], v[g * per_group + i]);
      }
    }
    float gmin = gmax[0];
    float tmax = gmax[0];
#pragma unroll
    for (int g = 1; g < groups; ++g)
    {
      gmin = fminf(gmin, gmax[g]);
      tmax = fmaxf(tmax, gmax[g]);
    }
    const u32 w_lo = __reduce_min_sync(0xffffffffu, key_ops<float>::twiddle<false>(gmin));
    const u32 w_hi = __reduce_max_sync(0xffffffffu, key_ops<float>::twiddle<false>(tmax));
    if (lane == 0)
    {
      storage.bracket[warp][0] = w_lo;
      storage.bracket[warp][1] = w_hi;
    }
    zero_hist(0);
    __syncthreads();
    B4V4_STAMP(1);

    u32 lo_t = storage.bracket[0][0];
    u32 hi_t = storage.bracket[0][1];
#pragma unroll
    for (int w = 1; w < warps; ++w)
    {
      lo_t = min(lo_t, storage.bracket[w][0]);
      hi_t = max(hi_t, storage.bracket[w][1]);
    }
    const float lo_v = key_ops<float>::value(lo_t);

    // Candidate count (four independent accumulators) and positions. The compare is in the
    // value domain, so -0.0 and +0.0 are equal here and a candidate may sit below the twiddled
    // bound, which the finish does not rely on.
    u32 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
    for (int i = 0; i < ipt; i += 4)
    {
      c0 += (v[i] >= lo_v) ? 1u : 0u;
      if (i + 1 < ipt)
      {
        c1 += (v[i + 1] >= lo_v) ? 1u : 0u;
      }
      if (i + 2 < ipt)
      {
        c2 += (v[i + 2] >= lo_v) ? 1u : 0u;
      }
      if (i + 3 < ipt)
      {
        c3 += (v[i + 3] >= lo_v) ? 1u : 0u;
      }
    }
    const u32 cnt = (c0 + c1) + (c2 + c3);
    u32 pos, total;
    scan_t(storage.scan).ExclusiveSum(cnt, pos, total);
    const int c = static_cast<int>(total);
    B4V4_STAMP(2);

    if (c <= cap && c >= k)
    {
#pragma unroll
      for (int i = 0; i < ipt; ++i)
      {
        if (v[i] >= lo_v)
        {
          storage.cands[pos] = make_raw_word(__float_as_uint(v[i]), inv_of(i));
          ++pos;
        }
      }
      __syncthreads();
      B4V4_STAMP(3);
      constexpr float flt_max = 3.4028234663852886e38f;
      const float hi_v        = fminf(key_ops<float>::value(hi_t), flt_max);
      const float lo_c        = fmaxf(lo_v, -flt_max);
      const float width       = hi_v - lo_c;
      const float scale =
        (width > 0.0f && width < flt_max) ? __fdividef(8388600.0f, width) : (width > 0.0f ? 1.0e-32f : 1.0f);
      const int slots = ceil_div_rt(c, Threads);
      u32 ctk[smax];
      u32 cinv[smax];
      u32 cu[smax];
#pragma unroll
      for (int g = 0; g < smax; g += sg)
      {
        if (g < slots)
        {
#pragma unroll
          for (int j = g; j < g + sg && j < smax; ++j)
          {
            const int q      = j * Threads + tid;
            const bool valid = q < c;
            const word_t w   = storage.cands[valid ? q : (c - 1)];
            const u32 raw    = raw_bits(w);
            ctk[j]           = valid ? ko::template twiddle<normalize>(key_from_bits(raw)) : 0u;
            cinv[j]          = valid ? word_inv(w) : 0u;
            cu[j]            = image_of_value(__uint_as_float(raw), lo_c, scale);
          }
        }
        else
        {
#pragma unroll
          for (int j = g; j < g + sg && j < smax; ++j)
          {
            ctk[j]  = 0u;
            cinv[j] = 0u;
            cu[j]   = 0u;
          }
        }
      }
      cand_slots<smax> sl{ctk, cinv, cu, c, slots};
      B4V4_STAMP(4);
      finish<smax, true>(sl, k, out);
    }
    else
    {
      // the tile fallback twiddles from the fp32 values, so the keys die after the widen
      u32 tk[ipt];
#pragma unroll
      for (int i = 0; i < ipt; ++i)
      {
        tk[i] = ko::template twiddle<normalize>(key_from_bits(__float_as_uint(v[i])));
      }
      tile_slots sl{tk, tid};
      finish<ipt, false>(sl, k, out);
    }
  }

private:
  TempStorage& storage;
  int tid;

  struct state_t
  {
    u32 um, up;
    u32 km, kp;
    u32 im, ip;
    int k_rem;
    int sel_total;
    int count;
    u32 kmin, kmax; // pivot key min and max, valid after pivot_minmax
    int pass;
    int last_src; // 0 image, 1 key, 2 index
    int last_shift;
    u32 last_mask;
    int pivot_bin;
  };

  enum class source
  {
    image,
    key,
    index
  };

  enum class emit_mode
  {
    all,
    tie_any,
    w2
  };

  template <int M>
  struct cand_slots
  {
    const u32 (&ctk)[M];
    const u32 (&cinv)[M];
    const u32 (&cu)[M];
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
    __device__ __forceinline__ u32 u(int j) const
    {
      return cu[j];
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
    __device__ __forceinline__ u32 u(int) const
    {
      return 0u;
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
  __device__ __forceinline__ static u32 image_of_value(float v, float lo_v, float scale)
  {
    const float f = fmaf(v - lo_v, scale, 8388608.0f);
    const int b   = __float_as_int(f);
    const int c   = min(max(b, static_cast<int>(u_lo)), static_cast<int>(u_lo + 0x7FFFFFu));
    return static_cast<u32>(c);
  }
  // fp32 value of a key (bf16 is the top half of fp32, exact)
  __device__ __forceinline__ static float key_value(KeyT k)
  {
    if constexpr (packed)
    {
      return __uint_as_float(static_cast<u32>(__bfloat16_as_ushort(k)) << 16);
    }
    else
    {
      return k;
    }
  }
  __device__ __forceinline__ static KeyT key_from_bits(u32 raw)
  {
    if constexpr (packed)
    {
      return __ushort_as_bfloat16(static_cast<unsigned short>(raw >> 16));
    }
    else
    {
      return __uint_as_float(raw);
    }
  }
  // candidate word from the fp32 bit pattern of the value: bf16 keeps its 16 bits above the
  // inverted index (the low half of the widened value is zero), fp32 keeps two words
  __device__ __forceinline__ static word_t make_raw_word(u32 raw, u32 inv)
  {
    if constexpr (packed)
    {
      return raw | inv;
    }
    else
    {
      return make_uint2(raw, inv);
    }
  }
  __device__ __forceinline__ static u32 raw_bits(word_t w)
  {
    if constexpr (packed)
    {
      return w & 0xFFFF0000u;
    }
    else
    {
      return w.x;
    }
  }

  __device__ __forceinline__ void zero_hist(int buf)
  {
#pragma unroll
    for (int b = tid; b < num_bins; b += Threads)
    {
      storage.hist[buf][b] = 0u;
    }
  }

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
      if (b < num_bins)
      {
        storage.binpos[b] = e;
      }
      if (b < num_bins && e < static_cast<u32>(k_rem) && inc >= static_cast<u32>(k_rem))
      {
        storage.state[0] = e;
        storage.state[1] = static_cast<u32>(b);
        storage.state[2] = counts[i];
      }
    }
  }

  template <class Slots>
  __device__ __forceinline__ static bool in_pivot(const Slots& sl, int j, const state_t& st)
  {
    return ((sl.u(j) & st.um) == st.up) & ((sl.tk(j) & st.km) == st.kp) & ((sl.inv(j) & st.im) == st.ip);
  }

  // One pass: histogram of digit [shift, shift + width) of the source word over the valid pivot
  // items, scan and pivot choice, state update.
  template <int M, source Src, class Slots>
  __device__ __forceinline__ void run_pass(const Slots& sl, int shift, int width, state_t& st)
  {
    const int buf        = st.pass & 1;
    const u32 digit_mask = (1u << width) - 1u;
    u32* hist            = storage.hist[buf];
    u32* dummy           = &storage.dummy[tid];
#pragma unroll
    for (int g = 0; g < M; g += sg)
    {
      if (g < sl.n)
      {
#pragma unroll
        for (int j = g; j < g + sg && j < M; ++j)
        {
          const bool p = sl.valid(j) && in_pivot(sl, j, st);
          u32 w;
          if constexpr (Src == source::image)
          {
            w = sl.u(j);
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
          u32* addr   = p ? &hist[digit_mask - d] : dummy;
          atomicAdd_block(addr, 1u);
        }
      }
    }
    zero_hist(buf ^ 1);
    __syncthreads();
    scan_and_choose(buf, st.k_rem);
    __syncthreads();
    const int sel = static_cast<int>(storage.state[0]);
    const u32 b   = storage.state[1];
    st.count      = static_cast<int>(storage.state[2]);
    st.k_rem -= sel;
    st.sel_total += sel;
    const u32 d   = digit_mask - b;
    st.last_src   = (Src == source::image) ? 0 : (Src == source::key) ? 1 : 2;
    st.last_shift = shift;
    st.last_mask  = digit_mask;
    st.pivot_bin  = static_cast<int>(b);
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
    B4V4_STAMP(5 + st.pass);
  }

  // Block min and max of the key image over the pivot items (lazy tie-class test)
  template <int M, class Slots>
  __device__ __forceinline__ void pivot_minmax(const Slots& sl, state_t& st)
  {
    u32 mn = 0xFFFFFFFFu;
    u32 mx = 0u;
#pragma unroll
    for (int g = 0; g < M; g += sg)
    {
      if (g < sl.n)
      {
#pragma unroll
        for (int j = g; j < g + sg && j < M; ++j)
        {
          const bool p = sl.valid(j) && in_pivot(sl, j, st);
          mn           = p ? min(mn, sl.tk(j)) : mn;
          mx           = p ? max(mx, sl.tk(j)) : mx;
        }
      }
    }
    const u32 wmn = __reduce_min_sync(0xffffffffu, mn);
    const u32 wmx = __reduce_max_sync(0xffffffffu, mx);
    if ((tid & 31) == 0)
    {
      storage.red[tid / 32][0] = wmn;
      storage.red[tid / 32][1] = wmx;
    }
    __syncthreads();
    mn = storage.red[0][0];
    mx = storage.red[0][1];
#pragma unroll
    for (int w = 1; w < warps; ++w)
    {
      mn = min(mn, storage.red[w][0]);
      mx = max(mx, storage.red[w][1]);
    }
    st.kmin = mn;
    st.kmax = mx;
  }

  template <class Slots>
  __device__ __forceinline__ static void
  classify(const Slots& sl, int j, const state_t& st, bool& above, bool& piv)
  {
    const u32 uu  = sl.u(j) & st.um;
    const u32 kk  = sl.tk(j) & st.km;
    const u32 ii  = sl.inv(j) & st.im;
    const bool gu = uu > st.up;
    const bool eu = uu == st.up;
    const bool gk = kk > st.kp;
    const bool ek = kk == st.kp;
    const bool gi = ii > st.ip;
    const bool ei = ii == st.ip;
    above         = gu | (eu & (gk | (ek & gi)));
    piv           = eu & ek & ei;
  }

  // Emission after a single pass: the position of a selected item is the exclusive prefix of its
  // bin plus a per-bin ticket (the zeroed next histogram buffer serves as the ticket array), no
  // classify sweep, no scan, no barrier before the stores.
  template <int M, class Slots>
  __device__ __forceinline__ void emit_single_pass(const Slots& sl, const state_t& st, emit_mode mode, int* out)
  {
    u32* tick       = storage.hist[st.pass & 1];
    u32* dummy      = &storage.dummy[tid];
    const int shift = st.last_shift;
    const u32 dmask = st.last_mask;
    const int pb    = st.pivot_bin;
    const bool all  = (mode == emit_mode::all);
    const bool w2   = (mode == emit_mode::w2);
    const bool tany = (mode == emit_mode::tie_any);
    const u32 need  = static_cast<u32>(st.k_rem);
#pragma unroll
    for (int g = 0; g < M; g += sg)
    {
      if (g < sl.n)
      {
#pragma unroll
        for (int j = g; j < g + sg && j < M; ++j)
        {
          const u32 w      = (st.last_src == 0) ? sl.u(j) : (st.last_src == 1) ? sl.tk(j) : sl.inv(j);
          const int bin    = static_cast<int>(dmask - ((w >> shift) & dmask));
          const bool v     = sl.valid(j);
          const bool above = v && (bin < pb);
          const bool piv   = v && (bin == pb);
          u32* addr        = (above || piv) ? &tick[bin] : dummy;
          const u32 r      = atomicAdd_block(addr, 1u);
          const u32 p      = storage.binpos[bin] + r;
          const int idxv   = static_cast<int>(inv_mask - sl.inv(j));
          if (above || (piv && (all || (tany && r < need))))
          {
            out[p] = idxv;
          }
          if (piv && w2)
          {
            storage.pivot[r] = make_word(sl.tk(j), sl.inv(j));
          }
        }
      }
    }
    B4V4_STAMP(12);
    if (w2)
    {
      __syncthreads();
      B4V4_STAMP(13);
      if (tid < 32)
      {
        warp_finish(st.count, st.k_rem, out + st.sel_total);
      }
    }
    B4V4_STAMP(14);
  }

  // Emission after several passes: one classify sweep into bitmasks, one packed scan, one store sweep
  template <int M, class Slots>
  __device__ __forceinline__ void emit(const Slots& sl, const state_t& st, emit_mode mode, int* out)
  {
    if (st.pass == 1)
    {
      emit_single_pass<M>(sl, st, mode, out);
      return;
    }
    constexpr int words = (M + 31) / 32;
    const bool all      = (mode == emit_mode::all);
    u32 sel_bits[words];
    u32 piv_bits[words];
#pragma unroll
    for (int w = 0; w < words; ++w)
    {
      sel_bits[w] = 0u;
      piv_bits[w] = 0u;
    }
#pragma unroll
    for (int g = 0; g < M; g += sg)
    {
      if (g < sl.n)
      {
#pragma unroll
        for (int j = g; j < g + sg && j < M; ++j)
        {
          bool above, piv;
          classify(sl, j, st, above, piv);
          const bool v = sl.valid(j);
          const bool s = v && (above || (piv && all));
          const bool p = v && piv && !all;
          sel_bits[j / 32] |= (s ? 1u : 0u) << (j % 32);
          piv_bits[j / 32] |= (p ? 1u : 0u) << (j % 32);
        }
      }
    }
    u32 n_sel = 0;
    u32 n_piv = 0;
#pragma unroll
    for (int w = 0; w < words; ++w)
    {
      n_sel += __popc(sel_bits[w]);
      n_piv += __popc(piv_bits[w]);
    }
    B4V4_STAMP(10);
    u32 excl;
    scan_t(storage.scan).ExclusiveSum((n_sel << 16) | n_piv, excl);
    B4V4_STAMP(11);
    u32 pos_sel    = excl >> 16;
    u32 pos_piv    = excl & 0xFFFFu;
    const u32 need = static_cast<u32>(st.k_rem);
#pragma unroll
    for (int g = 0; g < M; g += sg)
    {
      if (g < sl.n)
      {
#pragma unroll
        for (int j = g; j < g + sg && j < M; ++j)
        {
          const bool s   = (sel_bits[j / 32] >> (j % 32)) & 1u;
          const bool p   = (piv_bits[j / 32] >> (j % 32)) & 1u;
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
    }
    B4V4_STAMP(12);
    if (mode == emit_mode::w2)
    {
      __syncthreads();
      B4V4_STAMP(13);
      if (tid < 32)
      {
        warp_finish(st.count, st.k_rem, out + st.sel_total);
      }
    }
    B4V4_STAMP(14);
  }

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
      const u32 tk   = pw.x;
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

  template <int M, bool HasImage, class Slots>
  __device__ __forceinline__ void finish(const Slots& sl, int k, int* out)
  {
    state_t st;
    st.um = st.up = st.km = st.kp = st.im = st.ip = 0u;
    st.k_rem     = k;
    st.sel_total = 0;
    st.count     = 0;
    st.kmin      = 0u;
    st.kmax      = 0xFFFFFFFFu;
    st.pass      = 0;
    st.last_src  = 0;
    st.last_shift = 0;
    st.last_mask = 0u;
    st.pivot_bin = 0;

    bool done        = false;
    bool index_phase = false;
    emit_mode mode   = emit_mode::all;
    auto check       = [&]() {
      if (st.count == st.k_rem)
      {
        mode = emit_mode::all;
        done = true;
      }
      else if (st.count <= fin)
      {
        mode = emit_mode::w2;
        done = true;
      }
      else if (!index_phase)
      {
        pivot_minmax<M>(sl, st);
        if (st.kmin == st.kmax)
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
      }
    };

    if constexpr (HasImage)
    {
      run_pass<M, source::image>(sl, 15, 8, st);
      check();
      if (!done && !index_phase)
      {
        run_pass<M, source::image>(sl, 7, 8, st);
        check();
      }
      if (!done && !index_phase)
      {
        run_pass<M, source::image>(sl, 0, 7, st);
        check();
      }
    }
    if (!done && !index_phase)
    {
      // key bytes, from the highest differing byte of the pivot set's min and max when known
      int byte = HasImage ? (31 - ::cuda::std::countl_zero(st.kmin ^ st.kmax)) / 8 : (key_bits / 8 - 1);
      for (; byte >= 0 && !done && !index_phase; --byte)
      {
        run_pass<M, source::key>(sl, byte * 8, 8, st);
        check();
      }
    }
    if constexpr (TB == tie_break::smaller_index)
    {
      if (!done)
      {
        index_phase = true;
        int shift   = idx_bits;
        while (!done && shift > 0)
        {
          const int w = (shift >= 8) ? 8 : shift;
          shift -= w;
          run_pass<M, source::index>(sl, shift, w, st);
          check();
        }
      }
    }
    emit<M>(sl, st, mode, out);
  }
};

} // namespace b4v4
