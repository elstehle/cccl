// N3: streaming block top-k with a running exact threshold over chunked input.
//
// The row arrives in chunks of ChunkSize items (striped over the block: item i of the chunk at
// thread t has source index base + i x Threads + t). The primitive never looks at a chunk before
// insert_chunk is called with it and never again afterwards, which is the "insert chunk, keep k"
// contract a device agent over many chunks per query could use. State between chunks lives in
// shared memory: a candidate buffer of (key, index) entries, a histogram of the buffer under a
// fixed-point image of the key (NumBins equal-width value bins over the first chunk's value
// bracket, T25), and a threshold.
//
// Per chunk after the first: every item is compared against the threshold (one compare), the
// per-thread candidate count and the thread's histogram bins are packed into one word and
// scanned by one block scan (T29), which gives each thread its store offset and, from the
// histogram prefix, the pivot bin of the buffer: the bin in which the k-th largest buffered key
// lies. Its lower edge is a lower bound on the k-th largest key seen so far and becomes the
// threshold for the next chunk (one chunk of lag, the histogram read in the scan does not yet
// contain this chunk's candidates). The candidates are then stored at their offsets (predicated
// stores, in thread-major order) and counted into the histogram with one branch-free shared
// atomic per item slot (a private dummy word for the slots that are no candidates, T32).
// Nothing is ever merged or reselected during the stream.
//
// The finish scans the histogram once more, classifies every buffered entry as above the pivot
// bin, in it, or below, emits the above items through one packed block scan and resolves the
// pivot bin: exact count (all of it), at most Threads items (an all-pairs rank over the staged
// pivot items by key descending then index ascending, exact under both requirements), or a
// refinement level that re-bins the buffer on the twiddled integer key over the pivot set's own
// min and max (never stalls, every level separates distinct keys). A tie class (pivot set min
// equals max) is taken in buffer order under not_guaranteed and refined on the index image under
// prefer_smaller_index (NumBins bins over the index range, at most Threads items per bin).
//
// Buffer overflow (the next chunk's candidates would not fit: sorted ascending input, a useless
// first bracket) runs the same refinement, compacts the buffer in place blockwise (all loads of a
// block before any store, kept entries only move down), rebuilds the histogram and rescans.
//
// Contract: block scope, Threads 256 or 512 (multiples of 32 with NumBins % Threads == 0),
// ChunkSize a multiple of Threads, k in [1, ChunkSize] at run time, source indices below MaxIndex
// (16-bit index fields, MaxIndex <= 65536, and MaxIndex <= NumBins x Threads for the index
// refine), Cap < 65536 entries with Cap >= k + Threads + ChunkSize. Keys fp32 or bf16 (compared
// as fp32 values, bf16 widened exactly). Output: k source indices in the caller's shared array,
// unspecified order. Requirements not_guaranteed and prefer_smaller_index (SmallerIndex), the
// same code except the tie-class exit. -0.0 equals +0.0 everywhere (value compares, the image
// subtracts the bracket base, the integer image folds the sign of zero). NaN: never a candidate
// after the first chunk (compares false), inside the first chunk its image lands in the top bin,
// unsupported. Temp storage: histogram, scan storage, reduction partials, dummy words, control
// words, a stage of Threads entries and the buffer of Cap entries (4 bytes per entry for bf16,
// 8 for fp32). Synchronization: __syncthreads inside, the caller separates consecutive calls
// that reuse the temp storage by a barrier.
#pragma once

#include <cub/block/block_scan.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cuda_bf16.h>

#ifdef N3_PROFILE
#  define N3_STAMP(i)                                                   \
    do                                                                  \
    {                                                                   \
      if (threadIdx.x == 0)                                             \
      {                                                                 \
        const int c_          = ::n3::g_stamp_cursor;                   \
        ::n3::g_stamps[c_]    = clock64();                              \
        ::n3::g_stamp_ids[c_] = (i);                                    \
        ::n3::g_stamp_cursor  = c_ + 1;                                 \
      }                                                                 \
    } while (0)
#else
#  define N3_STAMP(i) \
    do                \
    {                 \
    } while (0)
#endif

namespace n3
{

#ifdef N3_PROFILE
__device__ long long g_stamps[64];
__device__ int g_stamp_ids[64];
__device__ int g_stamp_cursor;
#endif

using u32 = ::cuda::std::uint32_t;

__device__ __forceinline__ u32 twiddle(float f)
{
  f           = __fadd_rn(f, 0.0f); // -0.0 onto +0.0
  const u32 b = __float_as_uint(f);
  const u32 m = static_cast<u32>(static_cast<int>(b) >> 31) | 0x80000000u;
  return b ^ m;
}

__device__ __forceinline__ float untwiddle(u32 t)
{
  const u32 m = (static_cast<int>(t) >= 0) ? 0xFFFFFFFFu : 0x80000000u;
  return __uint_as_float(t ^ m);
}

template <typename KeyT>
struct key_cvt;

template <>
struct key_cvt<float>
{
  __device__ __forceinline__ static float value(float f)
  {
    return f;
  }
};

template <>
struct key_cvt<__nv_bfloat16>
{
  __device__ __forceinline__ static float value(__nv_bfloat16 h)
  {
    return __uint_as_float(static_cast<u32>(__bfloat16_as_ushort(h)) << 16);
  }
};

template <typename KeyT, int Threads, int ChunkSize, int MaxIndex, int Cap, int NumBins, bool SmallerIndex>
struct block_topk_stream
{
  static constexpr int cpt     = ChunkSize / Threads;
  static constexpr int bpt     = NumBins / Threads;
  static constexpr int warps   = Threads / 32;
  static constexpr bool packed = sizeof(KeyT) == 2;
  static constexpr int cg      = 4; // entries per thread per compaction block
  static constexpr int lg_nb   = (NumBins == 256) ? 8 : (NumBins == 512) ? 9 : (NumBins == 1024) ? 10 : (NumBins == 2048) ? 11 : -1;
  static constexpr float flt_max = 3.4028234663852886e38f;
  static constexpr float flt_inf = __builtin_huge_valf();

  using entry_t = ::cuda::std::conditional_t<packed, u32, uint2>;
  using scan_t  = cub::BlockScan<u32, Threads, cub::BLOCK_SCAN_WARP_SCANS>;

  static_assert(Threads % 32 == 0, "whole warps");
  static_assert(ChunkSize % Threads == 0 && cpt >= 1, "a chunk is a striped tile of the block");
  static_assert(NumBins % Threads == 0 && lg_nb > 0, "bins per thread");
  static_assert(MaxIndex <= 65536, "16-bit index field");
  static_assert(MaxIndex <= NumBins * Threads, "the index refine needs at most Threads items per index bin");
  static_assert(Cap < 65536 && ChunkSize < 65536, "16-bit scan fields");

  enum : int
  {
    C_NBUF = 0,
    C_THRN,
    C_P,
    C_A,
    C_PC,
    C_XA,
    C_WORDS
  };

  struct TempStorage
  {
    u32 hist[NumBins];
    typename scan_t::TempStorage scan;
    u32 red[warps][2];
    u32 dummy[Threads];
    u32 ctrl[C_WORDS];
    entry_t stage[Threads];
    entry_t buf[Cap];
  };

  // kind 0: fixed-point image of the key value over [lo_c, lo_c + (NumBins - 1) / scale]
  // kind 1: integer image of the twiddled key over [lo_t, ...] with a right shift of sh
  // kind 2: index image, the fixed-point image of -(index) (smaller index is larger), only for
  //         entries whose key equals cv; keys above cv are above, keys below cv are below
  struct image_params
  {
    float lo_c;
    float lo_min;
    float scale;
    float cv;
    u32 lo_t;
    int sh;
    int p;
    int kind;
  };

  struct resolution
  {
    int mode; // 0 all pivot items, 1 first R pivot items in buffer order, 2 rank finish
    int A;
    int R;
    int P;
  };

  TempStorage& s;
  const int tid;
  image_params vq; // the stream image (kind 0), uniform
  u32 n_buf; // uniform
  float thr; // uniform

  __device__ __forceinline__ block_topk_stream(TempStorage& ts)
      : s(ts)
      , tid(static_cast<int>(threadIdx.x))
  {}

  // ---------------------------------------------------------------------------------------------
  // Entries
  // ---------------------------------------------------------------------------------------------

  __device__ __forceinline__ static entry_t make_entry(float v, int idx)
  {
    if constexpr (packed)
    {
      return (__float_as_uint(v) & 0xFFFF0000u) | static_cast<u32>(idx);
    }
    else
    {
      return make_uint2(__float_as_uint(v), static_cast<u32>(idx));
    }
  }

  __device__ __forceinline__ static entry_t zero_entry()
  {
    if constexpr (packed)
    {
      return 0u;
    }
    else
    {
      return make_uint2(0u, 0u);
    }
  }

  __device__ __forceinline__ static float entry_key(entry_t e)
  {
    if constexpr (packed)
    {
      return __uint_as_float(e & 0xFFFF0000u);
    }
    else
    {
      return __uint_as_float(e.x);
    }
  }

  __device__ __forceinline__ static int entry_idx(entry_t e)
  {
    if constexpr (packed)
    {
      return static_cast<int>(e & 0xFFFFu);
    }
    else
    {
      return static_cast<int>(e.y);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Images
  // ---------------------------------------------------------------------------------------------

  __device__ __forceinline__ static int image_f(float v, float lo_c, float scale)
  {
    const float f = fmaf(v - lo_c, scale, 8388608.0f);
    int u         = __float_as_int(f);
    u             = max(u, 0x4B000000);
    u             = min(u, 0x4B000000 + NumBins - 1);
    return u - 0x4B000000;
  }

  __device__ __forceinline__ static float make_scale(float width)
  {
    float sc = static_cast<float>(NumBins - 1) / width;
    if (!(sc > 0.0f) || sc == flt_inf)
    {
      sc = (width == flt_inf) ? 1.5e-36f : 1.0f;
    }
    return sc;
  }

  __device__ __forceinline__ static int image_of(float key, int idx, const image_params& q)
  {
    if (q.kind == 0)
    {
      return image_f(key, q.lo_c, q.scale);
    }
    if (q.kind == 1)
    {
      const u32 t = twiddle(key);
      const u32 d = max(t, q.lo_t) - q.lo_t;
      return static_cast<int>(min(d >> q.sh, static_cast<u32>(NumBins - 1)));
    }
    return image_f(-static_cast<float>(idx), q.lo_c, q.scale);
  }

  // 0 below, 1 pivot, 2 above
  __device__ __forceinline__ static int classify(float key, int idx, const image_params& q)
  {
    if (q.kind == 2)
    {
      if (key > q.cv)
      {
        return 2;
      }
      if (key < q.cv)
      {
        return 0;
      }
    }
    else if (key < q.lo_min)
    {
      return 0;
    }
    const int im = image_of(key, idx, q);
    return (im > q.p) ? 2 : ((im == q.p) ? 1 : 0);
  }

  // Largest value t such that every key whose kind-0 image is at least p is strictly above t,
  // or lo_min when the edge is not representable. The edge is nudged down in units of the
  // magnitude of t and lo_c, not in ulps of t, because the image subtracts lo_c first and a
  // one-ulp step of t vanishes in that difference.
  __device__ __forceinline__ static float lower_edge(int p, const image_params& q)
  {
    float t = q.lo_c + (static_cast<float>(p) - 0.5f) / q.scale;
    if (!(t > q.lo_c) || t == flt_inf || t != t)
    {
      return q.lo_min;
    }
    float eps = fmaxf(fabsf(t), fabsf(q.lo_c)) * 2.4e-7f;
#pragma unroll 1
    for (int i = 0; i < 6; ++i)
    {
      if (image_f(t, q.lo_c, q.scale) < p)
      {
        return t;
      }
      t -= eps;
      eps *= 4.0f;
    }
    return (image_f(t, q.lo_c, q.scale) < p) ? t : q.lo_min;
  }

  __device__ __forceinline__ static int span(int n)
  {
    return ((n + Threads - 1) / Threads) | 1; // odd stride: conflict-free blocked loads (T31)
  }

  // ---------------------------------------------------------------------------------------------
  // Histogram scan: thread t owns slot u = Threads - 1 - t, bins [u bpt, (u + 1) bpt) read from the
  // highest down, so the exclusive prefix over threads is the count above the thread's bins. The
  // owner of the bin where the count from the top crosses k writes p, A (above) and P (bin count)
  // and, when asked, the next threshold. Caller barriers afterwards.
  // ---------------------------------------------------------------------------------------------

  __device__ __forceinline__ void scan_hist(u32 k, const image_params& q, bool set_thr)
  {
    const int u = Threads - 1 - tid;
    u32 h[bpt];
    u32 hs = 0;
#pragma unroll
    for (int j = 0; j < bpt; ++j)
    {
      h[j] = s.hist[u * bpt + (bpt - 1 - j)];
      hs += h[j];
    }
    const u32 xa = s.ctrl[C_XA];
    u32 pre;
    scan_t(s.scan).ExclusiveSum(hs, pre);
    u32 above = pre + xa;
#pragma unroll
    for (int j = 0; j < bpt; ++j)
    {
      const int b = u * bpt + (bpt - 1 - j);
      if (above < k && k <= above + h[j])
      {
        s.ctrl[C_P]  = static_cast<u32>(b);
        s.ctrl[C_A]  = above;
        s.ctrl[C_PC] = h[j];
        if (set_thr)
        {
          s.ctrl[C_THRN] = __float_as_uint(fmaxf(thr, lower_edge(b, q)));
        }
      }
      above += h[j];
    }
  }

  // Zero the histogram and count every buffered entry under q. Barriers inside, ends with one.
  __device__ void rebin(const image_params& q)
  {
#pragma unroll
    for (int j = 0; j < bpt; ++j)
    {
      s.hist[tid * bpt + j] = 0u;
    }
    if (tid == 0)
    {
      s.ctrl[C_XA] = 0u;
    }
    __syncthreads();
    const int n   = static_cast<int>(n_buf);
    const int sp  = span(n);
    const int beg = tid * sp;
    const int end = min(beg + sp, n);
    u32 xa        = 0;
#pragma unroll 1
    for (int e = beg; e < end; ++e)
    {
      const entry_t en = s.buf[e];
      const float key  = entry_key(en);
      const int idx    = entry_idx(en);
      if (q.kind == 2)
      {
        if (key > q.cv)
        {
          ++xa;
        }
        else if (key == q.cv)
        {
          atomicAdd(&s.hist[image_of(key, idx, q)], 1u);
        }
      }
      else if (!(key < q.lo_min))
      {
        atomicAdd(&s.hist[image_of(key, idx, q)], 1u);
      }
    }
    if (q.kind == 2 && xa != 0)
    {
      atomicAdd(&s.ctrl[C_XA], xa);
    }
    __syncthreads();
  }

  // Min and max key over the pivot entries of q. One barrier inside.
  __device__ void pivot_minmax(const image_params& q, float& mn, float& mx)
  {
    const int n   = static_cast<int>(n_buf);
    const int sp  = span(n);
    const int beg = tid * sp;
    const int end = min(beg + sp, n);
    u32 tmn       = 0xFFFFFFFFu;
    u32 tmx       = 0u;
#pragma unroll 1
    for (int e = beg; e < end; ++e)
    {
      const entry_t en = s.buf[e];
      const float key  = entry_key(en);
      if (classify(key, entry_idx(en), q) == 1)
      {
        const u32 t = twiddle(key);
        tmn         = min(tmn, t);
        tmx         = max(tmx, t);
      }
    }
    const u32 wmn = __reduce_min_sync(0xffffffffu, tmn);
    const u32 wmx = __reduce_max_sync(0xffffffffu, tmx);
    if ((tid & 31) == 0)
    {
      s.red[tid >> 5][0] = wmn;
      s.red[tid >> 5][1] = wmx;
    }
    __syncthreads();
    u32 bmn = s.red[0][0];
    u32 bmx = s.red[0][1];
#pragma unroll
    for (int w = 1; w < warps; ++w)
    {
      bmn = min(bmn, s.red[w][0]);
      bmx = max(bmx, s.red[w][1]);
    }
    mn = untwiddle(bmn);
    mx = untwiddle(bmx);
  }

  // Find the pivot of the buffer under q, refining q until the pivot set is resolvable.
  // Returns with the histogram consistent with q and a barrier passed.
  __device__ resolution resolve(u32 k, image_params& q)
  {
    resolution r;
#pragma unroll 1
    for (;;)
    {
      scan_hist(k, q, false);
      __syncthreads();
      q.p         = static_cast<int>(s.ctrl[C_P]);
      const int A = static_cast<int>(s.ctrl[C_A]);
      const int P = static_cast<int>(s.ctrl[C_PC]);
      const int R = static_cast<int>(k) - A;
      r.A         = A;
      r.R         = R;
      r.P         = P;
      if (P == R)
      {
        r.mode = 0;
        return r;
      }
      if (P <= Threads)
      {
        r.mode = 2;
        return r;
      }
      float mn, mx;
      pivot_minmax(q, mn, mx);
      if (mn == mx)
      {
        if (!SmallerIndex)
        {
          r.mode = 1;
          return r;
        }
        q.kind   = 2;
        q.cv     = mn;
        q.lo_c   = -static_cast<float>(MaxIndex - 1);
        q.lo_min = q.lo_c;
        q.scale  = make_scale(static_cast<float>(MaxIndex - 1));
      }
      else
      {
        q.kind          = 1;
        q.lo_min        = mn;
        q.lo_t          = twiddle(mn);
        const u32 range = twiddle(mx) - q.lo_t;
        const int nb    = 32 - __clz(range);
        q.sh            = max(nb - lg_nb, 0);
      }
      rebin(q);
    }
  }

  // Emit the above items and the resolved pivot items into out. Ends without a barrier.
  __device__ void emit(const image_params& q, const resolution& r, int* out)
  {
    const int n   = static_cast<int>(n_buf);
    const int sp  = span(n);
    const int beg = tid * sp;
    const int end = min(beg + sp, n);
    u32 ca = 0, cp = 0;
#pragma unroll 1
    for (int e = beg; e < end; ++e)
    {
      const entry_t en = s.buf[e];
      const int c      = classify(entry_key(en), entry_idx(en), q);
      ca += (c == 2) ? 1u : 0u;
      cp += (c == 1) ? 1u : 0u;
    }
    N3_STAMP(9);
    u32 pre;
    scan_t(s.scan).ExclusiveSum((ca << 16) | cp, pre);
    u32 apos = pre >> 16;
    u32 ppos = pre & 0xFFFFu;
#pragma unroll 1
    for (int e = beg; e < end; ++e)
    {
      const entry_t en = s.buf[e];
      const int idx    = entry_idx(en);
      const int c      = classify(entry_key(en), idx, q);
      if (c == 2)
      {
        out[apos] = idx;
        ++apos;
      }
      else if (c == 1)
      {
        if (r.mode == 2)
        {
          s.stage[ppos] = en;
        }
        else if (ppos < static_cast<u32>(r.R))
        {
          out[r.A + ppos] = idx;
        }
        ++ppos;
      }
    }
    N3_STAMP(10);
    if (r.mode == 2)
    {
      __syncthreads();
      if (tid < r.P)
      {
        const entry_t me = s.stage[tid];
        const float key  = entry_key(me);
        const int idx    = entry_idx(me);
        int rank         = 0;
#pragma unroll 4
        for (int j = 0; j < r.P; ++j)
        {
          const entry_t o = s.stage[j];
          const float ko  = entry_key(o);
          const int io    = entry_idx(o);
          rank += ((ko > key) | ((ko == key) & (io < idx))) ? 1 : 0;
        }
        if (rank < r.R)
        {
          out[r.A + rank] = idx;
        }
      }
    }
  }

  // In-place compaction: keep the above entries and the first `limit` pivot entries (buffer
  // order), preserving order. Blockwise: all loads of a block precede its stores (barrier), kept
  // entries only move to lower positions, so later blocks are never overwritten. Ends with a
  // barrier and n_buf updated. The histogram is not maintained, the caller rebuilds it.
  __device__ void compact(const image_params& q, u32 limit)
  {
    const int n = static_cast<int>(n_buf);
    u32 wpos    = 0;
    u32 lim     = limit;
#pragma unroll 1
    for (int blk = 0; blk < n; blk += Threads * cg)
    {
      entry_t e[cg];
      int cls[cg];
      u32 ca = 0, cp = 0;
#pragma unroll
      for (int i = 0; i < cg; ++i)
      {
        const int pos    = blk + tid * cg + i;
        const bool valid = pos < n;
        e[i]             = valid ? s.buf[pos] : zero_entry();
        cls[i]           = valid ? classify(entry_key(e[i]), entry_idx(e[i]), q) : 0;
        ca += (cls[i] == 2) ? 1u : 0u;
        cp += (cls[i] == 1) ? 1u : 0u;
      }
      __syncthreads();
      u32 pre, tot;
      scan_t(s.scan).ExclusiveSum((ca << 16) | cp, pre, tot);
      u32 ab = pre >> 16;
      u32 pb = pre & 0xFFFFu;
#pragma unroll
      for (int i = 0; i < cg; ++i)
      {
        if (cls[i] == 2)
        {
          s.buf[wpos + ab + min(pb, lim)] = e[i];
          ++ab;
        }
        else if (cls[i] == 1)
        {
          if (pb < lim)
          {
            s.buf[wpos + ab + pb] = e[i];
          }
          ++pb;
        }
      }
      const u32 kept_p = min(tot & 0xFFFFu, lim);
      wpos += (tot >> 16) + kept_p;
      lim -= kept_p;
    }
    __syncthreads();
    n_buf = wpos;
    if (tid == 0)
    {
      s.ctrl[C_NBUF] = wpos;
    }
  }

  // The next chunk's candidates do not fit: resolve the buffer's pivot, compact, rebuild the
  // histogram under the stream image, rescan with the threshold. Ends with a barrier.
  __device__ void overflow_compact(u32 k)
  {
    image_params q     = vq;
    const resolution r = resolve(k, q);
    compact(q, (r.mode == 1) ? static_cast<u32>(r.R) : 0xFFFFFFFFu);
    rebin(vq);
    scan_hist(k, vq, true);
    __syncthreads();
  }

  // ---------------------------------------------------------------------------------------------
  // Streaming interface
  // ---------------------------------------------------------------------------------------------

  __device__ __forceinline__ void begin()
  {
    N3_STAMP(0);
#pragma unroll
    for (int j = 0; j < bpt; ++j)
    {
      s.hist[tid * bpt + j] = 0u;
    }
    if (tid == 0)
    {
      s.ctrl[C_THRN] = __float_as_uint(-flt_inf);
      s.ctrl[C_XA]   = 0u;
      s.ctrl[C_NBUF] = 0u;
    }
    thr   = -flt_inf;
    n_buf = 0;
    // the first chunk's bracket barrier orders this zeroing before the first atomics
  }

  template <bool First>
  __device__ __forceinline__ void insert_chunk(const KeyT (&ck)[cpt], int base, int k)
  {
    const u32 ku = static_cast<u32>(k);
    float v[cpt];
#pragma unroll
    for (int i = 0; i < cpt; ++i)
    {
      v[i] = key_cvt<KeyT>::value(ck[i]);
    }

    if constexpr (First)
    {
      float mn = v[0];
      float mx = v[0];
#pragma unroll
      for (int i = 1; i < cpt; ++i)
      {
        mn = fminf(mn, v[i]);
        mx = fmaxf(mx, v[i]);
      }
      const u32 wmn = __reduce_min_sync(0xffffffffu, twiddle(mn));
      const u32 wmx = __reduce_max_sync(0xffffffffu, twiddle(mx));
      if ((tid & 31) == 0)
      {
        s.red[tid >> 5][0] = wmn;
        s.red[tid >> 5][1] = wmx;
      }
      __syncthreads();
      u32 bmn = s.red[0][0];
      u32 bmx = s.red[0][1];
#pragma unroll
      for (int w = 1; w < warps; ++w)
      {
        bmn = min(bmn, s.red[w][0]);
        bmx = max(bmx, s.red[w][1]);
      }
      vq.lo_min        = untwiddle(bmn);
      const float hi   = untwiddle(bmx);
      vq.lo_c          = fmaxf(vq.lo_min, -flt_max);
      const float hi_c = fminf(hi, flt_max);
      vq.scale         = make_scale(hi_c - vq.lo_c);
      vq.kind          = 0;
      vq.p             = 0;
      vq.cv            = 0.0f;
      vq.lo_t          = 0u;
      vq.sh            = 0;
      N3_STAMP(1);
      // every item of the first chunk is a candidate, stored at its striped position
#pragma unroll
      for (int i = 0; i < cpt; ++i)
      {
        s.buf[i * Threads + tid] = make_entry(v[i], base + i * Threads + tid);
        atomicAdd(&s.hist[image_f(v[i], vq.lo_c, vq.scale)], 1u);
      }
      n_buf = ChunkSize;
      __syncthreads();
      N3_STAMP(2);
      scan_hist(ku, vq, true);
      __syncthreads();
      N3_STAMP(3);
      thr = __uint_as_float(s.ctrl[C_THRN]);
    }
    else
    {
      // candidate count against the current threshold, four accumulators
      u32 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
      for (int i = 0; i < cpt; i += 4)
      {
        c0 += (v[i] > thr) ? 1u : 0u;
        if (i + 1 < cpt)
        {
          c1 += (v[i + 1] > thr) ? 1u : 0u;
        }
        if (i + 2 < cpt)
        {
          c2 += (v[i + 2] > thr) ? 1u : 0u;
        }
        if (i + 3 < cpt)
        {
          c3 += (v[i + 3] > thr) ? 1u : 0u;
        }
      }
      const u32 c = (c0 + c1) + (c2 + c3);
      N3_STAMP(4);
      // the thread's histogram bins (slot u, highest bin first) and the packed scan
      const int u = Threads - 1 - tid;
      u32 h[bpt];
      u32 hs = 0;
#pragma unroll
      for (int j = 0; j < bpt; ++j)
      {
        h[j] = s.hist[u * bpt + (bpt - 1 - j)];
        hs += h[j];
      }
      u32 pre, tot;
      scan_t(s.scan).ExclusiveSum((c << 16) | hs, pre, tot);
      const u32 tot_c = tot >> 16;
      u32 above       = pre & 0xFFFFu;
#pragma unroll
      for (int j = 0; j < bpt; ++j)
      {
        const int b = u * bpt + (bpt - 1 - j);
        if (above < ku && ku <= above + h[j])
        {
          s.ctrl[C_P]    = static_cast<u32>(b);
          s.ctrl[C_A]    = above;
          s.ctrl[C_PC]   = h[j];
          s.ctrl[C_THRN] = __float_as_uint(fmaxf(thr, lower_edge(b, vq)));
        }
        above += h[j];
      }
      N3_STAMP(5);
      if (n_buf + tot_c > static_cast<u32>(Cap))
      {
        __syncthreads();
        overflow_compact(ku);
      }
      u32 pos = n_buf + (pre >> 16);
      // store sweep: predicated stores, branch-free histogram atomics
#pragma unroll
      for (int i = 0; i < cpt; ++i)
      {
        const bool cand = v[i] > thr;
        if (cand)
        {
          s.buf[pos] = make_entry(v[i], base + i * Threads + tid);
        }
        u32* target = cand ? &s.hist[image_f(v[i], vq.lo_c, vq.scale)] : &s.dummy[tid];
        atomicAdd(target, 1u);
        pos += cand ? 1u : 0u;
      }
      n_buf += tot_c;
      __syncthreads();
      N3_STAMP(6);
      thr = __uint_as_float(s.ctrl[C_THRN]);
    }
  }

  __device__ __forceinline__ void finish(int k, int* out)
  {
    image_params q     = vq;
    N3_STAMP(7);
    const resolution r = resolve(static_cast<u32>(k), q);
    N3_STAMP(8);
    emit(q, r, out);
    N3_STAMP(11);
  }
};

} // namespace n3
