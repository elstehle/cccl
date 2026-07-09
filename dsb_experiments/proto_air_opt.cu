// Deep-dive: where does cub::detail::block_topk_air spend its time at
// (BLOCK=256, N=1024, K=16, float+index pairs, sm_100), and how much of it is removable?
//
// Companion to BLOCK_TOPK_RESULTS.md (which found air = 2499 cyc random). This file:
//   * air_ref    : the real cub header (baseline)
//   * air_reimpl : faithful standalone reimplementation (parity gate for the profile)
//   * prof mode  : air_reimpl instrumented with clock64 stamps at every barrier boundary ->
//                  per-phase cycle table (twiddle, per-pass init/histogram/scan/choose,
//                  scatter, gathers), min across reps/calls (steady-state marginal)
//   * air_fused  : v2 — compile-time bits/K, double-buffered histograms (init folded into the
//                  histogram phase), scan fused with choose (exclusive = inclusive - input, no
//                  smem writeback, no separate choose phase): 3 barriers/pass instead of 5
//   * air_wscan  : v3 — like v2 but replaces BlockScan with an explicit warp-shfl scan +
//                  cross-warp aggregate broadcast (isolates BlockScan's cost)
//
// Modes: ./proto_air_opt [correct|lat|prof|thr|res|all]
// Methodology identical to proto_topk.cu (slope latency, set-semantics validation).

#include <cub/block/block_scan.cuh>
#include <cub/block/specializations/block_topk_air.cuh>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

constexpr unsigned kFull = 0xffffffffu;
constexpr int kBlock     = 256;
constexpr int kIpt       = 4;
constexpr int kN         = 1024;
constexpr int kK         = 16;

__device__ int g_sink;

__device__ __forceinline__ unsigned twiddle(float f)
{
  unsigned u = __float_as_uint(f);
  return u ^ (((unsigned) ((int) u >> 31)) | 0x80000000u);
}

__device__ __forceinline__ float untwiddle(unsigned t)
{
  unsigned m = (~((unsigned) ((int) t >> 31))) | 0x80000000u;
  return __uint_as_float(t ^ m);
}

// ------------------------------------------------------------------ air_ref (the real header)
struct ProtoAirRef
{
  static constexpr const char* name = "air_ref (cub header)";
  using topk_t                      = cub::detail::block_topk_air<float, kBlock, kIpt, int>;
  struct Smem
  {
    typename topk_t::TempStorage ts;
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    float k[4];
    int val[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      k[i]   = v[i];
      val[i] = idx[i];
    }
    topk_t(sh.ts).template select_pairs<cub::detail::topk::select::max, true>(k, val, kK, kN);
    if (threadIdx.x < kK / kIpt)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        out_v[threadIdx.x * 4 + i] = k[i];
        out_i[threadIdx.x * 4 + i] = val[i];
      }
    }
    __syncthreads();
  }
};

// ------------------------------------------------------------------ faithful reimplementation
// Mirrors block_topk_air.cuh select_topk<max, full_tile> for float/int pairs, k=16, bits [0,32):
// same stage order, same barriers, same runtime pass loop, same 2-class scatter. Stamp is a
// no-op for measurement parity and a thread0 clock64 recorder for the profile.
using block_scan_t = cub::BlockScan<unsigned, kBlock, cub::BLOCK_SCAN_WARP_SCANS>;

struct AirSmem
{
  unsigned hist[256];
  typename block_scan_t::TempStorage scan_temp;
  unsigned bucket, cands, selected;
  unsigned sel_off[2];
  union
  {
    float keys[kN];
    int vals[kN];
  } exch;
};

constexpr int kStamps = 24;

struct NoStamp
{
  __device__ __forceinline__ void operator()(int) const {}
};

struct SmemStamp
{
  long long* t;
  __device__ __forceinline__ void operator()(int i) const
  {
    if (threadIdx.x == 0)
    {
      t[i] = clock64();
    }
  }
};

template <class Stamp>
__device__ __forceinline__ void
air_reimpl_run(const float (&v)[4], const int (&idx)[4], AirSmem& sh, float* out_v, int* out_i, Stamp stamp, int* d_passes)
{
  stamp(0);
  unsigned uk[4];
  int val[4];
  unsigned flip                = 0;
  constexpr unsigned tw_mzero  = 0x7fffffffu; // twiddle(-0.0f)
  constexpr unsigned tw_pzero  = 0x80000000u; // twiddle(+0.0f)
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    uk[i]  = twiddle(v[i]);
    val[i] = idx[i];
    if (uk[i] == tw_mzero)
    {
      flip |= 1u << i;
      uk[i] = tw_pzero;
    }
  }
  stamp(1);

  unsigned kth_prefix = 0, prefix_mask = 0;
  int k              = kK;
  int total_selected = 0;
  int num_candidates = kN;
  int pass           = 0;
#pragma unroll 1
  for (; pass < 4; ++pass)
  {
    const int pass_begin = 24 - 8 * pass;
    // init histograms
    sh.hist[threadIdx.x] = 0;
    __syncthreads();
    stamp(2 + pass * 4 + 0);
    // histogram over candidates
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if ((uk[i] & prefix_mask) == kth_prefix)
      {
        const int digit = (uk[i] >> pass_begin) & 0xff;
        atomicAdd(&sh.hist[255 - digit], 1u); // select::max bucket remap
      }
    }
    __syncthreads();
    stamp(2 + pass * 4 + 1);
    // prefix sum over buckets (read, BlockScan, write back — as in the header)
    unsigned tb = sh.hist[threadIdx.x];
    block_scan_t(sh.scan_temp).InclusiveSum(tb, tb);
    sh.hist[threadIdx.x] = tb;
    __syncthreads();
    stamp(2 + pass * 4 + 2);
    // choose bucket
    {
      const unsigned prev = (threadIdx.x == 0) ? 0 : sh.hist[threadIdx.x - 1];
      const unsigned cur  = sh.hist[threadIdx.x];
      if (prev < (unsigned) k && cur >= (unsigned) k)
      {
        sh.bucket   = threadIdx.x;
        sh.cands    = cur - prev;
        sh.selected = prev;
      }
    }
    __syncthreads();
    stamp(2 + pass * 4 + 3);
    k -= (int) sh.selected;
    num_candidates = (int) sh.cands;
    total_selected += (int) sh.selected;
    const unsigned kth_digit = 255 - sh.bucket;
    kth_prefix |= kth_digit << pass_begin;
    prefix_mask |= 0xffu << pass_begin;
    if (num_candidates == k)
    {
      ++pass;
      break;
    }
  }
  __syncthreads(); // repurpose shared memory
  stamp(18);
  if (threadIdx.x == 0 && d_passes)
  {
    *d_passes = pass;
  }

  const bool select_all = (num_candidates + total_selected == kK);
  if (threadIdx.x == 0)
  {
    sh.sel_off[0] = 0;
    sh.sel_off[1] = (unsigned) total_selected;
  }
  int scatter_idx[4] = {-1, -1, -1, -1};
  __syncthreads();
  stamp(19);
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    const unsigned kp  = uk[i] & prefix_mask;
    const bool is_sel  = kp > kth_prefix;
    const bool is_cand = kp == kth_prefix;
    const int cls      = (!select_all && is_cand) ? 1 : 0;
    if (is_sel || is_cand)
    {
      const unsigned off = atomicAdd(&sh.sel_off[cls], 1u);
      sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
      scatter_idx[i]     = (int) off;
    }
  }
  __syncthreads();
  stamp(20);
  float outk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    const int bi = threadIdx.x * 4 + i;
    outk[i]      = (bi < kK) ? sh.exch.keys[bi] : 0.f;
  }
  __syncthreads(); // keys read back; repurpose exchange for values
  stamp(21);
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    if (scatter_idx[i] >= 0)
    {
      sh.exch.vals[scatter_idx[i]] = val[i];
    }
  }
  __syncthreads();
  stamp(22);
  if (threadIdx.x < kK / kIpt)
  {
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      out_v[threadIdx.x * 4 + i] = outk[i];
      out_i[threadIdx.x * 4 + i] = sh.exch.vals[threadIdx.x * 4 + i];
    }
  }
  __syncthreads();
  stamp(23);
}

struct ProtoAirReimpl
{
  static constexpr const char* name = "air_reimpl (parity gate)";
  using Smem                        = AirSmem;
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    air_reimpl_run(v, idx, sh, out_v, out_i, NoStamp{}, nullptr);
  }
};

// ------------------------------------------------------------------ v2: fused / barrier diet
// Changes vs air (algorithm identical):
//   * pass loop fully unrolled -> immediate shifts/masks (no runtime bit arithmetic)
//   * double-buffered histograms: the buffer for pass p+1 is zeroed during pass p's histogram
//     phase (it was last read before pass p's state barrier) -> the init phase and its barrier
//     disappear from every pass
//   * scan fused with choose: exclusive = inclusive - input; the crossing test runs on
//     registers right after BlockScan; no histogram writeback, no choose phase, no extra barrier
//   -> 3 barriers/pass (histogram, BlockScan-internal, state) instead of 5, and ~260 fewer
//      smem accesses per pass.
struct ProtoAirFused
{
  static constexpr const char* name = "air_fused (v2: unroll+dbuf+fused scan)";
  struct Smem
  {
    unsigned hist[2][256];
    typename block_scan_t::TempStorage scan_temp;
    unsigned bucket, cands, selected;
    unsigned sel_off[2];
    union
    {
      float keys[kN];
      int vals[kN];
    } exch;
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    unsigned uk[4];
    int val[4];
    unsigned flip               = 0;
    constexpr unsigned tw_mzero = 0x7fffffffu;
    constexpr unsigned tw_pzero = 0x80000000u;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      uk[i]  = twiddle(v[i]);
      val[i] = idx[i];
      if (uk[i] == tw_mzero)
      {
        flip |= 1u << i;
        uk[i] = tw_pzero;
      }
    }
    sh.hist[0][threadIdx.x] = 0;
    sh.hist[1][threadIdx.x] = 0;
    __syncthreads();

    unsigned kth_prefix = 0, prefix_mask = 0;
    int k              = kK;
    int total_selected = 0;
    int num_candidates = kN;
#pragma unroll
    for (int pass = 0; pass < 4; ++pass)
    {
      const int pass_begin = 24 - 8 * pass;
      unsigned* cur        = sh.hist[pass & 1];
      unsigned* nxt        = sh.hist[(pass + 1) & 1];
      // histogram over candidates; zero the other buffer in the same phase
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if ((uk[i] & prefix_mask) == kth_prefix)
        {
          const int digit = (uk[i] >> pass_begin) & 0xff;
          atomicAdd(&cur[255 - digit], 1u);
        }
      }
      if (pass > 0 && pass < 3)
      {
        nxt[threadIdx.x] = 0;
      }
      __syncthreads();
      // fused scan + choose: no writeback, crossing test on registers
      const unsigned cnt = cur[threadIdx.x];
      unsigned incl;
      block_scan_t(sh.scan_temp).InclusiveSum(cnt, incl);
      const unsigned excl = incl - cnt;
      if (excl < (unsigned) k && incl >= (unsigned) k)
      {
        sh.bucket   = threadIdx.x;
        sh.cands    = incl - excl;
        sh.selected = excl;
      }
      __syncthreads();
      k -= (int) sh.selected;
      num_candidates = (int) sh.cands;
      total_selected += (int) sh.selected;
      const unsigned kth_digit = 255 - sh.bucket;
      kth_prefix |= kth_digit << pass_begin;
      prefix_mask |= 0xffu << pass_begin;
      if (num_candidates == k)
      {
        break;
      }
    }
    __syncthreads();

    const bool select_all = (num_candidates + total_selected == kK);
    if (threadIdx.x == 0)
    {
      sh.sel_off[0] = 0;
      sh.sel_off[1] = (unsigned) total_selected;
    }
    int scatter_idx[4] = {-1, -1, -1, -1};
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const unsigned kp  = uk[i] & prefix_mask;
      const bool is_sel  = kp > kth_prefix;
      const bool is_cand = kp == kth_prefix;
      const int cls      = (!select_all && is_cand) ? 1 : 0;
      if (is_sel || is_cand)
      {
        const unsigned off = atomicAdd(&sh.sel_off[cls], 1u);
        sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
        scatter_idx[i]     = (int) off;
      }
    }
    __syncthreads();
    float outk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const int bi = threadIdx.x * 4 + i;
      outk[i]      = (bi < kK) ? sh.exch.keys[bi] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (scatter_idx[i] >= 0)
      {
        sh.exch.vals[scatter_idx[i]] = val[i];
      }
    }
    __syncthreads();
    if (threadIdx.x < kK / kIpt)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        out_v[threadIdx.x * 4 + i] = outk[i];
        out_i[threadIdx.x * 4 + i] = sh.exch.vals[threadIdx.x * 4 + i];
      }
    }
    __syncthreads();
  }
};

// ------------------------------------------------------------------ v3: explicit warp scan
// v2 with BlockScan replaced by an explicit two-level scan: warp shfl inclusive scan, lane 31
// posts the warp aggregate, one barrier, every thread folds the preceding warps' aggregates in
// registers (8 broadcast LDS). Isolates what BLOCK_SCAN_WARP_SCANS itself costs.
struct ProtoAirWscan
{
  static constexpr const char* name = "air_wscan (v3: explicit warp scan)";
  struct Smem
  {
    unsigned hist[2][256];
    unsigned wagg[8];
    unsigned bucket, cands, selected;
    unsigned sel_off[2];
    union
    {
      float keys[kN];
      int vals[kN];
    } exch;
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned uk[4];
    int val[4];
    unsigned flip               = 0;
    constexpr unsigned tw_mzero = 0x7fffffffu;
    constexpr unsigned tw_pzero = 0x80000000u;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      uk[i]  = twiddle(v[i]);
      val[i] = idx[i];
      if (uk[i] == tw_mzero)
      {
        flip |= 1u << i;
        uk[i] = tw_pzero;
      }
    }
    sh.hist[0][threadIdx.x] = 0;
    sh.hist[1][threadIdx.x] = 0;
    __syncthreads();

    unsigned kth_prefix = 0, prefix_mask = 0;
    int k              = kK;
    int total_selected = 0;
    int num_candidates = kN;
#pragma unroll
    for (int pass = 0; pass < 4; ++pass)
    {
      const int pass_begin = 24 - 8 * pass;
      unsigned* cur        = sh.hist[pass & 1];
      unsigned* nxt        = sh.hist[(pass + 1) & 1];
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if ((uk[i] & prefix_mask) == kth_prefix)
        {
          const int digit = (uk[i] >> pass_begin) & 0xff;
          atomicAdd(&cur[255 - digit], 1u);
        }
      }
      if (pass > 0 && pass < 3)
      {
        nxt[threadIdx.x] = 0;
      }
      __syncthreads();
      // warp-level inclusive scan of this thread's bin
      const unsigned cnt = cur[threadIdx.x];
      unsigned incl      = cnt;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1)
      {
        const unsigned o = __shfl_up_sync(kFull, incl, d);
        if (lane >= d)
        {
          incl += o;
        }
      }
      if (lane == 31)
      {
        sh.wagg[warp] = incl;
      }
      __syncthreads();
      // fold preceding warps' aggregates (broadcast reads), fused crossing test
#pragma unroll
      for (int w = 0; w < 8; ++w)
      {
        incl += (w < warp) ? sh.wagg[w] : 0u;
      }
      const unsigned excl = incl - cnt;
      if (excl < (unsigned) k && incl >= (unsigned) k)
      {
        sh.bucket   = threadIdx.x;
        sh.cands    = incl - excl;
        sh.selected = excl;
      }
      __syncthreads();
      k -= (int) sh.selected;
      num_candidates = (int) sh.cands;
      total_selected += (int) sh.selected;
      const unsigned kth_digit = 255 - sh.bucket;
      kth_prefix |= kth_digit << pass_begin;
      prefix_mask |= 0xffu << pass_begin;
      if (num_candidates == k)
      {
        break;
      }
    }
    __syncthreads();

    const bool select_all = (num_candidates + total_selected == kK);
    if (threadIdx.x == 0)
    {
      sh.sel_off[0] = 0;
      sh.sel_off[1] = (unsigned) total_selected;
    }
    int scatter_idx[4] = {-1, -1, -1, -1};
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const unsigned kp  = uk[i] & prefix_mask;
      const bool is_sel  = kp > kth_prefix;
      const bool is_cand = kp == kth_prefix;
      const int cls      = (!select_all && is_cand) ? 1 : 0;
      if (is_sel || is_cand)
      {
        const unsigned off = atomicAdd(&sh.sel_off[cls], 1u);
        sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
        scatter_idx[i]     = (int) off;
      }
    }
    __syncthreads();
    float outk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const int bi = threadIdx.x * 4 + i;
      outk[i]      = (bi < kK) ? sh.exch.keys[bi] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (scatter_idx[i] >= 0)
      {
        sh.exch.vals[scatter_idx[i]] = val[i];
      }
    }
    __syncthreads();
    if (threadIdx.x < kK / kIpt)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        out_v[threadIdx.x * 4 + i] = outk[i];
        out_i[threadIdx.x * 4 + i] = sh.exch.vals[threadIdx.x * 4 + i];
      }
    }
    __syncthreads();
  }
};

// ------------------------------------------------------------------ v4: eager scatter
// v2 + the class-0 ("certainly selected") scatter folded into the pass loop: an item that became
// certain in pass p (candidate under the old prefix, strictly above the new one) is scattered
// during pass p+1's histogram phase, sharing its barrier. The epilogue only handles the final
// pass's pending items and the boundary candidates. Exchange no longer aliases the histograms.
struct ProtoAirEager
{
  static constexpr const char* name = "air_eager (v4: scatter fused into passes)";
  struct Smem
  {
    unsigned hist[2][256];
    typename block_scan_t::TempStorage scan_temp;
    unsigned bucket, cands, selected;
    unsigned sel_cnt, sel_off1;
    union
    {
      float keys[kN];
      int vals[kN];
    } exch;
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    unsigned uk[4];
    int val[4];
    unsigned flip               = 0;
    constexpr unsigned tw_mzero = 0x7fffffffu;
    constexpr unsigned tw_pzero = 0x80000000u;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      uk[i]  = twiddle(v[i]);
      val[i] = idx[i];
      if (uk[i] == tw_mzero)
      {
        flip |= 1u << i;
        uk[i] = tw_pzero;
      }
    }
    sh.hist[0][threadIdx.x] = 0;
    sh.hist[1][threadIdx.x] = 0;
    if (threadIdx.x == 0)
    {
      sh.sel_cnt = 0;
    }
    __syncthreads();

    unsigned kth_prefix = 0, prefix_mask = 0;
    int k              = kK;
    int total_selected = 0;
    int num_candidates = kN;
    unsigned pending   = 0; // items that became certain last pass, not yet scattered
    int scatter_idx[4] = {-1, -1, -1, -1};
#pragma unroll
    for (int pass = 0; pass < 4; ++pass)
    {
      const int pass_begin = 24 - 8 * pass;
      unsigned* cur        = sh.hist[pass & 1];
      unsigned* nxt        = sh.hist[(pass + 1) & 1];
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if ((uk[i] & prefix_mask) == kth_prefix)
        {
          const int digit = (uk[i] >> pass_begin) & 0xff;
          atomicAdd(&cur[255 - digit], 1u);
        }
      }
      if (pass > 0 && pass < 3)
      {
        nxt[threadIdx.x] = 0;
      }
      // eager scatter of last pass's certain items, sharing the histogram phase
      if (pending)
      {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
          if ((pending >> i) & 1)
          {
            const unsigned off = atomicAdd(&sh.sel_cnt, 1u);
            sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
            scatter_idx[i]     = (int) off;
          }
        }
        pending = 0;
      }
      __syncthreads();
      const unsigned cnt = cur[threadIdx.x];
      unsigned incl;
      block_scan_t(sh.scan_temp).InclusiveSum(cnt, incl);
      const unsigned excl = incl - cnt;
      if (excl < (unsigned) k && incl >= (unsigned) k)
      {
        sh.bucket   = threadIdx.x;
        sh.cands    = incl - excl;
        sh.selected = excl;
      }
      __syncthreads();
      k -= (int) sh.selected;
      num_candidates = (int) sh.cands;
      total_selected += (int) sh.selected;
      const unsigned old_prefix = kth_prefix;
      const unsigned old_mask   = prefix_mask;
      const unsigned kth_digit  = 255 - sh.bucket;
      kth_prefix |= kth_digit << pass_begin;
      prefix_mask |= 0xffu << pass_begin;
      // items that just became certain: candidates under the old prefix, strictly above the new
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        const bool was_cand = (uk[i] & old_mask) == old_prefix;
        const bool now_sel  = (uk[i] & prefix_mask) > kth_prefix;
        pending |= (was_cand && now_sel) ? (1u << i) : 0u;
      }
      if (num_candidates == k)
      {
        break;
      }
    }
    // epilogue: final pending items, then boundary candidates
    if (pending)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if ((pending >> i) & 1)
        {
          const unsigned off = atomicAdd(&sh.sel_cnt, 1u);
          sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
          scatter_idx[i]     = (int) off;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      sh.sel_off1 = (unsigned) total_selected; // == sel_cnt now
    }
    const bool select_all = (num_candidates + total_selected == kK);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if ((uk[i] & prefix_mask) == kth_prefix)
      {
        const unsigned off = atomicAdd(select_all ? &sh.sel_cnt : &sh.sel_off1, 1u);
        sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
        scatter_idx[i]     = (int) off;
      }
    }
    __syncthreads();
    float outk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const int bi = threadIdx.x * 4 + i;
      outk[i]      = (bi < kK) ? sh.exch.keys[bi] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (scatter_idx[i] >= 0)
      {
        sh.exch.vals[scatter_idx[i]] = val[i];
      }
    }
    __syncthreads();
    if (threadIdx.x < kK / kIpt)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        out_v[threadIdx.x * 4 + i] = outk[i];
        out_i[threadIdx.x * 4 + i] = sh.exch.vals[threadIdx.x * 4 + i];
      }
    }
    __syncthreads();
  }
};

// ------------------------------------------------------------------ v5: warp specialization + split barriers
// The histogram->scan edge uses a producer/consumer split: warps 1..7 post their histogram
// atomics and *arrive* on named barrier 1 without blocking, then eager-scatter last pass's
// certain items while warp 0 (which *syncs* on barrier 1, i.e. waits for all histogram traffic)
// scans the 256 bins alone: stride-9 padded bins -> conflict-free 8-bins-per-lane loads, warp
// shfl prefix, in-lane walk to the crossing. Everyone rejoins at __syncthreads. The class-0
// scatter thus runs concurrently with the find-bucket step instead of sequentially.
struct ProtoAirWspec
{
  static constexpr const char* name = "air_wspec (v5: split-bar warp-spec scan)";
  struct Smem
  {
    unsigned hist[2][288]; // stride-9 padded bins: bin b lives at b + b/8
    unsigned bucket, cands, selected;
    unsigned sel_cnt, sel_off1;
    union
    {
      float keys[kN];
      int vals[kN];
    } exch;
  };
  __device__ __forceinline__ static unsigned slot_of(unsigned b)
  {
    return b + (b >> 3);
  }
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned uk[4];
    int val[4];
    unsigned flip               = 0;
    constexpr unsigned tw_mzero = 0x7fffffffu;
    constexpr unsigned tw_pzero = 0x80000000u;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      uk[i]  = twiddle(v[i]);
      val[i] = idx[i];
      if (uk[i] == tw_mzero)
      {
        flip |= 1u << i;
        uk[i] = tw_pzero;
      }
    }
#pragma unroll
    for (int i = threadIdx.x; i < 288; i += kBlock)
    {
      sh.hist[0][i] = 0;
      sh.hist[1][i] = 0;
    }
    if (threadIdx.x == 0)
    {
      sh.sel_cnt = 0;
    }
    __syncthreads();

    unsigned kth_prefix = 0, prefix_mask = 0;
    int k              = kK;
    int total_selected = 0;
    int num_candidates = kN;
    unsigned pending   = 0;
    int scatter_idx[4] = {-1, -1, -1, -1};
#pragma unroll
    for (int pass = 0; pass < 4; ++pass)
    {
      const int pass_begin = 24 - 8 * pass;
      unsigned* cur        = sh.hist[pass & 1];
      unsigned* nxt        = sh.hist[(pass + 1) & 1];
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if ((uk[i] & prefix_mask) == kth_prefix)
        {
          const unsigned digit = (uk[i] >> pass_begin) & 0xffu;
          atomicAdd(&cur[slot_of(255 - digit)], 1u);
        }
      }
      if (pass > 0 && pass < 3)
      {
#pragma unroll
        for (int i = threadIdx.x; i < 288; i += kBlock)
        {
          nxt[i] = 0;
        }
      }
      if (warp == 0)
      {
        // wait until every warp's histogram traffic (issued before its arrive) is visible
        asm volatile("barrier.cta.sync 1, 256;" ::: "memory");
        // one-warp scan: lane l owns buckets 8l..8l+7 (conflict-free at stride 9)
        unsigned h[8];
        unsigned s = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
          h[j] = cur[9 * lane + j];
          s += h[j];
        }
        unsigned excl_lane = s; // -> exclusive prefix of lane sums
#pragma unroll
        for (int d = 1; d < 32; d <<= 1)
        {
          const unsigned o = __shfl_up_sync(kFull, excl_lane, d);
          if (lane >= d)
          {
            excl_lane += o;
          }
        }
        excl_lane -= s;
        unsigned cum = excl_lane;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
          const unsigned prev = cum;
          cum += h[j];
          if (prev < (unsigned) k && cum >= (unsigned) k)
          {
            sh.bucket   = 8 * lane + j;
            sh.cands    = cum - prev;
            sh.selected = prev;
          }
        }
        // warp 0 scatters its own pending items after the scan
        if (pending)
        {
#pragma unroll
          for (int i = 0; i < 4; ++i)
          {
            if ((pending >> i) & 1)
            {
              const unsigned off = atomicAdd(&sh.sel_cnt, 1u);
              sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
              scatter_idx[i]     = (int) off;
            }
          }
          pending = 0;
        }
      }
      else
      {
        // non-blocking arrive, then overlap the class-0 scatter with warp 0's scan
        asm volatile("barrier.cta.arrive 1, 256;" ::: "memory");
        if (pending)
        {
#pragma unroll
          for (int i = 0; i < 4; ++i)
          {
            if ((pending >> i) & 1)
            {
              const unsigned off = atomicAdd(&sh.sel_cnt, 1u);
              sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
              scatter_idx[i]     = (int) off;
            }
          }
          pending = 0;
        }
      }
      __syncthreads(); // rejoin; state visible
      k -= (int) sh.selected;
      num_candidates = (int) sh.cands;
      total_selected += (int) sh.selected;
      const unsigned old_prefix = kth_prefix;
      const unsigned old_mask   = prefix_mask;
      const unsigned kth_digit  = 255 - sh.bucket;
      kth_prefix |= kth_digit << pass_begin;
      prefix_mask |= 0xffu << pass_begin;
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        const bool was_cand = (uk[i] & old_mask) == old_prefix;
        const bool now_sel  = (uk[i] & prefix_mask) > kth_prefix;
        pending |= (was_cand && now_sel) ? (1u << i) : 0u;
      }
      if (num_candidates == k)
      {
        break;
      }
    }
    if (pending)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if ((pending >> i) & 1)
        {
          const unsigned off = atomicAdd(&sh.sel_cnt, 1u);
          sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
          scatter_idx[i]     = (int) off;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      sh.sel_off1 = (unsigned) total_selected;
    }
    const bool select_all = (num_candidates + total_selected == kK);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if ((uk[i] & prefix_mask) == kth_prefix)
      {
        const unsigned off = atomicAdd(select_all ? &sh.sel_cnt : &sh.sel_off1, 1u);
        sh.exch.keys[off]  = ((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]);
        scatter_idx[i]     = (int) off;
      }
    }
    __syncthreads();
    float outk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const int bi = threadIdx.x * 4 + i;
      outk[i]      = (bi < kK) ? sh.exch.keys[bi] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (scatter_idx[i] >= 0)
      {
        sh.exch.vals[scatter_idx[i]] = val[i];
      }
    }
    __syncthreads();
    if (threadIdx.x < kK / kIpt)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        out_v[threadIdx.x * 4 + i] = outk[i];
        out_i[threadIdx.x * 4 + i] = sh.exch.vals[threadIdx.x * 4 + i];
      }
    }
    __syncthreads();
  }
};

// ------------------------------------------------------------------ v6: v2 + packed state + pair epilogue
// Profile-driven additions on top of air_fused:
//   * pass state (bucket | candidates | selected) packed into ONE 32-bit word -> the per-pass
//     state broadcast is a single shared load instead of three dependent ones
//   * keys and values scattered TOGETHER as 8-byte pairs and gathered once -> the epilogue
//     shrinks from [setup | scatter keys | gather keys | scatter vals | gather vals] (4 barriers)
//     to [scatter pairs | gather pairs] (2 barriers)
//   * scatter counters preset in the prologue; class-1 position = total_selected + zero-based
//     ticket, so the setup phase (and its barrier) disappears entirely
struct ProtoAirPair
{
  static constexpr const char* name = "air_pair (v6: +packed state, pair scatter)";
  struct Pair
  {
    float k;
    int v;
  };
  // histograms/scan are dead once the pass loop ends (last read precedes the final state
  // barrier); the pair exchange is only written after it -> union them (saves 2.2 KB)
  struct Smem
  {
    union
    {
      struct
      {
        unsigned hist[2][256];
        typename block_scan_t::TempStorage scan_temp;
      } passes;
      Pair exch[kN];
    } stage;
    unsigned state; // bucket(8) | candidates(11) | selected(5)
    unsigned cntA, cntB;
  };
  __device__ __forceinline__ static void
  run(const float (&v)[4], const int (&idx)[4], Smem& sh, float* out_v, int* out_i)
  {
    unsigned uk[4];
    int val[4];
    unsigned flip               = 0;
    constexpr unsigned tw_mzero = 0x7fffffffu;
    constexpr unsigned tw_pzero = 0x80000000u;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      uk[i]  = twiddle(v[i]);
      val[i] = idx[i];
      if (uk[i] == tw_mzero)
      {
        flip |= 1u << i;
        uk[i] = tw_pzero;
      }
    }
    sh.stage.passes.hist[0][threadIdx.x] = 0;
    sh.stage.passes.hist[1][threadIdx.x] = 0;
    if (threadIdx.x == 0)
    {
      sh.cntA = 0;
      sh.cntB = 0;
    }
    __syncthreads();

    unsigned kth_prefix = 0, prefix_mask = 0;
    int k              = kK;
    int total_selected = 0;
    int num_candidates = kN;
#pragma unroll
    for (int pass = 0; pass < 4; ++pass)
    {
      const int pass_begin = 24 - 8 * pass;
      unsigned* cur        = sh.stage.passes.hist[pass & 1];
      unsigned* nxt        = sh.stage.passes.hist[(pass + 1) & 1];
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if ((uk[i] & prefix_mask) == kth_prefix)
        {
          const int digit = (uk[i] >> pass_begin) & 0xff;
          atomicAdd(&cur[255 - digit], 1u);
        }
      }
      if (pass > 0 && pass < 3)
      {
        nxt[threadIdx.x] = 0;
      }
      __syncthreads();
      const unsigned cnt = cur[threadIdx.x];
      unsigned incl;
      block_scan_t(sh.stage.passes.scan_temp).InclusiveSum(cnt, incl);
      const unsigned excl = incl - cnt;
      if (excl < (unsigned) k && incl >= (unsigned) k)
      {
        sh.state = (threadIdx.x << 16) | ((incl - excl) << 5) | excl; // bucket | cands | selected
      }
      __syncthreads();
      const unsigned st       = sh.state;
      const unsigned selected = st & 0x1fu;
      k -= (int) selected;
      num_candidates = (int) ((st >> 5) & 0x7ffu);
      total_selected += (int) selected;
      const unsigned kth_digit = 255 - (st >> 16);
      kth_prefix |= kth_digit << pass_begin;
      prefix_mask |= 0xffu << pass_begin;
      if (num_candidates == k)
      {
        break;
      }
    }
    // epilogue: single pair scatter, single pair gather (no setup phase, no post-loop barrier:
    // the last state barrier already orders everything, and the counters were preset)
    const bool select_all = (num_candidates + total_selected == kK);
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const unsigned kp  = uk[i] & prefix_mask;
      const bool is_sel  = kp > kth_prefix;
      const bool is_cand = kp == kth_prefix;
      if (is_sel || is_cand)
      {
        const bool cls1    = !select_all && is_cand;
        const unsigned t   = atomicAdd(cls1 ? &sh.cntB : &sh.cntA, 1u);
        const unsigned off = cls1 ? (unsigned) total_selected + t : t;
        sh.stage.exch[off]  = Pair{((flip >> i) & 1) ? -0.0f : untwiddle(uk[i]), val[i]};
      }
    }
    __syncthreads();
    if (threadIdx.x < kK)
    {
      const Pair p       = sh.stage.exch[threadIdx.x];
      out_v[threadIdx.x] = p.k;
      out_i[threadIdx.x] = p.v;
    }
    __syncthreads();
  }
};

// ------------------------------------------------------------------ kernels (same harness as proto_topk.cu)
template <class P>
struct Box
{
  typename P::Smem s;
  float out_v[kK];
  int out_i[kK];
};

template <class P>
__global__ void __launch_bounds__(kBlock) correct_kernel(const float* in, float* out_v, int* out_i)
{
  __shared__ Box<P> box;
  float v[4];
  int idx[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    idx[i] = threadIdx.x * 4 + i;
    v[i]   = in[idx[i]];
  }
  P::run(v, idx, box.s, box.out_v, box.out_i);
  if (threadIdx.x < kK)
  {
    out_v[threadIdx.x] = box.out_v[threadIdx.x];
    out_i[threadIdx.x] = box.out_i[threadIdx.x];
  }
}

constexpr int kLatReps = 24;

template <class P>
__global__ void __launch_bounds__(kBlock) lat_kernel(const float* in, int chain, long long* out)
{
  __shared__ Box<P> box;
  float v0[4];
  int idx[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    idx[i] = threadIdx.x * 4 + i;
    v0[i]  = in[idx[i]];
  }
  P::run(v0, idx, box.s, box.out_v, box.out_i);
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int rep = 0; rep < kLatReps; ++rep)
  {
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll 1
    for (int n = 0; n < chain; ++n)
    {
      float v[4];
      v[0] = fmaf(box.out_v[0], 0.0f, v0[0]);
      v[1] = v0[1];
      v[2] = v0[2];
      v[3] = v0[3];
      P::run(v, idx, box.s, box.out_v, box.out_i);
    }
    float f = box.out_v[0];
    asm volatile("" ::"f"(f));
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    *out = best;
  }
  if (box.out_i[0] == -12345)
  {
    g_sink = box.out_i[0];
  }
}

// prof: chained calls; per-stage deltas, min across calls & reps (steady state)
__global__ void __launch_bounds__(kBlock) prof_kernel(const float* in, long long* d_acc, int* d_passes)
{
  __shared__ Box<ProtoAirReimpl> box;
  __shared__ long long ts[kStamps];
  __shared__ long long acc[kStamps];
  float v0[4];
  int idx[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    idx[i] = threadIdx.x * 4 + i;
    v0[i]  = in[idx[i]];
  }
  if (threadIdx.x == 0)
  {
    for (int i = 0; i < kStamps; ++i)
    {
      acc[i] = LLONG_MAX;
    }
  }
  air_reimpl_run(v0, idx, box.s, box.out_v, box.out_i, NoStamp{}, nullptr); // warmup
#pragma unroll 1
  for (int rep = 0; rep < kLatReps; ++rep)
  {
    __syncthreads();
#pragma unroll 1
    for (int n = 0; n < 4; ++n)
    {
      float v[4];
      v[0] = fmaf(box.out_v[0], 0.0f, v0[0]);
      v[1] = v0[1];
      v[2] = v0[2];
      v[3] = v0[3];
      air_reimpl_run(v, idx, box.s, box.out_v, box.out_i, SmemStamp{ts}, d_passes);
      if (threadIdx.x == 0)
      {
#pragma unroll 1
        for (int i = 0; i + 1 < kStamps; ++i)
        {
          const long long d = ts[i + 1] - ts[i];
          if (d >= 0 && d < acc[i])
          {
            acc[i] = d;
          }
        }
      }
    }
  }
  if (threadIdx.x == 0)
  {
    for (int i = 0; i < kStamps; ++i)
    {
      d_acc[i] = acc[i];
    }
  }
}

template <class P>
__global__ void __launch_bounds__(kBlock) thr_kernel(const float* in, int outer, float* sink)
{
  __shared__ Box<P> box;
  float v0[4];
  int idx[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    idx[i] = threadIdx.x * 4 + i;
    v0[i]  = in[idx[i]];
  }
  P::run(v0, idx, box.s, box.out_v, box.out_i);
#pragma unroll 1
  for (int n = 0; n < outer; ++n)
  {
    float v[4];
    v[0] = fmaf(box.out_v[0], 0.0f, v0[0]);
    v[1] = v0[1];
    v[2] = v0[2];
    v[3] = v0[3];
    P::run(v, idx, box.s, box.out_v, box.out_i);
  }
  if (threadIdx.x < kK)
  {
    sink[blockIdx.x * kK + threadIdx.x] = box.out_v[threadIdx.x];
  }
}

// ------------------------------------------------------------------ host (patterns/validate as before)
struct Lcg
{
  unsigned s;
  explicit Lcg(unsigned seed)
      : s(seed)
  {}
  unsigned next()
  {
    s = 1664525u * s + 1013904223u;
    return s;
  }
  float uniform()
  {
    return (next() >> 8) * (1.0f / 16777216.0f);
  }
};

static std::vector<float> gen_pattern(const std::string& p, unsigned seed)
{
  std::vector<float> v(kN);
  Lcg rng(seed * 2654435761u + 12345u);
  auto normal = [&]() {
    float u1 = std::max(rng.uniform(), 1e-7f);
    float u2 = rng.uniform();
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.28318530718f * u2);
  };
  auto quantize = [](float base) {
    float r  = std::rint(base);
    float fr = std::rint((base - r) * 32.0f);
    return r + fr / 32.0f;
  };
  if (p == "random")
  {
    for (auto& x : v)
    {
      x = normal();
    }
  }
  else if (p == "quantized_random")
  {
    for (auto& x : v)
    {
      x = quantize(normal());
    }
  }
  else if (p == "relu_quantized")
  {
    for (auto& x : v)
    {
      x = quantize(std::max(normal(), 0.f));
    }
  }
  else if (p == "tie_heavy")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = (float) (j % 64) / 64.f;
    }
  }
  else if (p == "pivot_tie4" || p == "pivot_tie40")
  {
    const int gt = (p == "pivot_tie4") ? 4 : 40;
    for (auto& x : v)
    {
      x = 1.f;
    }
    int placed = 0;
    while (placed < gt)
    {
      unsigned pos = rng.next() % kN;
      if (v[pos] != 2.f)
      {
        v[pos] = 2.f;
        ++placed;
      }
    }
  }
  else if (p == "all_equal")
  {
    for (auto& x : v)
    {
      x = 1.f;
    }
  }
  else if (p == "sorted_asc")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = (float) j / kN;
    }
  }
  else if (p == "neg_zero")
  {
    // exercise the -0.0 restoration path: top-16 includes -0.0 and +0.0
    for (int j = 0; j < kN; ++j)
    {
      v[j] = -1.f - (float) j;
    }
    for (int j = 0; j < 8; ++j)
    {
      v[100 + j] = -0.0f;
      v[200 + j] = +0.0f;
    }
  }
  else
  {
    printf("unknown pattern %s\n", p.c_str());
    exit(1);
  }
  return v;
}

static bool validate(const std::vector<float>& in, const float* ov, const int* oi, std::string& why)
{
  std::vector<float> sorted = in;
  std::sort(sorted.begin(), sorted.end(), [](float a, float b) {
    return a > b;
  });
  std::vector<float> want(sorted.begin(), sorted.begin() + kK);
  std::vector<float> got(ov, ov + kK);
  std::sort(want.begin(), want.end());
  std::sort(got.begin(), got.end());
  for (int i = 0; i < kK; ++i)
  {
    if (!(want[i] == got[i]))
    {
      char b[128];
      snprintf(b, sizeof b, "value multiset mismatch at %d: want %g got %g", i, want[i], got[i]);
      why = b;
      return false;
    }
  }
  bool used[kN] = {};
  for (int i = 0; i < kK; ++i)
  {
    if (oi[i] < 0 || oi[i] >= kN)
    {
      why = "index out of range";
      return false;
    }
    if (used[oi[i]])
    {
      why = "duplicate index " + std::to_string(oi[i]);
      return false;
    }
    used[oi[i]] = true;
    if (!(in[oi[i]] == ov[i]))
    {
      char b[128];
      snprintf(b, sizeof b, "index %d holds %g but output value is %g", oi[i], in[oi[i]], ov[i]);
      why = b;
      return false;
    }
  }
  return true;
}

static void fit(const double* x, const double* y, int m, double& a, double& b)
{
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = 0; i < m; ++i)
  {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  b = (m * sxy - sx * sy) / (m * sxx - sx * sx);
  a = (sy - b * sx) / m;
}

#define CHECK(call)                                                          \
  do                                                                         \
  {                                                                          \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess)                                                    \
    {                                                                        \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

template <class P>
void run_correct()
{
  const char* pats[] = {
    "random",
    "quantized_random",
    "relu_quantized",
    "tie_heavy",
    "pivot_tie4",
    "pivot_tie40",
    "all_equal",
    "sorted_asc",
    "neg_zero"};
  float *d_in, *d_ov;
  int* d_oi;
  CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CHECK(cudaMalloc(&d_ov, kK * sizeof(float)));
  CHECK(cudaMalloc(&d_oi, kK * sizeof(int)));
  int fails = 0, runs = 0;
  for (const char* pat : pats)
  {
    const int seeds = (std::string(pat) == "random" || std::string(pat) == "quantized_random") ? 8 : 2;
    for (int seed = 0; seed < seeds; ++seed)
    {
      auto in = gen_pattern(pat, seed);
      CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
      correct_kernel<P><<<1, kBlock>>>(d_in, d_ov, d_oi);
      CHECK(cudaDeviceSynchronize());
      float ov[kK];
      int oi[kK];
      CHECK(cudaMemcpy(ov, d_ov, sizeof ov, cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(oi, d_oi, sizeof oi, cudaMemcpyDeviceToHost));
      std::string why;
      ++runs;
      if (!validate(in, ov, oi, why))
      {
        ++fails;
        printf("    FAIL %-16s seed %d: %s\n", pat, seed, why.c_str());
      }
    }
  }
  printf("  %-42s %s (%d runs)\n", P::name, fails ? "FAIL" : "PASS", runs);
  cudaFree(d_in);
  cudaFree(d_ov);
  cudaFree(d_oi);
}

template <class P>
void run_lat()
{
  const char* pats[] = {"random", "tie_heavy", "pivot_tie40", "sorted_asc"};
  float* d_in;
  long long* d_out;
  CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CHECK(cudaMalloc(&d_out, sizeof(long long)));
  printf("  %-42s", P::name);
  for (const char* pat : pats)
  {
    auto in = gen_pattern(pat, 0);
    CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
    const int chains[] = {1, 2, 4, 8, 16};
    double x[5], y[5];
    for (int i = 0; i < 5; ++i)
    {
      lat_kernel<P><<<1, kBlock>>>(d_in, chains[i], d_out);
      CHECK(cudaDeviceSynchronize());
      long long c;
      CHECK(cudaMemcpy(&c, d_out, sizeof c, cudaMemcpyDeviceToHost));
      x[i] = chains[i];
      y[i] = (double) c;
    }
    double a, b;
    fit(x, y, 5, a, b);
    printf("  %s=%6.0f", pat, b);
  }
  printf("   cyc/call (slope)\n");
  cudaFree(d_in);
  cudaFree(d_out);
}

static const char* stage_names[kStamps - 1] = {
  "twiddle+load",
  "p0 init(+bar)",
  "p0 histogram",
  "p0 scan+wb",
  "p0 choose",
  "p1 init(+bar)",
  "p1 histogram",
  "p1 scan+wb",
  "p1 choose",
  "p2 init(+bar)",
  "p2 histogram",
  "p2 scan+wb",
  "p2 choose",
  "p3 init(+bar)",
  "p3 histogram",
  "p3 scan+wb",
  "p3 choose",
  "post-loop bar",
  "scatter setup",
  "scatter",
  "gather keys",
  "scatter vals",
  "gather vals+out",
};

void run_prof()
{
  const char* pats[] = {"random", "tie_heavy", "pivot_tie40", "sorted_asc"};
  float* d_in;
  long long* d_acc;
  int* d_passes;
  CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CHECK(cudaMalloc(&d_acc, kStamps * sizeof(long long)));
  CHECK(cudaMalloc(&d_passes, sizeof(int)));
  printf("\n=== per-phase cycles (air_reimpl, min across %d reps x 4 calls) ===\n", kLatReps);
  printf("  %-16s", "stage");
  for (const char* pat : pats)
  {
    printf("  %12s", pat);
  }
  printf("\n");
  long long acc[4][kStamps];
  int passes[4];
  for (int pi = 0; pi < 4; ++pi)
  {
    auto in = gen_pattern(pats[pi], 0);
    CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
    prof_kernel<<<1, kBlock>>>(d_in, d_acc, d_passes);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(acc[pi], d_acc, kStamps * sizeof(long long), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&passes[pi], d_passes, sizeof(int), cudaMemcpyDeviceToHost));
  }
  for (int s = 0; s < kStamps - 1; ++s)
  {
    // per-pass stages beyond the executed pass count report stale deltas; blank them
    printf("  %-16s", stage_names[s]);
    for (int pi = 0; pi < 4; ++pi)
    {
      const int pass_of_stage = (s >= 1 && s < 17) ? (s - 1) / 4 : -1;
      if (pass_of_stage >= 0 && pass_of_stage >= passes[pi])
      {
        printf("  %12s", "-");
      }
      else if (acc[pi][s] == LLONG_MAX)
      {
        printf("  %12s", "?");
      }
      else
      {
        printf("  %12lld", acc[pi][s]);
      }
    }
    printf("\n");
  }
  printf("  %-16s", "TOTAL(sum)");
  for (int pi = 0; pi < 4; ++pi)
  {
    long long tot = 0;
    for (int s = 0; s < kStamps - 1; ++s)
    {
      const int pass_of_stage = (s >= 1 && s < 17) ? (s - 1) / 4 : -1;
      if ((pass_of_stage < 0 || pass_of_stage < passes[pi]) && acc[pi][s] != LLONG_MAX)
      {
        tot += acc[pi][s];
      }
    }
    printf("  %12lld", tot);
  }
  printf("\n  %-16s", "passes");
  for (int pi = 0; pi < 4; ++pi)
  {
    printf("  %12d", passes[pi]);
  }
  printf("\n");
  cudaFree(d_in);
  cudaFree(d_acc);
  cudaFree(d_passes);
}

template <class P>
void run_thr()
{
  constexpr int grid = 2048, outer = 32;
  float *d_in, *d_sink;
  CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CHECK(cudaMalloc(&d_sink, (size_t) grid * kK * sizeof(float)));
  auto in = gen_pattern("random", 0);
  CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
  thr_kernel<P><<<grid, kBlock>>>(d_in, outer, d_sink);
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best_ms = 1e30f;
  for (int rep = 0; rep < 5; ++rep)
  {
    cudaEventRecord(e0);
    thr_kernel<P><<<grid, kBlock>>>(d_in, outer, d_sink);
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best_ms = std::min(best_ms, ms);
  }
  const double calls  = (double) grid * (outer + 1);
  const double gelems = calls * kN / (best_ms * 1e-3) / 1e9;
  printf("  %-42s %8.1f G elem/s   (%.3f ms)\n", P::name, gelems, best_ms);
  cudaFree(d_in);
  cudaFree(d_sink);
}

template <class P>
void run_res()
{
  cudaFuncAttributes a;
  CHECK(cudaFuncGetAttributes(&a, (const void*) thr_kernel<P>));
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*) thr_kernel<P>, kBlock, 0);
  printf("  %-42s regs=%3d  smem=%5zu B  local(spill)=%4zu B  maxblk/SM=%d\n",
         P::name,
         a.numRegs,
         a.sharedSizeBytes,
         a.localSizeBytes,
         occ);
}

#define FOREACH_PROTO(X) \
  X(ProtoAirRef)         \
  X(ProtoAirReimpl)      \
  X(ProtoAirFused)       \
  X(ProtoAirWscan)       \
  X(ProtoAirEager)       \
  X(ProtoAirWspec)       \
  X(ProtoAirPair)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness ===\n");
#define RUNC(P) run_correct<P>();
    FOREACH_PROTO(RUNC)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== latency: slope cyc/call ===\n");
#define RUNL(P) run_lat<P>();
    FOREACH_PROTO(RUNL)
  }
  if (mode == "prof" || mode == "all")
  {
    run_prof();
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== throughput ===\n");
#define RUNT(P) run_thr<P>();
    FOREACH_PROTO(RUNT)
  }
  if (mode == "res" || mode == "all")
  {
    printf("\n=== resources ===\n");
#define RUNR(P) run_res<P>();
    FOREACH_PROTO(RUNR)
  }
  return 0;
}
