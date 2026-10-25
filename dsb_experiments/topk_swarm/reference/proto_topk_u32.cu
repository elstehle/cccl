// uint32 keys edition of the block top-K comparison at (BLOCK=256, N=1024, K=16, pairs, sm_100).
// Companion to BLOCK_TOPK_RESULTS.md / BLOCK_TOPK_AIR_OPT_RESULTS.md (float editions).
//
// What changes for u32 vs float:
//   * no twiddle (identity bit-ordering) and no -0.0 flip-back handling anywhere
//   * key 0 is now a VALID key and collides with the atomic prototypes' empty-slot sentinel.
//     atomic_adaptive's value-based resolution absorbs this by dropping the `slot != 0` guard
//     in the post-hoc histogram (zero-valued items then match the trailing zero slots and land
//     in one bin, which is exactly right); atomic_iter's tier logic was already value-based and
//     needs no change. Dedicated `with_zeros` / `all_zero` patterns validate this edge.
//
// Prototypes: air_ref (cub header), air_fused (v2), air_pair (v6), atomic_adaptive, atomic_iter.
// Modes: ./proto_topk_u32 [correct|lat|thr|res|all]. Methodology identical to the float suites.

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

using block_scan_t = cub::BlockScan<unsigned, kBlock, cub::BLOCK_SCAN_WARP_SCANS>;

template <typename T>
__device__ __forceinline__ void cswap_desc(T& a, T& b)
{
  T lo = ::min(a, b);
  a    = ::max(a, b);
  b    = lo;
}

template <typename T>
__device__ __forceinline__ void sort4_desc(T (&v)[4])
{
  cswap_desc(v[0], v[1]);
  cswap_desc(v[2], v[3]);
  cswap_desc(v[0], v[2]);
  cswap_desc(v[1], v[3]);
  cswap_desc(v[1], v[2]);
}

// ------------------------------------------------------------------ air_ref (the real header)
struct U32AirRef
{
  static constexpr const char* name = "air_ref (cub header, u32)";
  using topk_t                      = cub::detail::block_topk_air<unsigned, kBlock, kIpt, int>;
  struct Smem
  {
    typename topk_t::TempStorage ts;
  };
  __device__ __forceinline__ static void
  run(const unsigned (&v)[4], const int (&idx)[4], Smem& sh, unsigned* out_v, int* out_i)
  {
    unsigned k[4];
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

// ------------------------------------------------------------------ air_fused (v2, u32)
struct U32AirFused
{
  static constexpr const char* name = "air_fused (v2, u32)";
  struct Smem
  {
    unsigned hist[2][256];
    typename block_scan_t::TempStorage scan_temp;
    unsigned bucket, cands, selected;
    unsigned sel_off[2];
    union
    {
      unsigned keys[kN];
      int vals[kN];
    } exch;
  };
  __device__ __forceinline__ static void
  run(const unsigned (&v)[4], const int (&idx)[4], Smem& sh, unsigned* out_v, int* out_i)
  {
    const unsigned (&uk)[4] = v; // identity bit-ordering for u32
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
        sh.exch.keys[off]  = uk[i];
        scatter_idx[i]     = (int) off;
      }
    }
    __syncthreads();
    unsigned outk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const int bi = threadIdx.x * 4 + i;
      outk[i]      = (bi < kK) ? sh.exch.keys[bi] : 0u;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (scatter_idx[i] >= 0)
      {
        sh.exch.vals[scatter_idx[i]] = idx[i];
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

// ------------------------------------------------------------------ air_pair (v6, u32)
struct U32AirPair
{
  static constexpr const char* name = "air_pair (v6, u32)";
  struct Pair
  {
    unsigned k;
    int v;
  };
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
  run(const unsigned (&v)[4], const int (&idx)[4], Smem& sh, unsigned* out_v, int* out_i)
  {
    const unsigned (&uk)[4]              = v;
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
        sh.state = (threadIdx.x << 16) | ((incl - excl) << 5) | excl;
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
    const bool select_all = (num_candidates + total_selected == kK);
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const unsigned kp  = uk[i] & prefix_mask;
      const bool is_sel  = kp > kth_prefix;
      const bool is_cand = kp == kth_prefix;
      if (is_sel || is_cand)
      {
        const bool cls1     = !select_all && is_cand;
        const unsigned t    = atomicAdd(cls1 ? &sh.cntB : &sh.cntA, 1u);
        const unsigned off  = cls1 ? (unsigned) total_selected + t : t;
        sh.stage.exch[off]  = Pair{uk[i], idx[i]};
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

// ------------------------------------------------------------------ atomic_adaptive (u32)
// Identical to the float version minus the twiddle. The `slot != 0` guard in the post-hoc
// histogram is dropped so real zero-valued keys match the trailing zero slots (all zero-valued
// items collapse into one histogram bin, which is exactly their rank behavior).
struct U32Adaptive
{
  static constexpr const char* name = "atomic_adaptive (u32)";
  struct Smem
  {
    unsigned slot[kK];
    unsigned rh[kK];
    unsigned cnt, cnt2;
    unsigned mstar, nAstar;
    unsigned cntA, cntB;
  };
  template <int R>
  __device__ __forceinline__ static void resolve(
    const unsigned (&tw)[4],
    const int (&idx)[4],
    Smem& sh,
    unsigned slX,
    unsigned c,
    unsigned myc,
    unsigned* out_v,
    int* out_i)
  {
    if (c == kK)
    {
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if (tw[i] >= slX)
        {
          unsigned t = atomicAdd(&sh.cntA, 1u);
          out_v[t]   = tw[i];
          out_i[t]   = idx[i];
        }
      }
      return;
    }
    if (myc)
    {
      unsigned sl[R];
#pragma unroll
      for (int j = 0; j < R; ++j)
      {
        sl[j] = sh.slot[j];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if (tw[i] >= slX)
        {
          unsigned jj = 0;
          bool eq     = false;
#pragma unroll
          for (int j = 0; j < R; ++j)
          {
            jj += (sl[j] > tw[i]);
            eq |= (sl[j] == tw[i]); // no != 0 guard: key 0 must match the zero slots
          }
          if (eq)
          {
            atomicAdd(&sh.rh[jj], 1u);
          }
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
      const int lane = threadIdx.x;
      unsigned r     = (lane < R) ? sh.rh[lane] : 0;
      unsigned cc    = r;
#pragma unroll
      for (int d = 1; d < kK; d <<= 1)
      {
        unsigned o = __shfl_up_sync(kFull, cc, d);
        if (lane >= d)
        {
          cc += o;
        }
      }
      if (lane < R && cc >= kK && (cc - r) < kK)
      {
        sh.mstar  = sh.slot[lane];
        sh.nAstar = cc - r;
      }
    }
    __syncthreads();
    const unsigned Mstar = sh.mstar;
    const unsigned nA    = sh.nAstar;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (tw[i] > Mstar)
      {
        unsigned t = atomicAdd(&sh.cntA, 1u);
        out_v[t]   = tw[i];
        out_i[t]   = idx[i];
      }
      else if (tw[i] == Mstar)
      {
        unsigned t   = atomicAdd(&sh.cntB, 1u);
        unsigned pos = nA + t;
        if (pos < kK)
        {
          out_v[pos] = tw[i];
          out_i[pos] = idx[i];
        }
      }
    }
  }
  __device__ __forceinline__ static void
  run(const unsigned (&v)[4], const int (&idx)[4], Smem& sh, unsigned* out_v, int* out_i)
  {
    if (threadIdx.x < kK)
    {
      sh.slot[threadIdx.x] = 0;
      sh.rh[threadIdx.x]   = 0;
    }
    if (threadIdx.x == kBlock - 1)
    {
      sh.cnt  = 0;
      sh.cnt2 = 0;
      sh.cntA = 0;
      sh.cntB = 0;
    }
    unsigned tw[4];
    unsigned s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      tw[i] = v[i];
      s[i]  = v[i];
    }
    sort4_desc(s);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r)
    {
      atomicMax(&sh.slot[r], s[0]);
      __syncthreads();
      const unsigned M = sh.slot[r];
      const bool w     = (s[0] == M);
      s[0]             = w ? s[1] : s[0];
      s[1]             = w ? s[2] : s[1];
      s[2]             = w ? s[3] : s[2];
      s[3]             = w ? 0u : s[3];
    }
    const unsigned sl1 = sh.slot[1];
    unsigned myc2      = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      myc2 += (tw[i] >= sl1);
    }
    atomicAdd(&sh.cnt2, myc2);
    __syncthreads();
    const unsigned c2 = sh.cnt2;
    if (c2 >= kK)
    {
      resolve<2>(tw, idx, sh, sl1, c2, myc2, out_v, out_i);
      __syncthreads();
      return;
    }
#pragma unroll
    for (int r = 2; r < kK; ++r)
    {
      atomicMax(&sh.slot[r], s[0]);
      __syncthreads();
      const unsigned M = sh.slot[r];
      const bool w     = (s[0] == M);
      s[0]             = w ? s[1] : s[0];
      s[1]             = w ? s[2] : s[1];
      s[2]             = w ? s[3] : s[2];
      s[3]             = w ? 0u : s[3];
    }
    const unsigned sl15 = sh.slot[kK - 1];
    unsigned myc        = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      myc += (tw[i] >= sl15);
    }
    atomicAdd(&sh.cnt, myc);
    __syncthreads();
    resolve<kK>(tw, idx, sh, sl15, sh.cnt, myc, out_v, out_i);
    __syncthreads();
  }
};

// ------------------------------------------------------------------ atomic_iter (break variant, u32)
// Value-based tiers were already 0-safe: if key 0 reaches the top-16, trailing rounds leave
// M* = 0 and the tier scatter (strictly-above pack + boundary-tie fill) still partitions
// correctly.
struct U32AtomicIter
{
  static constexpr const char* name = "atomic_iter (break, u32)";
  struct Smem
  {
    unsigned slot[kK];
    unsigned cnt, cntA, cntB;
  };
  __device__ __forceinline__ static void
  run(const unsigned (&v)[4], const int (&idx)[4], Smem& sh, unsigned* out_v, int* out_i)
  {
    if (threadIdx.x < kK)
    {
      sh.slot[threadIdx.x] = 0;
    }
    if (threadIdx.x == kBlock - 1)
    {
      sh.cnt  = 0;
      sh.cntA = 0;
      sh.cntB = 0;
    }
    unsigned tw[4];
    unsigned s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      tw[i] = v[i];
      s[i]  = v[i];
    }
    sort4_desc(s);
    __syncthreads();

    unsigned M = 0, Mprev = 0;
#pragma unroll
    for (int r = 0; r < kK; ++r)
    {
      atomicMax(&sh.slot[r], s[0]);
      __syncthreads();
      Mprev            = M;
      M                = sh.slot[r];
      const unsigned c = sh.cnt;
      if (c >= kK)
      {
        M = Mprev;
        break;
      }
      unsigned nm = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j)
      {
        const bool w = (s[0] == M) && (M != 0);
        nm += w;
        s[0] = w ? s[1] : s[0];
        s[1] = w ? s[2] : s[1];
        s[2] = w ? s[3] : s[2];
        s[3] = w ? 0u : s[3];
      }
      atomicAdd(&sh.cnt, nm);
    }
    const unsigned Mstar = M;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (tw[i] > Mstar)
      {
        unsigned t = atomicAdd(&sh.cntA, 1u);
        out_v[t]   = tw[i];
        out_i[t]   = idx[i];
      }
    }
    __syncthreads();
    const unsigned nA = sh.cntA;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      if (tw[i] == Mstar)
      {
        unsigned t = atomicAdd(&sh.cntB, 1u);
        int pos    = kK - 1 - (int) t;
        if (pos >= (int) nA)
        {
          out_v[pos] = tw[i];
          out_i[pos] = idx[i];
        }
      }
    }
    __syncthreads();
  }
};

// ------------------------------------------------------------------ kernels
template <class P>
struct Box
{
  typename P::Smem s;
  unsigned out_v[kK];
  int out_i[kK];
};

template <class P>
__global__ void __launch_bounds__(kBlock) correct_kernel(const unsigned* in, unsigned* out_v, int* out_i)
{
  __shared__ Box<P> box;
  unsigned v[4];
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

// integer chain link: two independently laundered copies of the previous output cannot be
// folded, so v[0] = v0[0] ^ dep1 ^ dep2 serializes calls while being value-neutral
__device__ __forceinline__ unsigned chain_link(unsigned base, unsigned out0)
{
  unsigned d1 = out0, d2 = out0;
  asm volatile("" : "+r"(d1));
  asm volatile("" : "+r"(d2));
  return base ^ d1 ^ d2;
}

template <class P>
__global__ void __launch_bounds__(kBlock) lat_kernel(const unsigned* in, int chain, long long* out)
{
  __shared__ Box<P> box;
  unsigned v0[4];
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
      unsigned v[4];
      v[0] = chain_link(v0[0], box.out_v[0]);
      v[1] = v0[1];
      v[2] = v0[2];
      v[3] = v0[3];
      P::run(v, idx, box.s, box.out_v, box.out_i);
    }
    unsigned f = box.out_v[0];
    asm volatile("" ::"r"(f));
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

template <class P>
__global__ void __launch_bounds__(kBlock) thr_kernel(const unsigned* in, int outer, unsigned* sink)
{
  __shared__ Box<P> box;
  unsigned v0[4];
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
    unsigned v[4];
    v[0] = chain_link(v0[0], box.out_v[0]);
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

// ------------------------------------------------------------------ host
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

// order-preserving float -> u32 map (host mirror of the device twiddle): reusing the float
// patterns through this map gives u32 inputs whose digit distributions match what the float
// suites saw after twiddling, so results are directly comparable.
static unsigned h_twiddle(float f)
{
  unsigned u;
  std::memcpy(&u, &f, 4);
  return u ^ (((unsigned) ((int) u >> 31)) | 0x80000000u);
}

static std::vector<unsigned> gen_pattern(const std::string& p, unsigned seed)
{
  std::vector<unsigned> v(kN);
  Lcg rng(seed * 2654435761u + 12345u);
  auto normal = [&]() {
    float u1 = std::max(rng.uniform(), 1e-7f);
    float u2 = rng.uniform();
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.28318530718f * u2);
  };
  if (p == "uniform_u32")
  {
    for (auto& x : v)
    {
      x = rng.next();
    }
  }
  else if (p == "random")
  {
    for (auto& x : v)
    {
      x = h_twiddle(normal());
    }
  }
  else if (p == "tie_heavy")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = h_twiddle((float) (j % 64) / 64.f);
    }
  }
  else if (p == "pivot_tie40")
  {
    for (auto& x : v)
    {
      x = h_twiddle(1.f);
    }
    int placed = 0;
    while (placed < 40)
    {
      unsigned pos = rng.next() % kN;
      if (v[pos] != h_twiddle(2.f))
      {
        v[pos] = h_twiddle(2.f);
        ++placed;
      }
    }
  }
  else if (p == "sorted_asc")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = (unsigned) j * 4096u;
    }
  }
  else if (p == "with_zeros")
  {
    // key 0 dominates; only 10 nonzero keys -> the top-16 must include six 0s
    for (auto& x : v)
    {
      x = 0;
    }
    int placed = 0;
    while (placed < 10)
    {
      unsigned pos = rng.next() % kN;
      if (v[pos] == 0)
      {
        v[pos] = rng.next() | 1u;
        ++placed;
      }
    }
  }
  else if (p == "all_zero")
  {
    for (auto& x : v)
    {
      x = 0;
    }
  }
  else if (p == "all_equal")
  {
    for (auto& x : v)
    {
      x = 0xdeadbeefu;
    }
  }
  else
  {
    printf("unknown pattern %s\n", p.c_str());
    exit(1);
  }
  return v;
}

static bool validate(const std::vector<unsigned>& in, const unsigned* ov, const int* oi, std::string& why)
{
  std::vector<unsigned> sorted = in;
  std::sort(sorted.begin(), sorted.end(), [](unsigned a, unsigned b) {
    return a > b;
  });
  std::vector<unsigned> want(sorted.begin(), sorted.begin() + kK);
  std::vector<unsigned> got(ov, ov + kK);
  std::sort(want.begin(), want.end());
  std::sort(got.begin(), got.end());
  for (int i = 0; i < kK; ++i)
  {
    if (want[i] != got[i])
    {
      char b[128];
      snprintf(b, sizeof b, "value multiset mismatch at %d: want %u got %u", i, want[i], got[i]);
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
    if (in[oi[i]] != ov[i])
    {
      char b[128];
      snprintf(b, sizeof b, "index %d holds %u but output value is %u", oi[i], in[oi[i]], ov[i]);
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
    "uniform_u32", "random", "tie_heavy", "pivot_tie40", "sorted_asc", "with_zeros", "all_zero", "all_equal"};
  unsigned *d_in, *d_ov;
  int* d_oi;
  CHECK(cudaMalloc(&d_in, kN * sizeof(unsigned)));
  CHECK(cudaMalloc(&d_ov, kK * sizeof(unsigned)));
  CHECK(cudaMalloc(&d_oi, kK * sizeof(int)));
  int fails = 0, runs = 0;
  for (const char* pat : pats)
  {
    const int seeds = (std::string(pat) == "uniform_u32" || std::string(pat) == "random") ? 8 : 2;
    for (int seed = 0; seed < seeds; ++seed)
    {
      auto in = gen_pattern(pat, seed);
      CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(unsigned), cudaMemcpyHostToDevice));
      correct_kernel<P><<<1, kBlock>>>(d_in, d_ov, d_oi);
      CHECK(cudaDeviceSynchronize());
      unsigned ov[kK];
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
  printf("  %-38s %s (%d runs)\n", P::name, fails ? "FAIL" : "PASS", runs);
  cudaFree(d_in);
  cudaFree(d_ov);
  cudaFree(d_oi);
}

template <class P>
void run_lat()
{
  const char* pats[] = {"uniform_u32", "random", "tie_heavy", "pivot_tie40", "sorted_asc"};
  unsigned* d_in;
  long long* d_out;
  CHECK(cudaMalloc(&d_in, kN * sizeof(unsigned)));
  CHECK(cudaMalloc(&d_out, sizeof(long long)));
  printf("  %-38s", P::name);
  for (const char* pat : pats)
  {
    auto in = gen_pattern(pat, 0);
    CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(unsigned), cudaMemcpyHostToDevice));
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

template <class P>
void run_thr()
{
  constexpr int grid = 2048, outer = 32;
  unsigned *d_in, *d_sink;
  CHECK(cudaMalloc(&d_in, kN * sizeof(unsigned)));
  CHECK(cudaMalloc(&d_sink, (size_t) grid * kK * sizeof(unsigned)));
  auto in = gen_pattern("uniform_u32", 0);
  CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(unsigned), cudaMemcpyHostToDevice));
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
  printf("  %-38s %8.1f G elem/s   (%.3f ms)\n", P::name, gelems, best_ms);
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
  printf("  %-38s regs=%3d  smem=%5zu B  local(spill)=%4zu B  maxblk/SM=%d\n",
         P::name,
         a.numRegs,
         a.sharedSizeBytes,
         a.localSizeBytes,
         occ);
}

#define FOREACH_PROTO(X) \
  X(U32AirRef)           \
  X(U32AirFused)         \
  X(U32AirPair)          \
  X(U32Adaptive)         \
  X(U32AtomicIter)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness (u32, incl. zero-key edge) ===\n");
#define RUNC(P) run_correct<P>();
    FOREACH_PROTO(RUNC)
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== latency: slope cyc/call (u32) ===\n");
#define RUNL(P) run_lat<P>();
    FOREACH_PROTO(RUNL)
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== throughput (u32, uniform input) ===\n");
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
