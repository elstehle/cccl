// Max-tuning air_pair (v6): radix-digit-width sweep + small-candidate early finish + scatter of
// original values (drops the float untwiddle/-0.0 flip machinery from the epilogue entirely).
//
//   AirTuned<KeyT, RBITS, FINISH>
//     RBITS  in {8,10,11,12}: bins = 2^RBITS, passes = ceil(32/RBITS) (4,4,3,3). Wider digits
//            buy fewer worst-case passes at the price of a bigger scan (RBITS>8: BPT bins/thread,
//            stride-9 padded histogram so the scan reads are bank-conflict-free) and a heavier
//            per-pass reset (folded into the histogram phase as before).
//     FINISH in {0,64}: if a pass ends with k < candidates <= FINISH, skip all remaining radix
//            passes: gather the candidate (ordered) keys, warp 0 bitonic-sorts them (keys-only,
//            32 or 64 wide) and publishes the exact k-th threshold + strictly-above count; the
//            standard 2-class scatter then runs with prefix_mask = ~0. Bounds tail latency:
//            at most ONE pass after the candidate set fits.
//     KeyT   in {float, unsigned}: float keys are compared in twiddled space but the scatter
//            writes the ORIGINAL register value, so no untwiddle and no -0.0 normalize/flip
//            tracking is needed anywhere (a consistent refinement of the float order; -0.0
//            then ranks just below +0.0, which any valid top-k tie-break permits).
//
// Modes: ./proto_air_tune [correct|lat|thr|res|all]. Harness identical to the other suites.

#include <cub/block/block_scan.cuh>
#include <cub/block/specializations/block_topk_air.cuh>
#include <cub/warp/warp_bitonic_sort.cuh>

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

__device__ __forceinline__ unsigned twiddle(float f)
{
  unsigned u = __float_as_uint(f);
  return u ^ (((unsigned) ((int) u >> 31)) | 0x80000000u);
}

template <typename T>
struct KeyOps;
template <>
struct KeyOps<float>
{
  static __device__ __forceinline__ unsigned to_ordered(float f)
  {
    return twiddle(f);
  }
};
template <>
struct KeyOps<unsigned>
{
  static __device__ __forceinline__ unsigned to_ordered(unsigned u)
  {
    return u;
  }
};

struct GreaterCmp
{
  template <typename T>
  __device__ bool operator()(const T& a, const T& b) const
  {
    return a > b;
  }
};

// ------------------------------------------------------------------ air_ref (header) per dtype
template <typename KeyT>
struct AirRefT
{
  using key_t  = KeyT;
  using topk_t = cub::detail::block_topk_air<KeyT, kBlock, kIpt, int>;
  struct Smem
  {
    typename topk_t::TempStorage ts;
  };
  __device__ __forceinline__ static void
  run(const KeyT (&v)[4], const int (&idx)[4], Smem& sh, KeyT* out_v, int* out_i)
  {
    KeyT k[4];
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

struct AirRefF : AirRefT<float>
{
  static constexpr const char* name = "air_ref float (header)";
};
struct AirRefU : AirRefT<unsigned>
{
  static constexpr const char* name = "air_ref u32 (header)";
};

// ------------------------------------------------------------------ the tuned template
template <typename KeyT, int RBITS, int FINISH>
struct AirTuned
{
  using key_t                = KeyT;
  static constexpr int NB    = 1 << RBITS;
  static constexpr int NBP   = (RBITS == 8) ? NB : NB + (NB >> 3); // stride-9 pad when BPT > 1
  static constexpr int BPT   = (NB > kBlock) ? NB / kBlock : 1;
  static constexpr int NPASS = (32 + RBITS - 1) / RBITS;
  struct Pair
  {
    KeyT k;
    int v;
  };
  struct Smem
  {
    union
    {
      struct
      {
        unsigned hist[2][NBP];
        typename block_scan_t::TempStorage scan_temp;
      } passes;
      Pair exch[kN];
    } stage;
    unsigned state; // bucket(<=12b) << 16 | candidates(11b) << 5 | selected(5b)
    unsigned cntA, cntB;
    unsigned fin_thr, fin_gt, gcnt;
    unsigned cand[FINISH > 0 ? FINISH : 1];
  };
  static __device__ __forceinline__ unsigned slot(unsigned b)
  {
    return (RBITS == 8) ? b : b + (b >> 3);
  }
  __device__ __forceinline__ static void
  run(const KeyT (&v)[4], const int (&idx)[4], Smem& sh, KeyT* out_v, int* out_i)
  {
    unsigned uk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      uk[i] = KeyOps<KeyT>::to_ordered(v[i]);
    }
#pragma unroll
    for (int i = threadIdx.x; i < NBP; i += kBlock)
    {
      sh.stage.passes.hist[0][i] = 0;
      sh.stage.passes.hist[1][i] = 0;
    }
    if (threadIdx.x == 0)
    {
      sh.cntA = 0;
      sh.cntB = 0;
      sh.gcnt = 0;
    }
    __syncthreads();

    unsigned kth_prefix = 0, pmask = 0;
    int k              = kK;
    int total_selected = 0;
    int num_candidates = kN;
#pragma unroll
    for (int pass = 0; pass < NPASS; ++pass)
    {
      const int end        = 32 - pass * RBITS;
      const int begin      = (end - RBITS > 0) ? end - RBITS : 0;
      const int nbits      = end - begin;
      const unsigned dmask = (1u << nbits) - 1u; // nbits <= RBITS <= 12
      unsigned* cur        = sh.stage.passes.hist[pass & 1];
      unsigned* nxt        = sh.stage.passes.hist[(pass + 1) & 1];
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        if ((uk[i] & pmask) == kth_prefix)
        {
          const unsigned d = (uk[i] >> begin) & dmask;
          atomicAdd(&cur[slot(NB - 1u - d)], 1u); // bucket 0 = largest digit
        }
      }
      if (pass > 0 && pass + 1 < NPASS)
      {
#pragma unroll
        for (int i = threadIdx.x; i < NBP; i += kBlock)
        {
          nxt[i] = 0;
        }
      }
      __syncthreads();
      // fused scan + choose over BPT contiguous bins per thread
      unsigned h[BPT];
      unsigned s = 0;
#pragma unroll
      for (int j = 0; j < BPT; ++j)
      {
        h[j] = cur[slot(threadIdx.x * BPT + j)];
        s += h[j];
      }
      unsigned incl;
      block_scan_t(sh.stage.passes.scan_temp).InclusiveSum(s, incl);
      unsigned cum = incl - s;
#pragma unroll
      for (int j = 0; j < BPT; ++j)
      {
        const unsigned prev = cum;
        cum += h[j];
        if (prev < (unsigned) k && cum >= (unsigned) k)
        {
          sh.state = ((threadIdx.x * BPT + j) << 16) | ((cum - prev) << 5) | prev;
        }
      }
      __syncthreads();
      const unsigned st       = sh.state;
      const unsigned selected = st & 0x1fu;
      k -= (int) selected;
      num_candidates = (int) ((st >> 5) & 0x7ffu);
      total_selected += (int) selected;
      const unsigned kth_digit = (NB - 1u) - (st >> 16);
      kth_prefix |= kth_digit << begin;
      pmask |= dmask << begin;
      if (num_candidates == k)
      {
        break;
      }
      if constexpr (FINISH > 0)
      {
        if (pass + 1 < NPASS && num_candidates <= FINISH)
        {
          // gather the (ordered) candidate keys; warp 0 sorts and publishes the exact threshold
#pragma unroll
          for (int i = 0; i < 4; ++i)
          {
            if ((uk[i] & pmask) == kth_prefix)
            {
              const unsigned t = atomicAdd(&sh.gcnt, 1u);
              sh.cand[t]       = uk[i];
            }
          }
          __syncthreads();
          if (threadIdx.x < 32)
          {
            const int lane = threadIdx.x;
            unsigned thr, gt;
            if (num_candidates <= 32)
            {
              unsigned key1[1] = {lane < num_candidates ? sh.cand[lane] : 0u};
              cub::detail::WarpBitonicSort<1, unsigned>{}.Sort(key1, GreaterCmp{});
              thr = __shfl_sync(kFull, key1[0], k - 1);
              gt  = __popc(__ballot_sync(kFull, key1[0] > thr));
            }
            else
            {
              unsigned key2[2];
#pragma unroll
              for (int j = 0; j < 2; ++j)
              {
                const int pos = lane + 32 * j;
                key2[j]       = pos < num_candidates ? sh.cand[pos] : 0u;
              }
              cub::detail::WarpBitonicSort<2, unsigned>{}.Sort(key2, GreaterCmp{});
              thr = __shfl_sync(kFull, key2[0], k - 1);
              gt  = __popc(__ballot_sync(kFull, key2[0] > thr)) + __popc(__ballot_sync(kFull, key2[1] > thr));
            }
            if (lane == 0)
            {
              sh.fin_thr = thr;
              sh.fin_gt  = gt;
            }
          }
          __syncthreads();
          kth_prefix = sh.fin_thr;
          pmask      = 0xffffffffu;
          total_selected += (int) sh.fin_gt;
          break;
        }
      }
    }
    // epilogue: pair scatter of ORIGINAL values (no untwiddle, no flip), single gather.
    // Candidates always go through the class-1 stream based at total_selected; when they exactly
    // fill k this yields the same positions the select_all special case would.
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
      const unsigned kp  = uk[i] & pmask;
      const bool is_sel  = kp > kth_prefix;
      const bool is_cand = kp == kth_prefix;
      if (is_sel || is_cand)
      {
        const unsigned t   = atomicAdd(is_cand ? &sh.cntB : &sh.cntA, 1u);
        const unsigned off = is_cand ? (unsigned) total_selected + t : t;
        sh.stage.exch[off] = Pair{v[i], idx[i]};
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

// float sweep
struct TF_R8_F0 : AirTuned<float, 8, 0>
{
  static constexpr const char* name = "tuned f32 R8  F0  (v6 + orig-value scatter)";
};
struct TF_R8_F64 : AirTuned<float, 8, 64>
{
  static constexpr const char* name = "tuned f32 R8  F64";
};
struct TF_R10_F0 : AirTuned<float, 10, 0>
{
  static constexpr const char* name = "tuned f32 R10 F0";
};
struct TF_R11_F0 : AirTuned<float, 11, 0>
{
  static constexpr const char* name = "tuned f32 R11 F0";
};
struct TF_R11_F64 : AirTuned<float, 11, 64>
{
  static constexpr const char* name = "tuned f32 R11 F64";
};
struct TF_R12_F0 : AirTuned<float, 12, 0>
{
  static constexpr const char* name = "tuned f32 R12 F0";
};
struct TF_R12_F64 : AirTuned<float, 12, 64>
{
  static constexpr const char* name = "tuned f32 R12 F64";
};
// u32 spot-checks of the promising configs
struct TU_R8_F0 : AirTuned<unsigned, 8, 0>
{
  static constexpr const char* name = "tuned u32 R8  F0";
};
struct TU_R8_F64 : AirTuned<unsigned, 8, 64>
{
  static constexpr const char* name = "tuned u32 R8  F64";
};
struct TU_R11_F64 : AirTuned<unsigned, 11, 64>
{
  static constexpr const char* name = "tuned u32 R11 F64";
};

// ------------------------------------------------------------------ harness
template <class P>
struct Box
{
  typename P::Smem s;
  typename P::key_t out_v[kK];
  int out_i[kK];
};

template <class P>
__global__ void __launch_bounds__(kBlock) correct_kernel(const typename P::key_t* in, typename P::key_t* out_v, int* out_i)
{
  __shared__ Box<P> box;
  typename P::key_t v[4];
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

__device__ __forceinline__ float chain_link(float base, float out0)
{
  return fmaf(out0, 0.0f, base);
}
__device__ __forceinline__ unsigned chain_link(unsigned base, unsigned out0)
{
  unsigned d1 = out0, d2 = out0;
  asm volatile("" : "+r"(d1));
  asm volatile("" : "+r"(d2));
  return base ^ d1 ^ d2;
}

template <class P>
__global__ void __launch_bounds__(kBlock) lat_kernel(const typename P::key_t* in, int chain, long long* out)
{
  using K = typename P::key_t;
  __shared__ Box<P> box;
  K v0[4];
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
      K v[4];
      v[0] = chain_link(v0[0], box.out_v[0]);
      v[1] = v0[1];
      v[2] = v0[2];
      v[3] = v0[3];
      P::run(v, idx, box.s, box.out_v, box.out_i);
    }
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
__global__ void __launch_bounds__(kBlock) thr_kernel(const typename P::key_t* in, int outer, typename P::key_t* sink)
{
  using K = typename P::key_t;
  __shared__ Box<P> box;
  K v0[4];
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
    K v[4];
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

static unsigned h_twiddle(float f)
{
  unsigned u;
  std::memcpy(&u, &f, 4);
  return u ^ (((unsigned) ((int) u >> 31)) | 0x80000000u);
}

static std::vector<float> gen_f(const std::string& p, unsigned seed)
{
  std::vector<float> v(kN);
  Lcg rng(seed * 2654435761u + 12345u);
  auto normal = [&]() {
    float u1 = std::max(rng.uniform(), 1e-7f);
    float u2 = rng.uniform();
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.28318530718f * u2);
  };
  if (p == "random")
  {
    for (auto& x : v)
    {
      x = normal();
    }
  }
  else if (p == "tie_heavy")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = (float) (j % 64) / 64.f;
    }
  }
  else if (p == "pivot_tie40")
  {
    for (auto& x : v)
    {
      x = 1.f;
    }
    int placed = 0;
    while (placed < 40)
    {
      unsigned pos = rng.next() % kN;
      if (v[pos] != 2.f)
      {
        v[pos] = 2.f;
        ++placed;
      }
    }
  }
  else if (p == "sorted_asc")
  {
    for (int j = 0; j < kN; ++j)
    {
      v[j] = (float) j / kN;
    }
  }
  else if (p == "all_equal")
  {
    for (auto& x : v)
    {
      x = 1.f;
    }
  }
  else if (p == "neg_zero")
  {
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

template <typename K>
static std::vector<K> gen_pattern(const std::string& p, unsigned seed);
template <>
std::vector<float> gen_pattern<float>(const std::string& p, unsigned seed)
{
  return gen_f(p, seed);
}
template <>
std::vector<unsigned> gen_pattern<unsigned>(const std::string& p, unsigned seed)
{
  if (p == "with_zeros")
  {
    std::vector<unsigned> v(kN, 0u);
    Lcg rng(seed * 2654435761u + 777u);
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
    return v;
  }
  if (p == "all_zero")
  {
    return std::vector<unsigned>(kN, 0u);
  }
  auto f = gen_f(p == "neg_zero" ? "random" : p, seed); // neg_zero is float-specific
  std::vector<unsigned> v(kN);
  for (int i = 0; i < kN; ++i)
  {
    v[i] = h_twiddle(f[i]);
  }
  return v;
}

template <typename K>
static bool validate(const std::vector<K>& in, const K* ov, const int* oi, std::string& why)
{
  std::vector<K> sorted = in;
  std::sort(sorted.begin(), sorted.end(), [](K a, K b) {
    return a > b;
  });
  std::vector<K> want(sorted.begin(), sorted.begin() + kK);
  std::vector<K> got(ov, ov + kK);
  std::sort(want.begin(), want.end());
  std::sort(got.begin(), got.end());
  for (int i = 0; i < kK; ++i)
  {
    if (!(want[i] == got[i]))
    {
      why = "value multiset mismatch at " + std::to_string(i);
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
      why = "index/value mismatch at " + std::to_string(i);
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
  using K                 = typename P::key_t;
  const bool is_f         = std::is_same_v<K, float>;
  std::vector<std::string> pats = {"random", "tie_heavy", "pivot_tie40", "sorted_asc", "all_equal"};
  if (is_f)
  {
    pats.push_back("neg_zero");
  }
  else
  {
    pats.push_back("with_zeros");
    pats.push_back("all_zero");
  }
  K *d_in, *d_ov;
  int* d_oi;
  CHECK(cudaMalloc(&d_in, kN * sizeof(K)));
  CHECK(cudaMalloc(&d_ov, kK * sizeof(K)));
  CHECK(cudaMalloc(&d_oi, kK * sizeof(int)));
  int fails = 0, runs = 0;
  for (const auto& pat : pats)
  {
    const int seeds = (pat == "random") ? 8 : 2;
    for (int seed = 0; seed < seeds; ++seed)
    {
      auto in = gen_pattern<K>(pat, seed);
      CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(K), cudaMemcpyHostToDevice));
      correct_kernel<P><<<1, kBlock>>>(d_in, d_ov, d_oi);
      CHECK(cudaDeviceSynchronize());
      K ov[kK];
      int oi[kK];
      CHECK(cudaMemcpy(ov, d_ov, sizeof ov, cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(oi, d_oi, sizeof oi, cudaMemcpyDeviceToHost));
      std::string why;
      ++runs;
      if (!validate(in, ov, oi, why))
      {
        ++fails;
        printf("    FAIL %-16s seed %d: %s\n", pat.c_str(), seed, why.c_str());
      }
    }
  }
  printf("  %-46s %s (%d runs)\n", P::name, fails ? "FAIL" : "PASS", runs);
  cudaFree(d_in);
  cudaFree(d_ov);
  cudaFree(d_oi);
}

template <class P>
void run_lat()
{
  using K            = typename P::key_t;
  const char* pats[] = {"random", "tie_heavy", "pivot_tie40", "sorted_asc"};
  K* d_in;
  long long* d_out;
  CHECK(cudaMalloc(&d_in, kN * sizeof(K)));
  CHECK(cudaMalloc(&d_out, sizeof(long long)));
  printf("  %-46s", P::name);
  for (const char* pat : pats)
  {
    auto in = gen_pattern<K>(pat, 0);
    CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(K), cudaMemcpyHostToDevice));
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
  using K            = typename P::key_t;
  constexpr int grid = 2048, outer = 32;
  K *d_in, *d_sink;
  CHECK(cudaMalloc(&d_in, kN * sizeof(K)));
  CHECK(cudaMalloc(&d_sink, (size_t) grid * kK * sizeof(K)));
  auto in = gen_pattern<K>("random", 0);
  CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(K), cudaMemcpyHostToDevice));
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
  printf("  %-46s %8.1f G elem/s   (%.3f ms)\n", P::name, gelems, best_ms);
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
  printf("  %-46s regs=%3d  smem=%5zu B  local(spill)=%4zu B  maxblk/SM=%d\n",
         P::name,
         a.numRegs,
         a.sharedSizeBytes,
         a.localSizeBytes,
         occ);
}

#define FOREACH_PROTO(X) \
  X(AirRefF)             \
  X(TF_R8_F0)            \
  X(TF_R8_F64)           \
  X(TF_R10_F0)           \
  X(TF_R11_F0)           \
  X(TF_R11_F64)          \
  X(TF_R12_F0)           \
  X(TF_R12_F64)          \
  X(AirRefU)             \
  X(TU_R8_F0)            \
  X(TU_R8_F64)           \
  X(TU_R11_F64)

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
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== throughput (random input) ===\n");
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
