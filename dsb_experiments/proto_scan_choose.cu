// Prototype: a faster scan+choose for the block top-k radix pass (branch exp/scan-choose-opt).
//
// The L0-vs-L6 phase profile (BLOCK_TOPK_AIR_ABLATION.md) shows the fused scan+choose stage is
// the optimized kernel's dominant invariant: ~377 cyc/pass at 256 bins, unchanged by all other
// optimizations. Structure of the incumbent (BLOCK_SCAN_WARP_SCANS, fused):
//
//   LDS count -> 5-shfl inclusive warp scan (~150, serial chain) -> lane31 posts the warp
//   AGGREGATE (= the scan's LAST output) -> barrier (releases only after the slowest warp's
//   full scan) -> per-thread fold of the 8 aggregates (~63) -> crossing test -> state -> barrier
//
// The aggregate is on the wrong side of the scan: it does not need the scan at all —
// redux.sync.add produces it in ~22 cyc. "Aggregate-first" (AF) restructure:
//
//   LDS count -> redux aggregate (22) -> post aggregate -> barrier (releases ~120 EARLIER)
//   -> in-warp shuffle scan with the aggregate-fold's independent loads/adds interleaved into
//   the shuffle chain's stall slots -> crossing test -> state -> barrier
//
// The fold tail hides inside the scan's shadow. Explored for (a) 256 bins (RadixBits=8,
// 1 bin/thread) and (b) 2048 bins (RadixBits=11, 8 contiguous padded bins/thread, warp segment
// = 256 bins). Also measured: BLOCK_SCAN_RAKING, a single-warp serial scan (reference), and a
// no-scan floor probe (peripherals: histogram LDS + state write + closing barrier).
//
// Chain serialization: a laundered zero derived from the previous call's crossing bucket is
// added to the histogram indices, so call n's first LDS depends on call n-1's result.
//
// Modes: ./proto_scan_choose [correct|lat|all]

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

constexpr unsigned kFull = 0xffffffffu;
constexpr int kBlock     = 256;
constexpr int kN         = 1024;
constexpr int kK         = 16;

__device__ int g_sink;

// stride-9 padding for multi-bin-per-thread layouts (conflict-free scan reads)
template <int BPT>
__device__ __host__ __forceinline__ constexpr int slot_of(int b)
{
  return (BPT == 1) ? b : b + (b >> 3);
}

struct State
{
  unsigned bucket, cands, sel;
};

// ------------------------------------------------------------------ implementations
// Contract: run(hist, k, zero, sh, st) with `zero` a laundered 0 added to histogram indices;
// exactly one thread writes st; ends with __syncthreads(); every thread returns st.bucket.

// S0: the incumbent — cub::BlockScan WARP_SCANS, fused crossing test (L6 / R11-tuned shape)
template <int NBINS>
struct ScanCub
{
  static constexpr const char* name = "cub WARP_SCANS fused (incumbent)";
  static constexpr int BPT          = NBINS / kBlock;
  static constexpr int HSIZE        = slot_of<BPT>(NBINS - 1) + 1;
  using scan_t                      = cub::BlockScan<unsigned, kBlock, cub::BLOCK_SCAN_WARP_SCANS>;
  struct Sh
  {
    typename scan_t::TempStorage st;
  };
  __device__ __forceinline__ static unsigned
  run(const unsigned* hist, unsigned k, unsigned zero, Sh& sh, State& st)
  {
    unsigned h[BPT];
    unsigned s = 0;
#pragma unroll
    for (int j = 0; j < BPT; ++j)
    {
      h[j] = hist[slot_of<BPT>(threadIdx.x * BPT + j) + zero];
      s += h[j];
    }
    unsigned incl;
    scan_t(sh.st).InclusiveSum(s, incl);
    unsigned cum = incl - s;
#pragma unroll
    for (int j = 0; j < BPT; ++j)
    {
      const unsigned prev = cum;
      cum += h[j];
      if (prev < k && cum >= k)
      {
        st.bucket = threadIdx.x * BPT + j;
        st.cands  = cum - prev;
        st.sel    = prev;
      }
    }
    __syncthreads();
    return st.bucket;
  }
};

// S1: aggregate-first two-level scan — redux posts the warp aggregate before the scan runs;
// the early barrier lets the cross-warp fold overlap the in-warp shuffle chain.
template <int NBINS>
struct ScanAF
{
  static constexpr const char* name = "aggregate-first (redux + early bar)";
  static constexpr int BPT          = NBINS / kBlock;
  static constexpr int HSIZE        = slot_of<BPT>(NBINS - 1) + 1;
  struct Sh
  {
    unsigned agg[8];
  };
  __device__ __forceinline__ static unsigned
  run(const unsigned* hist, unsigned k, unsigned zero, Sh& sh, State& st)
  {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned h[BPT];
    unsigned s = 0;
#pragma unroll
    for (int j = 0; j < BPT; ++j)
    {
      h[j] = hist[slot_of<BPT>(threadIdx.x * BPT + j) + zero];
      s += h[j];
    }
    const unsigned wsum = __reduce_add_sync(kFull, s); // aggregate without the scan
    if (lane == 0)
    {
      sh.agg[warp] = wsum;
    }
    __syncthreads(); // releases ~120 cyc earlier than a post-scan barrier
    // cross-warp base: independent loads/adds, interleaved by the scheduler with the shuffles
    unsigned base = 0;
#pragma unroll
    for (int w = 0; w < 7; ++w)
    {
      base += (w < warp) ? sh.agg[w] : 0u;
    }
    unsigned incl = s;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1)
    {
      const unsigned o = __shfl_up_sync(kFull, incl, d);
      if (lane >= d)
      {
        incl += o;
      }
    }
    unsigned cum = base + incl - s;
#pragma unroll
    for (int j = 0; j < BPT; ++j)
    {
      const unsigned prev = cum;
      cum += h[j];
      if (prev < k && cum >= k)
      {
        st.bucket = threadIdx.x * BPT + j;
        st.cands  = cum - prev;
        st.sel    = prev;
      }
    }
    __syncthreads();
    return st.bucket;
  }
};

// S2: cub::BlockScan RAKING (control: different CUB strategy, same fused test)
template <int NBINS>
struct ScanRaking
{
  static constexpr const char* name = "cub RAKING fused (control)";
  static constexpr int BPT          = NBINS / kBlock;
  static constexpr int HSIZE        = slot_of<BPT>(NBINS - 1) + 1;
  using scan_t                      = cub::BlockScan<unsigned, kBlock, cub::BLOCK_SCAN_RAKING>;
  struct Sh
  {
    typename scan_t::TempStorage st;
  };
  __device__ __forceinline__ static unsigned
  run(const unsigned* hist, unsigned k, unsigned zero, Sh& sh, State& st)
  {
    unsigned h[BPT];
    unsigned s = 0;
#pragma unroll
    for (int j = 0; j < BPT; ++j)
    {
      h[j] = hist[slot_of<BPT>(threadIdx.x * BPT + j) + zero];
      s += h[j];
    }
    unsigned incl;
    scan_t(sh.st).InclusiveSum(s, incl);
    unsigned cum = incl - s;
#pragma unroll
    for (int j = 0; j < BPT; ++j)
    {
      const unsigned prev = cum;
      cum += h[j];
      if (prev < k && cum >= k)
      {
        st.bucket = threadIdx.x * BPT + j;
        st.cands  = cum - prev;
        st.sel    = prev;
      }
    }
    __syncthreads();
    return st.bucket;
  }
};

// S3: single-warp serial scan (reference; 256 bins only — 8 padded bins per lane of warp 0)
template <int NBINS>
struct Scan1Warp
{
  static constexpr const char* name = "1-warp scan (reference)";
  static constexpr int BPT          = 8; // harness builds the padded layout this impl reads
  static constexpr int HSIZE        = slot_of<8>(NBINS - 1) + 1; // always padded (8 bins/lane)
  struct Sh
  {
    int dummy;
  };
  __device__ __forceinline__ static unsigned
  run(const unsigned* hist, unsigned k, unsigned zero, Sh& sh, State& st)
  {
    static_assert(NBINS == 256, "reference impl sized for 256 bins");
    if (threadIdx.x < 32)
    {
      const int lane = threadIdx.x;
      unsigned h[8];
      unsigned s = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j)
      {
        h[j] = hist[slot_of<8>(lane * 8 + j) + zero];
        s += h[j];
      }
      unsigned excl = s;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1)
      {
        const unsigned o = __shfl_up_sync(kFull, excl, d);
        if (lane >= d)
        {
          excl += o;
        }
      }
      excl -= s;
      unsigned cum = excl;
#pragma unroll
      for (int j = 0; j < 8; ++j)
      {
        const unsigned prev = cum;
        cum += h[j];
        if (prev < k && cum >= k)
        {
          st.bucket = lane * 8 + j;
          st.cands  = cum - prev;
          st.sel    = prev;
        }
      }
    }
    __syncthreads();
    return st.bucket;
  }
};

// S4: no-scan floor probe — peripherals only (histogram LDS, state write, closing barrier).
// NOT a correct scan; excluded from correctness. The scan's net cost = impl - floor.
template <int NBINS>
struct ScanFloor
{
  static constexpr const char* name = "no-scan floor (peripherals)";
  static constexpr int BPT          = NBINS / kBlock;
  static constexpr int HSIZE        = slot_of<BPT>(NBINS - 1) + 1;
  struct Sh
  {
    int dummy;
  };
  __device__ __forceinline__ static unsigned
  run(const unsigned* hist, unsigned k, unsigned zero, Sh& sh, State& st)
  {
    unsigned s = 0;
#pragma unroll
    for (int j = 0; j < BPT; ++j)
    {
      s += hist[slot_of<BPT>(threadIdx.x * BPT + j) + zero];
    }
    if (s > 100000u + k) // never true; keeps s live without a real crossing
    {
      st.bucket = s;
      st.cands  = s;
      st.sel    = s;
    }
    if (threadIdx.x == 0)
    {
      st.bucket = 0; // deterministic chaining value
    }
    __syncthreads();
    return st.bucket;
  }
};

// ------------------------------------------------------------------ harness
__device__ __forceinline__ unsigned launder_zero(unsigned x)
{
  unsigned d1 = x, d2 = x;
  asm volatile("" : "+r"(d1));
  asm volatile("" : "+r"(d2));
  return d1 ^ d2;
}

template <class Impl, int NBINS>
__global__ void __launch_bounds__(kBlock) lat_kernel(const unsigned* bins, int chain, long long* out, unsigned* st_out)
{
  __shared__ typename Impl::Sh sh;
  __shared__ unsigned hist[Impl::HSIZE];
  __shared__ State st;
  // build the histogram once (not timed)
  for (int i = threadIdx.x; i < Impl::HSIZE; i += kBlock)
  {
    hist[i] = 0;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
    atomicAdd(&hist[slot_of<Impl::BPT>((int) bins[threadIdx.x * 4 + i])], 1u);
  }
  __syncthreads();

  unsigned prev  = Impl::run(hist, kK, 0, sh, st); // warmup
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int rep = 0; rep < 24; ++rep)
  {
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll 1
    for (int n = 0; n < chain; ++n)
    {
      prev = Impl::run(hist, kK, launder_zero(prev), sh, st);
    }
    asm volatile("" ::"r"(prev));
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    *out      = best;
    st_out[0] = st.bucket;
    st_out[1] = st.cands;
    st_out[2] = st.sel;
  }
  if (prev == 0xdeadbeefu)
  {
    g_sink = (int) prev;
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

// bin index stream for a pattern: bucket = (NBINS-1) - top-digit of the twiddled float
template <int NBINS>
static std::vector<unsigned> gen_bins(const std::string& p, unsigned seed)
{
  Lcg rng(seed * 2654435761u + 12345u);
  auto normal = [&]() {
    float u1 = std::max(rng.uniform(), 1e-7f);
    float u2 = rng.uniform();
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.28318530718f * u2);
  };
  std::vector<float> f(kN);
  if (p == "random")
  {
    for (auto& x : f)
    {
      x = normal();
    }
  }
  else if (p == "tie_heavy")
  {
    for (int j = 0; j < kN; ++j)
    {
      f[j] = (float) (j % 64) / 64.f;
    }
  }
  else if (p == "uniform_bits")
  {
    // uniform digit distribution: crossing after ~k/(N/NBINS) buckets
    std::vector<unsigned> v(kN);
    for (int j = 0; j < kN; ++j)
    {
      v[j] = rng.next() % NBINS;
    }
    return v;
  }
  else
  {
    printf("unknown pattern %s\n", p.c_str());
    exit(1);
  }
  const int shift = 32 - (NBINS == 256 ? 8 : 11);
  std::vector<unsigned> v(kN);
  for (int j = 0; j < kN; ++j)
  {
    v[j] = (NBINS - 1) - (h_twiddle(f[j]) >> shift);
  }
  return v;
}

static void expected_state(const std::vector<unsigned>& bins, int nbins, unsigned k, State& st)
{
  std::vector<unsigned> hist(nbins, 0);
  for (unsigned b : bins)
  {
    hist[b]++;
  }
  unsigned cum = 0;
  for (int b = 0; b < nbins; ++b)
  {
    const unsigned prev = cum;
    cum += hist[b];
    if (prev < k && cum >= k)
    {
      st = {(unsigned) b, cum - prev, prev};
      return;
    }
  }
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

template <class Impl, int NBINS>
void run_one(const char* label, bool check)
{
  const char* pats[] = {"random", "tie_heavy", "uniform_bits"};
  unsigned* d_bins;
  long long* d_out;
  unsigned* d_st;
  CHECK(cudaMalloc(&d_bins, kN * sizeof(unsigned)));
  CHECK(cudaMalloc(&d_out, sizeof(long long)));
  CHECK(cudaMalloc(&d_st, 3 * sizeof(unsigned)));
  printf("  %-40s", label);
  for (const char* pat : pats)
  {
    auto bins = gen_bins<NBINS>(pat, 0);
    CHECK(cudaMemcpy(d_bins, bins.data(), kN * sizeof(unsigned), cudaMemcpyHostToDevice));
    const int chains[] = {1, 2, 4, 8, 16, 32};
    double x[6], y[6];
    for (int i = 0; i < 6; ++i)
    {
      lat_kernel<Impl, NBINS><<<1, kBlock>>>(d_bins, chains[i], d_out, d_st);
      CHECK(cudaDeviceSynchronize());
      long long c;
      CHECK(cudaMemcpy(&c, d_out, sizeof c, cudaMemcpyDeviceToHost));
      x[i] = chains[i];
      y[i] = (double) c;
    }
    double a, b;
    fit(x, y, 6, a, b);
    printf("  %s=%5.0f", pat, b);
    if (check)
    {
      unsigned got[3];
      CHECK(cudaMemcpy(got, d_st, sizeof got, cudaMemcpyDeviceToHost));
      State want{};
      expected_state(bins, NBINS, kK, want);
      if (got[0] != want.bucket || got[1] != want.cands || got[2] != want.sel)
      {
        printf("  [FAIL want b=%u c=%u s=%u got b=%u c=%u s=%u]", want.bucket, want.cands, want.sel, got[0], got[1], got[2]);
      }
    }
  }
  printf("   cyc/call (slope)%s\n", check ? "" : "  [floor probe]");
  cudaFree(d_bins);
  cudaFree(d_out);
  cudaFree(d_st);
}

int main(int argc, char** argv)
{
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

  printf("\n=== scan+choose in isolation, 256 bins (1 bin/thread) ===\n");
  run_one<ScanCub<256>, 256>("cub WARP_SCANS fused (incumbent)", true);
  run_one<ScanRaking<256>, 256>("cub RAKING fused (control)", true);
  run_one<ScanAF<256>, 256>("aggregate-first (redux + early bar)", true);
  run_one<Scan1Warp<256>, 256>("1-warp scan (reference)", true);
  run_one<ScanFloor<256>, 256>("no-scan floor (peripherals)", false);

  printf("\n=== scan+choose in isolation, 2048 bins (8 padded bins/thread) ===\n");
  run_one<ScanCub<2048>, 2048>("cub WARP_SCANS fused (incumbent)", true);
  run_one<ScanRaking<2048>, 2048>("cub RAKING fused (control)", true);
  run_one<ScanAF<2048>, 2048>("aggregate-first (redux + early bar)", true);
  run_one<ScanFloor<2048>, 2048>("no-scan floor (peripherals)", false);
  return 0;
}
