// Validation + perf harness for the fea/block-topk-air-latency integration branch.
//
// Exercises the REAL cub::detail::block_topk_air header across the general contract that the
// fixed-shape ablation harness did not cover:
//   * runtime k in {1, 5, 16, 100, 1000}
//   * partial tiles: IsFullTile=false with valid_items in {700, 701, 1024}
//   * both select directions (max and min)
//   * runtime bit sub-ranges (begin_bit=8 / end_bit=24 on low/high-byte-masked u32 keys, whose
//     reference equals the full comparison) -> the rolled pass-loop fallback
//   * template-parameter combinations: UnrollBitPasses on/off, FuseKeyValueExchange auto/forced
//   * exact ±0.0 restoration (neg_zero pattern with the boundary made of 8x -0.0 + 8x +0.0)
//
// Build against the integration branch normally; build with -DOLD_HEADER against the base
// branch (whose header has no config parameters) for the baseline perf rows.
//
// Modes: ./proto_air_integration [correct|lat|thr|res|all]

#include <cub/block/specializations/block_topk_air.cuh>

#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

constexpr int kBlock = 256;
constexpr int kIpt   = 4;
constexpr int kN     = 1024;

__device__ int g_sink;

#ifndef OLD_HEADER
template <typename K, typename V, bool U, bool F>
using air_t = cub::detail::block_topk_air<K, kBlock, kIpt, V, 8, U, F>;
#else
template <typename K, typename V, bool U, bool F>
using air_t = cub::detail::block_topk_air<K, kBlock, kIpt, V>;
#endif

// ------------------------------------------------------------------ configs
template <typename KeyT, typename ValueT, bool Unroll, bool Fuse>
struct CfgPairs
{
  using key_t                     = KeyT;
  using value_t                   = ValueT;
  static constexpr bool keys_only = false;
  using topk_t                    = air_t<KeyT, ValueT, Unroll, Fuse>;
  struct Smem
  {
    typename topk_t::TempStorage ts;
  };
  template <cub::detail::topk::select Dir, bool Full>
  __device__ static void
  run(KeyT (&k)[kIpt], ValueT (&v)[kIpt], Smem& sh, int kk, int valid, int bb, int eb)
  {
    topk_t(sh.ts).template select_pairs<Dir, Full>(k, v, kk, valid, bb, eb);
  }
};

template <typename KeyT, bool Unroll>
struct CfgKeys
{
  using key_t                     = KeyT;
  using value_t                   = int; // dummy
  static constexpr bool keys_only = true;
  using topk_t                    = air_t<KeyT, cub::NullType, Unroll, false>;
  struct Smem
  {
    typename topk_t::TempStorage ts;
  };
  template <cub::detail::topk::select Dir, bool Full>
  __device__ static void run(KeyT (&k)[kIpt], int (&)[kIpt], Smem& sh, int kk, int valid, int bb, int eb)
  {
    topk_t(sh.ts).template select_keys<Dir, Full>(k, kk, valid, bb, eb);
  }
};

// ------------------------------------------------------------------ kernels
template <class C, cub::detail::topk::select Dir, bool Full>
__global__ void __launch_bounds__(kBlock)
correct_kernel(const typename C::key_t* in, int k, int valid, int bb, int eb, typename C::key_t* out_v, int* out_i)
{
  using K = typename C::key_t;
  using V = typename C::value_t;
  __shared__ typename C::Smem sh;
  K keys[kIpt];
  V vals[kIpt];
#pragma unroll
  for (int i = 0; i < kIpt; ++i)
  {
    const int idx = threadIdx.x * kIpt + i;
    keys[i]       = in[idx];
    vals[i]       = (V) idx;
  }
  C::template run<Dir, Full>(keys, vals, sh, k, valid, bb, eb);
#pragma unroll
  for (int i = 0; i < kIpt; ++i)
  {
    const int bi = threadIdx.x * kIpt + i;
    if (bi < k)
    {
      out_v[bi] = keys[i];
      out_i[bi] = C::keys_only ? -2 : (int) vals[i];
    }
  }
}

constexpr int kLatReps = 24;

__device__ __forceinline__ float chain_link(float base, float out0)
{
  return fmaf(out0, 0.0f, base);
}

template <class C>
__global__ void __launch_bounds__(kBlock) lat_kernel(const float* in, int chain, long long* out)
{
  __shared__ typename C::Smem sh;
  __shared__ float bounce;
  float v0[kIpt];
#pragma unroll
  for (int i = 0; i < kIpt; ++i)
  {
    v0[i] = in[threadIdx.x * kIpt + i];
  }
  if (threadIdx.x == 0)
  {
    bounce = 0.f;
  }
  __syncthreads();
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
      float keys[kIpt];
      int vals[kIpt];
      keys[0] = chain_link(v0[0], bounce);
      keys[1] = v0[1];
      keys[2] = v0[2];
      keys[3] = v0[3];
#pragma unroll
      for (int i = 0; i < kIpt; ++i)
      {
        vals[i] = threadIdx.x * kIpt + i;
      }
      C::template run<cub::detail::topk::select::max, true>(keys, vals, sh, 16, kN, 0, 8 * (int) sizeof(float));
      if (threadIdx.x == 0)
      {
        bounce = keys[0]; // thread 0 holds output slots 0..3 -> serializes the chain
      }
      __syncthreads();
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
  if (bounce == -12345.f)
  {
    g_sink = 1;
  }
}

template <class C>
__global__ void __launch_bounds__(kBlock) thr_kernel(const float* in, int outer, float* sink)
{
  __shared__ typename C::Smem sh;
  __shared__ float bounce;
  float v0[kIpt];
#pragma unroll
  for (int i = 0; i < kIpt; ++i)
  {
    v0[i] = in[threadIdx.x * kIpt + i];
  }
  if (threadIdx.x == 0)
  {
    bounce = 0.f;
  }
  __syncthreads();
#pragma unroll 1
  for (int n = 0; n <= outer; ++n)
  {
    float keys[kIpt];
    int vals[kIpt];
    keys[0] = chain_link(v0[0], bounce);
    keys[1] = v0[1];
    keys[2] = v0[2];
    keys[3] = v0[3];
#pragma unroll
    for (int i = 0; i < kIpt; ++i)
    {
      vals[i] = threadIdx.x * kIpt + i;
    }
    C::template run<cub::detail::topk::select::max, true>(keys, vals, sh, 16, kN, 0, 8 * (int) sizeof(float));
    if (threadIdx.x == 0)
    {
      bounce = keys[0];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    sink[blockIdx.x] = bounce;
  }
}

// ------------------------------------------------------------------ host: patterns
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
  else if (p == "all_equal")
  {
    for (auto& x : v)
    {
      x = 1.f;
    }
  }
  else if (p == "neg_zero")
  {
    // top-16 (max) is exactly 8x -0.0 and 8x +0.0 -> validates exact sign restoration
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

static unsigned h_twiddle(float f)
{
  unsigned u;
  std::memcpy(&u, &f, 4);
  return u ^ (((unsigned) ((int) u >> 31)) | 0x80000000u);
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
  std::vector<unsigned> v(kN);
  if (p == "masked_low8" || p == "masked_high8")
  {
    // keys whose low (resp. high) byte is zero: a bit-sub-range covering the nonzero bytes
    // selects identically to the full comparison
    Lcg rng(seed * 2654435761u + 777u);
    for (auto& x : v)
    {
      x = (p == "masked_low8") ? (rng.next() & 0xffffff00u) : (rng.next() & 0x00ffffffu);
    }
    return v;
  }
  auto f = gen_f(p == "neg_zero" ? "random" : p, seed);
  for (int i = 0; i < kN; ++i)
  {
    v[i] = h_twiddle(f[i]);
  }
  return v;
}
template <>
std::vector<__half> gen_pattern<__half>(const std::string& p, unsigned seed)
{
  auto f = gen_f(p, seed);
  std::vector<__half> v(kN);
  for (int i = 0; i < kN; ++i)
  {
    v[i] = __float2half(f[i]);
  }
  return v;
}
template <>
std::vector<double> gen_pattern<double>(const std::string& p, unsigned seed)
{
  auto f = gen_f(p, seed);
  std::vector<double> v(kN);
  for (int i = 0; i < kN; ++i)
  {
    v[i] = (double) f[i];
  }
  return v;
}

static double to_cmp(float x)
{
  return (double) x;
}
static double to_cmp(unsigned x)
{
  return (double) x;
}
static double to_cmp(double x)
{
  return x;
}
static double to_cmp(__half x)
{
  return (double) __half2float(x);
}
static bool is_neg_zero(float x)
{
  return x == 0.f && std::signbit(x);
}
static bool is_neg_zero(double x)
{
  return x == 0. && std::signbit(x);
}
static bool is_neg_zero(unsigned)
{
  return false;
}
static bool is_neg_zero(__half x)
{
  return is_neg_zero(__half2float(x));
}

template <typename K>
static bool validate(
  const std::vector<K>& in,
  int valid,
  int k,
  bool select_min,
  bool keys_only,
  bool check_neg_zero,
  const K* ov,
  const int* oi,
  std::string& why)
{
  std::vector<double> ref(valid);
  for (int i = 0; i < valid; ++i)
  {
    ref[i] = to_cmp(in[i]);
  }
  std::vector<double> sorted = ref;
  if (select_min)
  {
    std::sort(sorted.begin(), sorted.end());
  }
  else
  {
    std::sort(sorted.begin(), sorted.end(), [](double a, double b) {
      return a > b;
    });
  }
  std::vector<double> want(sorted.begin(), sorted.begin() + k);
  std::vector<double> got(k);
  for (int i = 0; i < k; ++i)
  {
    got[i] = to_cmp(ov[i]);
  }
  std::sort(want.begin(), want.end());
  std::sort(got.begin(), got.end());
  for (int i = 0; i < k; ++i)
  {
    if (!(want[i] == got[i]))
    {
      char b[160];
      snprintf(b, sizeof b, "value multiset mismatch at %d/%d: want %g got %g", i, k, want[i], got[i]);
      why = b;
      return false;
    }
  }
  if (check_neg_zero)
  {
    // neg_zero pattern, k=16, max: exactly 8 of the outputs must be bit-exact -0.0
    int nz = 0;
    for (int i = 0; i < k; ++i)
    {
      nz += is_neg_zero(ov[i]) ? 1 : 0;
    }
    if (nz != 8)
    {
      why = "-0.0 restoration: expected 8 negative zeros, got " + std::to_string(nz);
      return false;
    }
  }
  if (keys_only)
  {
    return true;
  }
  std::vector<char> used(valid, 0);
  for (int i = 0; i < k; ++i)
  {
    if (oi[i] < 0 || oi[i] >= valid)
    {
      why = "index out of range: " + std::to_string(oi[i]);
      return false;
    }
    if (used[oi[i]])
    {
      why = "duplicate index " + std::to_string(oi[i]);
      return false;
    }
    used[oi[i]] = 1;
    if (!(ref[oi[i]] == to_cmp(ov[i])))
    {
      why = "index/value mismatch at " + std::to_string(i);
      return false;
    }
  }
  return true;
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

// ------------------------------------------------------------------ correctness driver
struct ShapeCase
{
  int k;
  int valid;
  bool full;
};
static const ShapeCase shape_cases[] = {
  {16, kN, true}, {16, kN, false}, {5, 700, false}, {100, 701, false}, {1, kN, true}, {1000, kN, true}};

template <class C, cub::detail::topk::select Dir, bool Full, typename K>
static void one_case(
  const std::vector<K>& in, int k, int valid, int bb, int eb, const char* what, int& fails, int& runs)
{
  K *d_in, *d_ov;
  int* d_oi;
  CHECK(cudaMalloc(&d_in, kN * sizeof(K)));
  CHECK(cudaMalloc(&d_ov, kN * sizeof(K)));
  CHECK(cudaMalloc(&d_oi, kN * sizeof(int)));
  CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(K), cudaMemcpyHostToDevice));
  correct_kernel<C, Dir, Full><<<1, kBlock>>>(d_in, k, valid, bb, eb, d_ov, d_oi);
  CHECK(cudaDeviceSynchronize());
  std::vector<K> ov(k);
  std::vector<int> oi(k);
  CHECK(cudaMemcpy(ov.data(), d_ov, k * sizeof(K), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(oi.data(), d_oi, k * sizeof(int), cudaMemcpyDeviceToHost));
  std::string why;
  ++runs;
  const bool check_nz = (std::string(what).find("neg_zero") != std::string::npos) && k == 16
                     && Dir == cub::detail::topk::select::max;
  if (!validate(in, valid, k, Dir == cub::detail::topk::select::min, C::keys_only, check_nz, ov.data(), oi.data(), why))
  {
    ++fails;
    printf("    FAIL %s: %s\n", what, why.c_str());
  }
  cudaFree(d_in);
  cudaFree(d_ov);
  cudaFree(d_oi);
}

template <class C>
static void run_correct(const char* name)
{
  using K = typename C::key_t;
  std::vector<std::string> pats = {"random", "tie_heavy", "pivot_tie40", "all_equal"};
  if (!std::is_same<K, unsigned>::value)
  {
    pats.push_back("neg_zero");
  }
  int fails = 0, runs = 0;
  for (const auto& pat : pats)
  {
    auto in = gen_pattern<K>(pat, 1);
    for (const auto& sc : shape_cases)
    {
      char what[128];
      snprintf(what, sizeof what, "%s %s k=%d valid=%d full=%d", name, pat.c_str(), sc.k, sc.valid, sc.full);
      const int eb = 8 * (int) sizeof(K);
      if (sc.full)
      {
        one_case<C, cub::detail::topk::select::max, true>(in, sc.k, sc.valid, 0, eb, what, fails, runs);
        one_case<C, cub::detail::topk::select::min, true>(in, sc.k, sc.valid, 0, eb, what, fails, runs);
      }
      else
      {
        one_case<C, cub::detail::topk::select::max, false>(in, sc.k, sc.valid, 0, eb, what, fails, runs);
        one_case<C, cub::detail::topk::select::min, false>(in, sc.k, sc.valid, 0, eb, what, fails, runs);
      }
    }
  }
  // runtime bit-sub-range cases (rolled pass-loop fallback) on masked u32 keys
  if constexpr (std::is_same<K, unsigned>::value)
  {
    auto lo = gen_pattern<unsigned>("masked_low8", 1);
    one_case<C, cub::detail::topk::select::max, true>(lo, 16, kN, 8, 32, "u32 masked_low8 bits[8,32)", fails, runs);
    one_case<C, cub::detail::topk::select::min, true>(lo, 16, kN, 8, 32, "u32 masked_low8 min bits[8,32)", fails, runs);
    auto hi = gen_pattern<unsigned>("masked_high8", 1);
    one_case<C, cub::detail::topk::select::max, true>(hi, 16, kN, 0, 24, "u32 masked_high8 bits[0,24)", fails, runs);
    one_case<C, cub::detail::topk::select::max, false>(hi, 7, 900, 0, 24, "u32 masked_high8 partial", fails, runs);
  }
  printf("  %-36s %s (%d runs)\n", name, fails ? "FAIL" : "PASS", runs);
}

// ------------------------------------------------------------------ perf drivers
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

template <class C>
static void run_lat(const char* name)
{
  const char* pats[] = {"random", "tie_heavy", "pivot_tie40"};
  float* d_in;
  long long* d_out;
  CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CHECK(cudaMalloc(&d_out, sizeof(long long)));
  printf("  %-36s", name);
  for (const char* pat : pats)
  {
    auto in = gen_pattern<float>(pat, 0);
    CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
    const int chains[] = {1, 2, 4, 8, 16};
    double x[5], y[5];
    for (int i = 0; i < 5; ++i)
    {
      lat_kernel<C><<<1, kBlock>>>(d_in, chains[i], d_out);
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

template <class C>
static void run_thr(const char* name)
{
  constexpr int grid = 2048, outer = 32;
  float *d_in, *d_sink;
  CHECK(cudaMalloc(&d_in, kN * sizeof(float)));
  CHECK(cudaMalloc(&d_sink, grid * sizeof(float)));
  auto in = gen_pattern<float>("random", 0);
  CHECK(cudaMemcpy(d_in, in.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
  thr_kernel<C><<<grid, kBlock>>>(d_in, outer, d_sink);
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best_ms = 1e30f;
  for (int rep = 0; rep < 5; ++rep)
  {
    cudaEventRecord(e0);
    thr_kernel<C><<<grid, kBlock>>>(d_in, outer, d_sink);
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best_ms = std::min(best_ms, ms);
  }
  const double gelems = (double) grid * (outer + 1) * kN / (best_ms * 1e-3) / 1e9;
  printf("  %-36s %8.1f G elem/s   (%.3f ms)\n", name, gelems, best_ms);
  cudaFree(d_in);
  cudaFree(d_sink);
}

template <class C>
static void run_res(const char* name)
{
  cudaFuncAttributes a;
  CHECK(cudaFuncGetAttributes(&a, (const void*) thr_kernel<C>));
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*) thr_kernel<C>, kBlock, 0);
  printf("  %-36s regs=%3d  smem=%5zu B  local(spill)=%4zu B  maxblk/SM=%d\n",
         name,
         a.numRegs,
         a.sharedSizeBytes,
         a.localSizeBytes,
         occ);
}

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
#ifndef OLD_HEADER
  const char* build = "NEW header (fea/block-topk-air-latency)";
#else
  const char* build = "OLD header (base)";
#endif
  printf("device: %s (sm_%d%d) — %s\n", p.name, p.major, p.minor, build);

  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness (runtime k / partial tiles / both directions / bit ranges) ===\n");
    run_correct<CfgPairs<float, int, true, true>>("f32+i32  default");
    run_correct<CfgPairs<float, int, false, true>>("f32+i32  Unroll=false");
    run_correct<CfgPairs<float, int, true, false>>("f32+i32  Fuse=false");
    run_correct<CfgKeys<float, true>>("f32 keys default");
    run_correct<CfgKeys<float, false>>("f32 keys Unroll=false");
    run_correct<CfgPairs<unsigned, int, true, true>>("u32+i32  default (+bit ranges)");
    run_correct<CfgPairs<__half, int, true, true>>("f16+i32  default");
    run_correct<CfgPairs<double, int, true, false>>("f64+i32  default(Fuse auto-off)");
    run_correct<CfgPairs<double, int, true, true>>("f64+i32  Fuse=true forced");
    run_correct<CfgPairs<float, long long, true, false>>("f32+i64  default(Fuse auto-off)");
    run_correct<CfgPairs<float, long long, true, true>>("f32+i64  Fuse=true forced");
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== latency: slope cyc/call (f32+i32, k=16, full tile, max) ===\n");
    run_lat<CfgPairs<float, int, true, true>>("f32+i32 default");
    run_lat<CfgPairs<float, int, false, true>>("f32+i32 Unroll=false");
    run_lat<CfgPairs<float, int, true, false>>("f32+i32 Fuse=false");
    run_lat<CfgKeys<float, true>>("f32 keys default");
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== throughput (random) ===\n");
    run_thr<CfgPairs<float, int, true, true>>("f32+i32 default");
    run_thr<CfgPairs<float, int, false, true>>("f32+i32 Unroll=false");
    run_thr<CfgPairs<float, int, true, false>>("f32+i32 Fuse=false");
    run_thr<CfgKeys<float, true>>("f32 keys default");
  }
  if (mode == "res" || mode == "all")
  {
    printf("\n=== resources ===\n");
    run_res<CfgPairs<float, int, true, true>>("f32+i32 default");
    run_res<CfgPairs<float, int, false, true>>("f32+i32 Unroll=false");
    run_res<CfgPairs<float, int, true, false>>("f32+i32 Fuse=false");
    run_res<CfgKeys<float, true>>("f32 keys default");
  }
  return 0;
}
