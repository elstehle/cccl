// Max achievable throughput for WarpBitonicSort (exp/sub-warp-bitonic-sort), unstable vs stable.
//
// Segment sizes 4..512:
//   4, 8, 16, 32   = sub-warp logical warps (LW = size, IPT = 1; 32/LW segments per hw warp)
//   64..512        = LW 32, IPT = size/32 (2, 4, 8, 16)
// Methods:
//   unstable = WarpBitonicSort<float>
//   stable   = WarpBitonicSort<u64> on the bit-twiddled (key32 << 32) | rank pack, rank = the
//              within-segment striped position (the winning wrapper from STABLE_WRAP_RESULTS.md)
// Keys-only and pairs (int values via the ValueT channel). Data generated in registers (LCG),
// every output consumed through the DCE sink -> measures the pure compute/issue ceiling, no
// global-memory traffic.
//
// Throughput methodology: grid = WAVES x (occupancy-max blocks) x SMs with WAVES in {1, 4, 16}
// to expose and then amortize the tail effect; fixed iterations per thread; best of 5
// cudaEvent timings; element-normalized Gelem/s. Block sizes 128 and 256 both measured.
//
// Build: nvcc -std=c++17 -arch=sm_100 -O3 -I<branch>/cub -I<branch>/libcudacxx/include \
//        -I<branch>/thrust proto_wbs_maxthr.cu
// Modes: ./proto_wbs_maxthr [correct|thr|res|all]

#include <cub/warp/warp_bitonic_sort.cuh>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

__device__ int g_dcesink[16];

__device__ __forceinline__ unsigned read_smid()
{
  unsigned x;
  asm("mov.u32 %0, %%smid;" : "=r"(x));
  return x;
}

template <int W>
__device__ __forceinline__ void sink_f(float (&v)[W])
{
  if (read_smid() == (unsigned) -1)
  {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < W; ++i)
    {
      sum += v[i];
    }
    *reinterpret_cast<float*>(g_dcesink) += sum;
  }
}

__device__ __forceinline__ float lcg_f(unsigned& s)
{
  s = 1664525u * s + 1013904223u;
  return (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
}

__device__ __forceinline__ unsigned twiddle_in(float f)
{
  const unsigned u = __float_as_uint(f);
  return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
__device__ __forceinline__ float twiddle_out(unsigned u)
{
  u ^= ((u >> 31) ? 0x80000000u : 0xFFFFFFFFu);
  return __uint_as_float(u);
}

struct LessAny
{
  template <typename T>
  __device__ __forceinline__ bool operator()(const T& a, const T& b) const
  {
    return a < b;
  }
};

// ------------------------------------------------------------------ sort dispatch (striped)
// METHOD: 0 = unstable float, 1 = stable twiddle-packed u64
template <int LW, int IPT, int METHOD, bool PAIRS>
__device__ __forceinline__ void do_sort(float (&k)[IPT], int (&v)[IPT])
{
  if constexpr (METHOD == 0)
  {
    if constexpr (PAIRS)
    {
      using BS = cub::detail::WarpBitonicSort<float, IPT, LW, int>;
      typename BS::TempStorage ts;
      BS(ts).Sort(k, v, LessAny{});
    }
    else
    {
      using BS = cub::detail::WarpBitonicSort<float, IPT, LW>;
      typename BS::TempStorage ts;
      BS(ts).Sort(k, LessAny{});
    }
  }
  else
  {
    const int subl = (threadIdx.x & 31) & (LW - 1);
    unsigned long long a[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      a[i] = ((unsigned long long) twiddle_in(k[i]) << 32) | (unsigned) (subl + LW * i);
    }
    if constexpr (PAIRS)
    {
      using BS = cub::detail::WarpBitonicSort<unsigned long long, IPT, LW, int>;
      typename BS::TempStorage ts;
      BS(ts).Sort(a, v, LessAny{});
    }
    else
    {
      using BS = cub::detail::WarpBitonicSort<unsigned long long, IPT, LW>;
      typename BS::TempStorage ts;
      BS(ts).Sort(a, LessAny{});
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = twiddle_out((unsigned) (a[i] >> 32));
    }
  }
}

// ------------------------------------------------------------------ correctness
template <int LW, int IPT, int METHOD, bool PAIRS>
__global__ void correct_k(const float* ink, const int* inv, float* outk, int* outv)
{
  const int lane = threadIdx.x & 31;
  const int subl = lane & (LW - 1);
  const int base = (lane / LW) * (LW * IPT); // segment's slice of the 32*IPT warp elements
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = ink[base + subl + LW * i]; // striped within segment
    v[i] = inv[base + subl + LW * i];
  }
  do_sort<LW, IPT, METHOD, PAIRS>(k, v);
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    outk[base + subl + LW * i] = k[i];
    outv[base + subl + LW * i] = v[i];
  }
}

// ------------------------------------------------------------------ throughput
constexpr int kThrIters = 64;

template <int LW, int IPT, int METHOD, bool PAIRS, int BLOCK>
__global__ void __launch_bounds__(BLOCK) thr_k(int num_iter)
{
  unsigned s = 12345u + (threadIdx.x + blockIdx.x * BLOCK + 7) * 2654435761u;
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    v[i] = i;
  }
#pragma unroll 1
  for (int iter = 0; iter < num_iter; ++iter)
  {
    unsigned ss = s + iter * 2654435761u;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = lcg_f(ss);
    }
    do_sort<LW, IPT, METHOD, PAIRS>(k, v);
    float cv[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      cv[i] = k[i] + (PAIRS ? (float) v[i] : 0.f); // consume every output
    }
    sink_f(cv);
  }
}

// ------------------------------------------------------------------ host
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

template <int LW, int IPT, int METHOD, bool PAIRS>
static bool check_one(unsigned seed, int pattern)
{
  const int N    = 32 * IPT; // one hardware warp's elements, 32/LW segments
  const int SEGS = 32 / LW;
  const int SEGN = LW * IPT;
  std::vector<float> ink(N);
  std::vector<int> inv(N);
  unsigned s = seed;
  for (int i = 0; i < N; ++i)
  {
    s      = 1664525u * s + 1013904223u;
    ink[i] = pattern == 0 ? (float) ((s >> 8) & 0xffffu) : pattern == 1 ? (float) ((s >> 8) % 5) : 1.f;
    inv[i] = i % SEGN; // within-segment input position (identity per segment)
  }
  float *d_ik, *d_ok;
  int *d_iv, *d_ov;
  CHECK(cudaMalloc(&d_ik, N * 4));
  CHECK(cudaMalloc(&d_ok, N * 4));
  CHECK(cudaMalloc(&d_iv, N * 4));
  CHECK(cudaMalloc(&d_ov, N * 4));
  CHECK(cudaMemcpy(d_ik, ink.data(), N * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_iv, inv.data(), N * 4, cudaMemcpyHostToDevice));
  correct_k<LW, IPT, METHOD, PAIRS><<<1, 32>>>(d_ik, d_iv, d_ok, d_ov);
  CHECK(cudaDeviceSynchronize());
  std::vector<float> ok(N);
  std::vector<int> ov(N);
  CHECK(cudaMemcpy(ok.data(), d_ok, N * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ov.data(), d_ov, N * 4, cudaMemcpyDeviceToHost));
  cudaFree(d_ik);
  cudaFree(d_ok);
  cudaFree(d_iv);
  cudaFree(d_ov);
  for (int g = 0; g < SEGS; ++g)
  {
    std::vector<std::pair<float, int>> ref(SEGN);
    for (int i = 0; i < SEGN; ++i)
    {
      ref[i] = {ink[g * SEGN + i], i};
    }
    std::stable_sort(ref.begin(), ref.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    for (int i = 0; i < SEGN; ++i)
    {
      const bool key_ok = ok[g * SEGN + i] == ref[i].first;
      // stability observable only for the stable method with values carried
      const bool val_ok = !(PAIRS && METHOD == 1) || (ov[g * SEGN + i] == ref[i].second);
      if (!key_ok || !val_ok)
      {
        printf("      LW%d IPT%d M%d P%d pat%d seg%d pos%d: got (%g,%d) want (%g,%d)\n",
               LW, IPT, METHOD, (int) PAIRS, pattern, g, i, ok[g * SEGN + i], ov[g * SEGN + i],
               ref[i].first, ref[i].second);
        return false;
      }
    }
  }
  return true;
}

template <int LW, int IPT>
static void run_correct()
{
  bool ok[4] = {true, true, true, true};
  for (int pat = 0; pat <= 2; ++pat)
  {
    for (unsigned t = 0; t < (pat <= 1 ? 8u : 1u); ++t)
    {
      const unsigned sd = 4242u + t * 131u + pat * 7919u;
      ok[0] = ok[0] && check_one<LW, IPT, 0, false>(sd, pat);
      ok[1] = ok[1] && check_one<LW, IPT, 0, true>(sd, pat);
      ok[2] = ok[2] && check_one<LW, IPT, 1, false>(sd, pat);
      ok[3] = ok[3] && check_one<LW, IPT, 1, true>(sd, pat);
    }
  }
  printf("  size %3d (LW%2d IPT%2d):  unstable k/p %s/%s   stable k/p %s/%s  (stable pairs incl. stability)\n",
         LW * IPT, LW, IPT, ok[0] ? "PASS" : "FAIL", ok[1] ? "PASS" : "FAIL", ok[2] ? "PASS" : "FAIL",
         ok[3] ? "PASS" : "FAIL");
}

template <int LW, int IPT, int METHOD, bool PAIRS, int BLOCK>
static void thr_row(int num_SMs)
{
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_k<LW, IPT, METHOD, PAIRS, BLOCK>, BLOCK, 0));
  const int waves[3] = {1, 4, 16};
  double gs[3];
  for (int w = 0; w < 3; ++w)
  {
    const int grid = waves[w] * maxblk * num_SMs;
    thr_k<LW, IPT, METHOD, PAIRS, BLOCK><<<grid, BLOCK>>>(kThrIters);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best_ms = 1e30f;
    for (int rep = 0; rep < 5; ++rep)
    {
      cudaEventRecord(e0);
      thr_k<LW, IPT, METHOD, PAIRS, BLOCK><<<grid, BLOCK>>>(kThrIters);
      cudaEventRecord(e1);
      CHECK(cudaDeviceSynchronize());
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      best_ms = std::min(best_ms, ms);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    const double elems = (double) grid * (BLOCK / 32) * kThrIters * (double) (32 * IPT);
    gs[w]              = elems / (best_ms * 1e-3) / 1e9;
  }
  printf("    %-8s %s blk%3d occ%2d:  1w=%7.1f   4w=%7.1f  16w=%7.1f  Gelem/s\n",
         METHOD == 0 ? "unstable" : "stable", PAIRS ? "pairs" : "keys ", BLOCK, maxblk, gs[0], gs[1], gs[2]);
}

template <int LW, int IPT>
static void run_thr(int num_SMs)
{
  printf("  size %3d (LW%2d IPT%2d):\n", LW * IPT, LW, IPT);
  thr_row<LW, IPT, 0, false, 128>(num_SMs);
  thr_row<LW, IPT, 0, false, 256>(num_SMs);
  thr_row<LW, IPT, 1, false, 128>(num_SMs);
  thr_row<LW, IPT, 1, false, 256>(num_SMs);
  thr_row<LW, IPT, 0, true, 128>(num_SMs);
  thr_row<LW, IPT, 0, true, 256>(num_SMs);
  thr_row<LW, IPT, 1, true, 128>(num_SMs);
  thr_row<LW, IPT, 1, true, 256>(num_SMs);
}

template <int LW, int IPT, int METHOD, bool PAIRS, int BLOCK>
static void res_row()
{
  cudaFuncAttributes at{};
  CHECK(cudaFuncGetAttributes(&at, thr_k<LW, IPT, METHOD, PAIRS, BLOCK>));
  int maxblk = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, thr_k<LW, IPT, METHOD, PAIRS, BLOCK>, BLOCK, 0));
  printf("    %-8s %s blk%3d: regs=%3d spills=%4zuB occ=%2d blk/SM (%d thr/SM)\n",
         METHOD == 0 ? "unstable" : "stable", PAIRS ? "pairs" : "keys ", BLOCK, at.numRegs,
         (size_t) at.localSizeBytes, maxblk, maxblk * BLOCK);
}

template <int LW, int IPT>
static void run_res()
{
  printf("  size %3d (LW%2d IPT%2d):\n", LW * IPT, LW, IPT);
  res_row<LW, IPT, 0, false, 256>();
  res_row<LW, IPT, 1, false, 256>();
  res_row<LW, IPT, 0, true, 256>();
  res_row<LW, IPT, 1, true, 256>();
}

#define FOR_SIZES(X) X(4, 1) X(8, 1) X(16, 1) X(32, 1) X(32, 2) X(32, 4) X(32, 8) X(32, 16)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness (all timed instantiations; stable pairs checked vs std::stable_sort) ===\n");
#define C(LW, IPT) run_correct<LW, IPT>();
    FOR_SIZES(C)
  }
  if (mode == "thr" || mode == "all")
  {
    printf("\n=== THROUGHPUT: waves x occupancy grid, %d iters/thread, Gelem/s (best of 5) ===\n", kThrIters);
#define T(LW, IPT) run_thr<LW, IPT>(p.multiProcessorCount);
    FOR_SIZES(T)
  }
  if (mode == "res" || mode == "all")
  {
    printf("\n=== RESOURCES (thr kernels, block 256) ===\n");
#define R(LW, IPT) run_res<LW, IPT>();
    FOR_SIZES(R)
  }
  return 0;
}
