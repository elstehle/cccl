// Validation for branch fix/block-merge-sort-partial-tile (base: NVIDIA/cccl upstream main).
//
// Two partial-tile interfaces on Warp/BlockMergeSort:
//   A) Sort(keys[, items], op, valid_items, oob_default)  — pad-then-sort-full. Contract: prefix
//      sorted AND suffix == oob_default (now guaranteed). Precondition: oob_default ordered after
//      all valid keys, uniform across threads. Merge rounds carry no valid_items logic at all.
//   B) Sort(keys[, items], op, valid_items)               — NEW, clamped rounds, no sentinel
//      needed. Contract: prefix sorted, suffix unspecified.  (compiled only with -DFIXED=1)
//
// Build baseline: nvcc <flags> -I<main>/cub -I<main>/libcudacxx/include -I<main>/thrust proto_merge_fix.cu
// Build fixed:    nvcc <flags> -DFIXED=1 -I<shadow> -I<main>/cub ... proto_merge_fix.cu
// Modes: correct | stability | device | lat | all
//   correct:   A-prefix, A-suffix==oob (expected to FAIL on baseline for mid-partials), B-prefix
//   stability: StableSort with heavy ties, exact match vs std::stable_sort
//   device:    DeviceMergeSort::SortPairs + StableSortKeys end-to-end (exercises the agent change)
//   lat:       chain-slope latency, full tile + partial tile (A), baseline-vs-fixed across builds

#include <cub/block/block_merge_sort.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef FIXED
#  define FIXED 0
#endif

constexpr int kWarp  = 32;
constexpr int kBlock = 256;

struct LessK
{
  template <typename T>
  __device__ __host__ __forceinline__ bool operator()(const T& a, const T& b) const
  {
    return a < b;
  }
};

__device__ __forceinline__ float lcg_f(unsigned& s)
{
  s = 1664525u * s + 1013904223u;
  return (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
}

// ------------------------------------------------------------------ correctness kernels
// OVERLOAD: 0 = A (oob_default), 1 = B (valid_items only, FIXED builds), 2 = full-tile Sort
template <int IPT, int SCOPE, int OVERLOAD>
__global__ void correct_k(const float* in, float* out, int valid_items)
{
  const int tid = threadIdx.x;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in[tid * IPT + i];
  }
  if constexpr (SCOPE == 0)
  {
    using WMS = cub::WarpMergeSort<float, IPT, kWarp>;
    __shared__ typename WMS::TempStorage temp;
    WMS sort(temp);
    if constexpr (OVERLOAD == 0)
    {
      sort.Sort(k, LessK{}, valid_items, INFINITY);
    }
#if FIXED
    else if constexpr (OVERLOAD == 1)
    {
      sort.Sort(k, LessK{}, valid_items);
    }
#endif
    else
    {
      sort.Sort(k, LessK{});
    }
  }
  else
  {
    using BMS = cub::BlockMergeSort<float, kBlock, IPT>;
    __shared__ typename BMS::TempStorage temp;
    BMS sort(temp);
    if constexpr (OVERLOAD == 0)
    {
      sort.Sort(k, LessK{}, valid_items, INFINITY);
    }
#if FIXED
    else if constexpr (OVERLOAD == 1)
    {
      sort.Sort(k, LessK{}, valid_items);
    }
#endif
    else
    {
      sort.Sort(k, LessK{});
    }
  }
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out[tid * IPT + i] = k[i];
  }
}

// ------------------------------------------------------------------ stability kernels (pairs)
// OVERLOAD: 0 = A partial, 1 = B partial (FIXED), 2 = full tile
template <int IPT, int SCOPE, int OVERLOAD>
__global__ void stable_k(const float* in_k, float* out_k, int* out_v, int valid_items)
{
  const int tid = threadIdx.x;
  float k[IPT];
  int v[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in_k[tid * IPT + i];
    v[i] = tid * IPT + i;
  }
  if constexpr (SCOPE == 0)
  {
    using WMS = cub::WarpMergeSort<float, IPT, kWarp, int>;
    __shared__ typename WMS::TempStorage temp;
    WMS sort(temp);
    if constexpr (OVERLOAD == 0)
    {
      sort.StableSort(k, v, LessK{}, valid_items, INFINITY);
    }
#if FIXED
    else if constexpr (OVERLOAD == 1)
    {
      sort.StableSort(k, v, LessK{}, valid_items);
    }
#endif
    else
    {
      sort.StableSort(k, v, LessK{});
    }
  }
  else
  {
    using BMS = cub::BlockMergeSort<float, kBlock, IPT, int>;
    __shared__ typename BMS::TempStorage temp;
    BMS sort(temp);
    if constexpr (OVERLOAD == 0)
    {
      sort.StableSort(k, v, LessK{}, valid_items, INFINITY);
    }
#if FIXED
    else if constexpr (OVERLOAD == 1)
    {
      sort.StableSort(k, v, LessK{}, valid_items);
    }
#endif
    else
    {
      sort.StableSort(k, v, LessK{});
    }
  }
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out_k[tid * IPT + i] = k[i];
    out_v[tid * IPT + i] = v[i];
  }
}

// ------------------------------------------------------------------ latency kernels
constexpr int kLatReps = 24;

// MODE: 0 = overload A partial (valid = N-3), 1 = full-tile Sort
template <int IPT, int SCOPE, int MODE, int DO_SORT>
__global__ void lat_k(unsigned seed0, int chain, long long* out_t)
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  unsigned s       = seed0 + (threadIdx.x + 7) * 2654435761u;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s);
  }
  using WMS = cub::WarpMergeSort<float, IPT, kWarp>;
  using BMS = cub::BlockMergeSort<float, kBlock, IPT>;
  __shared__ union
  {
    typename WMS::TempStorage w;
    typename BMS::TempStorage b;
  } temp;
  constexpr int valid = NT * IPT - 3;
  long long best      = LLONG_MAX;
#pragma unroll 1
  for (int rep = 0; rep < kLatReps; ++rep)
  {
    if constexpr (SCOPE == 0)
    {
      __syncwarp();
    }
    else
    {
      __syncthreads();
    }
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll 1
    for (int n = 0; n < chain; ++n)
    {
      unsigned ss = s ^ __float_as_uint(k[0]);
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        k[i] = lcg_f(ss);
      }
      if (DO_SORT)
      {
        if constexpr (SCOPE == 0)
        {
          if constexpr (MODE == 0)
          {
            WMS(temp.w).Sort(k, LessK{}, valid, INFINITY);
          }
          else
          {
            WMS(temp.w).Sort(k, LessK{});
          }
        }
        else
        {
          if constexpr (MODE == 0)
          {
            BMS(temp.b).Sort(k, LessK{}, valid, INFINITY);
          }
          else
          {
            BMS(temp.b).Sort(k, LessK{});
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      asm volatile("" ::"f"(k[i]));
    }
    asm volatile("" ::: "memory");
    if constexpr (SCOPE == 1)
    {
      __syncthreads();
    }
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    *out_t = best;
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

static std::vector<float> rand_input(int n, unsigned seed, int distinct = 0)
{
  std::vector<float> v(n);
  unsigned s = seed;
  for (int i = 0; i < n; ++i)
  {
    s    = 1664525u * s + 1013904223u;
    v[i] = distinct ? (float) ((s >> 8) % distinct) : (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
  }
  return v;
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

template <int IPT, int SCOPE, int OVERLOAD>
static void check_correct(int valid_items, int& fails, char* buf)
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  const int N      = NT * IPT;
  auto in          = rand_input(N, 71u + IPT * 13 + SCOPE * 7 + valid_items);
  std::vector<float> ref(in.begin(), in.begin() + valid_items);
  std::sort(ref.begin(), ref.end());
  float *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CHECK(cudaMemcpy(d_in, in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  correct_k<IPT, SCOPE, OVERLOAD><<<1, NT>>>(d_in, d_out, valid_items);
  CHECK(cudaDeviceSynchronize());
  std::vector<float> out(N);
  CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
  bool prefix = std::equal(ref.begin(), ref.end(), out.begin());
  bool suffix = true;
  if (OVERLOAD == 0)
  {
    for (int i = valid_items; i < N; ++i)
    {
      suffix = suffix && std::isinf(out[i]);
    }
  }
  const bool ok = prefix && suffix;
  if (!ok)
  {
    ++fails;
  }
  sprintf(buf + strlen(buf), " v=%d:%s%s", valid_items, prefix ? "P" : "!p", OVERLOAD == 0 ? (suffix ? "S" : "!s") : "");
}

template <int IPT, int SCOPE>
static void run_correct(int& fails)
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  const int N      = NT * IPT;
  const int valids[5] = {0, 1, N / 2 + 3, N - 1, N};
  char bufA[256] = "", bufB[256] = "";
  for (int i = 0; i < 5; ++i)
  {
    check_correct<IPT, SCOPE, 0>(valids[i] > N ? N : valids[i], fails, bufA);
  }
#if FIXED
  for (int i = 0; i < 5; ++i)
  {
    check_correct<IPT, SCOPE, 1>(valids[i] > N ? N : valids[i], fails, bufB);
  }
#endif
  printf("  %s size %4d (IPT %2d)  A(oob):%s%s\n",
         SCOPE == 0 ? "warp " : "block", N, IPT, bufA, FIXED ? "" : "   [B: n/a in baseline]");
#if FIXED
  printf("                          B(no-oob):%s\n", bufB);
#endif
}

template <int IPT, int SCOPE, int OVERLOAD>
static void check_stable(int valid_items, int& fails, char* buf)
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  const int N      = NT * IPT;
  auto in          = rand_input(N, 5u + IPT + SCOPE, /*distinct=*/7); // heavy ties
  std::vector<std::pair<float, int>> ref;
  for (int i = 0; i < valid_items; ++i)
  {
    ref.push_back({in[i], i});
  }
  std::stable_sort(ref.begin(), ref.end(), [](auto& a, auto& b) {
    return a.first < b.first;
  });
  float *d_in, *d_ok;
  int* d_ov;
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK(cudaMalloc(&d_ok, N * sizeof(float)));
  CHECK(cudaMalloc(&d_ov, N * sizeof(int)));
  CHECK(cudaMemcpy(d_in, in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  stable_k<IPT, SCOPE, OVERLOAD><<<1, NT>>>(d_in, d_ok, d_ov, valid_items);
  CHECK(cudaDeviceSynchronize());
  std::vector<float> ok(N);
  std::vector<int> ov(N);
  CHECK(cudaMemcpy(ok.data(), d_ok, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ov.data(), d_ov, N * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_ok);
  cudaFree(d_ov);
  bool good = true;
  for (int i = 0; i < valid_items; ++i)
  {
    good = good && (ok[i] == ref[i].first) && (ov[i] == ref[i].second);
  }
  if (!good)
  {
    ++fails;
  }
  sprintf(buf + strlen(buf), " v=%d:%s", valid_items, good ? "P" : "FAIL");
}

template <int IPT, int SCOPE>
static void run_stable(int& fails)
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  const int N      = NT * IPT;
  char buf[256]    = "";
  check_stable<IPT, SCOPE, 2>(N, fails, buf); // full tile
  check_stable<IPT, SCOPE, 0>(N / 2 + 3, fails, buf); // A partial
  check_stable<IPT, SCOPE, 0>(N - 1, fails, buf);
#if FIXED
  check_stable<IPT, SCOPE, 1>(N / 2 + 3, fails, buf); // B partial
  check_stable<IPT, SCOPE, 1>(N - 1, fails, buf);
#endif
  printf("  %s size %4d (IPT %2d)  full,A(v=%d),A(v=%d)%s:%s\n",
         SCOPE == 0 ? "warp " : "block", N, IPT, N / 2 + 3, N - 1, FIXED ? ",B,B" : "", buf);
}

static void run_device(int& fails)
{
  const int sizes[] = {1, 37, (1 << 20) + 12345};
  for (int n : sizes)
  {
    auto keys = rand_input(n, 99u + n, /*distinct=*/9);
    std::vector<int> vals(n);
    for (int i = 0; i < n; ++i)
    {
      vals[i] = i;
    }
    std::vector<std::pair<float, int>> ref(n);
    for (int i = 0; i < n; ++i)
    {
      ref[i] = {keys[i], i};
    }
    std::stable_sort(ref.begin(), ref.end(), [](auto& a, auto& b) {
      return a.first < b.first;
    });
    float* d_k;
    int* d_v;
    CHECK(cudaMalloc(&d_k, n * sizeof(float)));
    CHECK(cudaMalloc(&d_v, n * sizeof(int)));
    CHECK(cudaMemcpy(d_k, keys.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, vals.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    void* d_tmp      = nullptr;
    size_t tmp_bytes = 0;
    CHECK(cub::DeviceMergeSort::StableSortPairs(d_tmp, tmp_bytes, d_k, d_v, n, LessK{}));
    CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    CHECK(cub::DeviceMergeSort::StableSortPairs(d_tmp, tmp_bytes, d_k, d_v, n, LessK{}));
    CHECK(cudaDeviceSynchronize());
    std::vector<float> ok(n);
    std::vector<int> ov(n);
    CHECK(cudaMemcpy(ok.data(), d_k, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ov.data(), d_v, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_tmp);
    bool good = true;
    for (int i = 0; i < n; ++i)
    {
      good = good && (ok[i] == ref[i].first) && (ov[i] == ref[i].second);
    }
    if (!good)
    {
      ++fails;
    }
    printf("  DeviceMergeSort::StableSortPairs n=%-9d %s\n", n, good ? "PASS" : "FAIL");
  }
}

template <int IPT, int SCOPE, int MODE, int DO>
static double slope_raw()
{
  constexpr int NT = (SCOPE == 0) ? kWarp : kBlock;
  long long* d;
  CHECK(cudaMalloc(&d, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], y[5];
  for (int i = 0; i < 5; ++i)
  {
    lat_k<IPT, SCOPE, MODE, DO><<<1, NT>>>(12345u, chains[i], d);
    CHECK(cudaDeviceSynchronize());
    long long c;
    CHECK(cudaMemcpy(&c, d, sizeof c, cudaMemcpyDeviceToHost));
    x[i] = chains[i];
    y[i] = (double) c;
  }
  double a, b;
  fit(x, y, 5, a, b);
  cudaFree(d);
  return b;
}

template <int IPT, int SCOPE>
static void run_lat()
{
  const double gen  = slope_raw<IPT, SCOPE, 0, 0>();
  const double part = slope_raw<IPT, SCOPE, 0, 1>() - gen;
  const double full = slope_raw<IPT, SCOPE, 1, 1>() - gen;
  printf("  %s size %4d (IPT %2d):  full=%8.1f   partialA(v=N-3)=%8.1f\n",
         SCOPE == 0 ? "warp " : "block", ((SCOPE == 0) ? kWarp : kBlock) * IPT, IPT, full, part);
}

#define FOR_CFG(X) X(1, 0) X(2, 0) X(4, 0) X(8, 0) X(1, 1) X(4, 1) X(8, 1)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d, %d SMs)  build=%s\n", p.name, p.major, p.minor, p.multiProcessorCount,
         FIXED ? "FIXED" : "BASELINE");
  int fails = 0;
  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness: A-prefix/A-suffix (P/S), B-prefix; valids {0,1,N/2+3,N-1,N} ===\n");
#define C(IPT, S) run_correct<IPT, S>(fails);
    FOR_CFG(C)
  }
  if (mode == "stability" || mode == "all")
  {
    printf("\n=== stability vs std::stable_sort (7 distinct keys) ===\n");
#define ST(IPT, S) run_stable<IPT, S>(fails);
    FOR_CFG(ST)
  }
  if (mode == "device" || mode == "all")
  {
    printf("\n=== DeviceMergeSort end-to-end (partial last tile) ===\n");
    run_device(fails);
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== LATENCY: slope cyc/call (gen-subtracted); compare across builds ===\n");
#define L(IPT, S) run_lat<IPT, S>();
    FOR_CFG(L)
  }
  printf("\nRESULT: %d failing checks%s\n", fails,
         (!FIXED && fails) ? " (baseline A-suffix failures are the documented stock gap)" : "");
  return fails ? 1 : 0;
}
