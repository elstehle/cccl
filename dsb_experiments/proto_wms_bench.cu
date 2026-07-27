// WarpMergeSort latency benchmark + per-phase cycle breakdown, single warp (32 threads),
// keys-only float, sizes 32..384 (IPT = 1,2,3,4,5,6,8,10,12). Node umb-b200-237, sm_100.
//
// Methodology (per DEVICE_SIDE_BENCHMARKING_ISSUE.md): merge sort is DATA-DEPENDENT (it does
// less work on already-sorted input), so an in-place sort->sort chain would measure the best
// case. We therefore feed varied random input per call and serialize the chain through the
// previous call's output (a genuine RAW dependency); this is the "back-to-back random-input
// latency". The per-call input generation is isolated with a generate-only control kernel and
// subtracted, so the reported number is the sort's marginal cost. Cycles via clock64, min of
// reps. The per-phase breakdown comes from a faithful instrumented mirror of WarpMergeSort
// (validated == cub == std::sort) with clock64 stamps at every phase boundary.
//
// Modes: ./proto_wms_bench [correct|lat|prof|all]

#include <cub/warp/warp_merge_sort.cuh>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

constexpr unsigned kFull = 0xffffffffu;
constexpr int kWarp      = 32;

__device__ int g_sink;

struct CustomLess
{
  template <typename T>
  __device__ __forceinline__ bool operator()(const T& a, const T& b) const
  {
    return a < b;
  }
};

__device__ __forceinline__ float lcg_f(unsigned& s)
{
  s = 1664525u * s + 1013904223u;
  return (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
}

// ------------------------------------------------------------------ faithful instrumented mirror
// Replicates BlockMergeSortStrategy::Sort for a single 32-thread arch warp, keys-only, ascending,
// full tile. Stamps: 0 start, 1 after thread-sort, then per round r in [0,5): 2+3r after STS+sync,
// 3+3r after MergePath, 4+3r after SerialMerge.
constexpr int kRounds = 5; // log2(32)
constexpr int kStamps = 2 + kRounds * 3;

struct NoStamp
{
  __device__ __forceinline__ void operator()(int) const {}
};
struct WarpStamp
{
  long long* t;
  __device__ __forceinline__ void operator()(int i) const
  {
    __syncwarp();
    if ((threadIdx.x & 31) == 0)
    {
      t[i] = clock64();
    }
  }
};

__device__ __forceinline__ int
mp_while(const float* k1, const float* k2, int c1, int c2, int diag) // MergePath, data-dependent while
{
  int b = diag < c2 ? 0 : diag - c2;
  int e = ::min(diag, c1);
  while (b < e)
  {
    const int mid   = (b + e) >> 1;
    const float key1 = k1[mid];
    const float key2 = k2[diag - 1 - mid];
    if (key2 < key1)
    {
      e = mid;
    }
    else
    {
      b = mid + 1;
    }
  }
  return b;
}

template <int IPT>
__device__ __forceinline__ void serial_merge(
  const float* sh, int k1b, int k2b, int k1c, int k2c, float (&out)[IPT], float oob)
{
  const int k1e   = k1b + k1c;
  const int k2e   = k2b + k2c;
  float key1      = k1c != 0 ? sh[k1b] : oob;
  float key2      = k2c != 0 ? sh[k2b] : oob;
#pragma unroll
  for (int item = 0; item < IPT; ++item)
  {
    const bool p = (k2b < k2e) && ((k1b >= k1e) || (key2 < key1));
    out[item]    = p ? key2 : key1;
    if (p)
    {
      key2 = sh[++k2b];
    }
    else
    {
      key1 = sh[++k1b];
    }
  }
}

template <int IPT>
__device__ __forceinline__ void thread_odd_even(float (&keys)[IPT])
{
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
#pragma unroll
    for (int j = (1 & i); j < IPT - 1; j += 2)
    {
      if (keys[j + 1] < keys[j])
      {
        const float t = keys[j];
        keys[j]       = keys[j + 1];
        keys[j + 1]   = t;
      }
    }
  }
}

template <int IPT, class Stamp>
__device__ __forceinline__ void mirror_sort(float (&keys)[IPT], float* sh, Stamp stamp)
{
  const int lane = threadIdx.x & 31;
  stamp(0);
  thread_odd_even(keys);
  stamp(1);
#pragma unroll
  for (int round = 0; round < kRounds; ++round)
  {
    const int target = 2 << round; // 2,4,8,16,32
    const int merged = target >> 1;
    const int mask   = target - 1;
    __syncwarp();
#pragma unroll
    for (int item = 0; item < IPT; ++item)
    {
      sh[IPT * lane + item] = keys[item];
    }
    __syncwarp();
    stamp(2 + round * 3 + 0);

    const int first = ~mask & lane;
    const int start = IPT * first;
    const int size  = IPT * merged;
    const int tig   = mask & lane;
    const int diag  = IPT * tig;
    const int k1b   = start;
    const int k2b   = start + size;

    const int pdiag = mp_while(&sh[k1b], &sh[k2b], size, size, diag);
    stamp(2 + round * 3 + 1);

    const int k1b_loc = k1b + pdiag;
    const int k2b_loc = k2b + diag - pdiag;
    serial_merge<IPT>(sh, k1b_loc, k2b_loc, (k1b + size) - k1b_loc, (k2b + size) - k2b_loc, keys, keys[0]);
    stamp(2 + round * 3 + 2);
  }
}

// ------------------------------------------------------------------ kernels
template <int IPT>
__global__ void __launch_bounds__(kWarp) correct_mirror(const float* in, float* out)
{
  __shared__ float sh[kWarp * IPT + 1];
  const int lane = threadIdx.x & 31;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in[lane * IPT + i];
  }
  mirror_sort<IPT>(k, sh, NoStamp{});
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out[lane * IPT + i] = k[i];
  }
}

template <int IPT>
__global__ void __launch_bounds__(kWarp) correct_cub(const float* in, float* out)
{
  using WMS = cub::WarpMergeSort<float, IPT, kWarp>;
  __shared__ typename WMS::TempStorage temp;
  const int lane = threadIdx.x & 31;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = in[lane * IPT + i];
  }
  WMS(temp).Sort(k, CustomLess{});
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    out[lane * IPT + i] = k[i];
  }
}

constexpr int kLatReps = 32;

// do_sort=1: generate + cub sort; do_sort=0: generate only (control). Chain serialized on k[0].
template <int IPT, int DO_SORT>
__global__ void __launch_bounds__(kWarp) lat_cub(unsigned seed0, int chain, long long* out)
{
  using WMS = cub::WarpMergeSort<float, IPT, kWarp>;
  __shared__ typename WMS::TempStorage temp;
  const int lane = threadIdx.x & 31;
  unsigned s     = seed0 + (lane + 7) * 2654435761u;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s);
  }
  if (DO_SORT)
  {
    WMS(temp).Sort(k, CustomLess{}); // warmup
  }
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int rep = 0; rep < kLatReps; ++rep)
  {
    __syncwarp();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll 1
    for (int n = 0; n < chain; ++n)
    {
      unsigned ss = s ^ __float_as_uint(k[0]); // serialize on previous output
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        k[i] = lcg_f(ss);
      }
      if (DO_SORT)
      {
        WMS(temp).Sort(k, CustomLess{});
      }
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      asm volatile("" ::"f"(k[i]));
    }
    asm volatile("" ::: "memory");
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (lane == 0)
  {
    *out = best;
  }
}

template <int IPT>
__global__ void __launch_bounds__(kWarp) prof_mirror(unsigned seed0, long long* d_acc)
{
  __shared__ float sh[kWarp * IPT + 1];
  __shared__ long long ts[kStamps];
  __shared__ long long acc[kStamps];
  const int lane = threadIdx.x & 31;
  unsigned s     = seed0 + (lane + 7) * 2654435761u;
  float k[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    k[i] = lcg_f(s);
  }
  if (lane == 0)
  {
    for (int i = 0; i < kStamps; ++i)
    {
      acc[i] = LLONG_MAX;
    }
  }
  mirror_sort<IPT>(k, sh, NoStamp{}); // warmup
#pragma unroll 1
  for (int rep = 0; rep < kLatReps; ++rep)
  {
    unsigned ss = s ^ __float_as_uint(k[0]);
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      k[i] = lcg_f(ss);
    }
    mirror_sort<IPT>(k, sh, WarpStamp{ts});
    if (lane == 0)
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
  if (lane == 0)
  {
    for (int i = 0; i < kStamps; ++i)
    {
      d_acc[i] = acc[i];
    }
    if (k[0] < -1.f)
    {
      g_sink = (int) k[0];
    }
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

template <int IPT>
bool check_one()
{
  const int N = kWarp * IPT;
  std::vector<float> in(N), out(N), ref;
  unsigned s = 999u;
  for (int i = 0; i < N; ++i)
  {
    s     = 1664525u * s + 1013904223u;
    in[i] = (float) ((s >> 8) & 0xffffu) * (1.0f / 65536.0f);
  }
  ref = in;
  std::sort(ref.begin(), ref.end());
  float *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CHECK(cudaMemcpy(d_in, in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  bool ok = true;
  for (int which = 0; which < 2; ++which)
  {
    if (which == 0)
    {
      correct_mirror<IPT><<<1, kWarp>>>(d_in, d_out);
    }
    else
    {
      correct_cub<IPT><<<1, kWarp>>>(d_in, d_out);
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i)
    {
      if (out[i] != ref[i])
      {
        printf("    IPT=%2d %s MISMATCH at %d: got %g want %g\n", IPT, which ? "cub" : "mirror", i, out[i], ref[i]);
        ok = false;
        break;
      }
    }
  }
  cudaFree(d_in);
  cudaFree(d_out);
  return ok;
}

template <int IPT>
double lat_slope(int do_sort)
{
  long long* d_out;
  CHECK(cudaMalloc(&d_out, sizeof(long long)));
  const int chains[] = {1, 2, 4, 8, 16};
  double x[5], y[5];
  for (int i = 0; i < 5; ++i)
  {
    if (do_sort)
    {
      lat_cub<IPT, 1><<<1, kWarp>>>(12345u, chains[i], d_out);
    }
    else
    {
      lat_cub<IPT, 0><<<1, kWarp>>>(12345u, chains[i], d_out);
    }
    CHECK(cudaDeviceSynchronize());
    long long c;
    CHECK(cudaMemcpy(&c, d_out, sizeof c, cudaMemcpyDeviceToHost));
    x[i] = chains[i];
    y[i] = (double) c;
  }
  double a, b;
  fit(x, y, 5, a, b);
  cudaFree(d_out);
  return b;
}

template <int IPT>
void prof_one(long long acc[kStamps])
{
  long long* d_acc;
  CHECK(cudaMalloc(&d_acc, kStamps * sizeof(long long)));
  prof_mirror<IPT><<<1, kWarp>>>(12345u, d_acc);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(acc, d_acc, kStamps * sizeof(long long), cudaMemcpyDeviceToHost));
  cudaFree(d_acc);
}

template <int IPT>
void run_lat()
{
  const double full = lat_slope<IPT>(1);
  const double gen  = lat_slope<IPT>(0);
  printf("  size %3d (IPT %2d):  sort=%7.1f  (full=%7.1f  gen=%6.1f)  cyc/call\n", kWarp * IPT, IPT, full - gen, full, gen);
}

template <int IPT>
void run_prof()
{
  long long a[kStamps];
  prof_one<IPT>(a);
  long long thread_sort = a[0];
  long long sts = 0, mp = 0, sm = 0;
  for (int r = 0; r < kRounds; ++r)
  {
    sts += a[1 + r * 3 + 0];
    mp += a[1 + r * 3 + 1];
    sm += a[1 + r * 3 + 2];
  }
  const long long tot = thread_sort + sts + mp + sm;
  printf("  size %3d (IPT %2d):  thr_sort=%5lld  STS=%5lld  MergePath=%5lld  SerialMerge=%5lld  | total=%6lld\n",
         kWarp * IPT,
         IPT,
         thread_sort,
         sts,
         mp,
         sm,
         tot);
}

template <int IPT>
void run_prof_rounds()
{
  long long a[kStamps];
  prof_one<IPT>(a);
  printf("  size %3d (IPT %2d) per-round [STS/MP/SM]:", kWarp * IPT, IPT);
  for (int r = 0; r < kRounds; ++r)
  {
    printf("  r%d[%lld/%lld/%lld]", r, a[2 + r * 3], a[3 + r * 3], a[4 + r * 3]);
  }
  printf("  thr=%lld\n", a[0]);
}

#define FOR_SIZES(X) X(1) X(2) X(3) X(4) X(5) X(6) X(8) X(10) X(12)

int main(int argc, char** argv)
{
  std::string mode = argc > 1 ? argv[1] : "all";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

  if (mode == "correct" || mode == "all")
  {
    printf("\n=== correctness (mirror & cub vs std::sort) ===\n");
    bool ok = true;
#define C(IPT) ok &= check_one<IPT>();
    FOR_SIZES(C)
    printf("  %s\n", ok ? "ALL PASS" : "FAILURES ABOVE");
  }
  if (mode == "lat" || mode == "all")
  {
    printf("\n=== latency: cub WarpMergeSort, slope cyc/call (sort = full - gen) ===\n");
#define L(IPT) run_lat<IPT>();
    FOR_SIZES(L)
  }
  if (mode == "prof" || mode == "all")
  {
    printf("\n=== per-phase breakdown (instrumented mirror, min of %d reps), cycles ===\n", kLatReps);
#define P(IPT) run_prof<IPT>();
    FOR_SIZES(P)
    printf("\n=== per-round detail [STS/MergePath/SerialMerge] ===\n");
#define PR(IPT) run_prof_rounds<IPT>();
    FOR_SIZES(PR)
  }
  return 0;
}
