// Prototype: the "chain-length slope" latency method, demonstrated + validated.
//
// Model:   total_cycles(N) = intercept + slope * N
//   slope     = per-call marginal (steady-state) latency  <-- the number we want
//   intercept = all FIXED per-measurement costs (timer read, pipeline fill of the
//               first call, tail drain of the last, any boundary hoist error, and
//               for block-level the __syncthreads bracket). Cancelled by the slope.
//
// We sweep N over a wide range, do a least-squares fit (slope, intercept, R^2), and
// show (a) linearity where the method is valid and (b) the i-cache/code-size window
// where too-large N breaks linearity.

#include <cub/warp/warp_bitonic_sort.cuh>

#include <cuda/std/limits>

#include <cmath>
#include <cstdint>
#include <cstdio>

__device__ int g_sink[64];

struct CustomLess
{
  template <typename T>
  __device__ bool operator()(const T& a, const T& b) const
  {
    return a < b;
  }
};

template <typename KeyT>
__device__ __forceinline__ KeyT lcg(uint32_t& s)
{
  s = 1664525u * s + 1013904223u;
  return static_cast<KeyT>(s & 0xffffu);
}

constexpr int kReps = 128;

// Statically-unrolled dependency chain of N in-place sorts (data-oblivious => the
// chain is a genuine RAW dependency and cannot be CSE'd or overlapped).
template <int IPT, typename KeyT, int N>
__global__ void __launch_bounds__(32) chain(long long* out)
{
  const int lane = threadIdx.x & 31;
  uint32_t s     = 12345u + (lane + 7) * 2654435761u;
  KeyT keys[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    keys[i] = lcg<KeyT>(s);
  }
  cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{}); // warmup

  long long best = cuda::std::numeric_limits<long long>::max();
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    __syncwarp();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll
    for (int n = 0; n < N; ++n)
    {
      cub::detail::WarpBitonicSort<IPT, KeyT>{}.Sort(keys, CustomLess{});
    }
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      asm volatile("" : : "f"(keys[i]));
    }
    asm volatile("" ::: "memory");
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
    if (keys[0] == KeyT(-1))
    {
      g_sink[lane] = static_cast<int>(keys[0]);
    }
  }
  if (lane == 0)
  {
    *out = best;
  }
}

template <class K>
long long measure(K kernel)
{
  long long* d;
  cudaMalloc(&d, sizeof(long long));
  kernel<<<1, 32>>>(d);
  cudaDeviceSynchronize();
  long long v = 0;
  cudaMemcpy(&v, d, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(d);
  return v;
}

// least-squares fit y = a + b*x over the first `m` points; also R^2
void fit(const double* x, const double* y, int m, double& a, double& b, double& r2)
{
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = 0; i < m; ++i)
  {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  b            = (m * sxy - sx * sy) / (m * sxx - sx * sx);
  a            = (sy - b * sx) / m;
  double ybar  = sy / m;
  double ssres = 0, sstot = 0;
  for (int i = 0; i < m; ++i)
  {
    double f = a + b * x[i];
    ssres += (y[i] - f) * (y[i] - f);
    sstot += (y[i] - ybar) * (y[i] - ybar);
  }
  r2 = 1.0 - ssres / sstot;
}

int main()
{
  constexpr int IPT = 4; // WarpBitonicSort<float> len=128
  const int Ns[]    = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  const int M       = 9;
  double x[M], y[M];
  x[0] = 1;   y[0] = measure(chain<IPT, float, 1>);
  x[1] = 2;   y[1] = measure(chain<IPT, float, 2>);
  x[2] = 4;   y[2] = measure(chain<IPT, float, 4>);
  x[3] = 8;   y[3] = measure(chain<IPT, float, 8>);
  x[4] = 16;  y[4] = measure(chain<IPT, float, 16>);
  x[5] = 32;  y[5] = measure(chain<IPT, float, 32>);
  x[6] = 64;  y[6] = measure(chain<IPT, float, 64>);
  x[7] = 128; y[7] = measure(chain<IPT, float, 128>);
  x[8] = 256; y[8] = measure(chain<IPT, float, 256>);

  printf("=== chain-length slope method: WarpBitonicSort<float> len=128 ===\n");
  printf("  %4s  %10s  %10s  %12s\n", "N", "total_cyc", "total/N", "marginal d/dN");
  for (int i = 0; i < M; ++i)
  {
    double marg = (i == 0) ? y[0] : (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    printf("  %4d  %10.0f  %10.1f  %12.1f\n", Ns[i], y[i], y[i] / x[i], marg);
  }

  double a, b, r2;
  fit(x, y, 6, a, b, r2); // fit over N=1..32 (linear window)
  printf("\n  fit over N=1..32 :  latency(slope) = %.1f cyc/call   fixed-overhead(intercept) = %.1f cyc   R^2 = %.6f\n", b, a, r2);
  fit(x, y, M, a, b, r2); // fit over full range incl. large N
  printf("  fit over N=1..256:  latency(slope) = %.1f cyc/call   fixed-overhead(intercept) = %.1f cyc   R^2 = %.6f\n", b, a, r2);
  printf("\n  interpretation: cold single-call = intercept+slope ~= %.0f cyc;  steady-state per-call = slope\n",
         y[0]);
  return 0;
}
