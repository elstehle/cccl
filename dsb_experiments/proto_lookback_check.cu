// Correctness gate for the aggregate-first BlockScanWarpScans prefix-callback prototype:
// exercises the patched decoupled look-back paths end-to-end via DeviceScan::ExclusiveSum /
// InclusiveSum (I32/I64) and DeviceSelect::If (I32 offsets) against host references, across
// sizes that cover single-tile, few-tile, and many-tile grids (incl. non-power-of-two).

#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

struct LessThan
{
  int threshold;
  __host__ __device__ bool operator()(int x) const
  {
    return x < threshold;
  }
};

static unsigned lcg(unsigned& s)
{
  s = 1664525u * s + 1013904223u;
  return s;
}

template <typename T>
bool check_scan(size_t n, unsigned seed, bool inclusive)
{
  std::vector<T> h(n);
  unsigned s = seed;
  for (auto& x : h)
  {
    x = (T) (lcg(s) % 7);
  }
  T *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, n * sizeof(T)));
  CHECK(cudaMalloc(&d_out, n * sizeof(T)));
  CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  void* d_tmp    = nullptr;
  size_t tmp_len = 0;
  if (inclusive)
  {
    cub::DeviceScan::InclusiveSum(d_tmp, tmp_len, d_in, d_out, n);
    CHECK(cudaMalloc(&d_tmp, tmp_len));
    cub::DeviceScan::InclusiveSum(d_tmp, tmp_len, d_in, d_out, n);
  }
  else
  {
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_len, d_in, d_out, n);
    CHECK(cudaMalloc(&d_tmp, tmp_len));
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_len, d_in, d_out, n);
  }
  CHECK(cudaDeviceSynchronize());
  std::vector<T> got(n);
  CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(T), cudaMemcpyDeviceToHost));
  T run    = 0;
  bool ok  = true;
  size_t i = 0;
  for (; i < n; ++i)
  {
    T want = inclusive ? (T) (run + h[i]) : run;
    if (got[i] != want)
    {
      ok = false;
      break;
    }
    run = (T) (run + h[i]);
  }
  if (!ok)
  {
    printf("    scan FAIL n=%zu at i=%zu\n", n, i);
  }
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_tmp);
  return ok;
}

bool check_select(size_t n, unsigned seed)
{
  std::vector<int> h(n);
  unsigned s = seed;
  for (auto& x : h)
  {
    x = (int) (lcg(s) % 1000);
  }
  LessThan pred{500};
  int *d_in, *d_out, *d_num;
  CHECK(cudaMalloc(&d_in, n * sizeof(int)));
  CHECK(cudaMalloc(&d_out, n * sizeof(int)));
  CHECK(cudaMalloc(&d_num, sizeof(int)));
  CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  void* d_tmp    = nullptr;
  size_t tmp_len = 0;
  cub::DeviceSelect::If(d_tmp, tmp_len, d_in, d_out, d_num, (int) n, pred);
  CHECK(cudaMalloc(&d_tmp, tmp_len));
  cub::DeviceSelect::If(d_tmp, tmp_len, d_in, d_out, d_num, (int) n, pred);
  CHECK(cudaDeviceSynchronize());
  int num = 0;
  CHECK(cudaMemcpy(&num, d_num, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> got(num);
  CHECK(cudaMemcpy(got.data(), d_out, num * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> want;
  want.reserve(n);
  for (int x : h)
  {
    if (pred(x))
    {
      want.push_back(x);
    }
  }
  bool ok = ((size_t) num == want.size()) && std::equal(want.begin(), want.end(), got.begin());
  if (!ok)
  {
    printf("    select FAIL n=%zu (num=%d want=%zu)\n", n, num, want.size());
  }
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_num);
  cudaFree(d_tmp);
  return ok;
}

int main()
{
  const size_t sizes[] = {1, 31, 1024, 4097, 100000, 1 << 20, (1 << 24) + 12345};
  int fails            = 0;
  for (size_t n : sizes)
  {
    for (unsigned seed = 0; seed < 3; ++seed)
    {
      fails += !check_scan<int>(n, seed, false);
      fails += !check_scan<int>(n, seed, true);
      fails += !check_scan<long long>(n, seed, false); // I64: fallback path control
      fails += !check_select(n, seed + 7);
    }
  }
  printf(fails ? "LOOKBACK CHECK: %d FAILURES\n" : "LOOKBACK CHECK: ALL PASS\n", fails);
  return fails ? 1 : 0;
}
