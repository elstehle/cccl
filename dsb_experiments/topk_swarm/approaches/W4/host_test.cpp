// Host emulation of the W4 merge tree (approaches W4). Runs the per-thread phases of
// harness/adapters/W4_core.cuh in a loop over the threads with the barriers between phases, on
// fp32 (64-bit words) and bf16 bit images (32-bit words), and checks the k indices against the
// exact reference (key descending, index ascending). Build and run on any host:
//   c++ -std=c++17 -O2 -I harness/adapters approaches/W4/host_test.cpp -o /tmp/w4_host && /tmp/w4_host
#include "W4_core.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

static int failures = 0;

template <typename KeyT>
float widen(KeyT k);
template <>
float widen<float>(float k)
{
  return k;
}
template <>
float widen<uint16_t>(uint16_t b)
{
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint16_t to_bf16(float f)
{
  // round to nearest even on the 16-bit prefix (finite inputs only)
  uint32_t u;
  memcpy(&u, &f, 4);
  const uint32_t lsb      = (u >> 16) & 1u;
  const uint32_t rounding = 0x7fffu + lsb;
  return static_cast<uint16_t>((u + rounding) >> 16);
}

// reference: first k of (key desc, index asc), plus the sorted order itself
template <typename KeyT>
std::vector<int> reference(const std::vector<KeyT>& keys, int k)
{
  std::vector<int> idx(keys.size());
  for (size_t i = 0; i < idx.size(); ++i)
  {
    idx[i] = static_cast<int>(i);
  }
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    const float x = widen(keys[a]), y = widen(keys[b]);
    return x > y || (x == y && a < b);
  });
  idx.resize(k);
  return idx;
}

template <typename KeyT>
void sort_runs(std::vector<KeyT>& keys, int runs)
{
  const int L = static_cast<int>(keys.size()) / runs;
  for (int r = 0; r < runs; ++r)
  {
    std::vector<int> idx(L);
    for (int i = 0; i < L; ++i)
    {
      idx[i] = r * L + i;
    }
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
      const float x = widen(keys[a]), y = widen(keys[b]);
      return x > y || (x == y && a < b);
    });
    std::vector<KeyT> tmp(L);
    for (int i = 0; i < L; ++i)
    {
      tmp[i] = keys[idx[i]];
    }
    for (int i = 0; i < L; ++i)
    {
      keys[r * L + i] = tmp[i];
    }
  }
}

template <typename KeyT>
std::vector<KeyT> make_keys(const char* pattern, int n, unsigned seed)
{
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> f(n);
  for (int j = 0; j < n; ++j)
  {
    if (!strcmp(pattern, "random"))
    {
      f[j] = dist(rng);
    }
    else if (!strcmp(pattern, "tie_heavy"))
    {
      f[j] = static_cast<float>(j % 64) / 64.f;
    }
    else if (!strcmp(pattern, "all_equal"))
    {
      f[j] = 1.0f;
    }
    else if (!strcmp(pattern, "signed_zero"))
    {
      f[j] = (j % 3 == 0) ? ((j % 2) ? -0.0f : 0.0f) : -std::fabs(dist(rng)) - 0.5f;
    }
    else if (!strcmp(pattern, "sorted_asc"))
    {
      f[j] = static_cast<float>(j) * 0.001f;
    }
    else
    {
      f[j] = std::fabs(dist(rng));
    }
  }
  std::vector<KeyT> out(n);
  for (int j = 0; j < n; ++j)
  {
    if constexpr (std::is_same<KeyT, float>::value)
    {
      out[j] = f[j];
    }
    else
    {
      out[j] = to_bf16(f[j]);
    }
  }
  return out;
}

template <typename KeyT>
bool check(const char* what, const std::vector<KeyT>& keys, const std::vector<int>& got, int k)
{
  const auto ref = reference(keys, k);
  for (int i = 0; i < k; ++i)
  {
    if (got[i] != ref[i])
    {
      // sets may still agree when ties differ only in order: W4 is exact, so demand the order
      printf("FAIL %s: slot %d got %d (key %g) want %d (key %g)\n", what, i, got[i],
             got[i] >= 0 && got[i] < static_cast<int>(keys.size()) ? widen(keys[got[i]]) : NAN, ref[i],
             widen(keys[ref[i]]));
      ++failures;
      return false;
    }
  }
  return true;
}


// one level of the tree for all threads, barriers emulated by the two loops
template <int Lv, typename Tree>
void run_level(int threads, typename Tree::TempStorage& ts, std::vector<int>& out)
{
  if constexpr (Lv < Tree::levels)
  {
    using L = typename Tree::template level<Lv>;
    std::vector<int> xs(threads);
    for (int t = 0; t < threads; ++t)
    {
      xs[t] = L::split_end(t, ts);
    }
    for (int t = 0; t < threads; ++t)
    {
      const int a_beg = (t % L::TPP == 0) ? 0 : xs[t - 1];
      L::merge(t, ts, out.data(), a_beg, xs[t]);
    }
    run_level<Lv + 1, Tree>(threads, ts, out);
  }
}

// ---- block chunk merge emulation
template <typename KeyT, int N, int K, int M>
void test_block(const char* pattern, unsigned seed)
{
  constexpr int Threads = 256;
  using core            = w4::chunk_core<KeyT, Threads, N, K, M>;
  using tree            = typename core::tree;
  static_assert(tree::valid(), "shape");
  auto keys = make_keys<KeyT>(pattern, N, seed);
  sort_runs(keys, M);
  auto ts = std::make_unique<typename core::TempStorage>();
  memset(ts.get(), 0, sizeof(typename core::TempStorage));
  std::vector<int> out(K, -1);
  for (int t = 0; t < Threads; ++t)
  {
    KeyT mine[core::IPT];
    for (int i = 0; i < core::IPT; ++i)
    {
      mine[i] = keys[t * core::IPT + i];
    }
    core::stage(mine, t, *ts);
  }
  run_level<0, tree>(Threads, *ts, out);
  char what[128];
  snprintf(what, sizeof(what), "block %s N %d K %d M %d %s seed %u", sizeof(KeyT) == 4 ? "fp32" : "bf16", N, K, M,
           pattern, seed);
  if (check(what, keys, out, K))
  {
    printf("ok   %s\n", what);
  }
}

// ---- warp truncated merge sort emulation
template <typename KeyT, int N, int K, int Halves = 1>
void test_warp(const char* pattern, unsigned seed)
{
  using core = w4::wmerge_core<KeyT, N, K, Halves>;
  using tree = typename core::tree;
  static_assert(tree::valid(), "shape");
  auto keys = make_keys<KeyT>(pattern, N, seed);
  auto ts = std::make_unique<typename core::TempStorage>();
  memset(ts.get(), 0, sizeof(typename core::TempStorage));
  std::vector<int> out(K, -1);
  for (int lane = 0; lane < 32; ++lane)
  {
    KeyT mine[core::IPT];
    for (int i = 0; i < core::IPT; ++i)
    {
      mine[i] = keys[lane * core::IPT + i];
    }
    core::prepare(mine, lane, *ts);
  }
  run_level<0, tree>(32, *ts, out);
  char what[128];
  snprintf(what, sizeof(what), "warp  %s N %d K %d H %d %s seed %u", sizeof(KeyT) == 4 ? "fp32" : "bf16", N, K, Halves,
           pattern, seed);
  if (check(what, keys, out, K))
  {
    printf("ok   %s\n", what);
  }
}

template <typename KeyT, int N, int K, int M>
void print_ts_block()
{
  using core = w4::chunk_core<KeyT, 256, N, K, M>;
  printf("ts block %s N %d K %d M %d: %zu bytes\n", sizeof(KeyT) == 4 ? "fp32" : "bf16", N, K, M, sizeof(typename core::TempStorage));
}
template <typename KeyT, int N, int K, int Halves = 1>
void print_ts_warp()
{
  using core = w4::wmerge_core<KeyT, N, K, Halves>;
  printf("ts warp %s N %d K %d: %zu bytes (x8 warps %zu, plus out %d)\n", sizeof(KeyT) == 4 ? "fp32" : "bf16", N, K,
         sizeof(typename core::TempStorage), 8 * sizeof(typename core::TempStorage), 2 * 8 * K * 4);
}

int main()
{
  print_ts_block<float, 8192, 2048, 4>();
  print_ts_block<uint16_t, 8192, 2048, 4>();
  print_ts_block<float, 8192, 1024, 8>();
  print_ts_block<float, 4096, 512, 8>();
  print_ts_warp<float, 2048, 1024>();
  print_ts_warp<uint16_t, 2048, 1024>();
  print_ts_warp<float, 2048, 256>();
  print_ts_warp<float, 2048, 64>();
  print_ts_warp<uint16_t, 2048, 64>();
  print_ts_warp<float, 1024, 256>();
  print_ts_warp<float, 2048, 1024, 2>();
  print_ts_warp<uint16_t, 2048, 1024, 2>();
  const char* patterns[] = {"random", "tie_heavy", "all_equal", "signed_zero", "sorted_asc", "relu"};
  for (const char* p : patterns)
  {
    for (unsigned s = 1; s <= 3; ++s)
    {
      test_block<float, 8192, 2048, 4>(p, s);
      test_block<uint16_t, 8192, 2048, 4>(p, s);
      test_block<float, 8192, 1024, 8>(p, s);
      test_block<uint16_t, 8192, 1024, 8>(p, s);
      test_block<float, 4096, 512, 8>(p, s);
      test_block<float, 8192, 256, 4>(p, s);
      test_warp<float, 2048, 1024>(p, s);
      test_warp<uint16_t, 2048, 1024>(p, s);
      test_warp<float, 2048, 64>(p, s);
      test_warp<uint16_t, 1024, 64>(p, s);
      test_warp<float, 1024, 256>(p, s);
      test_warp<uint16_t, 2048, 32>(p, s);
      test_warp<float, 512, 64>(p, s);
      test_warp<float, 2048, 1024, 2>(p, s);
      test_warp<uint16_t, 2048, 1024, 2>(p, s);
      test_warp<float, 2048, 64, 2>(p, s);
      test_warp<uint16_t, 1024, 64, 2>(p, s);
      test_warp<float, 1024, 256, 2>(p, s);
      test_warp<uint16_t, 2048, 32, 2>(p, s);
    }
  }
  printf("%s: %d failures\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
