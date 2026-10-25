// Host check of the W5 candidate count on the harness's own measurement inputs (seeds 1000 to
// 1015 of the random pattern, exactly what run.py measures), per cell of the focus grid. Compiled
// with nvcc on the node so the generator is the harness's (libstdc++ normal_distribution).
//   nvcc -std=c++17 -O2 -I<harness> -o seed_check seed_check.cu && ./seed_check
// Prints per cell the candidate count of the configured bound for each of the 16 inputs, its max,
// the capacity, and whether any input overflows (fallback taken).
#include "topk_harness.cuh"
#include "adapters/W5_impl.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>

static uint32_t tw32(float f)
{
  uint32_t u;
  memcpy(&u, &f, 4);
  const uint32_t mask = static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u;
  const uint32_t t    = u ^ mask;
  return t + (t == 0x7fffffffu ? 1u : 0u);
}

template <class KeyT>
static std::vector<uint32_t> words_of(const topk::host_input<KeyT>& in, int n)
{
  std::vector<uint32_t> w(n);
  const int idx_bits = w2::ceil_log2(n);
  for (int j = 0; j < n; ++j)
  {
    if constexpr (std::is_same_v<KeyT, float>)
    {
      w[j] = tw32(in.values[j]);
    }
    else
    {
      const uint32_t t = tw32(in.values[j]) >> 16; // bf16 twiddle equals the fp32 twiddle of the widened value, top 16 bits
      w[j]             = (t << 16) | (((1u << idx_bits) - 1u) ^ static_cast<uint32_t>(j));
    }
  }
  return w;
}

template <class KeyT>
static void cell(int n, int k)
{
  const int ipt = n / 32;
  const int J = w5::cert_depth(k), G = w5::cert_groups(k), D = w5::list_depth(k), cap = 32 * D;
  int worst = 0;
  bool ovf  = false;
  printf("%s N %5d k %4d  G %2d J %d D %2d cap %4d :", std::is_same_v<KeyT, float> ? "fp32" : "bf16", n, k, G, J, D, cap);
  for (int s = 0; s < 16; ++s)
  {
    auto in = topk::make_input<KeyT>("random", n, k, 1000u + s);
    auto w  = words_of<KeyT>(in, n);
    // per lane top J, striped
    std::vector<std::vector<uint32_t>> top(32);
    for (int lane = 0; lane < 32; ++lane)
    {
      std::vector<uint32_t> v;
      for (int i = 0; i < ipt; ++i)
      {
        v.push_back(w[i * 32 + lane]);
      }
      std::sort(v.rbegin(), v.rend());
      v.resize(J);
      top[lane] = v;
    }
    // groups of 32 / G adjacent lanes
    uint32_t t = 0xffffffffu;
    const int lpg = 32 / G;
    for (int g = 0; g < G; ++g)
    {
      std::vector<uint32_t> u;
      for (int l = g * lpg; l < (g + 1) * lpg; ++l)
      {
        u.insert(u.end(), top[l].begin(), top[l].end());
      }
      std::sort(u.rbegin(), u.rend());
      t = std::min(t, u[J - 1]);
    }
    int cnt = 0;
    for (int j = 0; j < n; ++j)
    {
      cnt += (w[j] >= t) ? 1 : 0;
    }
    worst = std::max(worst, cnt);
    ovf |= cnt > cap;
    printf(" %d", cnt);
  }
  printf("  | max %d %s\n", worst, ovf ? "OVERFLOW" : "ok");
}

int main()
{
  const int ns[] = {512, 1024, 2048, 4096};
  const int ks[] = {4, 8, 16, 32, 64, 128};
  for (int n : ns)
  {
    for (int k : ks)
    {
      if (2 * k <= n)
      {
        cell<__nv_bfloat16>(n, k);
        cell<float>(n, k);
      }
    }
  }
  return 0;
}
