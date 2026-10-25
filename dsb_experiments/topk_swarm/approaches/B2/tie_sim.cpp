// N4 kill experiment part 1: tie class size of the k-th truncated key on the harness fp32 random
// pattern (same generator as topk_harness.cuh normal_values), for the key widths the B2 packing
// leaves at each N (32 - log2 N bits) and for the idea card's 20 and 16 bits.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static std::vector<float> normal_values(int n, unsigned seed)
{
  std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ (static_cast<unsigned long long>(seed) * 0xD1B54A32D192ED03ull));
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}
static uint32_t twiddle(float f)
{
  uint32_t u; memcpy(&u, &f, 4);
  return u ^ ((uint32_t)((int32_t)u >> 31) | 0x80000000u);
}
int main()
{
  const int Ns[] = {1024, 2048, 4096, 8192, 16384, 32768};
  const int Ks[] = {4, 16, 64, 256};
  printf("%6s %4s %8s | %10s %8s %8s | %10s %8s %8s\n", "N", "k", "keybits", "P(finish)", "mean|S|", "max|S|", "P(fin)@20b", "mean", "max");
  for (int N : Ns)
  {
    int b = 0; while ((1 << b) < N) ++b;
    for (int K : Ks)
    {
      if (2 * K > N) continue;
      for (int variant = 0; variant < 1; ++variant)
      {
        double res[2][3] = {};
        const int keybits_list[2] = {32 - b, 20};
        for (int kb = 0; kb < 2; ++kb)
        {
          const int keybits = keybits_list[kb];
          const int drop = 32 - keybits;
          int fin = 0; double sum = 0; int mx = 0;
          const int seeds = 2000;
          for (int seed = 0; seed < seeds; ++seed)
          {
            auto v = normal_values(N, seed);
            std::vector<uint32_t> tw(N);
            for (int i = 0; i < N; ++i) tw[i] = twiddle(v[i]);
            std::vector<uint32_t> sorted = tw;
            std::sort(sorted.begin(), sorted.end(), std::greater<uint32_t>());
            const uint32_t tk = sorted[K - 1] >> drop;
            int all = 0, in = 0;
            for (int i = 0; i < N; ++i) all += (tw[i] >> drop) == tk;
            for (int i = 0; i < K; ++i) in += (sorted[i] >> drop) == tk;
            fin += (all != in);
            sum += all; mx = std::max(mx, all);
          }
          res[kb][0] = (double)fin / seeds; res[kb][1] = sum / seeds; res[kb][2] = mx;
        }
        printf("%6d %4d %8d | %10.4f %8.3f %8.0f | %10.4f %8.3f %8.0f\n", N, K, 32 - b, res[0][0], res[0][1], res[0][2], res[1][0], res[1][1], res[1][2]);
      }
    }
  }
  return 0;
}
