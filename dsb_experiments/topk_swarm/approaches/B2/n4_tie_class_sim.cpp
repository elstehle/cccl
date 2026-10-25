// Host simulation for the N4 kill experiment, step 1: tie class size at the k-th truncated key
// for fp32 normal(0,1) input (the harness random pattern) with the N-adaptive split
// (key bits = 32 - log2 N) and the fixed 20 and 16 bit splits of the idea card.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <vector>

static uint32_t twiddle(float f)
{
  uint32_t u;
  memcpy(&u, &f, 4);
  return (static_cast<int32_t>(u) < 0) ? (0u - u) : (u | 0x80000000u);
}

int main()
{
  std::mt19937 rng(1);
  std::normal_distribution<float> dist(0.f, 1.f);
  const int trials = 2000;
  printf("%6s %5s %7s | %8s %11s %9s %9s\n", "N", "k", "keybits", "mean|C|", "P(fallback)", "fb_rounds", "max|C|");
  for (int N : {1024, 4096, 16384, 32768})
  {
    const int lg = static_cast<int>(std::log2(N));
    for (int k : {4, 16, 64, 256})
    {
      if (k > N / 2)
        continue;
      std::set<int> kbs = {32 - lg, 20, 16};
      for (auto it = kbs.rbegin(); it != kbs.rend(); ++it)
      {
        const int kb = *it, b = 32 - kb;
        double sum_c = 0, fb = 0, sum_rounds = 0;
        int max_c = 0;
        std::vector<uint32_t> tk(N);
        std::vector<int> order(N);
        for (int t = 0; t < trials; ++t)
        {
          for (int i = 0; i < N; ++i)
            tk[i] = twiddle(dist(rng)) >> b;
          std::iota(order.begin(), order.end(), 0);
          std::nth_element(order.begin(), order.begin() + (k - 1), order.end(),
                           [&](int a, int c) { return tk[a] != tk[c] ? tk[a] > tk[c] : a < c; });
          const uint32_t tkk = tk[order[k - 1]];
          int c = 0, a = 0;
          for (int i = 0; i < N; ++i)
          {
            c += (tk[i] == tkk);
            a += (tk[i] > tkk);
          }
          sum_c += c;
          max_c = std::max(max_c, c);
          if (c > k - a)
          {
            fb += 1;
            sum_rounds += (k - a);
          }
        }
        printf("%6d %5d %7d | %8.3f %11.3f %9.2f %9d\n", N, k, kb, sum_c / trials, fb / trials,
               fb > 0 ? sum_rounds / fb : 0.0, max_c);
      }
    }
  }
  return 0;
}
