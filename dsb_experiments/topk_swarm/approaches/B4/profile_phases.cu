// Phase profile of block_topk_bound on one block: clock64 stamps per phase, random normal input
// rounded to the key type, striped tile. Build: nvcc -std=c++17 -O3 -arch=sm_100 -DB4_PROFILE
// -I<cccl>/cub -I<cccl>/libcudacxx/include -I<cccl>/thrust profile_phases.cu
#include "block_topk_bound.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

template <typename KeyT, int N, int K>
__global__ void __launch_bounds__(256) prof_kernel(const KeyT* in, int* out, long long* stamps, int reps)
{
  constexpr int T   = 256;
  constexpr int IPT = N / T;
  using topk_t = b4::block_topk_bound<KeyT, T, IPT, (K <= 256) ? 4096 : (K <= 1024) ? 8192 : 16384,
                                      (K + 2 * T - 1) / (2 * T) < 1 ? 1 : (K + 2 * T - 1) / (2 * T), b4::tie_break::any>;
  __shared__ typename topk_t::TempStorage ts;
  __shared__ int out_idx[K];
  for (int rep = 0; rep < reps; ++rep)
  {
    KeyT keys[IPT];
    for (int i = 0; i < IPT; ++i)
    {
      keys[i] = in[rep * N + i * T + threadIdx.x];
    }
    __syncthreads();
    topk_t(ts).select(keys, K, out_idx);
    __syncthreads();
    if (threadIdx.x == 0)
    {
      for (int i = 0; i < 16; ++i)
      {
        stamps[rep * 16 + i] = b4::g_stamps[i];
        b4::g_stamps[i]      = 0;
      }
      out[rep] = out_idx[0];
    }
  }
}

template <typename KeyT>
KeyT to_key(float f);
template <>
float to_key<float>(float f)
{
  return f;
}
template <>
__nv_bfloat16 to_key<__nv_bfloat16>(float f)
{
  return __float2bfloat16(f);
}

template <typename KeyT, int N, int K>
void run(const char* name, int reps)
{
  std::mt19937_64 rng(42);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<KeyT> h(static_cast<size_t>(N) * reps);
  for (auto& x : h)
  {
    x = to_key<KeyT>(dist(rng));
  }
  KeyT* d_in;
  int* d_out;
  long long* d_st;
  cudaMalloc(&d_in, h.size() * sizeof(KeyT));
  cudaMalloc(&d_out, reps * sizeof(int));
  cudaMalloc(&d_st, reps * 16 * sizeof(long long));
  cudaMemcpy(d_in, h.data(), h.size() * sizeof(KeyT), cudaMemcpyHostToDevice);
  prof_kernel<KeyT, N, K><<<1, 256>>>(d_in, d_out, d_st, reps); // warm up
  prof_kernel<KeyT, N, K><<<1, 256>>>(d_in, d_out, d_st, reps);
  cudaDeviceSynchronize();
  std::vector<long long> st(reps * 16);
  cudaMemcpy(st.data(), d_st, st.size() * sizeof(long long), cudaMemcpyDeviceToHost);
  // phases: 0 start, 1 bracket, 2 count scan, 3 compaction, 4 loads, 6.. passes, 10 emit count, 11 emit scan,
  // 12 emit stores, 13 w2 barrier, 14 end
  const int ids[] = {0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14};
  const char* names[] = {"start", "bracket", "cntscan", "compact", "loads", "pass1", "pass2", "pass3", "emitcnt", "emitscan", "emitst", "w2bar", "end"};
  printf("%s: per-phase cycles (median over %d reps), delta from the previous nonzero stamp\n", name, reps);
  for (int p = 1; p < 13; ++p)
  {
    std::vector<long long> d;
    for (int r = 0; r < reps; ++r)
    {
      long long cur = st[r * 16 + ids[p]];
      if (cur == 0)
      {
        continue;
      }
      long long prev = 0;
      for (int q = p - 1; q >= 0; --q)
      {
        if (st[r * 16 + ids[q]] != 0)
        {
          prev = st[r * 16 + ids[q]];
          break;
        }
      }
      d.push_back(cur - prev);
    }
    if (d.empty())
    {
      continue;
    }
    std::sort(d.begin(), d.end());
    printf("  %-9s n=%3d median %6lld min %6lld max %6lld\n", names[p], (int) d.size(), d[d.size() / 2], d.front(), d.back());
  }
  std::vector<long long> tot;
  for (int r = 0; r < reps; ++r)
  {
    tot.push_back(st[r * 16 + 14] - st[r * 16 + 0]);
  }
  std::sort(tot.begin(), tot.end());
  printf("  total     median %6lld min %6lld max %6lld\n", tot[tot.size() / 2], tot.front(), tot.back());
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_st);
}

int main()
{
  run<__nv_bfloat16, 1024, 16>("bf16 N1024 k16", 32);
  run<__nv_bfloat16, 4096, 16>("bf16 N4096 k16", 32);
  run<float, 4096, 64>("fp32 N4096 k64", 32);
  run<float, 32768, 256>("fp32 N32768 k256", 16);
  run<__nv_bfloat16, 32768, 64>("bf16 N32768 k64", 16);
  return 0;
}
