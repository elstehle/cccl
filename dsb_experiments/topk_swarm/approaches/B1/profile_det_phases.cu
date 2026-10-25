// Phase profile of B1_det2 on one block (T56 method): clock64 stamps per phase written by thread 0,
// random normal input rounded to the key type, striped tile, the deterministic path (prefer_smaller_index,
// B1_det2) next to the not_guaranteed path of the same header (B1_idx's code) on the same inputs.
// Build: nvcc -std=c++17 -O3 -arch=sm_100 -DB1_PROFILE -I<cccl>/cub -I<cccl>/libcudacxx/include
// -I<cccl>/thrust profile_det_phases.cu
#include "block_topk_b1_det2.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

template <typename KeyT, int N, int K, bool Det>
__global__ void __launch_bounds__(256) prof_kernel(const KeyT* in, int* out, long long* stamps, int reps)
{
  constexpr int T   = 256;
  constexpr int IPT = N / T;
  using topk_t      = topk_b1_det2::block_topk_b1<KeyT, T, IPT, 8, Det,
                                              Det ? topk_b1_det2::tie_break::smaller_index : topk_b1_det2::tie_break::none,
                                              topk_b1_det2::ticket_mode::per_item>;
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
    topk_t(ts).select_indices_striped(keys, K, out_idx);
    __syncthreads();
    if (threadIdx.x == 0)
    {
      for (int i = 0; i < 16; ++i)
      {
        stamps[rep * 16 + i]        = topk_b1_det2::g_stamps[i];
        topk_b1_det2::g_stamps[i]   = 0;
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

// phases: 0 start, 1 prologue (twiddle and normalization), 2..5 radix passes, 6 tie ballots and table
// barrier, 7 tie base scan and ranks, 8 epilogue (behind a barrier), 9 flag: 1 when the tie phase ran
static const int ids[]       = {0, 1, 2, 3, 4, 5, 6, 7, 8};
static const char* names[]   = {"start", "prologue", "pass1", "pass2", "pass3", "pass4", "tieballot", "tiescan", "epilogue"};
static const int num_phases  = 9;

static void report(const char* label, const std::vector<long long>& st, int reps, int want_flag)
{
  printf("  %-16s", label);
  int n_sel = 0;
  for (int r = 0; r < reps; ++r)
  {
    n_sel += (want_flag < 0 || st[r * 16 + 9] == want_flag) ? 1 : 0;
  }
  printf(" n=%2d |", n_sel);
  for (int p = 1; p < num_phases; ++p)
  {
    std::vector<long long> d;
    for (int r = 0; r < reps; ++r)
    {
      if (want_flag >= 0 && st[r * 16 + 9] != want_flag)
      {
        continue;
      }
      const long long cur = st[r * 16 + ids[p]];
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
      printf(" %9s %-4s", "-", "");
      continue;
    }
    std::sort(d.begin(), d.end());
    printf(" %9lld n%-3d", d[d.size() / 2], (int) d.size());
  }
  std::vector<long long> tot;
  for (int r = 0; r < reps; ++r)
  {
    if (want_flag >= 0 && st[r * 16 + 9] != want_flag)
    {
      continue;
    }
    tot.push_back(st[r * 16 + 8] - st[r * 16 + 0]);
  }
  if (!tot.empty())
  {
    std::sort(tot.begin(), tot.end());
    printf(" | total %7lld", tot[tot.size() / 2]);
  }
  printf("\n");
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
  printf("%s: median cycles per phase over %d calls (one block, 256 threads), columns", name, reps);
  for (int p = 1; p < num_phases; ++p)
  {
    printf(" %s", names[p]);
  }
  printf("\n");
  std::vector<long long> st(reps * 16);
  prof_kernel<KeyT, N, K, true><<<1, 256>>>(d_in, d_out, d_st, reps);
  prof_kernel<KeyT, N, K, true><<<1, 256>>>(d_in, d_out, d_st, reps);
  cudaDeviceSynchronize();
  cudaMemcpy(st.data(), d_st, st.size() * sizeof(long long), cudaMemcpyDeviceToHost);
  report("det2 all", st, reps, -1);
  report("det2 tie ran", st, reps, 1);
  report("det2 no tie", st, reps, 0);
  prof_kernel<KeyT, N, K, false><<<1, 256>>>(d_in, d_out, d_st, reps);
  prof_kernel<KeyT, N, K, false><<<1, 256>>>(d_in, d_out, d_st, reps);
  cudaDeviceSynchronize();
  cudaMemcpy(st.data(), d_st, st.size() * sizeof(long long), cudaMemcpyDeviceToHost);
  report("idx path all", st, reps, -1);
  report("idx path tie", st, reps, 1);
  report("idx path no tie", st, reps, 0);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_st);
}

int main()
{
  run<__nv_bfloat16, 1024, 16>("bf16 N1024 k16", 32);
  run<__nv_bfloat16, 1024, 256>("bf16 N1024 k256", 32);
  run<__nv_bfloat16, 4096, 64>("bf16 N4096 k64", 32);
  run<__nv_bfloat16, 16384, 256>("bf16 N16384 k256", 32);
  run<float, 1024, 16>("fp32 N1024 k16", 32);
  run<float, 16384, 16>("fp32 N16384 k16", 32);
  const cudaError_t e = cudaDeviceSynchronize();
  printf("status %s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
