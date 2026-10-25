// Phase profile of block_topk_stream on one block: clock64 stamps per phase (thread 0, a log with
// a cursor so every chunk's stamps survive), random normal input rounded to the key type, striped
// tile fed as chunks of 4096 like the harness adapter. Build on the node:
//   nvcc -std=c++17 -O3 -arch=sm_100 -DN3_PROFILE -I<cccl>/cub -I<cccl>/libcudacxx/include
//        -I<cccl>/thrust profile_phases.cu -o profile_phases
// Stamp ids: 0 begin, 1 bracket, 2 first chunk stored, 3 first threshold, then per chunk 4 counted,
// 5 scanned, 6 stored, finish: 7 start, 8 pivot resolved, 9 count sweep, 10 emit sweep, 11 rank.
#define N3_PROFILE 1
#include "block_topk_stream.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int MAXS = 64;

template <int J, class P, typename KeyT, int IPT, int cpt, int chunks, int chunk, int K>
__device__ __forceinline__ void feed(P& p, const KeyT (&keys)[IPT])
{
  if constexpr (J < chunks)
  {
    KeyT ck[cpt];
#pragma unroll
    for (int i = 0; i < cpt; ++i)
    {
      ck[i] = keys[J * cpt + i];
    }
    p.template insert_chunk<J == 0>(ck, J * chunk, K);
    feed<J + 1, P, KeyT, IPT, cpt, chunks, chunk, K>(p, keys);
  }
}


template <typename KeyT, int Threads, int N, int K, bool Smaller>
__global__ void __launch_bounds__(Threads) prof_kernel(const KeyT* in, int* out, long long* stamps, int* ids, int* counts, int reps)
{
  constexpr int IPT    = N / Threads;
  constexpr int chunk  = 4096;
  constexpr int cpt    = chunk / Threads;
  constexpr int chunks = N / chunk;
  using topk_t         = n3::block_topk_stream<KeyT, Threads, chunk, N, 16384, 1024, Smaller>;
  extern __shared__ __align__(16) unsigned char smem[];
  typename topk_t::TempStorage& ts = *reinterpret_cast<typename topk_t::TempStorage*>(smem);
  __shared__ int out_idx[K];
  for (int rep = 0; rep < reps; ++rep)
  {
    KeyT keys[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      keys[i] = in[rep * N + i * Threads + threadIdx.x];
    }
    if (threadIdx.x == 0)
    {
      n3::g_stamp_cursor = 0;
    }
    __syncthreads();
    topk_t p(ts);
    p.begin();
    feed<0, topk_t, KeyT, IPT, cpt, chunks, chunk, K>(p, keys);
    p.finish(K, out_idx);
    __syncthreads();
    if (threadIdx.x == 0)
    {
      const int c = n3::g_stamp_cursor;
      counts[rep] = c;
      for (int i = 0; i < c && i < MAXS; ++i)
      {
        stamps[rep * MAXS + i] = n3::g_stamps[i];
        ids[rep * MAXS + i]    = n3::g_stamp_ids[i];
      }
      out[rep] = out_idx[0];
    }
    __syncthreads();
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

template <typename KeyT, int Threads, int N, int K>
void run(const char* name, int reps)
{
  std::vector<KeyT> h(static_cast<size_t>(reps) * N);
  for (int r = 0; r < reps; ++r)
  {
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ (static_cast<unsigned long long>(r + 1) * 0xD1B54A32D192ED03ull));
    std::normal_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < N; ++i)
    {
      h[static_cast<size_t>(r) * N + i] = to_key<KeyT>(dist(rng));
    }
  }
  KeyT* d_in;
  int* d_out;
  long long* d_st;
  int* d_ids;
  int* d_cnt;
  cudaMalloc(&d_in, h.size() * sizeof(KeyT));
  cudaMalloc(&d_out, reps * sizeof(int));
  cudaMalloc(&d_st, reps * MAXS * sizeof(long long));
  cudaMalloc(&d_ids, reps * MAXS * sizeof(int));
  cudaMalloc(&d_cnt, reps * sizeof(int));
  cudaMemcpy(d_in, h.data(), h.size() * sizeof(KeyT), cudaMemcpyHostToDevice);
  using topk_t = n3::block_topk_stream<KeyT, Threads, 4096, N, 16384, 1024, false>;
  const size_t smem = sizeof(typename topk_t::TempStorage);
  auto kern = prof_kernel<KeyT, Threads, N, K, false>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  kern<<<1, Threads, smem>>>(d_in, d_out, d_st, d_ids, d_cnt, reps); // warm-up
  kern<<<1, Threads, smem>>>(d_in, d_out, d_st, d_ids, d_cnt, reps);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
  {
    printf("%s: CUDA error %s\n", name, cudaGetErrorString(e));
    exit(1);
  }
  std::vector<long long> st(reps * MAXS);
  std::vector<int> ids(reps * MAXS), cnt(reps);
  cudaMemcpy(st.data(), d_st, st.size() * sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(cnt.data(), d_cnt, cnt.size() * sizeof(int), cudaMemcpyDeviceToHost);
  const int c = cnt[0];
  printf("%s: %d stamps per call, smem %zu bytes, median over %d calls (cycles between consecutive stamps)\n", name, c, smem, reps);
  long long total = 0;
  for (int i = 1; i < c; ++i)
  {
    std::vector<long long> d;
    for (int r = 0; r < reps; ++r)
    {
      if (cnt[r] == c)
      {
        d.push_back(st[r * MAXS + i] - st[r * MAXS + i - 1]);
      }
    }
    std::sort(d.begin(), d.end());
    const long long med = d[d.size() / 2];
    total += med;
    printf("  %2d -> %2d : %7lld\n", ids[i - 1], ids[i], med);
  }
  printf("  total     : %7lld\n", total);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_st);
  cudaFree(d_ids);
  cudaFree(d_cnt);
}

int main(int argc, char** argv)
{
  const int reps = (argc > 1) ? atoi(argv[1]) : 32;
  run<float, 256, 32768, 2048>("fp32 T256 N32768 K2048", reps);
  run<float, 512, 32768, 2048>("fp32 T512 N32768 K2048", reps);
  run<__nv_bfloat16, 256, 32768, 2048>("bf16 T256 N32768 K2048", reps);
  run<__nv_bfloat16, 512, 32768, 2048>("bf16 T512 N32768 K2048", reps);
  return 0;
}
