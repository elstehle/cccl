// Phase profile of the W4 primitives outside the harness: one warp (W4_wmerge) or one block
// (W4_chunk4) runs the phases of the core with clock64 stamps between them on fresh random input,
// mean cycles per phase over the calls. The stamps and their syncs perturb the phases a little,
// the shape is what matters. Build on a node:
//   nvcc -std=c++17 -O3 -arch=sm_100 -I<repo>/cub -I<repo>/libcudacxx/include -I<repo>/thrust \
//        -I harness approaches/W4/phase_probe.cu -o phase_probe && ./phase_probe
#include "adapters/W4_impl.cuh"

#include <cstdio>
#include <random>
#include <vector>

namespace
{

constexpr int inputs = 16;

template <typename KeyT>
KeyT from_float(float f);
template <>
float from_float<float>(float f)
{
  return f;
}
template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float f)
{
  return __float2bfloat16(f);
}

// ---- warp form: phases = prepare, then per level (split, merge) = 1 + 2 x 5 = 11 stamps
template <typename KeyT, int N, int K>
__global__ void probe_warp(const KeyT* in, int* out, long long* acc, int calls)
{
  using core = w4::wmerge_core<KeyT, N, K>;
  using tree = typename core::tree;
  constexpr int IPT = core::IPT;
  __shared__ typename core::TempStorage ts;
  __shared__ int out_idx[K];
  const int lane = threadIdx.x;
  long long sum[12] = {};
  for (int c = 0; c < calls; ++c)
  {
    const KeyT* src = in + (c % inputs) * N;
    KeyT keys[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      keys[i] = src[lane * IPT + i];
    }
    __syncwarp();
    long long t[12];
    t[0] = clock64();
    core::prepare(keys, lane, ts);
    __syncwarp();
    t[1] = clock64();
    int s = 2;
    auto level = [&](auto lv_tag) {
      using lv        = decltype(lv_tag);
      const int a_beg = lv::split_begin(lane, ts);
      __syncwarp();
      t[s++]          = clock64();
      lv::merge(lane, ts, out_idx, a_beg, 0);
      __syncwarp();
      t[s++] = clock64();
    };
    level(typename tree::template level<0>{});
    level(typename tree::template level<1>{});
    level(typename tree::template level<2>{});
    level(typename tree::template level<3>{});
    level(typename tree::template level<4>{});
    for (int i = 0; i < 11; ++i)
    {
      sum[i] += t[i + 1] - t[i];
    }
    if (lane == 0)
    {
      out[c % inputs] = out_idx[c % K];
    }
  }
  if (lane == 0)
  {
    for (int i = 0; i < 11; ++i)
    {
      acc[i] = sum[i];
    }
  }
}

// ---- block form: stage, then per level (split, exchange, merge)
template <typename KeyT, int N, int K, int M>
__global__ void probe_block(const KeyT* in, int* out, long long* acc, int calls)
{
  using core = w4::chunk_core<KeyT, 256, N, K, M>;
  using tree = typename core::tree;
  constexpr int IPT = core::IPT;
  extern __shared__ __align__(16) unsigned char smem[];
  auto& ts     = *reinterpret_cast<typename core::TempStorage*>(smem);
  int* out_idx = reinterpret_cast<int*>(smem + sizeof(typename core::TempStorage));
  const int t  = threadIdx.x;
  long long sum[12] = {};
  for (int c = 0; c < calls; ++c)
  {
    const KeyT* src = in + (c % inputs) * N;
    KeyT keys[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      keys[i] = src[t * IPT + i];
    }
    __syncthreads();
    long long st[12];
    st[0] = clock64();
    core::stage(keys, t, ts);
    __syncthreads();
    st[1] = clock64();
    int s = 2;
    auto level = [&](auto lv_tag) {
      using lv        = decltype(lv_tag);
      const int a_end = lv::split_end(t, ts);
      ts.xs[t]        = a_end;
      __syncthreads();
      st[s++]         = clock64();
      const int a_beg = (t % lv::TPP == 0) ? 0 : ts.xs[t - 1];
      lv::merge(t, ts, out_idx, a_beg, a_end);
      __syncthreads();
      st[s++] = clock64();
    };
    level(typename tree::template level<0>{});
    level(typename tree::template level<1>{});
    if constexpr (tree::levels > 2)
    {
      level(typename tree::template level<2>{});
    }
    for (int i = 0; i < 1 + 2 * tree::levels; ++i)
    {
      sum[i] += st[i + 1] - st[i];
    }
    if (t == 0)
    {
      out[c % inputs] = out_idx[c % K];
    }
  }
  if (t == 0)
  {
    for (int i = 0; i < 1 + 2 * tree::levels; ++i)
    {
      acc[i] = sum[i];
    }
  }
}

template <typename KeyT>
std::vector<KeyT> make_inputs(int n, int runs)
{
  std::mt19937_64 rng(42);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<KeyT> v(static_cast<size_t>(inputs) * n);
  for (int i = 0; i < inputs; ++i)
  {
    std::vector<float> f(n);
    for (auto& x : f)
    {
      x = dist(rng);
    }
    for (int j = 0; j < n; ++j)
    {
      v[static_cast<size_t>(i) * n + j] = from_float<KeyT>(f[j]);
    }
    if (runs > 1)
    {
      // sort every run descending (the key image, ties by index through stable sort)
      const int L = n / runs;
      for (int r = 0; r < runs; ++r)
      {
        auto b = v.begin() + static_cast<size_t>(i) * n + r * L;
        std::stable_sort(b, b + L, [](KeyT a, KeyT c) {
          float x, y;
          if constexpr (sizeof(KeyT) == 4)
          {
            x = a;
            y = c;
          }
          else
          {
            x = __bfloat162float(a);
            y = __bfloat162float(c);
          }
          return x > y;
        });
      }
    }
  }
  return v;
}

template <typename KeyT, int N, int K>
void run_warp(const char* name)
{
  auto h = make_inputs<KeyT>(N, 1);
  KeyT* d_in;
  int* d_out;
  long long* d_acc;
  cudaMalloc(&d_in, h.size() * sizeof(KeyT));
  cudaMalloc(&d_out, inputs * sizeof(int));
  cudaMalloc(&d_acc, 12 * sizeof(long long));
  cudaMemcpy(d_in, h.data(), h.size() * sizeof(KeyT), cudaMemcpyHostToDevice);
  const int calls = 256;
  probe_warp<KeyT, N, K><<<1, 32>>>(d_in, d_out, d_acc, calls);
  probe_warp<KeyT, N, K><<<1, 32>>>(d_in, d_out, d_acc, calls);
  cudaError_t e = cudaDeviceSynchronize();
  long long acc[12] = {};
  cudaMemcpy(acc, d_acc, 11 * sizeof(long long), cudaMemcpyDeviceToHost);
  printf("%s N %d K %d: %s\n  prepare %lld", name, N, K, cudaGetErrorString(e), acc[0] / calls);
  long long tot = acc[0];
  for (int l = 0; l < 5; ++l)
  {
    printf(" | L%d split %lld merge %lld", l, acc[1 + 2 * l] / calls, acc[2 + 2 * l] / calls);
    tot += acc[1 + 2 * l] + acc[2 + 2 * l];
  }
  printf(" | total %lld\n", tot / calls);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_acc);
}

template <typename KeyT, int N, int K, int M>
void run_block(const char* name)
{
  using core = w4::chunk_core<KeyT, 256, N, K, M>;
  auto h     = make_inputs<KeyT>(N, M);
  KeyT* d_in;
  int* d_out;
  long long* d_acc;
  cudaMalloc(&d_in, h.size() * sizeof(KeyT));
  cudaMalloc(&d_out, inputs * sizeof(int));
  cudaMalloc(&d_acc, 12 * sizeof(long long));
  cudaMemcpy(d_in, h.data(), h.size() * sizeof(KeyT), cudaMemcpyHostToDevice);
  const int calls  = 256;
  const size_t dyn = sizeof(typename core::TempStorage) + K * sizeof(int);
  auto kern        = probe_block<KeyT, N, K, M>;
  cudaFuncSetAttribute(reinterpret_cast<const void*>(kern), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dyn));
  kern<<<1, 256, dyn>>>(d_in, d_out, d_acc, calls);
  kern<<<1, 256, dyn>>>(d_in, d_out, d_acc, calls);
  cudaError_t e = cudaDeviceSynchronize();
  long long acc[12] = {};
  cudaMemcpy(acc, d_acc, 12 * sizeof(long long), cudaMemcpyDeviceToHost);
  constexpr int levels = core::tree::levels;
  printf("%s N %d K %d M %d: %s\n  stage %lld", name, N, K, M, cudaGetErrorString(e), acc[0] / calls);
  long long tot = acc[0];
  for (int l = 0; l < levels; ++l)
  {
    printf(" | L%d split+xchg %lld merge %lld", l, acc[1 + 2 * l] / calls, acc[2 + 2 * l] / calls);
    tot += acc[1 + 2 * l] + acc[2 + 2 * l];
  }
  printf(" | total %lld\n", tot / calls);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_acc);
}

} // namespace

int main()
{
  run_warp<__nv_bfloat16, 512, 256>("warp bf16");
  run_warp<float, 512, 256>("warp fp32");
  run_warp<__nv_bfloat16, 2048, 1024>("warp bf16");
  run_warp<float, 2048, 1024>("warp fp32");
  run_warp<__nv_bfloat16, 2048, 64>("warp bf16");
  run_warp<float, 2048, 64>("warp fp32");
  run_block<__nv_bfloat16, 8192, 2048, 4>("block bf16");
  run_block<float, 8192, 2048, 4>("block fp32");
  run_block<float, 8192, 1024, 8>("block fp32");
  return 0;
}
