// Phase profiler for W5 v2 (not an adapter, not a harness measurement): one warp runs a chain of
// calls on the harness's 16 random inputs (striped, fresh input per call, the load address depends
// on the previous output as in the harness) and records a clock stamp after every phase through a
// dependent move (in-order issue: the stamp issues when the phase result is ready). Prints the mean
// cycles between stamps per call. Build on the node:
//   nvcc -std=c++17 -O3 -arch=sm_100 -I<cub> -I<libcudacxx> -I<thrust> -I<harness> -o phase_probe phase_probe.cu
#include "topk_harness.cuh"
#include "adapters/W5_impl_v2.cuh"

#include <cstdio>
#include <vector>

#define STAMP(i, v)                                 \
  do                                                \
  {                                                 \
    uint32_t probe_v_ = (v);                        \
    asm volatile("mov.b32 %0, %0;" : "+r"(probe_v_)); \
    asm volatile("" ::"r"(probe_v_));               \
    st[i] = clock64();                              \
  } while (0)

constexpr int NSTAMP = 10;

template <typename KeyT, int N, int K>
__device__ __forceinline__ void probed_run(
  const KeyT (&keys)[N / 32], typename w5v2::primitive<KeyT, N, K>::TempStorage& ts, int* out_idx, long long (&st)[NSTAMP])
{
  using P = w5v2::primitive<KeyT, N, K>;
  constexpr int IPT = P::IPT, J = P::J, S = P::S, D = P::D, CAP = P::CAP, CNT_BITS = P::CNT_BITS, CH = P::CH, NCH = P::NCH;
  constexpr bool F32 = P::F32;
  const int lane    = static_cast<int>(threadIdx.x) & 31;
  const uint32_t lt = w5::lanemask_lt();
  const w2::masked_reducer<32> red{0xffffffffu};
  float a[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    a[i] = P::raw_float(keys[i]);
  }
  uint32_t fold = 0;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    fold ^= __float_as_uint(a[i]);
  }
  STAMP(1, fold); // loads done
  float cert[J];
  w5v2::lane_top<J, IPT>(a, cert);
  STAMP(2, __float_as_uint(cert[J - 1])); // per-lane top J
  w5v2::group_merge<J, S>(cert);
  const uint32_t t_tw = red.min(w2::twiddle_f32(cert[J - 1]));
  const float t       = __uint_as_float(w5v2::untwiddle_f32(t_tw));
  STAMP(3, __float_as_uint(t)); // threshold
  uint32_t cnt[NCH];
#pragma unroll
  for (int q = 0; q < NCH; ++q)
  {
    uint32_t f[CH];
#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
      f[i] = (a[q * CH + i] >= t) ? 1u : 0u;
    }
#pragma unroll
    for (int s = CH / 2; s >= 1; s /= 2)
    {
#pragma unroll
      for (int i = 0; i < s; ++i)
      {
        f[i] += f[i + s];
      }
    }
    cnt[q] = f[0];
  }
  uint32_t c = 0;
#pragma unroll
  for (int q = 0; q < NCH; ++q)
  {
    c += cnt[q];
  }
  STAMP(4, c); // count
  uint32_t pre = 0, tot = 0;
#pragma unroll
  for (int b = 0; b < CNT_BITS; ++b)
  {
    const uint32_t bal = __ballot_sync(0xffffffffu, (c >> b) & 1u);
    pre += static_cast<uint32_t>(__popc(bal & lt)) << b;
    tot += static_cast<uint32_t>(__popc(bal)) << b;
  }
  STAMP(5, pre + tot); // prefix
  if (tot > static_cast<uint32_t>(CAP))
  {
    out_idx[0] = -1;
    return;
  }
  uint32_t base = pre;
  uint32_t last = 0;
#pragma unroll
  for (int q = 0; q < NCH; ++q)
  {
    uint32_t pos = base;
#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
      const int item = q * CH + i;
      if (a[item] >= t)
      {
        P::store_slot(ts.buf, pos, a[item], item * 32 + lane);
        ++pos;
      }
    }
    base += cnt[q];
    last ^= pos;
  }
  STAMP(6, last); // scatter issued
  __syncwarp();
  uint32_t list[D];
#pragma unroll
  for (int j = 0; j < D; ++j)
  {
    const uint32_t p = static_cast<uint32_t>(lane + 32 * j);
    list[j]          = (p < tot) ? P::load_word(ts.buf, p) : 0u;
  }
  uint32_t g = 0;
#pragma unroll
  for (int j = 0; j < D; ++j)
  {
    g ^= list[j];
  }
  STAMP(7, g); // gather
  w2::oe_sort_desc<0, D - 1>(list);
  STAMP(8, list[0]); // sort
  uint32_t m_last;
  if constexpr (F32)
  {
    m_last = w2::extract_rounds<32, K, D>(list, lane, red, typename P::emit_f32{ts.buf, out_idx});
  }
  else
  {
    m_last = w2::extract_rounds<32, K, D>(list, lane, red, w2::emit_index<P::IdxBits>{out_idx});
  }
  STAMP(9, m_last); // rounds
}

template <typename KeyT, int N, int K>
__global__ void probe_kernel(const typename topk::key_traits<KeyT>::bits_t* __restrict__ g_in, int calls, int zero_mask, long long* out, int* g_dep)
{
  using P           = w5v2::primitive<KeyT, N, K>;
  constexpr int IPT = N / 32;
  __shared__ typename P::TempStorage ts;
  __shared__ int out_idx[2][K];
  const int lane = threadIdx.x & 31;
  long long acc[NSTAMP] = {};
  long long st[NSTAMP];
  int dep = 0;
  for (int n = 0; n < calls; ++n)
  {
    const int which = (n + (dep & zero_mask)) & 15;
    KeyT keys[IPT];
    __syncwarp();
    st[0] = clock64();
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      keys[i] = topk::key_traits<KeyT>::from_bits(g_in[static_cast<size_t>(which) * N + i * 32 + lane]);
    }
    probed_run<KeyT, N, K>(keys, ts, out_idx[n & 1], st);
    __syncwarp();
    dep = out_idx[n & 1][0];
    if (n >= 8)
    {
      for (int i = 1; i < NSTAMP; ++i)
      {
        acc[i] += st[i] - st[i - 1];
      }
    }
  }
  if (lane == 0)
  {
    for (int i = 1; i < NSTAMP; ++i)
    {
      out[i] = acc[i];
    }
    g_dep[0] = dep;
  }
}

template <typename KeyT, int N, int K>
void run_cell(const char* name)
{
  using traits = topk::key_traits<KeyT>;
  std::vector<typename traits::bits_t> h(16 * N);
  for (int i = 0; i < 16; ++i)
  {
    auto in = topk::make_input<KeyT>("random", N, K, 1000u + i);
    std::copy(in.bits.begin(), in.bits.end(), h.begin() + static_cast<size_t>(i) * N);
  }
  typename traits::bits_t* d_in;
  long long* d_out;
  int* d_dep;
  cudaMalloc(&d_in, h.size() * sizeof(h[0]));
  cudaMalloc(&d_out, NSTAMP * sizeof(long long));
  cudaMalloc(&d_dep, sizeof(int));
  cudaMemcpy(d_in, h.data(), h.size() * sizeof(h[0]), cudaMemcpyHostToDevice);
  const int calls = 8 + 256;
  probe_kernel<KeyT, N, K><<<1, 32>>>(d_in, calls, 0, d_out, d_dep);
  probe_kernel<KeyT, N, K><<<1, 32>>>(d_in, calls, 0, d_out, d_dep);
  cudaDeviceSynchronize();
  long long out[NSTAMP];
  cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
  const char* ph[] = {"", "loads", "lanetop", "thresh", "count", "prefix", "scatter", "gather", "sort", "rounds"};
  long long total = 0;
  printf("%-5s N %5d k %4d |", name, N, K);
  for (int i = 1; i < NSTAMP; ++i)
  {
    printf(" %s %5.0f", ph[i], static_cast<double>(out[i]) / 256.0);
    total += out[i];
  }
  printf(" | total %6.0f\n", static_cast<double>(total) / 256.0);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_dep);
}

int main()
{
  run_cell<__nv_bfloat16, 512, 8>("bf16");
  run_cell<__nv_bfloat16, 512, 32>("bf16");
  run_cell<__nv_bfloat16, 1024, 8>("bf16");
  run_cell<__nv_bfloat16, 1024, 32>("bf16");
  run_cell<__nv_bfloat16, 1024, 64>("bf16");
  run_cell<__nv_bfloat16, 2048, 8>("bf16");
  run_cell<__nv_bfloat16, 2048, 32>("bf16");
  run_cell<float, 1024, 32>("fp32");
  run_cell<float, 2048, 16>("fp32");
  return 0;
}
