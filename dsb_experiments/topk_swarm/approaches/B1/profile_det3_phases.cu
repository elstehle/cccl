// Phase profile of B1_det3 against B1_det2 on one block (T56 method): clock64 stamps per phase written by
// thread 0, random normal input rounded to the key type, striped tile. Four forms on the same rows: B1_det2
// (per-slot ballots and ranks), B1_det3 with per-slot ballots and the per-candidate lookup, B1_det3 with
// warp-sparse ballots and the lookup (the adapter's form), and the not_guaranteed path of the B1_det2 header
// (B1_idx's code). Columns: prologue, passes, tieballot (ballots, table stores and the barrier), tiescan
// (base scan and base table, det3 only, warp 0's view), tielook (the per-candidate lookup for det3, the base
// shuffles and per-slot ranks for det2), epilogue (behind a barrier). For the warp-sparse form the tie
// columns are warp 0's timeline and the barrier before the epilogue absorbs the other warps, so compare the
// totals and the sum of the tie and epilogue columns.
// Build: nvcc -std=c++17 -O3 -arch=sm_100 -DB1_PROFILE -I<cccl>/cub -I<cccl>/libcudacxx/include
// -I<cccl>/thrust -Xptxas -v profile_det3_phases.cu
#include "block_topk_b1_det2.cuh"
#include "block_topk_b1_det3.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

enum form
{
  det2     = 0,
  det3slot = 1,
  det3     = 2,
  idx      = 3
};

template <typename KeyT, int T, int N, int K, int Form>
struct pick;

template <typename KeyT, int T, int N, int K>
struct pick<KeyT, T, N, K, det2>
{
  using type = topk_b1_det2::block_topk_b1<KeyT, T, N / T, 8, true, topk_b1_det2::tie_break::smaller_index,
                                           topk_b1_det2::ticket_mode::per_item>;
  static __device__ long long* stamps()
  {
    return topk_b1_det2::g_stamps;
  }
};
template <typename KeyT, int T, int N, int K>
struct pick<KeyT, T, N, K, idx>
{
  using type = topk_b1_det2::block_topk_b1<KeyT, T, N / T, 8, false, topk_b1_det2::tie_break::none,
                                           topk_b1_det2::ticket_mode::per_item>;
  static __device__ long long* stamps()
  {
    return topk_b1_det2::g_stamps;
  }
};
template <typename KeyT, int T, int N, int K>
struct pick<KeyT, T, N, K, det3slot>
{
  using type = topk_b1_det3::block_topk_b1<KeyT, T, N / T, 8, true, topk_b1_det3::tie_break::smaller_index,
                                           topk_b1_det3::ticket_mode::per_item, topk_b1_det3::tie_ballots::per_slot>;
  static __device__ long long* stamps()
  {
    return topk_b1_det3::g_stamps;
  }
};
template <typename KeyT, int T, int N, int K>
struct pick<KeyT, T, N, K, det3>
{
  using type = topk_b1_det3::block_topk_b1<KeyT, T, N / T, 8, true, topk_b1_det3::tie_break::smaller_index,
                                           topk_b1_det3::ticket_mode::per_item, topk_b1_det3::tie_ballots::warp_sparse>;
  static __device__ long long* stamps()
  {
    return topk_b1_det3::g_stamps;
  }
};

template <typename KeyT, int T, int N, int K, int Form>
__global__ void __launch_bounds__(T) prof_kernel(const KeyT* in, int* out, long long* stamps, int reps)
{
  constexpr int IPT = N / T;
  using P           = pick<KeyT, T, N, K, Form>;
  using topk_t      = typename P::type;
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
      long long* g = P::stamps();
      for (int i = 0; i < 16; ++i)
      {
        stamps[rep * 16 + i] = g[i];
        g[i]                 = 0;
      }
      // Order independent digest of the result set (ticket order differs between the forms)
      long long sum = 0, sq = 0;
      for (int i = 0; i < K; ++i)
      {
        sum += out_idx[i];
        sq += (long long) out_idx[i] * out_idx[i];
      }
      out[rep * 2]     = (int) sum;
      out[rep * 2 + 1] = (int) (sq % 1000000007LL);
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

// phases: 0 start, 1 prologue, 2..5 radix passes, 6 tie ballots and table barrier, 10 tie base scan and base
// table (det3), 7 tie lookup (det3) or base shuffles and ranks (det2), 8 epilogue (behind a barrier), 9 flag:
// 1 when the tie phase ran
static const int ids[]      = {0, 1, 2, 3, 4, 5, 6, 10, 7, 8};
static const char* names[]  = {"start", "prologue", "pass1", "pass2", "pass3", "pass4", "tieballot", "tiescan", "tielook", "epilogue"};
static const int num_phases = 10;

static void report(const char* label, const std::vector<long long>& st, int reps, int want_flag)
{
  printf("  %-14s", label);
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
      printf(" %7s %-4s", "-", "");
      continue;
    }
    std::sort(d.begin(), d.end());
    printf(" %7lld n%-3d", d[d.size() / 2], (int) d.size());
  }
  // total: start to end stamp. post: last pass stamp to end stamp (tie phase plus epilogue, the end stamp
  // sits behind a barrier, so this is the block's view and not warp 0's). Both are medians over the rows.
  std::vector<long long> tot, post;
  for (int r = 0; r < reps; ++r)
  {
    if (want_flag >= 0 && st[r * 16 + 9] != want_flag)
    {
      continue;
    }
    tot.push_back(st[r * 16 + 8] - st[r * 16 + 0]);
    long long last_pass = 0;
    for (int q = 5; q >= 2; --q)
    {
      if (st[r * 16 + q] != 0)
      {
        last_pass = st[r * 16 + q];
        break;
      }
    }
    post.push_back(st[r * 16 + 8] - last_pass);
  }
  if (!tot.empty())
  {
    std::sort(tot.begin(), tot.end());
    std::sort(post.begin(), post.end());
    printf(" | total %7lld post %6lld", tot[tot.size() / 2], post[post.size() / 2]);
  }
  printf("\n");
}

template <typename KeyT, int T, int N, int K, int Form>
void run_form(const char* label, const KeyT* d_in, int* d_out, long long* d_st, int reps, std::vector<int>& result)
{
  std::vector<long long> st(reps * 16);
  prof_kernel<KeyT, T, N, K, Form><<<1, T>>>(d_in, d_out, d_st, reps);
  prof_kernel<KeyT, T, N, K, Form><<<1, T>>>(d_in, d_out, d_st, reps);
  cudaDeviceSynchronize();
  cudaMemcpy(st.data(), d_st, st.size() * sizeof(long long), cudaMemcpyDeviceToHost);
  result.resize(reps * 2);
  cudaMemcpy(result.data(), d_out, reps * 2 * sizeof(int), cudaMemcpyDeviceToHost);
  char buf[64];
  snprintf(buf, sizeof(buf), "%s all", label);
  report(buf, st, reps, -1);
  snprintf(buf, sizeof(buf), "%s tie ran", label);
  report(buf, st, reps, 1);
  snprintf(buf, sizeof(buf), "%s no tie", label);
  report(buf, st, reps, 0);
}

template <typename KeyT, int T, int N, int K>
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
  cudaMalloc(&d_out, reps * 2 * sizeof(int));
  cudaMalloc(&d_st, reps * 16 * sizeof(long long));
  cudaMemcpy(d_in, h.data(), h.size() * sizeof(KeyT), cudaMemcpyHostToDevice);
  printf("%s: median cycles per phase over %d calls (one block, %d threads), columns", name, reps, T);
  for (int p = 1; p < num_phases; ++p)
  {
    printf(" %s", names[p]);
  }
  printf("\n");
  std::vector<int> r_det2, r_det3slot, r_det3, r_idx;
  run_form<KeyT, T, N, K, det2>("det2", d_in, d_out, d_st, reps, r_det2);
  run_form<KeyT, T, N, K, det3slot>("det3 slot", d_in, d_out, d_st, reps, r_det3slot);
  run_form<KeyT, T, N, K, det3>("det3 sparse", d_in, d_out, d_st, reps, r_det3);
  run_form<KeyT, T, N, K, idx>("idx path", d_in, d_out, d_st, reps, r_idx);
  // The three deterministic forms must produce the same index set (unique under prefer_smaller_index),
  // compared by an order independent digest. A weak check, the harness oracle is the real one.
  int mism_slot = 0, mism_sparse = 0, differs_idx = 0;
  for (int r = 0; r < reps; ++r)
  {
    mism_slot += (r_det2[2 * r] != r_det3slot[2 * r] || r_det2[2 * r + 1] != r_det3slot[2 * r + 1]) ? 1 : 0;
    mism_sparse += (r_det2[2 * r] != r_det3[2 * r] || r_det2[2 * r + 1] != r_det3[2 * r + 1]) ? 1 : 0;
    differs_idx += (r_det2[2 * r] != r_idx[2 * r] || r_det2[2 * r + 1] != r_idx[2 * r + 1]) ? 1 : 0;
  }
  printf("  result digests: det3 slot differs from det2 in %d rows, det3 sparse in %d rows, idx path (ticket order ties) in %d rows\n",
         mism_slot, mism_sparse, differs_idx);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_st);
}

int main()
{
  run<__nv_bfloat16, 256, 1024, 16>("bf16 N1024 k16", 32);
  run<__nv_bfloat16, 256, 1024, 256>("bf16 N1024 k256", 32);
  run<__nv_bfloat16, 256, 4096, 64>("bf16 N4096 k64", 32);
  run<__nv_bfloat16, 256, 4096, 256>("bf16 N4096 k256", 32);
  run<__nv_bfloat16, 256, 16384, 256>("bf16 N16384 k256", 32);
  run<float, 256, 1024, 16>("fp32 N1024 k16", 32);
  run<float, 256, 16384, 16>("fp32 N16384 k16", 32);
  run<__nv_bfloat16, 512, 32768, 64>("bf16 N32768 k64 T512", 32);
  const cudaError_t e = cudaDeviceSynchronize();
  printf("status %s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
