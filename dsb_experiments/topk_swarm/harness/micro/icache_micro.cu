// Instruction fetch cost of straight-line code against code size, warps per scheduler and node.
//
// Motivation: verify-B2 measured the same 128-thread, 5000 to 6500 instruction kernel at +25 to +33
// percent cycles on one B200 node against another while 256-thread kernels of the same size agreed.
// This file isolates the mechanism: a fully unrolled chain of LEN dependent IMADs (4 cycles each when
// fetch keeps up) or LEN independent IMADs in 4 interleaved chains (issue bound, 1 per cycle), run by
// 1, 4, 8 or 16 warps per block (0.25, 1, 2 and 4 warps per scheduler), code size 16 bytes per SASS
// instruction, so LEN 8192 is a 128 KB body. Cycles per instruction above the ideal (4.0 dependent,
// 1.0 independent) is fetch cost. A second kernel reports the SM clock in MHz from %clock64 over
// %globaltimer so that node clock differences are visible next to the cycle counts.
//
// The bodies live in icache_bodies.h as inline PTX (generated, see the swarm report), because the front end
// needs minutes for an 8192-way unrolled C++ loop and seconds for the same straight-line PTX.
// Build: nvcc -arch=sm_100 -O3 -std=c++17 -o icache_micro icache_micro.cu
// Run:   ./icache_micro --node <name> --date <yyyy-mm-dd>     (one JSON line per measurement)

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

__device__ int g_sink;
constexpr int kReps = 24;

__device__ __forceinline__ unsigned smid()
{
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}
__device__ __forceinline__ unsigned long long globaltimer()
{
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

#include "icache_bodies.h"

template <int LEN, int BLOCK, bool DEP>
__global__ void __launch_bounds__(BLOCK) body(long long* out, long long* first)
{
  unsigned sm    = smid();
  unsigned v0    = threadIdx.x * 2654435761u;
  unsigned v1    = v0 ^ 0x9e3779b9u;
  unsigned v2    = v0 + 0x7f4a7c15u;
  unsigned v3    = v0 * 3u;
  unsigned c     = threadIdx.x + 1u;
  unsigned d     = threadIdx.x * 7u + 3u;
  long long best = LLONG_MAX;
  long long cold = 0;
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
    unsigned v = body_asm<LEN, DEP>(v0, v1, v2, v3, c, d);
    v0 ^= v;
    if (sm == 0xffffffffu)
    {
      g_sink = static_cast<int>(v);
    }
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    if (r == 0)
    {
      cold = t1 - t0;
    }
    best = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    *out   = best;
    *first = cold;
  }
}

__global__ void clock_probe(double* mhz)
{
  unsigned long long g0 = globaltimer();
  long long c0          = clock64();
  while (globaltimer() - g0 < 20000000ull) // 20 ms
  {
  }
  long long c1          = clock64();
  unsigned long long g1 = globaltimer();
  *mhz                  = static_cast<double>(c1 - c0) * 1000.0 / static_cast<double>(g1 - g0);
}

struct Meta
{
  std::string node, date, arch, device, cuda, nvcc;
};
static Meta g_meta;

// B2-shaped straight-line body: ROUNDS rounds of 32 dependent IMADs, one fire-and-forget shared atomicMax
// into slot[round mod 16] (uniform address, so ptxas aggregates it, see T10), barrier, read of the slot.
template <int ROUNDS, int BLOCK, bool BRANCH = false>
__global__ void __launch_bounds__(BLOCK) body_rounds(long long* out, unsigned* smids, int reps)
{
  __shared__ unsigned slots[16];
  unsigned sm = smid();
  unsigned v  = threadIdx.x * 2654435761u + blockIdx.x;
  unsigned c  = threadIdx.x + 1u;
  for (int i = threadIdx.x; i < 16; i += BLOCK)
  {
    slots[i] = 0;
  }
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int r = 0; r < reps; ++r)
  {
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll
    for (int rd = 0; rd < ROUNDS; ++rd)
    {
      asm volatile(
        "mad.lo.u32 %0, %0, 3, %1;\n mad.lo.u32 %0, %0, 5, %1;\n mad.lo.u32 %0, %0, 7, %1;\n mad.lo.u32 %0, %0, 9, %1;\n"
        "mad.lo.u32 %0, %0, 3, %1;\n mad.lo.u32 %0, %0, 5, %1;\n mad.lo.u32 %0, %0, 7, %1;\n mad.lo.u32 %0, %0, 9, %1;\n"
        "mad.lo.u32 %0, %0, 3, %1;\n mad.lo.u32 %0, %0, 5, %1;\n mad.lo.u32 %0, %0, 7, %1;\n mad.lo.u32 %0, %0, 9, %1;\n"
        "mad.lo.u32 %0, %0, 3, %1;\n mad.lo.u32 %0, %0, 5, %1;\n mad.lo.u32 %0, %0, 7, %1;\n mad.lo.u32 %0, %0, 9, %1;\n"
        "mad.lo.u32 %0, %0, 3, %1;\n mad.lo.u32 %0, %0, 5, %1;\n mad.lo.u32 %0, %0, 7, %1;\n mad.lo.u32 %0, %0, 9, %1;\n"
        "mad.lo.u32 %0, %0, 3, %1;\n mad.lo.u32 %0, %0, 5, %1;\n mad.lo.u32 %0, %0, 7, %1;\n mad.lo.u32 %0, %0, 9, %1;\n"
        "mad.lo.u32 %0, %0, 3, %1;\n mad.lo.u32 %0, %0, 5, %1;\n mad.lo.u32 %0, %0, 7, %1;\n mad.lo.u32 %0, %0, 9, %1;\n"
        "mad.lo.u32 %0, %0, 3, %1;\n mad.lo.u32 %0, %0, 5, %1;\n mad.lo.u32 %0, %0, 7, %1;\n mad.lo.u32 %0, %0, 9, %1;\n"
        : "+r"(v)
        : "r"(c));
      if constexpr (BRANCH)
      {
        if (threadIdx.x == static_cast<unsigned>(rd & (BLOCK - 1)))
        {
          atomicMax(&slots[rd & 15], v); // one lane, divergent: BSSY, BRA, BSYNC per round
        }
      }
      else
      {
        atomicMax(&slots[rd & 15], v);
      }
      __syncthreads();
      v ^= slots[rd & 15];
    }
    if (sm == 0xffffffffu)
    {
      g_sink = static_cast<int>(v);
    }
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    best = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    out[blockIdx.x]   = best;
    smids[blockIdx.x] = sm;
  }
}

// B2-shaped round with what the real kernel has and the IMAD body lacks: a dependent list shift of
// 32 predicated selects (SEL on a data predicate) and a one-lane guarded atomic that ptxas wraps in
// BSSY, BRA, BSYNC (T32), so the 80 KB body is not straight-line any more.
template <int ROUNDS, int BLOCK, bool BRANCH>
__global__ void __launch_bounds__(BLOCK) body_rounds_sel(long long* out, unsigned* smids, int reps)
{
  __shared__ unsigned slots[16];
  unsigned sm = smid();
  unsigned list[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
  {
    list[i] = threadIdx.x * 2654435761u + static_cast<unsigned>(i) * 40503u;
  }
  for (int i = threadIdx.x; i < 16; i += BLOCK)
  {
    slots[i] = 0;
  }
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int r = 0; r < reps; ++r)
  {
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll
    for (int rd = 0; rd < ROUNDS; ++rd)
    {
      unsigned m = slots[rd & 15];
      bool win   = (list[0] & 0xffu) == static_cast<unsigned>(rd & 0x7f); // data predicate, rarely true
      // 32 selects: four passes of a shift of the 8-entry list
#pragma unroll
      for (int pass = 0; pass < 4; ++pass)
      {
#pragma unroll
        for (int i = 0; i < 7; ++i)
        {
          list[i] = win ? list[i + 1] : list[i];
        }
        list[7] = win ? (m ^ static_cast<unsigned>(pass)) : list[7];
      }
      if constexpr (BRANCH)
      {
        if (threadIdx.x == static_cast<unsigned>(rd & (BLOCK - 1)))
        {
          atomicMax(&slots[rd & 15], list[0]); // one lane, divergent: BSSY, BRA, BSYNC
        }
      }
      else
      {
        atomicMax(&slots[rd & 15], list[0]); // uniform address, all lanes: aggregated by ptxas
      }
      __syncthreads();
    }
    unsigned v = list[0] ^ list[3] ^ list[7];
    if (sm == 0xffffffffu)
    {
      g_sink = static_cast<int>(v);
    }
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    best = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    out[blockIdx.x]   = best;
    smids[blockIdx.x] = sm;
  }
}

template <int ROUNDS, int BLOCK, bool BRANCH>
void run_rounds_sel(int blocks, const char* tag)
{
  long long* d;
  unsigned* ds;
  cudaMalloc(&d, sizeof(long long) * blocks);
  cudaMalloc(&ds, sizeof(unsigned) * blocks);
  body_rounds_sel<ROUNDS, BLOCK, BRANCH><<<blocks, BLOCK>>>(d, ds, 4);
  cudaDeviceSynchronize();
  body_rounds_sel<ROUNDS, BLOCK, BRANCH><<<blocks, BLOCK>>>(d, ds, kReps);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "KERNEL ERROR: %s\n", cudaGetErrorString(err));
    return;
  }
  long long* h = new long long[blocks];
  unsigned* hs = new unsigned[blocks];
  cudaMemcpy(h, d, sizeof(long long) * blocks, cudaMemcpyDeviceToHost);
  cudaMemcpy(hs, ds, sizeof(unsigned) * blocks, cudaMemcpyDeviceToHost);
  long long mn = LLONG_MAX, mx = 0;
  for (int b = 0; b < blocks; ++b)
  {
    mn = ::min(mn, h[b]);
    mx = ::max(mx, h[b]);
  }
  printf("{\"name\":\"rounds-sel %s %s ROUNDS %d BLOCK %d blocks %d\",\"rounds\":%d,\"block\":%d,\"blocks\":%d,\"branch\":%s,"
         "\"cycles_min\":%lld,\"cycles_max\":%lld,\"per_round_min\":%.1f,\"per_round_max\":%.1f,"
         "\"arch\":\"%s\",\"device\":\"%s\",\"node\":\"%s\",\"date\":\"%s\",\"file\":\"harness/micro/icache_micro.cu\",\"per_block\":[",
         BRANCH ? "branchy" : "uniform", tag, ROUNDS, BLOCK, blocks, ROUNDS, BLOCK, blocks, BRANCH ? "true" : "false",
         mn, mx, static_cast<double>(mn) / ROUNDS, static_cast<double>(mx) / ROUNDS,
         g_meta.arch.c_str(), g_meta.device.c_str(), g_meta.node.c_str(), g_meta.date.c_str());
  for (int b = 0; b < blocks; ++b)
  {
    printf("%s[%u,%lld]", b ? "," : "", hs[b], h[b]);
  }
  printf("]}\n");
  fflush(stdout);
  delete[] h;
  delete[] hs;
  cudaFree(d);
  cudaFree(ds);
}

// Round with a one-lane guarded atomic (BSSY, BRA, BSYNC), barrier, LDS, then 32 independent IMADs on
// 4 accumulators (issue bound, one per cycle): high instruction rate right after a branch, like the 32
// predicated selects of the B2 list shift. 128 rounds is about 85 KB.
template <int ROUNDS, int BLOCK, bool BRANCH>
__global__ void __launch_bounds__(BLOCK) body_rounds_ipc(long long* out, unsigned* smids, int reps)
{
  __shared__ unsigned slots[16];
  unsigned sm = smid();
  unsigned a0 = threadIdx.x * 2654435761u, a1 = a0 ^ 0x9e3779b9u, a2 = a0 + 0x7f4a7c15u, a3 = a0 * 3u;
  unsigned c  = threadIdx.x + 1u;
  for (int i = threadIdx.x; i < 16; i += BLOCK)
  {
    slots[i] = 0;
  }
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int r = 0; r < reps; ++r)
  {
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll
    for (int rd = 0; rd < ROUNDS; ++rd)
    {
      if constexpr (BRANCH)
      {
        if (threadIdx.x == static_cast<unsigned>(rd & (BLOCK - 1)))
        {
          atomicMax(&slots[rd & 15], a0);
        }
      }
      else
      {
        atomicMax(&slots[rd & 15], a0);
      }
      __syncthreads();
      unsigned m = slots[rd & 15];
      asm volatile(
        "mad.lo.u32 %0, %0, 3, %4;\n mad.lo.u32 %1, %1, 5, %4;\n mad.lo.u32 %2, %2, 7, %4;\n mad.lo.u32 %3, %3, 9, %4;\n"
        "mad.lo.u32 %0, %0, 3, %5;\n mad.lo.u32 %1, %1, 5, %5;\n mad.lo.u32 %2, %2, 7, %5;\n mad.lo.u32 %3, %3, 9, %5;\n"
        "mad.lo.u32 %0, %0, 3, %4;\n mad.lo.u32 %1, %1, 5, %4;\n mad.lo.u32 %2, %2, 7, %4;\n mad.lo.u32 %3, %3, 9, %4;\n"
        "mad.lo.u32 %0, %0, 3, %5;\n mad.lo.u32 %1, %1, 5, %5;\n mad.lo.u32 %2, %2, 7, %5;\n mad.lo.u32 %3, %3, 9, %5;\n"
        "mad.lo.u32 %0, %0, 3, %4;\n mad.lo.u32 %1, %1, 5, %4;\n mad.lo.u32 %2, %2, 7, %4;\n mad.lo.u32 %3, %3, 9, %4;\n"
        "mad.lo.u32 %0, %0, 3, %5;\n mad.lo.u32 %1, %1, 5, %5;\n mad.lo.u32 %2, %2, 7, %5;\n mad.lo.u32 %3, %3, 9, %5;\n"
        "mad.lo.u32 %0, %0, 3, %4;\n mad.lo.u32 %1, %1, 5, %4;\n mad.lo.u32 %2, %2, 7, %4;\n mad.lo.u32 %3, %3, 9, %4;\n"
        "mad.lo.u32 %0, %0, 3, %5;\n mad.lo.u32 %1, %1, 5, %5;\n mad.lo.u32 %2, %2, 7, %5;\n mad.lo.u32 %3, %3, 9, %5;\n"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "r"(c), "r"(m));
    }
    unsigned v = a0 ^ a1 ^ a2 ^ a3;
    if (sm == 0xffffffffu)
    {
      g_sink = static_cast<int>(v);
    }
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    best = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    out[blockIdx.x]   = best;
    smids[blockIdx.x] = sm;
  }
}

template <int ROUNDS, int BLOCK, bool BRANCH>
void run_rounds_ipc(int blocks, const char* tag)
{
  long long* d;
  unsigned* ds;
  cudaMalloc(&d, sizeof(long long) * blocks);
  cudaMalloc(&ds, sizeof(unsigned) * blocks);
  body_rounds_ipc<ROUNDS, BLOCK, BRANCH><<<blocks, BLOCK>>>(d, ds, 4);
  cudaDeviceSynchronize();
  body_rounds_ipc<ROUNDS, BLOCK, BRANCH><<<blocks, BLOCK>>>(d, ds, kReps);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "KERNEL ERROR: %s\n", cudaGetErrorString(err));
    return;
  }
  long long* h = new long long[blocks];
  unsigned* hs = new unsigned[blocks];
  cudaMemcpy(h, d, sizeof(long long) * blocks, cudaMemcpyDeviceToHost);
  cudaMemcpy(hs, ds, sizeof(unsigned) * blocks, cudaMemcpyDeviceToHost);
  long long mn = LLONG_MAX, mx = 0;
  for (int b = 0; b < blocks; ++b)
  {
    mn = ::min(mn, h[b]);
    mx = ::max(mx, h[b]);
  }
  printf("{\"name\":\"rounds-ipc %s %s ROUNDS %d BLOCK %d blocks %d\",\"rounds\":%d,\"block\":%d,\"blocks\":%d,\"branch\":%s,"
         "\"cycles_min\":%lld,\"cycles_max\":%lld,\"per_round_min\":%.1f,\"per_round_max\":%.1f,"
         "\"arch\":\"%s\",\"device\":\"%s\",\"node\":\"%s\",\"date\":\"%s\",\"file\":\"harness/micro/icache_micro.cu\",\"per_block\":[",
         BRANCH ? "branchy" : "uniform", tag, ROUNDS, BLOCK, blocks, ROUNDS, BLOCK, blocks, BRANCH ? "true" : "false",
         mn, mx, static_cast<double>(mn) / ROUNDS, static_cast<double>(mx) / ROUNDS,
         g_meta.arch.c_str(), g_meta.device.c_str(), g_meta.node.c_str(), g_meta.date.c_str());
  for (int b = 0; b < blocks; ++b)
  {
    printf("%s[%u,%lld]", b ? "," : "", hs[b], h[b]);
  }
  printf("]}\n");
  fflush(stdout);
  delete[] h;
  delete[] hs;
  cudaFree(d);
  cudaFree(ds);
}

template <int ROUNDS, int BLOCK, bool BRANCH = false>
void run_rounds(int blocks, const char* tag)
{
  long long* d;
  unsigned* ds;
  cudaMalloc(&d, sizeof(long long) * blocks);
  cudaMalloc(&ds, sizeof(unsigned) * blocks);
  body_rounds<ROUNDS, BLOCK, BRANCH><<<blocks, BLOCK>>>(d, ds, 4); // warm up
  cudaDeviceSynchronize();
  body_rounds<ROUNDS, BLOCK, BRANCH><<<blocks, BLOCK>>>(d, ds, kReps);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "KERNEL ERROR: %s\n", cudaGetErrorString(err));
    return;
  }
  long long* h = new long long[blocks];
  unsigned* hs = new unsigned[blocks];
  cudaMemcpy(h, d, sizeof(long long) * blocks, cudaMemcpyDeviceToHost);
  cudaMemcpy(hs, ds, sizeof(unsigned) * blocks, cudaMemcpyDeviceToHost);
  long long mn = LLONG_MAX, mx = 0;
  double sum   = 0;
  for (int b = 0; b < blocks; ++b)
  {
    mn = ::min(mn, h[b]);
    mx = ::max(mx, h[b]);
    sum += static_cast<double>(h[b]);
  }
  printf("{\"name\":\"rounds %s%s ROUNDS %d BLOCK %d blocks %d\",\"rounds\":%d,\"block\":%d,\"blocks\":%d,"
         "\"cycles_min\":%lld,\"cycles_max\":%lld,\"cycles_mean\":%.0f,\"per_round_min\":%.1f,\"per_round_max\":%.1f,"
         "\"arch\":\"%s\",\"device\":\"%s\",\"cuda\":\"%s\",\"nvcc\":\"%s\",\"node\":\"%s\",\"date\":\"%s\","
         "\"file\":\"harness/micro/icache_micro.cu\",\"per_block\":[",
         BRANCH ? "branchy-imad " : "", tag, ROUNDS, BLOCK, blocks, ROUNDS, BLOCK, blocks, mn, mx, sum / blocks,
         static_cast<double>(mn) / ROUNDS, static_cast<double>(mx) / ROUNDS,
         g_meta.arch.c_str(), g_meta.device.c_str(), g_meta.cuda.c_str(), g_meta.nvcc.c_str(),
         g_meta.node.c_str(), g_meta.date.c_str());
  for (int b = 0; b < blocks; ++b)
  {
    printf("%s[%u,%lld]", b ? "," : "", hs[b], h[b]);
  }
  printf("]}\n");
  fflush(stdout);
  delete[] h;
  delete[] hs;
  cudaFree(d);
  cudaFree(ds);
}

// Per-SM memory latency map: one thread per block chases a random permutation of 128-byte lines through
// a buffer that fits L2 (4 MB) or exceeds it (512 MB), HOPS dependent loads, reports cycles per hop and
// %smid. Launched with 1 block (the SM the latency harness lands on) and with one block per SM.
__global__ void l2_chase(const unsigned* __restrict__ buf, unsigned start_stride, unsigned words, int hops, long long* out, unsigned* smids)
{
  unsigned sm  = smid();
  unsigned idx = buf[(static_cast<unsigned long long>(blockIdx.x) * start_stride) % words]; // first hop outside the timing
  idx          = buf[idx]; // first hop outside the timing (TLB and first line)
  long long t0 = clock64();
  for (int h = 0; h < hops; ++h)
  {
    idx = buf[idx];
  }
  long long t1 = clock64();
  if (idx == 0xffffffffu)
  {
    g_sink = 1;
  }
  out[blockIdx.x]   = t1 - t0;
  smids[blockIdx.x] = sm;
}

// Load-path chases over an L1-resident 16 KB buffer, one thread per block: the same permutation walked
// with ld.global.nc (LDG.E.CONSTANT, the read-only path my other chases use), plain ld.global (LDG.E),
// generic ld (LD.E), and a local-memory chase (a 1 KB per-thread array indexed dynamically, LDL, the path
// register spills take). Per SM and single block.
__global__ void path_chase(const unsigned* buf, unsigned words, int hops, int mode, long long* out, unsigned* smids)
{
  unsigned sm  = smid();
  unsigned idx = buf[(static_cast<unsigned long long>(blockIdx.x) * 32u) % words];
  unsigned local_arr[256];
  if (mode == 3)
  {
#pragma unroll 1
    for (int i = 0; i < 256; ++i)
    {
      local_arr[i] = (static_cast<unsigned>(i) * 37u + 11u) & 255u; // full cycle since 37 is odd
    }
    idx = idx & 255u;
  }
  long long t0 = clock64();
  if (mode == 0)
  {
    for (int h = 0; h < hops; ++h)
    {
      unsigned v;
      asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(buf + idx));
      idx = v;
    }
  }
  else if (mode == 1)
  {
    for (int h = 0; h < hops; ++h)
    {
      unsigned v;
      asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(buf + idx));
      idx = v;
    }
  }
  else if (mode == 2)
  {
    for (int h = 0; h < hops; ++h)
    {
      unsigned v;
      asm volatile("ld.u32 %0, [%1];" : "=r"(v) : "l"(buf + idx));
      idx = v;
    }
  }
  else
  {
    for (int h = 0; h < hops; ++h)
    {
      idx = local_arr[idx];
    }
  }
  long long t1 = clock64();
  if (idx == 0xffffffffu)
  {
    g_sink = 1;
  }
  out[blockIdx.x]   = t1 - t0;
  smids[blockIdx.x] = sm;
}

static void run_path_chase(unsigned* d, unsigned words, int blocks, int hops)
{
  const char* names[4] = {"ld.global.nc 16KB", "ld.global 16KB", "ld generic 16KB", "local 1KB LDL"};
  long long* o;
  unsigned* sm;
  cudaMalloc(&o, sizeof(long long) * blocks);
  cudaMalloc(&sm, sizeof(unsigned) * blocks);
  long long* h = new long long[blocks];
  unsigned* hs = new unsigned[blocks];
  for (int mode = 0; mode < 4; ++mode)
  {
    path_chase<<<blocks, 1>>>(d, words, hops, mode, o, sm);
    cudaDeviceSynchronize();
    path_chase<<<blocks, 1>>>(d, words, hops, mode, o, sm);
    cudaDeviceSynchronize();
    cudaMemcpy(h, o, sizeof(long long) * blocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(hs, sm, sizeof(unsigned) * blocks, cudaMemcpyDeviceToHost);
    long long mn = LLONG_MAX, mx = 0;
    for (int b = 0; b < blocks; ++b)
    {
      mn = ::min(mn, h[b]);
      mx = ::max(mx, h[b]);
    }
    printf("{\"name\":\"path %s blocks %d\",\"mode\":%d,\"blocks\":%d,\"hops\":%d,\"per_hop_min\":%.1f,\"per_hop_max\":%.1f,"
           "\"arch\":\"%s\",\"device\":\"%s\",\"node\":\"%s\",\"date\":\"%s\",\"file\":\"harness/micro/icache_micro.cu\",\"per_block\":[",
           names[mode], blocks, mode, blocks, hops, static_cast<double>(mn) / hops, static_cast<double>(mx) / hops,
           g_meta.arch.c_str(), g_meta.device.c_str(), g_meta.node.c_str(), g_meta.date.c_str());
    for (int b = 0; b < blocks; ++b)
    {
      printf("%s[%u,%.1f]", b ? "," : "", hs[b], static_cast<double>(h[b]) / hops);
    }
    printf("]}\n");
  }
  fflush(stdout);
  delete[] h;
  delete[] hs;
  cudaFree(o);
  cudaFree(sm);
}

static unsigned* make_chase(size_t bytes, unsigned& stride_words)
{
  // permutation over lines of 128 bytes: line i points to line (i * 2654435761 + 7) mod n with n prime-ish
  size_t words = bytes / 4, lines = bytes / 128;
  unsigned* h = new unsigned[words];
  for (size_t i = 0; i < words; ++i)
  {
    h[i] = 0;
  }
  // build one long cycle with an LCG step over lines (stride coprime to a power-of-two line count)
  size_t cur = 0;
  for (size_t i = 0; i < lines; ++i)
  {
    size_t nxt     = (cur * 1u + 40503u * 32u + 1u) % lines; // large odd stride mod power of two: full cycle
    h[cur * 32]    = static_cast<unsigned>(nxt * 32);
    cur            = nxt;
  }
  unsigned* d = nullptr;
  cudaMalloc(&d, bytes);
  cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
  delete[] h;
  stride_words = static_cast<unsigned>(((lines / 149 > 0) ? (lines / 149) : 1) * 32);
  return d;
}

static void run_chase(size_t bytes, const char* tag, int blocks, int hops)
{
  unsigned stride = 0;
  unsigned* d     = make_chase(bytes, stride);
  long long* o;
  unsigned* sm;
  cudaMalloc(&o, sizeof(long long) * blocks);
  cudaMalloc(&sm, sizeof(unsigned) * blocks);
  l2_chase<<<blocks, 1>>>(d, stride, static_cast<unsigned>(bytes / 4), hops, o, sm); // warm up (L2 fill for the 4 MB case)
  cudaDeviceSynchronize();
  long long* h = new long long[blocks];
  unsigned* hs = new unsigned[blocks];
  long long mn = LLONG_MAX, mx = 0;
  double sum = 0;
  for (int rep = 0; rep < 3; ++rep)
  {
    l2_chase<<<blocks, 1>>>(d, stride, static_cast<unsigned>(bytes / 4), hops, o, sm);
    cudaDeviceSynchronize();
    cudaMemcpy(h, o, sizeof(long long) * blocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(hs, sm, sizeof(unsigned) * blocks, cudaMemcpyDeviceToHost);
    if (rep == 2)
    {
      for (int b = 0; b < blocks; ++b)
      {
        mn = ::min(mn, h[b]);
        mx = ::max(mx, h[b]);
        sum += static_cast<double>(h[b]);
      }
    }
  }
  printf("{\"name\":\"chase %s blocks %d\",\"bytes\":%zu,\"blocks\":%d,\"hops\":%d,\"per_hop_min\":%.1f,\"per_hop_max\":%.1f,\"per_hop_mean\":%.1f,"
         "\"arch\":\"%s\",\"device\":\"%s\",\"node\":\"%s\",\"date\":\"%s\",\"file\":\"harness/micro/icache_micro.cu\",\"per_block\":[",
         tag, blocks, bytes, blocks, hops, static_cast<double>(mn) / hops, static_cast<double>(mx) / hops, sum / blocks / hops,
         g_meta.arch.c_str(), g_meta.device.c_str(), g_meta.node.c_str(), g_meta.date.c_str());
  for (int b = 0; b < blocks; ++b)
  {
    printf("%s[%u,%.1f]", b ? "," : "", hs[b], static_cast<double>(h[b]) / hops);
  }
  printf("]}\n");
  fflush(stdout);
  delete[] h;
  delete[] hs;
  cudaFree(o);
  cudaFree(sm);
  cudaFree(d);
}

template <int LEN, int BLOCK, bool DEP>
void run()
{
  long long *d, *f;
  cudaMalloc(&d, sizeof(long long));
  cudaMalloc(&f, sizeof(long long));
  body<LEN, BLOCK, DEP><<<1, BLOCK>>>(d, f); // warm up (module load, first touch)
  cudaDeviceSynchronize();
  long long best = 0, cold = 0;
  double bests[3];
  for (int k = 0; k < 3; ++k)
  {
    body<LEN, BLOCK, DEP><<<1, BLOCK>>>(d, f);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      fprintf(stderr, "KERNEL ERROR: %s\n", cudaGetErrorString(err));
      return;
    }
    cudaMemcpy(&best, d, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cold, f, sizeof(long long), cudaMemcpyDeviceToHost);
    bests[k] = static_cast<double>(best);
  }
  cudaFree(d);
  cudaFree(f);
  double b = bests[0] < bests[1] ? bests[0] : bests[1];
  b        = b < bests[2] ? b : bests[2];
  printf("{\"name\":\"icache %s LEN %d BLOCK %d\",\"mode\":\"%s\",\"len\":%d,\"block\":%d,\"warps_per_scheduler\":%.2f,"
         "\"cycles\":%.0f,\"cycles_per_instr\":%.3f,\"cold_first_rep\":%lld,\"launches\":[%.0f,%.0f,%.0f],"
         "\"arch\":\"%s\",\"device\":\"%s\",\"cuda\":\"%s\",\"nvcc\":\"%s\",\"node\":\"%s\",\"date\":\"%s\","
         "\"file\":\"harness/micro/icache_micro.cu\",\"method\":\"min over %d reps and 3 launches, barrier bracketed clock64\"}\n",
         DEP ? "dependent" : "independent", LEN, BLOCK, DEP ? "dependent" : "independent", LEN, BLOCK, BLOCK / 128.0,
         b, b / LEN, cold, bests[0], bests[1], bests[2],
         g_meta.arch.c_str(), g_meta.device.c_str(), g_meta.cuda.c_str(), g_meta.nvcc.c_str(),
         g_meta.node.c_str(), g_meta.date.c_str(), kReps);
  fflush(stdout);
}

template <int LEN>
void run_len()
{
  run<LEN, 128, true>();
  run<LEN, 256, true>();
  run<LEN, 512, true>();
  run<LEN, 128, false>();
  run<LEN, 256, false>();
  run<LEN, 512, false>();
}

void probe_clock(const char* when)
{
  double* d;
  cudaMalloc(&d, sizeof(double));
  double mhz = 0;
  for (int k = 0; k < 3; ++k)
  {
    clock_probe<<<1, 1>>>(d);
    cudaDeviceSynchronize();
    cudaMemcpy(&mhz, d, sizeof(double), cudaMemcpyDeviceToHost);
    printf("{\"name\":\"sm clock probe %s %d\",\"mhz\":%.1f,\"arch\":\"%s\",\"device\":\"%s\",\"node\":\"%s\",\"date\":\"%s\","
           "\"file\":\"harness/micro/icache_micro.cu\"}\n",
           when, k, mhz, g_meta.arch.c_str(), g_meta.device.c_str(), g_meta.node.c_str(), g_meta.date.c_str());
  }
  cudaFree(d);
  fflush(stdout);
}

int main(int argc, char** argv)
{
  g_meta.node = "unknown";
  g_meta.date = "unknown";
  for (int i = 1; i + 1 < argc; i += 2)
  {
    if (!strcmp(argv[i], "--node"))
    {
      g_meta.node = argv[i + 1];
    }
    else if (!strcmp(argv[i], "--date"))
    {
      g_meta.date = argv[i + 1];
    }
  }
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  char buf[64];
  snprintf(buf, sizeof(buf), "sm_%d%d", p.major, p.minor);
  g_meta.arch   = buf;
  g_meta.device = p.name;
  int rt        = 0;
  cudaRuntimeGetVersion(&rt);
  snprintf(buf, sizeof(buf), "%d.%d", rt / 1000, (rt % 1000) / 10);
  g_meta.cuda = buf;
  snprintf(buf, sizeof(buf), "%d.%d.%d", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
  g_meta.nvcc = buf;
  int khz = 0;
  cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
  fprintf(stderr, "device: %s (%s, %d SMs, attribute clock %d kHz) runtime %s nvcc %s\n",
          p.name, g_meta.arch.c_str(), p.multiProcessorCount, khz, g_meta.cuda.c_str(), g_meta.nvcc.c_str());
  probe_clock("start");
  run_len<1024>(); // 16 KB of SASS
  run_len<2048>(); // 32 KB
  run_len<4096>(); // 64 KB
  run_len<8192>(); // 128 KB
  // B2-shaped rounds: 128 rounds of 32 IMADs plus atomic, barrier, LDS is about 5000 instructions (80 KB)
  int sms = p.multiProcessorCount;
  run_rounds<128, 128>(1, "one block");
  run_rounds<128, 256>(1, "one block");
  run_rounds<32, 128>(1, "one block");
  run_rounds<128, 128>(sms, "one block per SM");
  run_rounds<128, 256>(sms, "one block per SM");
  run_rounds<128, 128>(1, "one block again");
  // the 80 KB IMAD body with a one-lane guarded atomic per round (BSSY, BRA, BSYNC), single block and per SM
  run_rounds<128, 128, true>(1, "one block");
  run_rounds<128, 256, true>(1, "one block");
  run_rounds<128, 128, true>(sms, "one block per SM");
  // branch then 32 independent IMADs per round: high issue rate right after the branch
  run_rounds_ipc<128, 128, true>(1, "one block");
  run_rounds_ipc<128, 128, false>(1, "one block");
  run_rounds_ipc<128, 256, true>(1, "one block");
  run_rounds_ipc<128, 128, true>(sms, "one block per SM");
  // select-heavy rounds with and without a divergent one-lane atomic branch, 128 rounds is about 90 KB of code
  run_rounds_sel<128, 128, false>(1, "one block");
  run_rounds_sel<128, 128, true>(1, "one block");
  run_rounds_sel<128, 256, false>(1, "one block");
  run_rounds_sel<128, 256, true>(1, "one block");
  run_rounds_sel<32, 128, true>(1, "one block");
  run_rounds_sel<128, 128, true>(sms, "one block per SM");
  // larger branchy bodies: 256 and 384 rounds are about 60 and 90 KB of code, the size of the slow kernels
  run_rounds_sel<256, 128, true>(1, "one block");
  run_rounds_sel<384, 128, true>(1, "one block");
  run_rounds_sel<384, 256, true>(1, "one block");
  run_rounds_sel<384, 128, false>(1, "one block");
  run_rounds_sel<384, 128, true>(sms, "one block per SM");
  // memory latency per SM: L2 resident (4 MB) and beyond L2 (512 MB)
  run_chase(4u << 20, "L2 4MB", 1, 4096);
  run_chase(4u << 20, "L2 4MB", sms, 4096);
  run_chase(512u << 20, "HBM 512MB", 1, 2048);
  run_chase(512u << 20, "HBM 512MB", sms, 2048);
  // small buffers (the size of a harness input): does a buffer's L2 home make some SMs far from it
  for (int k = 0; k < 3; ++k)
  {
    run_chase(16u << 10, "L2 16KB", sms, 2048);
    run_chase(64u << 10, "L2 64KB", sms, 2048);
    run_chase(256u << 10, "L2 256KB", sms, 2048);
  }
  run_chase(16u << 10, "L2 16KB", 1, 2048);
  run_chase(64u << 10, "L2 64KB", 1, 2048);
  // load paths over an L1 resident 16 KB buffer and local memory, per SM and single block
  {
    unsigned stride = 0;
    unsigned* d     = make_chase(16u << 10, stride);
    run_path_chase(d, (16u << 10) / 4, 1, 2048);
    run_path_chase(d, (16u << 10) / 4, sms, 2048);
    cudaFree(d);
  }
  probe_clock("end");
  return 0;
}
