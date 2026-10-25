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

template <int LEN, int BLOCK, bool DEP>
__global__ void __launch_bounds__(BLOCK) body(long long* out, long long* first)
{
  unsigned sm     = smid();
  unsigned v0     = threadIdx.x * 2654435761u;
  unsigned v1     = v0 ^ 0x9e3779b9u;
  unsigned v2     = v0 + 0x7f4a7c15u;
  unsigned v3     = v0 * 3u;
  long long best  = LLONG_MAX;
  long long cold  = 0;
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
    if constexpr (DEP)
    {
#pragma unroll
      for (int i = 0; i < LEN; ++i)
      {
        v0 = v0 * 3u + (threadIdx.x + static_cast<unsigned>(i));
      }
    }
    else
    {
#pragma unroll
      for (int i = 0; i < LEN; i += 4)
      {
        v0 = v0 * 3u + (threadIdx.x + static_cast<unsigned>(i));
        v1 = v1 * 3u + (threadIdx.x + static_cast<unsigned>(i) + 1u);
        v2 = v2 * 3u + (threadIdx.x + static_cast<unsigned>(i) + 2u);
        v3 = v3 * 3u + (threadIdx.x + static_cast<unsigned>(i) + 3u);
      }
    }
    unsigned v = v0 ^ v1 ^ v2 ^ v3;
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
  run<LEN, 32, true>();
  run<LEN, 128, true>();
  run<LEN, 256, true>();
  run<LEN, 512, true>();
  run<LEN, 32, false>();
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
  run_len<256>();
  run_len<1024>();
  run_len<4096>();
  run_len<8192>();
  probe_clock("end");
  return 0;
}
