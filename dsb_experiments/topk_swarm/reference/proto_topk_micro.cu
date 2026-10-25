// Prototype: building-block latency microbenchmarks for the block top-K design (B200, sm_100).
//
// Measures, via the chain-length slope method (see proto_slope.cu / DEVICE_SIDE_BENCHMARKING_ISSUE.md §3B),
// the marginal per-op latency in cycles of every primitive that could sit on the top-K critical path:
//   - redux.sync.max.u32 (single-instruction warp max, sm_80+)
//   - shfl_xor (single), full 5-step shuffle max tree (u32 and u64/packed)
//   - ballot_sync (winner-lane election)
//   - shared-memory atomicMax u32/u64: uncontended, 8-way, 32-way, 256-way, banked
//   - __syncwarp, __syncthreads (256 threads)
//   - LDS/STS round trip
//   - three composed block-wide max primitives (the §7 redux-vs-atomics question):
//       classic  : warp shfl-max -> smem partial -> 1 barrier -> redundant 8-way shfl-max
//       atomic256: 256 threads atomicMax one slot -> 1 barrier -> read
//       hybrid8  : warp shfl-max -> 8 lane0 atomicMax one slot -> 1 barrier -> read
//     (u32 and u64 variants; u64 models the packed (value,index) word for float arg-max)
//
// total_cycles(N) = intercept + slope * N ; slope = marginal per-op latency, intercept absorbs
// timer/fill/drain/barrier-bracket overhead. Boundary sloppiness lands in the intercept, not the slope.

#include <climits>
#include <cstdint>
#include <cstdio>

__device__ int g_sink;

constexpr int kReps      = 64;
constexpr int kMaxChain  = 32;
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ unsigned warp_max_shfl(unsigned x)
{
#pragma unroll
  for (int d = 16; d; d >>= 1)
  {
    x = ::max(x, __shfl_xor_sync(kFull, x, d));
  }
  return x;
}

__device__ __forceinline__ unsigned long long warp_max_shfl(unsigned long long x)
{
#pragma unroll
  for (int d = 16; d; d >>= 1)
  {
    x = ::max(x, __shfl_xor_sync(kFull, x, d));
  }
  return x;
}

struct NoSmem
{
  int dummy;
};

// ---------------------------------------------------------------- warp-level
struct ReduxU32
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return __reduce_max_sync(kFull, v) + threadIdx.x;
  }
};

struct ShflXor1
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return __shfl_xor_sync(kFull, v, 1) + 1;
  }
};

struct WarpMaxShflU32
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return warp_max_shfl(v) + threadIdx.x;
  }
};

struct WarpMaxShflU64
{
  using value_t = unsigned long long;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761ull;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return warp_max_shfl(v) + threadIdx.x;
  }
};

struct Ballot
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    return __ballot_sync(kFull, (v >> (threadIdx.x & 31)) & 1) + threadIdx.x;
  }
};

struct BallotFfsElect // ballot + ffs + "am I the winner" select: the index-resolution idiom
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    unsigned b   = __ballot_sync(kFull, (v >> (threadIdx.x & 31)) & 1);
    int winner   = __ffs(b) - 1;
    return v + (winner == (int)threadIdx.x ? 1u : 2u);
  }
};

struct SyncWarp
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    __syncwarp();
    return v + 1;
  }
};

struct LdsSts
{
  using value_t = unsigned;
  struct Smem
  {
    unsigned buf[32];
  };
  static constexpr int block = 32;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    volatile unsigned* p = sh.buf;
    p[threadIdx.x]       = v;
    return p[threadIdx.x] + 1;
  }
};

// ---------------------------------------------------------------- shared atomics
template <typename T>
struct AtomSmem
{
  T word;
  T slots[kMaxChain];
};

// Single-thread dependent chain: pure round-trip latency of one shared atomicMax.
template <typename T>
struct AtomUncontended
{
  using value_t = T;
  using Smem    = AtomSmem<T>;
  static constexpr int block = 32;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      sh.word = 0;
    }
    return threadIdx.x + 1;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      v = atomicMax(&sh.word, v + 1) + 1;
    }
    return v;
  }
};

// C threads hammer one word, each with a self-dependent chain: per-op cost under contention.
template <typename T, int C, int BLOCK>
struct AtomContended
{
  using value_t = T;
  using Smem    = AtomSmem<T>;
  static constexpr int block = BLOCK;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      sh.word = 0;
    }
    return threadIdx.x + 1;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    if (threadIdx.x < C)
    {
      v = atomicMax(&sh.word, v + 2) + 2;
    }
    return v;
  }
};

// One-atomic-per-warp (lane 0 of all 8 warps): the hybrid combine's contention level.
template <typename T>
struct AtomWarpLane0
{
  using value_t = T;
  using Smem    = AtomSmem<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x == 0)
    {
      sh.word = 0;
    }
    return threadIdx.x + 1;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    if ((threadIdx.x & 31) == 0)
    {
      v = atomicMax(&sh.word, v + 2) + 2;
    }
    return v;
  }
};

// 256 threads across 32 banked words (tid%32): contention 8 per word, bank-parallel.
struct AtomBanked
{
  using value_t = unsigned;
  struct Smem
  {
    unsigned words[32];
  };
  static constexpr int block = 256;
  __device__ value_t init(Smem& sh)
  {
    if (threadIdx.x < 32)
    {
      sh.words[threadIdx.x] = 0;
    }
    return threadIdx.x + 1;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    return atomicMax(&sh.words[threadIdx.x & 31], v + 2) + 2;
  }
};

// ---------------------------------------------------------------- barriers
struct SyncThreads256
{
  using value_t = unsigned;
  using Smem    = NoSmem;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem&)
  {
    __syncthreads();
    return v + 1;
  }
};

// ---------------------------------------------------------------- composed block-wide max
template <typename T>
struct BlockMaxSmem
{
  T partial[8];
  T slots[kMaxChain];
};

// classic: warp shfl-max -> smem partials -> 1 barrier -> every warp redundantly reduces the 8 partials.
template <typename T>
struct BlockMaxClassic
{
  using value_t = T;
  using Smem    = BlockMaxSmem<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem&, int) {}
  __device__ value_t step(value_t v, int, Smem& sh)
  {
    T x = warp_max_shfl(v);
    if ((threadIdx.x & 31) == 0)
    {
      sh.partial[threadIdx.x >> 5] = x;
    }
    __syncthreads();
    T y = sh.partial[threadIdx.x & 7];
#pragma unroll
    for (int d = 4; d; d >>= 1)
    {
      y = ::max(y, __shfl_xor_sync(kFull, y, d));
    }
    return y + threadIdx.x;
  }
};

// atomic256: all 256 threads atomicMax into slot[n] -> 1 barrier -> read.
template <typename T>
struct BlockMaxAtomic256
{
  using value_t = T;
  using Smem    = BlockMaxSmem<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem& sh, int n)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += block)
    {
      sh.slots[i] = 0;
    }
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    atomicMax(&sh.slots[n], v);
    __syncthreads();
    return sh.slots[n] + threadIdx.x;
  }
};

// hybrid8: warp shfl-max -> lane0-of-each-warp atomicMax into slot[n] -> 1 barrier -> read.
template <typename T>
struct BlockMaxHybrid8
{
  using value_t = T;
  using Smem    = BlockMaxSmem<T>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem& sh, int n)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += block)
    {
      sh.slots[i] = 0;
    }
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    T x = warp_max_shfl(v);
    if ((threadIdx.x & 31) == 0)
    {
      atomicMax(&sh.slots[n], x);
    }
    __syncthreads();
    return sh.slots[n] + threadIdx.x;
  }
};

// hybrid8 with redux instead of shuffle tree (u32 only).
struct BlockMaxHybrid8Redux
{
  using value_t = unsigned;
  using Smem    = BlockMaxSmem<unsigned>;
  static constexpr int block = 256;
  __device__ value_t init(Smem&)
  {
    return threadIdx.x * 2654435761u;
  }
  __device__ void reset(Smem& sh, int n)
  {
    for (int i = threadIdx.x; i < kMaxChain; i += block)
    {
      sh.slots[i] = 0;
    }
  }
  __device__ value_t step(value_t v, int n, Smem& sh)
  {
    unsigned x = __reduce_max_sync(kFull, v);
    if ((threadIdx.x & 31) == 0)
    {
      atomicMax(&sh.slots[n], x);
    }
    __syncthreads();
    return sh.slots[n] + threadIdx.x;
  }
};

// ---------------------------------------------------------------- harness
template <class Exp, int N>
__global__ void __launch_bounds__(Exp::block) bench(long long* out)
{
  __shared__ typename Exp::Smem sh;
  Exp e;
  auto v         = e.init(sh);
  long long best = LLONG_MAX;
#pragma unroll 1
  for (int r = 0; r < kReps; ++r)
  {
    e.reset(sh, N);
    __syncthreads();
    long long t0 = clock64();
    asm volatile("" ::: "memory");
#pragma unroll
    for (int n = 0; n < N; ++n)
    {
      v = e.step(v, n, sh);
    }
    // consume v (scoreboard wait) so the closing barrier follows completion, then align warps
    if (v == static_cast<decltype(v)>(0xdeadbeefu))
    {
      g_sink = (int)v;
    }
    asm volatile("" ::: "memory");
    __syncthreads();
    long long t1 = clock64();
    best         = ::min(best, t1 - t0);
  }
  if (threadIdx.x == 0)
  {
    *out = best;
  }
}

template <class Exp, int N>
long long go(long long* d)
{
  bench<Exp, N><<<1, Exp::block>>>(d);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    printf("KERNEL ERROR: %s\n", cudaGetErrorString(err));
    return -1;
  }
  long long v = 0;
  cudaMemcpy(&v, d, sizeof(long long), cudaMemcpyDeviceToHost);
  return v;
}

void fit(const double* x, const double* y, int m, double& a, double& b, double& r2)
{
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = 0; i < m; ++i)
  {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    sxy += x[i] * y[i];
  }
  b            = (m * sxy - sx * sy) / (m * sxx - sx * sx);
  a            = (sy - b * sx) / m;
  double ybar  = sy / m;
  double ssres = 0, sstot = 0;
  for (int i = 0; i < m; ++i)
  {
    double f = a + b * x[i];
    ssres += (y[i] - f) * (y[i] - f);
    sstot += (y[i] - ybar) * (y[i] - ybar);
  }
  r2 = (sstot > 0) ? 1.0 - ssres / sstot : 1.0;
}

template <class Exp>
void run(const char* name)
{
  long long* d;
  cudaMalloc(&d, sizeof(long long));
  double x[6] = {1, 2, 4, 8, 16, 32}, y[6];
  y[0] = (double)go<Exp, 1>(d);
  y[1] = (double)go<Exp, 2>(d);
  y[2] = (double)go<Exp, 4>(d);
  y[3] = (double)go<Exp, 8>(d);
  y[4] = (double)go<Exp, 16>(d);
  y[5] = (double)go<Exp, 32>(d);
  cudaFree(d);
  double a, b, r2;
  fit(x, y, 6, a, b, r2);
  printf("  %-34s slope=%7.1f cyc/op   intercept=%7.1f   R2=%.5f   raw(N=1,8,32)=%.0f,%.0f,%.0f\n",
         name, b, a, r2, y[0], y[3], y[5]);
}

int main()
{
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("device: %s  (sm_%d%d, %d SMs)\n\n", p.name, p.major, p.minor, p.multiProcessorCount);

  printf("--- warp-scope primitives (1 warp) ---\n");
  run<ReduxU32>("redux.sync.max.u32");
  run<ShflXor1>("shfl_xor (single)");
  run<WarpMaxShflU32>("warp max, 5-shfl tree, u32");
  run<WarpMaxShflU64>("warp max, 5-shfl tree, u64");
  run<Ballot>("ballot_sync");
  run<BallotFfsElect>("ballot + ffs + elect");
  run<SyncWarp>("syncwarp");
  run<LdsSts>("STS+LDS round trip");

  printf("\n--- shared-memory atomics ---\n");
  run<AtomUncontended<unsigned>>("atomicMax smem u32, uncontended");
  run<AtomUncontended<unsigned long long>>("atomicMax smem u64, uncontended");
  run<AtomContended<unsigned, 2, 32>>("atomicMax smem u32, contention 2");
  run<AtomContended<unsigned, 8, 32>>("atomicMax smem u32, contention 8");
  run<AtomContended<unsigned, 32, 32>>("atomicMax smem u32, contention 32");
  run<AtomContended<unsigned, 256, 256>>("atomicMax smem u32, contention 256");
  run<AtomWarpLane0<unsigned>>("atomicMax smem u32, 8 warps lane0");
  run<AtomWarpLane0<unsigned long long>>("atomicMax smem u64, 8 warps lane0");
  run<AtomContended<unsigned long long, 256, 256>>("atomicMax smem u64, contention 256");
  run<AtomBanked>("atomicMax smem u32, 256thr/32banks");

  printf("\n--- barriers ---\n");
  run<SyncThreads256>("__syncthreads (256 thr)");

  printf("\n--- composed block-wide max (256 thr), the redux<->atomics question ---\n");
  run<BlockMaxClassic<unsigned>>("blockmax classic u32 (shfl+smem)");
  run<BlockMaxClassic<unsigned long long>>("blockmax classic u64 (shfl+smem)");
  run<BlockMaxAtomic256<unsigned>>("blockmax atomic256 u32");
  run<BlockMaxAtomic256<unsigned long long>>("blockmax atomic256 u64");
  run<BlockMaxHybrid8<unsigned>>("blockmax hybrid8 u32 (shfl+atomic)");
  run<BlockMaxHybrid8<unsigned long long>>("blockmax hybrid8 u64 (shfl+atomic)");
  run<BlockMaxHybrid8Redux>("blockmax hybrid8 u32 (redux+atomic)");
  return 0;
}
