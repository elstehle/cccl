// Codegen micro for N5: which packed compare-exchange form does ptxas fold into two instructions?
// One kernel per form, a 32-stage chain of shuffle exchanges with a lane-dependent direction per
// stage (as in a bitonic network), compiled with -cubin and read with cuobjdump -sass. Not a timing
// benchmark. Build: nvcc -std=c++17 -O3 -arch=sm_100 -cubin -o exchange_forms.cubin exchange_forms.cu
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) { uint32_t r; asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ uint32_t hmin2(uint32_t a, uint32_t b) { uint32_t r; asm("min.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ uint32_t hneg2(uint32_t a) { uint32_t r; asm("neg.bf16x2 %0, %1;" : "=r"(r) : "r"(a)); return r; }

// 0: u32 min max select (the T42 reference form)
struct f_u32 { __device__ __forceinline__ static uint32_t ex(uint32_t k, uint32_t o, bool km) { return km ? max(k, o) : min(k, o); } };
// 1: bf16x2 through inline asm, select
struct f_bf16_asm { __device__ __forceinline__ static uint32_t ex(uint32_t k, uint32_t o, bool km) { return km ? hmax2(k, o) : hmin2(k, o); } };
// 2: bf16x2 through the header intrinsics, select
struct f_bf16_hdr { __device__ __forceinline__ static uint32_t ex(uint32_t k, uint32_t o, bool km) {
  __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&k), b = *reinterpret_cast<__nv_bfloat162*>(&o);
  __nv_bfloat162 mx = __hmax2(a, b), mn = __hmin2(a, b);
  __nv_bfloat162 r = km ? mx : mn; return *reinterpret_cast<uint32_t*>(&r); } };
// 3: u16x2 integer SIMD, select
struct f_u16x2 { __device__ __forceinline__ static uint32_t ex(uint32_t k, uint32_t o, bool km) { return km ? __vmaxu2(k, o) : __vminu2(k, o); } };
// 4: bf16x2, blend by a per-lane mask (LOP3)
struct f_bf16_blend { __device__ __forceinline__ static uint32_t ex(uint32_t k, uint32_t o, bool km) {
  const uint32_t m = km ? 0xffffffffu : 0u; return (hmax2(k, o) & m) | (hmin2(k, o) & ~m); } };
// 5: bf16x2 with a negated partner operand (does ptxas fold neg.bf16x2 into HMNMX2?)
struct f_bf16_neg { __device__ __forceinline__ static uint32_t ex(uint32_t k, uint32_t o, bool km) {
  const uint32_t r = hmax2(k, hneg2(o)); return km ? r : r ^ 0x80008000u; } };
// 6: bf16x2 with the partner negated by xor (same question for a LOP3 negation)
struct f_bf16_xneg { __device__ __forceinline__ static uint32_t ex(uint32_t k, uint32_t o, bool km) {
  const uint32_t r = hmax2(k, o ^ 0x80008000u); return km ? r : r ^ 0x80008000u; } };
// 7: u16x2 with vimax3 style: max then predicated min via select on the result of __vibmax? (plain for reference)
struct f_u16x2_hdr2 { __device__ __forceinline__ static uint32_t ex(uint32_t k, uint32_t o, bool km) {
  const uint32_t mx = __vmaxu2(k, o), mn = __vminu2(k, o); return km ? mx : mn; } };

template <class F>
__global__ void chain(const uint32_t* in, uint32_t* out)
{
  const int lane = threadIdx.x & 31;
  uint32_t k = in[threadIdx.x];
#pragma unroll
  for (int i = 0; i < 32; ++i)
  {
    const int stride = 1 << (i % 5);
    const uint32_t o = __shfl_xor_sync(0xffffffffu, k, stride);
    const bool upper = (lane & stride) != 0;
    const bool desc  = ((lane >> ((i / 5) % 5)) & 1) != 0;
    k = F::ex(k, o, upper != desc);
  }
  out[threadIdx.x] = k;
}

template __global__ void chain<f_u32>(const uint32_t*, uint32_t*);
template __global__ void chain<f_bf16_asm>(const uint32_t*, uint32_t*);
template __global__ void chain<f_bf16_hdr>(const uint32_t*, uint32_t*);
template __global__ void chain<f_u16x2>(const uint32_t*, uint32_t*);
template __global__ void chain<f_bf16_blend>(const uint32_t*, uint32_t*);
template __global__ void chain<f_bf16_neg>(const uint32_t*, uint32_t*);
template __global__ void chain<f_bf16_xneg>(const uint32_t*, uint32_t*);
template __global__ void chain<f_u16x2_hdr2>(const uint32_t*, uint32_t*);
