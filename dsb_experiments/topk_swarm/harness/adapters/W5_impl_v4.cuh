// W5 v4: v3 with a predicate-free gather. The v3 phase profile showed the bf16 gather at about 440
// cycles for 8 loads: every gathered word was twiddled with W2's twiddle, whose minus-zero
// normalization is a compare and a select per word, and every position carried a `p < tot`
// predicate and select. Here minus zero is normalized once at load time with one FADD per item
// (x + 0.0f is x for every x except minus zero, which becomes plus zero), the twiddle is the plain
// sign-fold without a compare, the gather loads every position unconditionally (positions below
// CAP + 32 are always inside the buffer) and masks invalid words with the sign of (p - tot).
// Everything else is v3's.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

#include "../topk_harness.cuh"
#include "W2_impl.cuh"
#include "W5_impl.cuh"
#include "W5_impl_v2.cuh"
#include "W5_impl_v3.cuh"

namespace w5v4
{

// order-preserving sign fold of float bits, no minus-zero fix (the tile never holds minus zero)
__device__ __forceinline__ uint32_t twiddle_plain(uint32_t u)
{
  const uint32_t mask = static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u;
  return u ^ mask;
}

template <typename KeyT, int N, int K>
struct primitive
{
  using v3 = w5v3::primitive<KeyT, N, K>;
  using v2 = w5v2::primitive<KeyT, N, K>;
  static constexpr int T        = 32;
  static constexpr int IPT      = v3::IPT;
  static constexpr int IdxBits  = v3::IdxBits;
  static constexpr int J        = v3::J;
  static constexpr int D        = v3::D;
  static constexpr int CAP      = v3::CAP;
  static constexpr uint32_t PM  = v3::PM;
  static constexpr bool F32     = v3::F32;
  static constexpr int NCH      = v3::NCH;
  static constexpr uint32_t idx_mask = (1u << IdxBits) - 1u;
  using slot_t                  = typename v3::slot_t;
  using TempStorage             = typename v3::TempStorage;

  __device__ __forceinline__ static void load(const KeyT (&keys)[IPT], float (&a)[IPT])
  {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      a[i] = v2::raw_float(keys[i]) + 0.0f;
    }
  }

  __device__ __forceinline__ static uint32_t load_word(const slot_t* buf, uint32_t pos)
  {
    if constexpr (F32)
    {
      return (twiddle_plain(buf[pos].x) & ~PM) | (PM ^ pos);
    }
    else
    {
      const uint32_t s = buf[pos];
      return (twiddle_plain(s & 0xffff0000u) & 0xffff0000u) | (idx_mask ^ (s & 0xffffu));
    }
  }

  __device__ __forceinline__ static void gather(const slot_t* buf, uint32_t tot, int lane, uint32_t (&list)[D])
  {
#pragma unroll
    for (int j = 0; j < D; ++j)
    {
      const uint32_t p     = static_cast<uint32_t>(lane + 32 * j);
      const uint32_t valid = static_cast<uint32_t>(static_cast<int32_t>(p - tot) >> 31); // all ones when p < tot
      list[j]              = load_word(buf, p) & valid;
    }
  }

  __device__ __forceinline__ static void run(const KeyT (&keys)[IPT], TempStorage& ts, int* out_idx)
  {
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const uint32_t lt = w5::lanemask_lt();
    const w2::masked_reducer<32> red{0xffffffffu};
    float a[IPT];
    load(keys, a);
    float cert[J];
    uint32_t t_tw;
    const float t = v3::bound(a, cert, t_tw, red);
    uint32_t cnt[NCH];
    const uint32_t c = v3::count(a, t, cnt);
    uint32_t pre;
    const uint32_t tot = v3::prefix(c, lt, pre);
    if (tot > static_cast<uint32_t>(CAP))
    {
      v3::fallback(a, cert, t_tw, out_idx, lane, lt, red);
      return;
    }
    v3::scatter(a, t, cnt, pre, lane, ts.buf);
    __syncwarp();
    uint32_t list[D];
    gather(ts.buf, tot, lane, list);
    v3::finish(list, ts.buf, tot, out_idx, lane, red);
  }
};

} // namespace w5v4
