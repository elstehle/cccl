// W5 v3: v2 with a predicate-free filter. The phase profiler of v2 (approaches/W5/phase_probe.cu)
// showed the compare-and-count and the scatter phases at about 10 cycles per lane item each while
// they issue 3 to 5 instructions per item: every item's float compare produces a predicate, there
// are seven predicate registers, and ptxas serializes compare and consume per item with P2R
// spills. Here the flag of an item is the sign bit of the subtraction (a - t), an integer 0 or 1
// without a predicate: a >= t gives a non-negative difference (plus zero when equal), a < t a
// negative one, and a flushed tiny difference keeps the sign, so the test is exact or conservative.
// The count is a plain integer tree of these flags, the scatter stores every item unconditionally
// (candidates at their dense position, the others at a per-lane dummy slot behind the buffer) with
// the position advanced by the flag, and the gather, sort, rounds, fp32 route and fallback are v2's.
// The phases are separate functions so the profiler can stamp between them.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

#include "../topk_harness.cuh"
#include "W2_impl.cuh"
#include "W5_impl.cuh"
#include "W5_impl_v2.cuh"

namespace w5v3
{

template <typename KeyT, int N, int K>
struct primitive
{
  using base = w5v2::primitive<KeyT, N, K>;
  static constexpr int T        = 32;
  static constexpr int IPT      = base::IPT;
  static constexpr int IdxBits  = base::IdxBits;
  static constexpr int J        = base::J;
  static constexpr int S        = base::S;
  static constexpr int D        = base::D;
  static constexpr int CAP      = base::CAP;
  static constexpr int LOG2CAP  = w2::ceil_log2(CAP);
  static constexpr uint32_t PM  = base::PM;
  static constexpr bool F32     = base::F32;
  static constexpr int CNT_BITS = base::CNT_BITS;
  static constexpr int CH       = base::CH;
  static constexpr int NCH      = base::NCH;
  using slot_t                  = typename base::slot_t;
  struct TempStorage
  {
    slot_t buf[CAP + 32]; // 32 dummy slots behind the buffer, one per lane
  };

  // 1 when the item is below the threshold (not a candidate), 0 otherwise, no predicate
  __device__ __forceinline__ static uint32_t below(float a, float t)
  {
    return __float_as_uint(a - t) >> 31;
  }

  // phase 1: tile as float bits
  __device__ __forceinline__ static void load(const KeyT (&keys)[IPT], float (&a)[IPT])
  {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      a[i] = base::raw_float(keys[i]);
    }
  }

  // phase 2: certified bound, returns the threshold as float and its twiddled image
  template <class Red>
  __device__ __forceinline__ static float bound(const float (&a)[IPT], float (&cert)[J], uint32_t& t_tw, const Red& red)
  {
    w5v2::lane_top<J, IPT>(a, cert);
    w5v2::group_merge<J, S>(cert);
    t_tw = red.min(w2::twiddle_f32(cert[J - 1]));
    return __uint_as_float(w5v2::untwiddle_f32(t_tw));
  }

  // phase 3: per-chunk candidate counts (tree of integer flags) and the lane total
  __device__ __forceinline__ static uint32_t count(const float (&a)[IPT], float t, uint32_t (&cnt)[NCH])
  {
    uint32_t c = 0;
#pragma unroll
    for (int q = 0; q < NCH; ++q)
    {
      uint32_t f[CH];
#pragma unroll
      for (int i = 0; i < CH; ++i)
      {
        f[i] = below(a[q * CH + i], t);
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
      cnt[q] = static_cast<uint32_t>(CH) - f[0];
      c += cnt[q];
    }
    return c;
  }

  // phase 4: ballot prefix over lanes, returns the total, pre is the lane's exclusive prefix
  __device__ __forceinline__ static uint32_t prefix(uint32_t c, uint32_t lt, uint32_t& pre)
  {
    pre          = 0;
    uint32_t tot = 0;
#pragma unroll
    for (int b = 0; b < CNT_BITS; ++b)
    {
      const uint32_t bal = __ballot_sync(0xffffffffu, (c >> b) & 1u);
      pre += static_cast<uint32_t>(__popc(bal & lt)) << b;
      tot += static_cast<uint32_t>(__popc(bal)) << b;
    }
    return tot;
  }

  // phase 5: unconditional scatter, candidates to their dense position, others to the lane's dummy
  __device__ __forceinline__ static uint32_t
  scatter(const float (&a)[IPT], float t, const uint32_t (&cnt)[NCH], uint32_t pre, int lane, slot_t* buf)
  {
    const uint32_t dummy = static_cast<uint32_t>(CAP + lane);
    uint32_t base_pos    = pre;
    uint32_t fold        = 0;
#pragma unroll
    for (int q = 0; q < NCH; ++q)
    {
      uint32_t pos = base_pos;
#pragma unroll
      for (int i = 0; i < CH; ++i)
      {
        const int item   = q * CH + i;
        const uint32_t b = below(a[item], t);
        const uint32_t p = b ? dummy : pos;
        base::store_slot(buf, p, a[item], item * 32 + lane);
        pos += 1u - b;
      }
      base_pos += cnt[q];
      fold ^= pos;
    }
    return fold;
  }

  // phase 6: gather the lane's positions into the finish list
  __device__ __forceinline__ static void gather(const slot_t* buf, uint32_t tot, int lane, uint32_t (&list)[D])
  {
#pragma unroll
    for (int j = 0; j < D; ++j)
    {
      const uint32_t p = static_cast<uint32_t>(lane + 32 * j);
      list[j]          = (p < tot) ? base::load_word(buf, p) : 0u;
    }
  }

  template <class Red>
  __device__ __forceinline__ static void
  fallback(float (&a)[IPT], const float (&cert)[J], uint32_t t_tw, int* out_idx, int lane, uint32_t lt, const Red& red)
  {
    const uint32_t hi  = red.max(w2::twiddle_f32(cert[0]));
    uint32_t (&w)[IPT] = reinterpret_cast<uint32_t(&)[IPT]>(a);
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      w[i] = w2::twiddle_f32(a[i]);
    }
    w5::radix_fallback<IPT, K>(w, out_idx, lane, lt, t_tw, hi, red);
  }

  template <class Red>
  __device__ __forceinline__ static uint32_t finish(uint32_t (&list)[D], slot_t* buf, uint32_t tot, int* out_idx, int lane, const Red& red)
  {
    w2::oe_sort_desc<0, D - 1>(list);
    if constexpr (F32)
    {
      const uint32_t m_last = w2::extract_rounds<T, K, D>(list, lane, red, typename base::emit_f32{buf, out_idx});
      base::tie_check(buf, tot, m_last, list[0], out_idx, lane, red);
      return m_last;
    }
    else
    {
      return w2::extract_rounds<T, K, D>(list, lane, red, w2::emit_index<IdxBits>{out_idx});
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
    const float t = bound(a, cert, t_tw, red);
    uint32_t cnt[NCH];
    const uint32_t c = count(a, t, cnt);
    uint32_t pre;
    const uint32_t tot = prefix(c, lt, pre);
    if (tot > static_cast<uint32_t>(CAP))
    {
      fallback(a, cert, t_tw, out_idx, lane, lt, red);
      return;
    }
    scatter(a, t, cnt, pre, lane, ts.buf);
    __syncwarp();
    uint32_t list[D];
    gather(ts.buf, tot, lane, list);
    finish(list, ts.buf, tot, out_idx, lane, red);
  }
};

} // namespace w5v3
