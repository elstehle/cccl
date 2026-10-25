// W5 v2: the filter runs on the raw floating keys (bf16 widened to fp32 bits by a shift), so the
// per-item work before the compaction is one float compare and a share of a float max tree, and
// only the candidates are twiddled and packed after the gather (at most D per lane). The W5 v1
// design packed every item first (eight instructions per bf16 item). Other changes against v1:
// the per-lane count is a tree, the scatter runs IPT / 8 independent position chains instead of
// one, and the per-lane top 2 (k 17 to 32) is a pair tree instead of sorted blocks. Everything else
// (grouped certified bound, ballot prefix, shared buffer, packed W2 finish, fp32 route, radix
// fallback) is W5_impl.cuh's. Not an adapter by itself.
//
// Order semantics: the filter keeps every item whose key is at least the threshold key under the
// float order (minus zero equals plus zero, ties of the threshold key all pass), the finish orders
// the candidates by the packed word (twiddled key, inverted index). The certified set guarantees
// that the k-th largest key is at least the threshold, so the true top-k is inside the candidates
// under both tie-break requirements.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

#include "../topk_harness.cuh"
#include "W2_impl.cuh"
#include "W5_impl.cuh"

namespace w5v2
{

using w5::cert_depth;
using w5::cert_groups;
using w5::list_depth;

__device__ __forceinline__ uint32_t untwiddle_f32(uint32_t t)
{
  return (t & 0x80000000u) ? (t ^ 0x80000000u) : ~t;
}

__device__ __forceinline__ void cswap_desc(float& a, float& b)
{
  const float lo = fminf(a, b);
  a              = fmaxf(a, b);
  b              = lo;
}

// Batcher odd-even merge sort of a[LO, HI] into descending order, float keys
template <int LO, int HI, int R, int P>
__device__ __forceinline__ void oe_merge(float (&a)[P])
{
  constexpr int STEP = 2 * R;
  if constexpr (STEP < HI - LO)
  {
    oe_merge<LO, HI, STEP>(a);
    oe_merge<LO + R, HI, STEP>(a);
#pragma unroll
    for (int i = LO + R; i < HI - R; i += STEP)
    {
      cswap_desc(a[i], a[i + R]);
    }
  }
  else
  {
    cswap_desc(a[LO], a[LO + R]);
  }
}

template <int LO, int HI, int P>
__device__ __forceinline__ void oe_sort_desc(float (&a)[P])
{
  if constexpr (HI - LO >= 1)
  {
    constexpr int MID = LO + (HI - LO) / 2;
    oe_sort_desc<LO, MID>(a);
    oe_sort_desc<MID + 1, HI>(a);
    oe_merge<LO, HI, 1>(a);
  }
}

// list (descending) becomes the L largest of list and a[OFF, OFF + L) (descending), sorted
template <int L, int OFF, int P>
__device__ __forceinline__ void top_merge_desc(float (&list)[L], const float (&a)[P])
{
#pragma unroll
  for (int i = 0; i < L; ++i)
  {
    list[i] = fmaxf(list[i], a[OFF + L - 1 - i]);
  }
#pragma unroll
  for (int stride = L >> 1; stride > 0; stride >>= 1)
  {
#pragma unroll
    for (int i = 0; i < L; ++i)
    {
      const int p = i ^ stride;
      if (p > i)
      {
        cswap_desc(list[i], list[p]);
      }
    }
  }
}

// top 2 of two sorted pairs
__device__ __forceinline__ void merge_pair(float& h, float& l, float h2, float l2)
{
  const float nh = fmaxf(h, h2);
  l              = fmaxf(fminf(h, h2), fmaxf(l, l2));
  h              = nh;
}

// Per-lane top J of the tile (float keys), the tile is not permuted
template <int J, int IPT>
__device__ __forceinline__ void lane_top(const float (&a)[IPT], float (&cert)[J])
{
  if constexpr (J == 1)
  {
    float t[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      t[i] = a[i];
    }
#pragma unroll
    for (int s = IPT / 2; s >= 1; s /= 2)
    {
#pragma unroll
      for (int i = 0; i < s; ++i)
      {
        t[i] = fmaxf(t[i], t[i + s]);
      }
    }
    cert[0] = t[0];
  }
  else if constexpr (J == 2)
  {
    // chunks of 8: pair tree inside the chunk, chain across chunks
    constexpr int C = (IPT < 8) ? IPT : 8;
    static_assert(IPT % C == 0, "tile must be a multiple of the chunk");
#pragma unroll
    for (int g = 0; g < IPT / C; ++g)
    {
      float h[C / 2];
      float l[C / 2];
#pragma unroll
      for (int i = 0; i < C / 2; ++i)
      {
        h[i] = fmaxf(a[g * C + 2 * i], a[g * C + 2 * i + 1]);
        l[i] = fminf(a[g * C + 2 * i], a[g * C + 2 * i + 1]);
      }
#pragma unroll
      for (int s = C / 4; s >= 1; s /= 2)
      {
#pragma unroll
        for (int i = 0; i < s; ++i)
        {
          merge_pair(h[i], l[i], h[i + s], l[i + s]);
        }
      }
      if (g == 0)
      {
        cert[0] = h[0];
        cert[1] = l[0];
      }
      else
      {
        merge_pair(cert[0], cert[1], h[0], l[0]);
      }
    }
  }
  else
  {
    constexpr int B = 8;
    static_assert(IPT % B == 0, "tile must be a multiple of the block size");
    float blk[B];
#pragma unroll
    for (int i = 0; i < B; ++i)
    {
      blk[i] = a[i];
    }
    oe_sort_desc<0, B - 1>(blk);
#pragma unroll
    for (int j = 0; j < J; ++j)
    {
      cert[j] = blk[j];
    }
#pragma unroll
    for (int g = 1; g < IPT / B; ++g)
    {
#pragma unroll
      for (int i = 0; i < B; ++i)
      {
        blk[i] = a[g * B + i];
      }
      oe_sort_desc<0, B - 1>(blk);
      top_merge_desc<J, 0, B>(cert, blk);
    }
  }
}

template <int J, int S>
__device__ __forceinline__ void group_merge(float (&cert)[J])
{
#pragma unroll
  for (int s = 0; s < S; ++s)
  {
    float other[J];
#pragma unroll
    for (int j = 0; j < J; ++j)
    {
      other[j] = __shfl_xor_sync(0xffffffffu, cert[j], 1 << s);
    }
    if constexpr (J == 2)
    {
      merge_pair(cert[0], cert[1], other[0], other[1]);
    }
    else
    {
      top_merge_desc<J, 0, J>(cert, other);
    }
  }
}

template <typename KeyT, int N, int K>
struct primitive
{
  static constexpr int T        = 32;
  static constexpr int IPT      = N / T;
  static constexpr int IdxBits  = w2::ceil_log2(N);
  static constexpr int J        = cert_depth(K);
  static constexpr int G        = cert_groups(K);
  static constexpr int S        = w2::ceil_log2(32 / G);
  static constexpr int D        = list_depth(K);
  static constexpr int CAP      = 32 * D;
  static constexpr uint32_t PM  = static_cast<uint32_t>(CAP - 1);
  static constexpr bool F32     = std::is_same_v<KeyT, float>;
  static constexpr int CNT_BITS = w2::ceil_log2(IPT + 1);
  static constexpr int CH       = (IPT < 8) ? IPT : 8; // items per scatter chain
  static constexpr int NCH      = IPT / CH;
  static constexpr uint32_t idx_mask = (1u << IdxBits) - 1u;
  static_assert(G * J >= K, "the certified set must hold k words");
  static_assert(CAP >= K, "the finish must hold k words");
  static_assert(IPT % CH == 0, "tile must be a multiple of the chain length");

  using slot_t = std::conditional_t<F32, uint2, uint32_t>;
  struct TempStorage
  {
    slot_t buf[CAP];
  };

  // raw key as float bits (bf16 widened by a shift, the low 16 bits are zero)
  __device__ __forceinline__ static float raw_float(KeyT key)
  {
    if constexpr (F32)
    {
      return key;
    }
    else
    {
      return __uint_as_float(static_cast<uint32_t>(__bfloat16_as_ushort(key)) << 16);
    }
  }

  __device__ __forceinline__ static void store_slot(slot_t* buf, uint32_t pos, float v, int idx)
  {
    if constexpr (F32)
    {
      buf[pos] = make_uint2(__float_as_uint(v), static_cast<uint32_t>(idx));
    }
    else
    {
      buf[pos] = __float_as_uint(v) | static_cast<uint32_t>(idx);
    }
  }

  // packed finish word of a buffer slot
  __device__ __forceinline__ static uint32_t load_word(const slot_t* buf, uint32_t pos)
  {
    if constexpr (F32)
    {
      return (w2::twiddle_f32(__uint_as_float(buf[pos].x)) & ~PM) | (PM ^ pos);
    }
    else
    {
      const uint32_t s = buf[pos];
      return (w2::twiddle_f32(__uint_as_float(s & 0xffff0000u)) & 0xffff0000u) | (idx_mask ^ (s & 0xffffu));
    }
  }

  struct emit_f32
  {
    const uint2* buf;
    int* out_idx;
    __device__ __forceinline__ void operator()(uint32_t m, int r) const
    {
      out_idx[r] = static_cast<int>(buf[PM ^ (m & PM)].y);
    }
  };

  template <class Red>
  __device__ __forceinline__ static void
  tie_check(const uint2* buf, uint32_t tot, uint32_t m_last, uint32_t head, int* out_idx, int lane, const Red& red)
  {
    const uint32_t m_next = red.max(head);
    if ((m_next ^ m_last) > PM)
    {
      return;
    }
    const uint32_t cls = m_last & ~PM;
    uint32_t mem       = 0;
    uint32_t cnt       = 0;
#pragma unroll
    for (int j = 0; j < D; ++j)
    {
      const uint32_t p  = static_cast<uint32_t>(lane + 32 * j);
      const uint2 v     = buf[p];
      const uint32_t tw = w2::twiddle_f32(__uint_as_float(v.x));
      const bool in     = (p < tot) && ((tw & ~PM) == cls);
      mem |= in ? (1u << j) : 0u;
      const uint32_t w = (tw & ~PM) | (PM ^ p);
      cnt += (in && w >= m_last) ? 1u : 0u;
    }
    const int e = static_cast<int>(red.add(cnt));
#pragma unroll 1
    for (int q = 0; q < e; ++q)
    {
      uint32_t cand = 0;
#pragma unroll
      for (int j = 0; j < D; ++j)
      {
        const uint2 v     = buf[lane + 32 * j];
        const uint32_t tw = w2::twiddle_f32(__uint_as_float(v.x));
        cand              = ((mem >> j) & 1u) ? ::max(cand, tw) : cand;
      }
      const uint32_t m = red.max(cand);
      uint32_t myidx   = 0xffffffffu;
#pragma unroll
      for (int j = 0; j < D; ++j)
      {
        const uint2 v     = buf[lane + 32 * j];
        const uint32_t tw = w2::twiddle_f32(__uint_as_float(v.x));
        myidx             = (((mem >> j) & 1u) && tw == m) ? ::min(myidx, v.y) : myidx;
      }
      const uint32_t widx = red.min(myidx);
      if (lane == (q & 31))
      {
        out_idx[K - e + q] = static_cast<int>(widx);
      }
#pragma unroll
      for (int j = 0; j < D; ++j)
      {
        const uint2 v = buf[lane + 32 * j];
        mem &= (v.y == widx) ? ~(1u << j) : 0xffffffffu;
      }
    }
  }

  __device__ __forceinline__ static void run(const KeyT (&keys)[IPT], TempStorage& ts, int* out_idx)
  {
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const uint32_t lt = w5::lanemask_lt();
    const w2::masked_reducer<32> red{0xffffffffu};

    float a[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      a[i] = raw_float(keys[i]);
    }

    // certified bound on raw keys, broadcast through the twiddled image
    float cert[J];
    lane_top<J, IPT>(a, cert);
    group_merge<J, S>(cert);
    const uint32_t t_tw = red.min(w2::twiddle_f32(cert[J - 1]));
    const float t       = __uint_as_float(untwiddle_f32(t_tw));

    // flags, per-chain counts (tree inside the chain), total
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
    uint32_t pre = 0;
    uint32_t tot = 0;
#pragma unroll
    for (int b = 0; b < CNT_BITS; ++b)
    {
      const uint32_t bal = __ballot_sync(0xffffffffu, (c >> b) & 1u);
      pre += static_cast<uint32_t>(__popc(bal & lt)) << b;
      tot += static_cast<uint32_t>(__popc(bal)) << b;
    }
    if (tot > static_cast<uint32_t>(CAP))
    {
      const uint32_t hi = red.max(w2::twiddle_f32(cert[0]));
      uint32_t (&w)[IPT] = reinterpret_cast<uint32_t(&)[IPT]>(a);
#pragma unroll
      for (int i = 0; i < IPT; ++i)
      {
        w[i] = w2::twiddle_f32(a[i]);
      }
      w5::radix_fallback<IPT, K>(w, out_idx, lane, lt, t_tw, hi, red);
      return;
    }

    // scatter: one position chain per chunk of CH items
    uint32_t base = pre;
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
          store_slot(ts.buf, pos, a[item], item * 32 + lane);
          ++pos;
        }
      }
      base += cnt[q];
    }
    __syncwarp();

    uint32_t list[D];
#pragma unroll
    for (int j = 0; j < D; ++j)
    {
      const uint32_t p = static_cast<uint32_t>(lane + 32 * j);
      list[j]          = (p < tot) ? load_word(ts.buf, p) : 0u;
    }
    w2::oe_sort_desc<0, D - 1>(list);

    if constexpr (F32)
    {
      const uint32_t m_last = w2::extract_rounds<T, K, D>(list, lane, red, emit_f32{ts.buf, out_idx});
      tie_check(ts.buf, tot, m_last, list[0], out_idx, lane, red);
    }
    else
    {
      w2::extract_rounds<T, K, D>(list, lane, red, w2::emit_index<IdxBits>{out_idx});
    }
  }
};

} // namespace w5v2
