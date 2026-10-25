// Implementation of approach W5 (threshold filter with a certified lower bound, compaction through
// shared memory, packed W2 finish on the candidates), warp scope. Not an adapter by itself.
//
// Items become the same unique 32-bit words as in W2 (bf16: twiddled key above an inverted index,
// idea N4 for fp32 after the compaction), so every threshold compare is a compare of unique words
// and the number of candidates is the exact rank of the threshold word, ties included.
//
// Bound. The 32 lanes form G groups of 32 / G adjacent lanes. Each lane selects its J largest words
// with a small per-lane network (blocks of 4 or 8 sorted with Batcher odd-even merge sort and
// merged into the running top J), the group's top J is formed with log2(32 / G) xor-shuffle merge
// steps, and the threshold is the warp minimum (one redux) over the groups of the group's J-th
// largest. G x J >= k, so at least k words are at or above the threshold, no retry is ever needed
// and the finish is exact. G and J per k come from the host simulation in approaches/W5
// (sim_gj.py): 16 groups of depth 1 at k <= 16 (fewer groups below k 16), depth 2 at k 32,
// depth 4 at k 64, depth 8 at k 128.
//
// Filter and compaction. One compare per item against the threshold gives the per-lane candidate
// count, log2(IPT + 1) ballots of its bits give the exclusive prefix over lanes and the total, and
// every lane stores its candidates as one run into the group's shared buffer. After a __syncwarp
// lane t reads positions t, t + 32, ... so the candidates are spread one per lane per position and
// each lane holds at most D of them (D = capacity / 32, 4 to 16 by k). The per-lane list is sorted
// with an odd-even network and the packed W2 rounds (one redux per round, T19, T41) emit the k
// largest words.
//
// fp32 (idea N4). The compaction buffer holds (twiddled key, source index) pairs, the finish word
// is the top 32 - log2(capacity) key bits above the inverted buffer position, so 23 key bits at
// capacity 512 and more at smaller capacities. After the rounds one redux of the remaining heads
// is the largest word not emitted. If it has the truncated key of the k-th word the tie class is
// resolved on full keys from the buffer (W2's next-word check and slow path, on the buffer instead
// of the register tile).
//
// Overflow. When more than the capacity of candidates pass the threshold (sorted blocked input
// cannot do it on this striped arrangement, tie floods and periodic patterns can) the warp runs an
// exact bit-serial radix select over the register tile, levels skipped above the first bit in which
// the warp maximum and the threshold differ, and emits by ballot tickets in index order, which is
// prefer_smaller_index by construction. It needs no second copy of the tile.
//
// Both tie-break requirements run the same code. NaN: as W2 (a NaN with the sign bit clear twiddles
// above +inf and is selected first, with the sign bit set it is never selected, the all-ones pattern
// may collide with the empty-list sentinel). Minus zero is mapped onto plus zero before packing.
// Focus-set inputs contain no NaN.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

#include "../topk_harness.cuh"
#include "W2_impl.cuh"

namespace w5
{

// ------------------------------------------------------------------------------------------------
// Configuration per k (host simulation approaches/W5/sim_gj.py, striped lanes, random input)
// ------------------------------------------------------------------------------------------------
// Certified depth per group
__host__ __device__ constexpr int cert_depth(int k)
{
  return k <= 16 ? 1 : k <= 32 ? 2 : k <= 64 ? 4 : 8;
}
// Number of lane groups (G x cert_depth >= k)
__host__ __device__ constexpr int cert_groups(int k)
{
  return k <= 16 ? w2::next_pow2(k) : 16;
}
// Per-lane list depth of the finish (capacity 32 x D words)
__host__ __device__ constexpr int list_depth(int k)
{
  return k <= 8 ? 4 : k <= 64 ? 8 : 16;
}

__device__ __forceinline__ uint32_t lanemask_lt()
{
  uint32_t m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

// ------------------------------------------------------------------------------------------------
// Per-lane top J of the tile without permuting it (the tile stays in slot order for the indices)
// ------------------------------------------------------------------------------------------------
template <int J, int IPT>
__device__ __forceinline__ void lane_top(const uint32_t (&a)[IPT], uint32_t (&cert)[J])
{
  if constexpr (J == 1)
  {
    uint32_t t[IPT];
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
        t[i] = ::max(t[i], t[i + s]);
      }
    }
    cert[0] = t[0];
  }
  else
  {
    constexpr int B = (J > 2) ? 8 : 4;
    static_assert(IPT % B == 0, "tile must be a multiple of the block size");
    uint32_t blk[B];
#pragma unroll
    for (int i = 0; i < B; ++i)
    {
      blk[i] = a[i];
    }
    w2::oe_sort_desc<0, B - 1>(blk);
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
      w2::oe_sort_desc<0, B - 1>(blk);
      uint32_t spare;
      w2::top_merge_desc<J, 0, B>(cert, blk, spare);
    }
  }
}

// Group top J over 2^S adjacent lanes by xor-shuffle merge steps
template <int J, int S>
__device__ __forceinline__ void group_merge(uint32_t (&cert)[J])
{
#pragma unroll
  for (int s = 0; s < S; ++s)
  {
    uint32_t other[J];
#pragma unroll
    for (int j = 0; j < J; ++j)
    {
      other[j] = __shfl_xor_sync(0xffffffffu, cert[j], 1 << s);
    }
    uint32_t spare;
    w2::top_merge_desc<J, 0, J>(cert, other, spare);
  }
}

// ------------------------------------------------------------------------------------------------
// Exact fallback: bit-serial radix select over the register tile (unique words or twiddled keys),
// then emission by ballot tickets, strictly greater items first, then the first `need` equal items
// in index order. lo is a certified lower bound of the k-th largest, hi the tile maximum.
// ------------------------------------------------------------------------------------------------
template <int IPT, int K, class Red>
__device__ __forceinline__ void
radix_fallback(const uint32_t (&a)[IPT], int* out_idx, int lane, uint32_t lt, uint32_t lo, uint32_t hi, const Red& red)
{
  const uint32_t diff = hi ^ lo;
  const int top       = (diff != 0u) ? (31 - __clz(diff)) : -1;
  uint32_t prefix     = (top >= 0) ? (hi & ~((2u << top) - 1u)) : hi;
  int need            = K;
#pragma unroll 1
  for (int b = top; b >= 0; --b)
  {
    const uint32_t target = (prefix >> b) | 1u;
    uint32_t cnt          = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      cnt += ((a[i] >> b) == target) ? 1u : 0u;
    }
    const int c = static_cast<int>(red.add(cnt));
    if (c >= need)
    {
      prefix |= (1u << b);
    }
    else
    {
      need -= c;
    }
  }
  int base_gt = 0;
  int base_eq = 0;
#pragma unroll
  for (int i = 0; i < IPT; ++i)
  {
    const bool gt     = a[i] > prefix;
    const bool eq     = a[i] == prefix;
    const uint32_t bg = __ballot_sync(0xffffffffu, gt);
    const uint32_t be = __ballot_sync(0xffffffffu, eq);
    const int idx     = i * 32 + lane;
    if (gt)
    {
      out_idx[base_gt + __popc(bg & lt)] = idx;
    }
    const int r = base_eq + __popc(be & lt);
    if (eq && r < need)
    {
      out_idx[K - need + r] = idx;
    }
    base_gt += __popc(bg);
    base_eq += __popc(be);
  }
}

// ------------------------------------------------------------------------------------------------
// The warp primitive: 32 lanes hold N keys striped (lane t has items t, t + 32, ...), write K
// indices to out_idx.
// ------------------------------------------------------------------------------------------------
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
  static_assert(G * J >= K, "the certified set must hold k words");
  static_assert(CAP >= K, "the finish must hold k words");

  using slot_t = std::conditional_t<F32, uint2, uint32_t>;
  struct TempStorage
  {
    slot_t buf[CAP];
  };

  __device__ __forceinline__ static uint32_t item_word(KeyT key, int idx)
  {
    if constexpr (F32)
    {
      return w2::twiddle_f32(key);
    }
    else
    {
      return w2::pack<KeyT, IdxBits>(key, static_cast<uint32_t>(idx));
    }
  }

  __device__ __forceinline__ static void store_slot(slot_t* buf, uint32_t pos, uint32_t w, int idx)
  {
    if constexpr (F32)
    {
      buf[pos] = make_uint2(w, static_cast<uint32_t>(idx));
    }
    else
    {
      buf[pos] = w;
    }
  }

  __device__ __forceinline__ static uint32_t load_word(const slot_t* buf, uint32_t pos)
  {
    if constexpr (F32)
    {
      return (buf[pos].x & ~PM) | (PM ^ pos);
    }
    else
    {
      return buf[pos];
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

  // fp32 next-word check and slow path on the buffer (see W2_impl.cuh n4_tie_pass_next)
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
      const uint32_t p = static_cast<uint32_t>(lane + 32 * j);
      const uint2 v    = buf[p];
      const bool in    = (p < tot) && ((v.x & ~PM) == cls);
      mem |= in ? (1u << j) : 0u;
      const uint32_t w = (v.x & ~PM) | (PM ^ p);
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
        const uint2 v = buf[lane + 32 * j];
        cand          = ((mem >> j) & 1u) ? ::max(cand, v.x) : cand;
      }
      const uint32_t m = red.max(cand);
      uint32_t myidx   = 0xffffffffu;
#pragma unroll
      for (int j = 0; j < D; ++j)
      {
        const uint2 v = buf[lane + 32 * j];
        myidx         = (((mem >> j) & 1u) && v.x == m) ? ::min(myidx, v.y) : myidx;
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
    const uint32_t lt = lanemask_lt();
    const w2::masked_reducer<32> red{0xffffffffu};

    uint32_t a[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      a[i] = item_word(keys[i], i * 32 + lane);
    }

    // certified bound
    uint32_t cert[J];
    lane_top<J, IPT>(a, cert);
    group_merge<J, S>(cert);
    const uint32_t t = red.min(cert[J - 1]);

    // per-lane count, prefix and total through ballots of the count bits
    uint32_t c = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      c += (a[i] >= t) ? 1u : 0u;
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
      const uint32_t hi = red.max(cert[0]);
      radix_fallback<IPT, K>(a, out_idx, lane, lt, t, hi, red);
      return;
    }

    // scatter the candidates as one run per lane
    uint32_t pos = pre;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
    {
      if (a[i] >= t)
      {
        store_slot(ts.buf, pos, a[i], i * 32 + lane);
        ++pos;
      }
    }
    __syncwarp();

    // per-lane list of the positions lane, lane + 32, ...
    uint32_t list[D];
#pragma unroll
    for (int j = 0; j < D; ++j)
    {
      const uint32_t p = static_cast<uint32_t>(lane + 32 * j);
      list[j]          = (p < tot) ? load_word(ts.buf, p) : 0u;
    }
    w2::oe_sort_desc<0, D - 1>(list);

    uint32_t m_last;
    if constexpr (F32)
    {
      m_last = w2::extract_rounds<T, K, D>(list, lane, red, emit_f32{ts.buf, out_idx});
      tie_check(ts.buf, tot, m_last, list[0], out_idx, lane, red);
    }
    else
    {
      m_last = w2::extract_rounds<T, K, D>(list, lane, red, w2::emit_index<IdxBits>{out_idx});
    }
  }
};

// Compile-time support predicate shared by the W5 adapters
__host__ __device__ constexpr bool supported_shape(topk::key_type kt, int threads, int n, int k, topk::requirement req)
{
  const bool key_ok   = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
  const bool shape_ok = (threads == 32) && (n >= 256) && ((n & (n - 1)) == 0) && (k >= 4) && (k <= 128) && (2 * k <= n);
  const bool width_ok = (kt == topk::key_type::bf16) ? (n <= 65536) : (n <= 32768);
  const bool req_ok   = (req != topk::requirement::prefer_larger_index);
  return key_ok && shape_ok && width_ok && req_ok;
}

} // namespace w5
