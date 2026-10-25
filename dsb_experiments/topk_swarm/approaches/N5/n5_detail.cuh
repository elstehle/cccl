// N5: two bf16 keys per 32-bit register, one SIMD compare-exchange per stage for both.
//
// The register-resident bitonic top-k network of W1 (approaches/W1/warp_level_topk.cuh, the eager
// tile loop in template recursion with the min and max exchange of T42) is run on words that hold
// two independent bf16 keys, one in each 16-bit half. Every stage compares position i with position
// j of the same sequence, so the two halves never interact: max.bf16x2 and min.bf16x2 (one HMNMX2
// each) do the exchange for both halves at once and one shuffle moves both. The network is keys
// only (the one-word W1 route carries the index in the low half of the word, which has no room
// here), so the source indices are recovered after the network from the k-th key of every
// sequence: the keys strictly above it are all selected, the keys equal to it fill the remaining
// slots in index order, which gives prefer_smaller_index exactly with the same code.
//
// Two ways to fill the halves:
//   rows_adapter   two independent rows of N keys per call (the harness's packed_rows arrangement,
//                  row 0 in the low half, row 1 in the high half of word j = item j * LW + lane).
//                  Output: row r's k indices at out_idx[r * K, (r + 1) * K).
//   split_adapter  one row of N keys split into two sets of N / 2 (the harness's packed_striped
//                  arrangement, word j of lane l holds items 2 (j LW + l) and 2 (j LW + l) + 1, the
//                  even items form set A in the low halves, the odd items set B in the high halves).
//                  After the network the two sets' top-MaxK are merged by one more bitonic merge
//                  (set B reversed by a lane-mirror shuffle, position-wise max, log2(MaxK) stages)
//                  on one-key words. Output: the row's k indices in out_idx[0, K).
//
// Index recovery (both adapters): the k-th key t of every half comes from the network's output
// position k - 1 with one shuffle (both halves at once). Each lane compares its input words with
// (t_lo, t_hi) by set.gt.u32.bf16x2 and set.eq.u32.bf16x2 (one HSET2 each per word, per-half
// 0xffff masks, T18). Strictly greater keys are counted per lane and placed by one inclusive scan
// over the logical warp (any order is valid for them, they are all selected). Equal keys are ranked
// in index order with one ballot per word and half (slot-major, lane-minor, which is the index
// order of both arrangements) and the first k minus count_above of them are written. Under the
// IEEE comparisons -0.0 equals +0.0, so the tie class is the oracle's class and the result is
// exact under prefer_smaller_index.
//
// Order semantics of the network: max.bf16x2 and min.bf16x2 order -0.0 below +0.0 (T18), a total
// order on the focus inputs. The recovery classes are defined by IEEE > and ==, under which the
// two zeros coincide: every key the network ranks at or above position k - 1 is > t or == t, and
// at most k - 1 keys are > t, so the selected set is exactly the oracle's set in every case.
// NaN: never selected (a NaN loses every max and min against a number, and both set instructions
// are false for NaN), so with fewer than k non-NaN keys the output has unwritten slots. Focus-set
// inputs carry no NaN. Denormals are compared as values (no flush in the bf16x2 forms).
//
// Shuffles use the full warp mask with the logical warp width: the harness calls the primitive
// from all 32 lanes of a hardware warp in lockstep (one call per logical warp, all logical warps at
// once), which is also the convention of T37 and T40 for sub-warp collectives.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

#include "../W1/warp_level_topk.cuh"
#include "topk_harness.cuh"

namespace n5
{
// ------------------------------------------------------------------------------------------------
// bf16x2 SIMD primitives on raw 32-bit words
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b)
{
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t hmin2(uint32_t a, uint32_t b)
{
  uint32_t r;
  asm("min.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
// per-half 0xffff where a > b (IEEE, false for NaN)
__device__ __forceinline__ uint32_t hgt2_mask(uint32_t a, uint32_t b)
{
  uint32_t r;
  asm("set.gt.u32.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
// per-half 0xffff where a == b (IEEE, -0.0 == +0.0, false for NaN)
__device__ __forceinline__ uint32_t heq2_mask(uint32_t a, uint32_t b)
{
  uint32_t r;
  asm("set.eq.u32.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// Exchange policy for w1::eager_topk. Both SIMD results and a select: ptxas is expected to turn the
// select into two predicated HMNMX2, as it does for the VIMNMX pair of the one-word route (T42).
struct ops_sel
{
  __device__ __forceinline__ static uint32_t max(uint32_t a, uint32_t b)
  {
    return hmax2(a, b);
  }
  __device__ __forceinline__ static void exchange(uint32_t& k, uint32_t other, bool keep_max)
  {
    const uint32_t mx = hmax2(k, other);
    const uint32_t mn = hmin2(k, other);
    k                 = keep_max ? mx : mn;
  }
  __device__ __forceinline__ static void exchange_regs(uint32_t& a, uint32_t& b, bool desc)
  {
    const uint32_t lo = hmin2(a, b);
    const uint32_t hi = hmax2(a, b);
    a                 = desc ? hi : lo;
    b                 = desc ? lo : hi;
  }
};

// The same exchange with the predicate set and applied inside one PTX block (fallback if ops_sel is
// not predicated by ptxas).
struct ops_pred
{
  __device__ __forceinline__ static uint32_t max(uint32_t a, uint32_t b)
  {
    return hmax2(a, b);
  }
  __device__ __forceinline__ static void exchange(uint32_t& k, uint32_t other, bool keep_max)
  {
    asm("{\n\t.reg .pred p;\n\t"
        "setp.ne.u32 p, %2, 0;\n\t"
        "@p max.bf16x2 %0, %0, %1;\n\t"
        "@!p min.bf16x2 %0, %0, %1;\n\t}"
        : "+r"(k)
        : "r"(other), "r"(static_cast<uint32_t>(keep_max)));
  }
  __device__ __forceinline__ static void exchange_regs(uint32_t& a, uint32_t& b, bool desc)
  {
    ops_sel::exchange_regs(a, b, desc);
  }
};

template <int LW, int M, int T, class Ops>
using net_t = w1::eager_topk<uint32_t, LW, M, T, w1::shfl_split32, Ops>;

constexpr unsigned full_mask = 0xffffffffu;

// ------------------------------------------------------------------------------------------------
// Shared pieces
// ------------------------------------------------------------------------------------------------
// Inclusive scan of a 32-bit word (two independent 16-bit fields) over the logical warp.
template <int LW>
__device__ __forceinline__ uint32_t inclusive_scan(uint32_t v, int lane)
{
#pragma unroll
  for (int d = 1; d < LW; d <<= 1)
  {
    const uint32_t n = __shfl_up_sync(full_mask, v, d, LW);
    v += (lane >= d) ? n : 0u;
  }
  return v;
}

// The word at network position K - 1 of tile 0 (register (K - 1) / LW of lane (K - 1) % LW).
template <int LW, int K, int R>
__device__ __forceinline__ uint32_t kth_word(const uint32_t (&w)[R])
{
  return __shfl_sync(full_mask, w[(K - 1) / LW], (K - 1) % LW, LW);
}

// Lane bookkeeping for ballots of a logical warp inside the hardware warp.
template <int LW>
struct lanes
{
  unsigned gmask; // the logical warp's lanes
  unsigned lt; // the logical warp's lanes below this one
  __device__ __forceinline__ lanes()
  {
    gmask           = topk::member_mask<LW>();
    const unsigned h = threadIdx.x & 31u;
    lt              = ((1u << h) - 1u) & gmask;
  }
};

__host__ __device__ constexpr int max_k_of(int lw, int k)
{
  return topk::round_up(k, lw);
}

__host__ __device__ constexpr bool lw_ok(int lw)
{
  return lw == 4 || lw == 8 || lw == 16 || lw == 32;
}

// ------------------------------------------------------------------------------------------------
// rows_adapter: two rows of N per call (packed_rows)
// ------------------------------------------------------------------------------------------------
__host__ __device__ constexpr bool
rows_supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
{
  const int max_k = max_k_of(threads, k);
  return kt == topk::key_type::bf16 && lw_ok(threads) && n % threads == 0 && k >= 1 && max_k <= n && n % max_k == 0
      && topk::is_pow2(n / max_k)
      && (req == topk::requirement::not_guaranteed || req == topk::requirement::prefer_smaller_index);
}

template <typename KeyT, int LW, int N, int K, topk::requirement Req, class Ops>
struct rows_adapter
{
  static constexpr int rows                            = 2;
  static constexpr int items_per_thread                = N / LW; // words per lane, one item index each
  static constexpr int max_k                           = max_k_of(LW, K);
  static constexpr int M                               = max_k / LW; // words per tile
  static constexpr int T                               = N / max_k; // tiles
  static constexpr topk::arrangement input_arrangement = topk::arrangement::packed_rows;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = uint32_t[items_per_thread];
  struct TempStorage
  {};
  using net = net_t<LW, M, T, Ops>;
  static_assert(net::R == items_per_thread, "tile words times tiles must cover the lane's words");

  __device__ __forceinline__ static void
  run(key_storage& in, int (&)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    constexpr int IPT = items_per_thread;
    const int lane    = topk::group_lane<LW>();
    const lanes<LW> ln;

    // 1. the network on a copy, the input words stay for the recovery
    uint32_t w[IPT];
#pragma unroll
    for (int j = 0; j < IPT; ++j)
    {
      w[j] = in[j];
    }
    net::run(w, lane, full_mask);

    // 2. the k-th key of both rows in one word
    const uint32_t t2 = kth_word<LW, K>(w);

    // 3. classes per word, strictly-greater counts per lane and row (16-bit fields, no carry)
    uint32_t ma[IPT], me[IPT];
    uint32_t acc = 0;
#pragma unroll
    for (int j = 0; j < IPT; ++j)
    {
      ma[j] = hgt2_mask(in[j], t2);
      me[j] = heq2_mask(in[j], t2);
      acc += ma[j] & 0x00010001u;
    }
    const uint32_t incl = inclusive_scan<LW>(acc, lane);
    const uint32_t tot  = __shfl_sync(full_mask, incl, LW - 1, LW);
    const uint32_t excl = incl - acc;
    int pa0             = static_cast<int>(excl & 0xffffu);
    int pa1             = static_cast<int>(excl >> 16);
    int pe0             = static_cast<int>(tot & 0xffffu); // the tie class starts after the greater class
    int pe1             = static_cast<int>(tot >> 16);
    int* out1           = out_idx + K;

    // 4. emission: greater keys by lane prefix, equal keys in index order by ballot rank
#pragma unroll
    for (int j = 0; j < IPT; ++j)
    {
      const int item  = j * LW + lane;
      const bool a0   = (ma[j] & 0xffffu) != 0;
      const bool a1   = (ma[j] >> 16) != 0;
      const bool e0   = (me[j] & 0xffffu) != 0;
      const bool e1   = (me[j] >> 16) != 0;
      if (a0)
      {
        out_idx[pa0] = item;
      }
      if (a1)
      {
        out1[pa1] = item;
      }
      pa0 += a0;
      pa1 += a1;
      const unsigned b0 = __ballot_sync(full_mask, e0) & ln.gmask;
      const unsigned b1 = __ballot_sync(full_mask, e1) & ln.gmask;
      const int r0      = pe0 + __popc(b0 & ln.lt);
      const int r1      = pe1 + __popc(b1 & ln.lt);
      if (e0 && r0 < K)
      {
        out_idx[r0] = item;
      }
      if (e1 && r1 < K)
      {
        out1[r1] = item;
      }
      pe0 += __popc(b0);
      pe1 += __popc(b1);
    }
  }
};

// ------------------------------------------------------------------------------------------------
// split_adapter: one row of N per call as two sets of N / 2 (packed_striped), merged at the end
// ------------------------------------------------------------------------------------------------
__host__ __device__ constexpr bool
split_supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
{
  const int max_k = max_k_of(threads, k);
  const int half  = n / 2;
  return kt == topk::key_type::bf16 && lw_ok(threads) && n % (2 * threads) == 0 && k >= 1 && max_k <= half
      && half % max_k == 0 && topk::is_pow2(half / max_k)
      && (req == topk::requirement::not_guaranteed || req == topk::requirement::prefer_smaller_index);
}

template <typename KeyT, int LW, int N, int K, topk::requirement Req, class Ops>
struct split_adapter
{
  static constexpr int items_per_thread                = N / LW;
  static constexpr int words                           = items_per_thread / 2;
  static constexpr int max_k                           = max_k_of(LW, K);
  static constexpr int M                               = max_k / LW; // words per tile
  static constexpr int T                               = (N / 2) / max_k; // tiles per set
  static constexpr topk::arrangement input_arrangement = topk::arrangement::packed_striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = uint32_t[words];
  struct TempStorage
  {};
  using net = net_t<LW, M, T, Ops>;
  static_assert(net::R == words, "tile words times tiles must cover the lane's words");

  __device__ __forceinline__ static void
  run(key_storage& in, int (&)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    const int lane = topk::group_lane<LW>();
    const lanes<LW> ln;

    // 1. the SIMD network on both sets
    uint32_t w[words];
#pragma unroll
    for (int j = 0; j < words; ++j)
    {
      w[j] = in[j];
    }
    net::run(w, lane, full_mask);

    // 2. merge the two sets' top-MaxK: set A's keys to the high half (low half +0.0), set B's keys
    //    stay high, B mirrored in position (lane LW - 1 - lane, register M - 1 - r), position-wise
    //    max gives a bitonic sequence, one descending merge sorts it
    uint32_t b[M];
#pragma unroll
    for (int r = 0; r < M; ++r)
    {
      b[r] = w[r] & 0xffff0000u;
    }
#pragma unroll
    for (int r = 0; r < M; ++r)
    {
      const uint32_t a  = w[r] << 16;
      const uint32_t br = __shfl_xor_sync(full_mask, b[M - 1 - r], LW - 1);
      w[r]              = hmax2(a, br);
    }
    net::template merge_tile0<net::L / 2>(w, lane, full_mask);

    // 3. the k-th key, in both halves
    const uint32_t tw = kth_word<LW, K>(w);
    const uint32_t t2 = __byte_perm(tw, 0u, 0x3232);

    // 4. classes per word, strictly-greater count per lane (both halves are the same row)
    uint32_t ma[words], me[words];
    uint32_t acc = 0;
#pragma unroll
    for (int j = 0; j < words; ++j)
    {
      ma[j] = hgt2_mask(in[j], t2);
      me[j] = heq2_mask(in[j], t2);
      acc += ma[j] & 0x00010001u;
    }
    const uint32_t ca   = (acc & 0xffffu) + (acc >> 16);
    const uint32_t incl = inclusive_scan<LW>(ca, lane);
    const uint32_t tot  = __shfl_sync(full_mask, incl, LW - 1, LW);
    int pa              = static_cast<int>(incl - ca);
    int pe              = static_cast<int>(tot);

    // 5. emission: greater keys by lane prefix, equal keys in index order (word, lane, half)
#pragma unroll
    for (int j = 0; j < words; ++j)
    {
      const int item0 = 2 * (j * LW + lane);
      const bool a0   = (ma[j] & 0xffffu) != 0;
      const bool a1   = (ma[j] >> 16) != 0;
      const bool e0   = (me[j] & 0xffffu) != 0;
      const bool e1   = (me[j] >> 16) != 0;
      if (a0)
      {
        out_idx[pa] = item0;
      }
      pa += a0;
      if (a1)
      {
        out_idx[pa] = item0 + 1;
      }
      pa += a1;
      const unsigned b0 = __ballot_sync(full_mask, e0) & ln.gmask;
      const unsigned b1 = __ballot_sync(full_mask, e1) & ln.gmask;
      const int r0      = pe + __popc(b0 & ln.lt) + __popc(b1 & ln.lt);
      const int r1      = r0 + e0;
      if (e0 && r0 < K)
      {
        out_idx[r0] = item0;
      }
      if (e1 && r1 < K)
      {
        out_idx[r1] = item0 + 1;
      }
      pe += __popc(b0) + __popc(b1);
    }
  }
};
} // namespace n5
