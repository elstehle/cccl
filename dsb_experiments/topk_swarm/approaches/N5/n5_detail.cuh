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
#include <type_traits>

#include "../W1/warp_level_topk.cuh"
#include "n5_net.cuh"
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

// Exchange policy for w1::eager_topk. The calls sit inside the ternary: the front end keeps a
// branch around the two asm statements (an asm result cannot be speculated into a select) and
// ptxas if-converts it into two predicated HMNMX2 that write the same register, the shape of the
// one-word VIMNMX route (T42). See approaches/N5/micro/exchange_forms.cu for the forms compared.
struct ops_bf16
{
  __device__ __forceinline__ static uint32_t max(uint32_t a, uint32_t b)
  {
    return hmax2(a, b);
  }
  __device__ __forceinline__ static void exchange(uint32_t& k, uint32_t other, bool keep_max)
  {
    k = keep_max ? hmax2(k, other) : hmin2(k, other);
  }
  __device__ __forceinline__ static void exchange_regs(uint32_t& a, uint32_t& b, bool desc)
  {
    const uint32_t lo = hmin2(a, b);
    const uint32_t hi = hmax2(a, b);
    a                 = desc ? hi : lo;
    b                 = desc ? lo : hi;
  }
};

// Both SIMD results computed, then a select: ptxas keeps both HMNMX2 unpredicated and selects
// with a predicated IMAD.MOV, three instructions per stage word (the first N5 batch ran this form).
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
    ops_bf16::exchange_regs(a, b, desc);
  }
};

// The exchange with the predicate set and applied inside one PTX block: the predicate is
// materialized as a register around every asm block (SEL, ISETP, IMAD.MOV), worse than both forms
// above (424 against 352 SASS instructions in the N 128 k 32 split kernel). Kept as the record.
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

// Network selection. net_w1: the W1 recursion as is (direction recomputed per stage from the lane
// id). net_dir: the N5 copy with the direction as one bit test on precomputed parity words
// (n5_net.cuh).
struct net_w1
{
  template <int LW, int M, int T, class Ops>
  struct bind
  {
    using net              = w1::eager_topk<uint32_t, LW, M, T, w1::shfl_split32, Ops>;
    static constexpr int R = net::R;
    static constexpr int L = net::L;
    __device__ __forceinline__ static void run(uint32_t (&w)[R], int lane, unsigned mask)
    {
      net::run(w, lane, mask);
    }
    __device__ __forceinline__ static void merge0(uint32_t (&w)[R], int lane, unsigned mask)
    {
      net::template merge_tile0<L / 2>(w, lane, mask);
    }
  };
};
struct net_dir
{
  template <int LW, int M, int T, class Ops>
  struct bind
  {
    using net              = eager_topk_dir<uint32_t, LW, M, T, w1::shfl_split32, Ops>;
    static constexpr int R = net::R;
    static constexpr int L = net::L;
    __device__ __forceinline__ static void run(uint32_t (&w)[R], int lane, unsigned mask)
    {
      const typename net::dir_t dir(lane);
      net::run(w, dir, mask);
    }
    __device__ __forceinline__ static void merge0(uint32_t (&w)[R], int lane, unsigned mask)
    {
      const typename net::dir_t dir(lane);
      net::template merge_tile0<L / 2>(w, dir, mask);
    }
  };
};

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
// Index recovery policies
//   recover_exact  greater keys placed by the lane scan, equal keys ranked in index order by one
//                  ballot per word and half, the first k minus count_greater of them written.
//                  Exact under both requirements, always the same code path.
//   recover_fast   greater and equal keys both placed by the lane scan (lane-major order). When
//                  the tie class fits the remaining slots (a warp-uniform test on the scan
//                  totals, the common case on random bf16 input) every equal key is written and
//                  no cap test runs. When it overflows, not_guaranteed caps the equal class at k
//                  (any equal keys fill the remaining slots) and prefer_smaller_index emits the
//                  equal keys by the exact ballot rank, so both requirements stay exact.
// ------------------------------------------------------------------------------------------------
struct recover_exact
{};
struct recover_fast
{};

// Greater keys of word j (per-half masks ma) at the lane's running positions pa0 (row 0 or the
// single row) and pa1 (row 1), items item0 (low half) and item1 (high half).
__device__ __forceinline__ void
emit_greater2(uint32_t ma, int item0, int item1, int& pa0, int& pa1, int* out0, int* out1)
{
  const bool a0 = (ma & 0xffffu) != 0;
  const bool a1 = ma > 0xffffu;
  if (a0)
  {
    out0[pa0] = item0;
  }
  if (a1)
  {
    out1[pa1] = item1;
  }
  pa0 += a0;
  pa1 += a1;
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

template <typename KeyT, int LW, int N, int K, topk::requirement Req, class Ops, class Recover = recover_exact, class Net = net_w1>
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
  using net = typename Net::template bind<LW, M, T, Ops>;
  static_assert(net::R == items_per_thread, "tile words times tiles must cover the lane's words");
  static constexpr bool exact = std::is_same<Recover, recover_exact>::value;
  static constexpr bool psi   = Req == topk::requirement::prefer_smaller_index;
  // one scan carries the four counts in 8-bit fields when every prefix fits (N at most 255)
  static constexpr bool one_scan = N <= 255;

  // equal keys of every word in index order (word, lane) per row, by ballot rank, the first
  // K - count_greater of them: pe0 and pe1 enter as the greater totals of the rows
  __device__ __forceinline__ static void
  emit_equal_exact(const uint32_t (&me)[items_per_thread], int pe0, int pe1, int lane, const lanes<LW>& ln, int* out_idx)
  {
    int* out1 = out_idx + K;
#pragma unroll
    for (int j = 0; j < items_per_thread; ++j)
    {
      const int item    = j * LW + lane;
      const bool e0     = (me[j] & 0xffffu) != 0;
      const bool e1     = me[j] > 0xffffu;
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

  __device__ __forceinline__ static void
  emit_greater(const uint32_t (&ma)[items_per_thread], int pa0, int pa1, int lane, int* out_idx)
  {
    int* out1 = out_idx + K;
#pragma unroll
    for (int j = 0; j < items_per_thread; ++j)
    {
      const int item = j * LW + lane;
      emit_greater2(ma[j], item, item, pa0, pa1, out_idx, out1);
    }
  }

  // both classes by lane prefix, Cap: the equal class stops at K (not_guaranteed), else all of it
  template <bool Cap>
  __device__ __forceinline__ static void emit_fast(
    const uint32_t (&ma)[items_per_thread],
    const uint32_t (&me)[items_per_thread],
    int pa0,
    int pa1,
    int pe0,
    int pe1,
    int lane,
    int* out_idx)
  {
    int* out1 = out_idx + K;
#pragma unroll
    for (int j = 0; j < items_per_thread; ++j)
    {
      const int item = j * LW + lane;
      emit_greater2(ma[j], item, item, pa0, pa1, out_idx, out1);
      const bool e0 = (me[j] & 0xffffu) != 0;
      const bool e1 = me[j] > 0xffffu;
      if (e0 && (!Cap || pe0 < K))
      {
        out_idx[pe0] = item;
      }
      if (e1 && (!Cap || pe1 < K))
      {
        out1[pe1] = item;
      }
      pe0 += e0;
      pe1 += e1;
    }
  }

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

    // 3. classes per word and per-lane counts (row 0 low field, row 1 high field)
    uint32_t ma[IPT], me[IPT];
    uint32_t acc_a = 0, acc_e = 0;
#pragma unroll
    for (int j = 0; j < IPT; ++j)
    {
      ma[j] = hgt2_mask(in[j], t2);
      me[j] = heq2_mask(in[j], t2);
      acc_a += ma[j] & 0x00010001u;
      acc_e += me[j] & 0x00010001u;
    }

    if constexpr (exact)
    {
      const uint32_t incl = inclusive_scan<LW>(acc_a, lane);
      const uint32_t tot  = __shfl_sync(full_mask, incl, LW - 1, LW);
      const uint32_t excl = incl - acc_a;
      emit_greater(ma, excl & 0xffffu, excl >> 16, lane, out_idx);
      emit_equal_exact(me, tot & 0xffffu, tot >> 16, lane, ln, out_idx);
    }
    else
    {
      uint32_t xa, xe, ta, te; // exclusive prefixes and totals, two 16-bit fields each
      if constexpr (one_scan)
      {
        const uint32_t v    = acc_a | (acc_e << 8);
        const uint32_t incl = inclusive_scan<LW>(v, lane);
        const uint32_t tot  = __shfl_sync(full_mask, incl, LW - 1, LW);
        const uint32_t excl = incl - v;
        xa                  = excl & 0x00ff00ffu;
        xe                  = (excl >> 8) & 0x00ff00ffu;
        ta                  = tot & 0x00ff00ffu;
        te                  = (tot >> 8) & 0x00ff00ffu;
      }
      else
      {
        const uint32_t ia = inclusive_scan<LW>(acc_a, lane);
        const uint32_t ie = inclusive_scan<LW>(acc_e, lane);
        ta                = __shfl_sync(full_mask, ia, LW - 1, LW);
        te                = __shfl_sync(full_mask, ie, LW - 1, LW);
        xa                = ia - acc_a;
        xe                = ie - acc_e;
      }
      const int pa0 = xa & 0xffffu, pa1 = xa >> 16;
      const int pe0 = (ta & 0xffffu) + (xe & 0xffffu), pe1 = (ta >> 16) + (xe >> 16);
      // the tie class overflows the remaining slots in a row (warp-uniform, rare on random input):
      // prefer_smaller_index needs the index-ordered ballot rank, not_guaranteed the capped prefix
      const uint32_t need = ta + te; // per row: count_greater + count_equal, 16-bit fields
      const bool over     = (need & 0xffffu) > static_cast<uint32_t>(K) || (need >> 16) > static_cast<uint32_t>(K);
      // uniform over the hardware warp: the fallback runs full-warp collectives, so every logical
      // warp of the hardware warp takes it together (the fallback is exact for any group)
      if (__any_sync(full_mask, over))
      {
        if constexpr (psi)
        {
          emit_greater(ma, pa0, pa1, lane, out_idx);
          emit_equal_exact(me, ta & 0xffffu, ta >> 16, lane, ln, out_idx);
        }
        else
        {
          emit_fast<true>(ma, me, pa0, pa1, pe0, pe1, lane, out_idx);
        }
        return;
      }
      emit_fast<false>(ma, me, pa0, pa1, pe0, pe1, lane, out_idx);
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

template <typename KeyT, int LW, int N, int K, topk::requirement Req, class Ops, class Recover = recover_exact, class Net = net_w1>
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
  using net = typename Net::template bind<LW, M, T, Ops>;
  static_assert(net::R == words, "tile words times tiles must cover the lane's words");
  static constexpr bool exact = std::is_same<Recover, recover_exact>::value;
  static constexpr bool psi   = Req == topk::requirement::prefer_smaller_index;

  // equal keys in index order (word, lane, half) by ballot rank, pe enters as the greater total
  __device__ __forceinline__ static void
  emit_equal_exact(const uint32_t (&me)[words], int pe, int lane, const lanes<LW>& ln, int* out_idx)
  {
#pragma unroll
    for (int j = 0; j < words; ++j)
    {
      const int item0   = 2 * (j * LW + lane);
      const bool e0     = (me[j] & 0xffffu) != 0;
      const bool e1     = me[j] > 0xffffu;
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

  __device__ __forceinline__ static void emit_greater(const uint32_t (&ma)[words], int pa, int lane, int* out_idx)
  {
#pragma unroll
    for (int j = 0; j < words; ++j)
    {
      const int item0 = 2 * (j * LW + lane);
      const bool a0   = (ma[j] & 0xffffu) != 0;
      const bool a1   = ma[j] > 0xffffu;
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
    }
  }

  template <bool Cap>
  __device__ __forceinline__ static void
  emit_fast(const uint32_t (&ma)[words], const uint32_t (&me)[words], int pa, int pe, int lane, int* out_idx)
  {
#pragma unroll
    for (int j = 0; j < words; ++j)
    {
      const int item0 = 2 * (j * LW + lane);
      const bool a0   = (ma[j] & 0xffffu) != 0;
      const bool a1   = ma[j] > 0xffffu;
      const bool e0   = (me[j] & 0xffffu) != 0;
      const bool e1   = me[j] > 0xffffu;
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
      if (e0 && (!Cap || pe < K))
      {
        out_idx[pe] = item0;
      }
      pe += e0;
      if (e1 && (!Cap || pe < K))
      {
        out_idx[pe] = item0 + 1;
      }
      pe += e1;
    }
  }

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
    net::merge0(w, lane, full_mask);

    // 3. the k-th key, in both halves
    const uint32_t tw = kth_word<LW, K>(w);
    const uint32_t t2 = __byte_perm(tw, 0u, 0x3232);

    // 4. classes per word and per-lane counts (both halves are the same row)
    uint32_t ma[words], me[words];
    uint32_t acc_a = 0, acc_e = 0;
#pragma unroll
    for (int j = 0; j < words; ++j)
    {
      ma[j] = hgt2_mask(in[j], t2);
      me[j] = heq2_mask(in[j], t2);
      acc_a += ma[j] & 0x00010001u;
      acc_e += me[j] & 0x00010001u;
    }
    const uint32_t ca = (acc_a & 0xffffu) + (acc_a >> 16);
    const uint32_t ce = (acc_e & 0xffffu) + (acc_e >> 16);

    if constexpr (exact)
    {
      const uint32_t incl = inclusive_scan<LW>(ca, lane);
      const uint32_t tot  = __shfl_sync(full_mask, incl, LW - 1, LW);
      emit_greater(ma, incl - ca, lane, out_idx);
      emit_equal_exact(me, tot, lane, ln, out_idx);
    }
    else
    {
      const uint32_t v    = ca | (ce << 16);
      const uint32_t incl = inclusive_scan<LW>(v, lane);
      const uint32_t tot  = __shfl_sync(full_mask, incl, LW - 1, LW);
      const uint32_t excl = incl - v;
      const int pa        = excl & 0xffffu;
      const int ta        = tot & 0xffffu;
      const int pe        = ta + (excl >> 16);
      const bool over     = ta + (tot >> 16) > static_cast<uint32_t>(K);
      if (__any_sync(full_mask, over)) // uniform over the hardware warp, see rows_adapter
      {
        if constexpr (psi)
        {
          emit_greater(ma, pa, lane, out_idx);
          emit_equal_exact(me, ta, lane, ln, out_idx);
        }
        else
        {
          emit_fast<true>(ma, me, pa, pe, lane, out_idx);
        }
        return;
      }
      emit_fast<false>(ma, me, pa, pe, lane, out_idx);
    }
  }
};
} // namespace n5
