// B3: compacting radix select at block scope, index-only output.
//
// Radix select in the style of block_topk_air (histogram of one digit per pass over the
// candidates, fused scan and pivot choice), with one structural change: after the first pass
// the items strictly above the pivot bucket are final and their indices are written to the
// output at once, and the pivot bucket's items (key and index) are compacted into shared memory
// in index order. Every later pass sweeps the compacted candidates, which each thread holds in
// registers, instead of the whole tile. When the pivot bucket does not fit the candidate buffer
// (tie floods, quantized data) the primitive falls back to sweeping the register tile as
// block_topk_air does, so every input is handled.
//
// Positions come from block scans over per-thread class counts in blocked order, never from
// atomic tickets, so the compacted candidates and the final tie resolution are in index order
// and prefer_smaller_index (or prefer_larger_index) costs nothing beyond the not_guaranteed path.
//
// Digit schedule: the first pass reads FirstDigitBits bits (8 is the AIR digit, sign plus seven
// exponent bits for floats, 9 is sign plus the whole exponent), later passes read 8 bits each,
// the last one the remainder. The histogram has 1 << FirstDigitBits bins.
//
// Contract: full tile only (N = Threads x ItemsPerThread), blocked arrangement, k in [1, N),
// N <= 32768 (16-bit fields in the packed scans and candidate words). Keys fp32, bf16 or fp16.
// NaN: a positive NaN orders above +inf and is selected first, a negative NaN orders below -inf.
// -0.0 is normalized to +0.0 before ranking, so the two zeros tie and the index decides.
// Shared memory: two histograms, the scan storage, the pass state and N / CapDivisor candidate
// words (4 bytes for 16-bit keys, 8 bytes for 32-bit keys).
#pragma once

#include <cub/block/block_scan.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace b3
{

enum class tie_break
{
  any, // any valid set, resolved like smaller_index here because it is free
  smaller_index,
  larger_index
};

// Key to an unsigned word whose ascending order is the key's descending order ("ck"), so that
// the selection is a smallest-k selection on ck with digits read as plain shifts.
template <typename KeyT>
struct ck_traits;

template <>
struct ck_traits<float>
{
  static constexpr int key_bits = 32;
  __device__ __forceinline__ static ::cuda::std::uint32_t to_ck(float f)
  {
    // -0.0 + 0.0 is +0.0 under round to nearest, every other value is unchanged
    const ::cuda::std::uint32_t b = __float_as_uint(__fadd_rn(f, 0.0f));
    const ::cuda::std::uint32_t m = (b >> 31) - 1u; // positive: all ones, negative: zero
    return b ^ (m & 0x7FFFFFFFu);
  }
};

template <>
struct ck_traits<__nv_bfloat16>
{
  static constexpr int key_bits = 16;
  __device__ __forceinline__ static ::cuda::std::uint32_t to_ck(__nv_bfloat16 h)
  {
    ::cuda::std::uint32_t b = __bfloat16_as_ushort(h);
    b                       = (b == 0x8000u) ? 0u : b;
    const ::cuda::std::uint32_t m = (b >> 15) - 1u;
    return b ^ (m & 0x7FFFu);
  }
};

template <>
struct ck_traits<__half>
{
  static constexpr int key_bits = 16;
  __device__ __forceinline__ static ::cuda::std::uint32_t to_ck(__half h)
  {
    ::cuda::std::uint32_t b = __half_as_ushort(h);
    b                       = (b == 0x8000u) ? 0u : b;
    const ::cuda::std::uint32_t m = (b >> 15) - 1u;
    return b ^ (m & 0x7FFFu);
  }
};

// Compacted candidate word: 16-bit keys pack key and index into one word, 32-bit keys use two.
template <int KeyBits>
struct cand_traits;

template <>
struct cand_traits<16>
{
  using word_t = ::cuda::std::uint32_t;
  __device__ __forceinline__ static word_t make(::cuda::std::uint32_t ck, int idx)
  {
    return (ck << 16) | static_cast<::cuda::std::uint32_t>(idx);
  }
  __device__ __forceinline__ static ::cuda::std::uint32_t ck(word_t w)
  {
    return w >> 16;
  }
  __device__ __forceinline__ static int idx(word_t w)
  {
    return static_cast<int>(w & 0xFFFFu);
  }
};

template <>
struct cand_traits<32>
{
  using word_t = uint2;
  __device__ __forceinline__ static word_t make(::cuda::std::uint32_t ck, int idx)
  {
    return make_uint2(ck, static_cast<::cuda::std::uint32_t>(idx));
  }
  __device__ __forceinline__ static ::cuda::std::uint32_t ck(word_t w)
  {
    return w.x;
  }
  __device__ __forceinline__ static int idx(word_t w)
  {
    return static_cast<int>(w.y);
  }
};


// Guarded shared memory operations emitted as predicated PTX. nvcc lowers `if (c) store` to a
// branch in PTX and ptxas keeps the branch (BSSY, BRA, BSYNC per item) once the tile is large,
// for the reference primitive as much as for this one. A compare and a predicated instruction
// inside one asm statement cannot become a branch. The address and value are computed outside.
namespace detail
{
using u32 = ::cuda::std::uint32_t;

__device__ __forceinline__ u32 smem_addr(const void* p)
{
  return static_cast<u32>(__cvta_generic_to_shared(p));
}

// if (a == b && (AlwaysValid || va < vb)) *addr += 1, fire and forget
template <bool AlwaysValid>
__device__ __forceinline__ void red_inc_if_eq(u32 addr, u32 a, u32 b, u32 va, u32 vb)
{
  const u32 one = 1u;
  if constexpr (AlwaysValid)
  {
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "setp.eq.u32 p, %1, %2;\n\t"
                 "@p red.shared.add.u32 [%0], %3;\n\t}" ::"r"(addr),
                 "r"(a),
                 "r"(b),
                 "r"(one));
  }
  else
  {
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "setp.lt.u32 p, %4, %5;\n\t"
                 "setp.eq.and.u32 p, %1, %2, p;\n\t"
                 "@p red.shared.add.u32 [%0], %3;\n\t}" ::"r"(addr),
                 "r"(a),
                 "r"(b),
                 "r"(one),
                 "r"(va),
                 "r"(vb));
  }
}

// if (x < lo && (AlwaysValid || va < vb)) *addr = v
template <bool AlwaysValid>
__device__ __forceinline__ void sts_if_lt(u32 addr, u32 v, u32 x, u32 lo, u32 va, u32 vb)
{
  if constexpr (AlwaysValid)
  {
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "setp.lt.u32 p, %2, %3;\n\t"
                 "@p st.shared.b32 [%0], %1;\n\t}" ::"r"(addr),
                 "r"(v),
                 "r"(x),
                 "r"(lo));
  }
  else
  {
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "setp.lt.u32 p, %4, %5;\n\t"
                 "setp.lt.and.u32 p, %2, %3, p;\n\t"
                 "@p st.shared.b32 [%0], %1;\n\t}" ::"r"(addr),
                 "r"(v),
                 "r"(x),
                 "r"(lo),
                 "r"(va),
                 "r"(vb));
  }
}

// if (lo <= x <= hi && ra < rb && (AlwaysValid || va < vb)) *addr = v
template <bool AlwaysValid>
__device__ __forceinline__ void
sts_if_in(u32 addr, u32 v, u32 x, u32 lo, u32 hi, u32 ra, u32 rb, u32 va, u32 vb)
{
  if constexpr (AlwaysValid)
  {
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "setp.ge.u32 p, %2, %3;\n\t"
                 "setp.le.and.u32 p, %2, %4, p;\n\t"
                 "setp.lt.and.u32 p, %5, %6, p;\n\t"
                 "@p st.shared.b32 [%0], %1;\n\t}" ::"r"(addr),
                 "r"(v),
                 "r"(x),
                 "r"(lo),
                 "r"(hi),
                 "r"(ra),
                 "r"(rb));
  }
  else
  {
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "setp.ge.u32 p, %2, %3;\n\t"
                 "setp.le.and.u32 p, %2, %4, p;\n\t"
                 "setp.lt.and.u32 p, %5, %6, p;\n\t"
                 "setp.lt.and.u32 p, %7, %8, p;\n\t"
                 "@p st.shared.b32 [%0], %1;\n\t}" ::"r"(addr),
                 "r"(v),
                 "r"(x),
                 "r"(lo),
                 "r"(hi),
                 "r"(ra),
                 "r"(rb),
                 "r"(va),
                 "r"(vb));
  }
}

// if (lo <= x <= hi) *(uint2*) addr = {v0, v1}, and the 32-bit form for packed candidates
__device__ __forceinline__ void sts64_if_in(u32 addr, u32 v0, u32 v1, u32 x, u32 lo, u32 hi)
{
  asm volatile("{\n\t.reg .pred p;\n\t"
               "setp.ge.u32 p, %3, %4;\n\t"
               "setp.le.and.u32 p, %3, %5, p;\n\t"
               "@p st.shared.v2.b32 [%0], {%1, %2};\n\t}" ::"r"(addr),
               "r"(v0),
               "r"(v1),
               "r"(x),
               "r"(lo),
               "r"(hi));
}
__device__ __forceinline__ void sts32_if_in(u32 addr, u32 v, u32 x, u32 lo, u32 hi)
{
  asm volatile("{\n\t.reg .pred p;\n\t"
               "setp.ge.u32 p, %2, %3;\n\t"
               "setp.le.and.u32 p, %2, %4, p;\n\t"
               "@p st.shared.b32 [%0], %1;\n\t}" ::"r"(addr),
               "r"(v),
               "r"(x),
               "r"(lo),
               "r"(hi));
}
} // namespace detail

// How guarded per-item memory operations are written: `branchy` leaves them to the compiler
// (if statements around store and position increment), `predicated` uses the asm helpers above,
// `lean` keeps plain C++ but puts exactly one store inside each if and advances the positions
// with selects, the shape ptxas predicates without a reconvergence region.
enum class store_mode
{
  branchy,
  predicated,
  lean
};

// Candidate buffer layout. Thread t holds compacted items [t * cpt, (t + 1) * cpt) in both.
//   plain : item q at word q, the loads of a warp stride cpt words and conflict on the banks
//   padded: item q at word q + q / cpt (one pad word per thread), odd stride, conflict free
enum class cand_layout
{
  plain,
  padded
};

template <typename KeyT,
          int Threads,
          int ItemsPerThread,
          int CapDivisor           = 4,
          int FirstDigitBits       = 8,
          cand_layout Layout       = cand_layout::plain,
          store_mode Stores        = store_mode::branchy>
class block_topk_compact
{
public:
  using u32 = ::cuda::std::uint32_t;
  static constexpr bool predicated = (Stores == store_mode::predicated);
  static constexpr bool lean       = (Stores == store_mode::lean);

  static constexpr int threads     = Threads;
  static constexpr int ipt         = ItemsPerThread;
  static constexpr int tile_items  = Threads * ItemsPerThread;
  static constexpr bool compacts   = (CapDivisor > 0); // CapDivisor 0: never compact, tile sweeps only
  static constexpr int cap         = compacts ? tile_items / CapDivisor : 1; // candidates kept after pass 1
  static constexpr int cpt         = (cap + Threads - 1) / Threads; // candidates per thread, blocked
  static constexpr bool padded     = (Layout == cand_layout::padded) && (cpt > 1);
  static constexpr int cand_stride = padded ? cpt + 1 : cpt; // words per thread in the buffer
  static constexpr int cand_words  = padded ? cand_stride * Threads : cap;
  using ckt                        = ck_traits<KeyT>;
  static constexpr int key_bits    = ckt::key_bits;
  static constexpr u32 key_mask    = (key_bits == 32) ? 0xFFFFFFFFu : ((1u << key_bits) - 1u);
  static constexpr int first_bits  = FirstDigitBits;
  static constexpr int rest_bits   = 8;
  static constexpr int num_passes  = 1 + (key_bits - first_bits + rest_bits - 1) / rest_bits;
  static constexpr int num_buckets = 1 << first_bits;
  static constexpr int bpt         = (num_buckets + Threads - 1) / Threads;
  using cand                       = cand_traits<key_bits>;
  using cand_word_t                = typename cand::word_t;
  using scan_t                     = cub::BlockScan<u32, Threads, cub::BLOCK_SCAN_WARP_SCANS>;

  static_assert(tile_items <= 32768, "16-bit fields in the packed scans and candidate words");
  static_assert(cap >= 1, "the candidate buffer must hold at least one item");
  static_assert(first_bits >= rest_bits && first_bits < key_bits, "the first digit sizes the histogram");

  // Digit `p` covers bits [pass_shift(p), pass_shift(p) + pass_bits(p))
  __host__ __device__ static constexpr int pass_shift(int p)
  {
    int s = key_bits - first_bits;
    for (int i = 0; i < p; ++i)
    {
      s = (s > rest_bits) ? s - rest_bits : 0;
    }
    return s;
  }
  __host__ __device__ static constexpr int pass_bits(int p)
  {
    return ((p == 0) ? key_bits : pass_shift(p - 1)) - pass_shift(p);
  }

  struct TempStorage
  {
    u32 histogram[2][num_buckets];
    typename scan_t::TempStorage scan;
    uint2 state; // x = (selected << first_bits) | bucket, y = candidates in the bucket
    cand_word_t cands[cand_words];
  };

  __device__ __forceinline__ block_topk_compact(TempStorage& s)
      : storage(s)
      , tid(static_cast<int>(threadIdx.x))
  {}

  // Writes the k source indices of the k largest keys to out[0, k), unsorted. The tile's item
  // i of thread t has index t * ipt + i.
  template <tie_break TB>
  __device__ __forceinline__ void select_max_indices(const KeyT (&keys)[ipt], int k, int* out)
  {
    u32 ck[ipt];
#pragma unroll
    for (int i = 0; i < ipt; ++i)
    {
      ck[i] = ckt::to_ck(keys[i]);
    }

    zero_histogram(storage.histogram[0]);
    __syncthreads();

    int k_rem  = k;
    u32 prefix = 0;
    u32 mask   = 0;
    const int base      = tid * ipt;
    const auto tile_idx = [base](int i) {
      return base + i;
    };
    const always_valid always{};

    // Pass over the whole tile on the most significant digit
    const int c1 = run_pass<ipt>(ck, always, ::cuda::std::integral_constant<int, 0>{}, k_rem, prefix, mask);
    const int s1 = k - k_rem;

    if (c1 == k_rem)
    {
      // Every item in the pivot bucket is selected: emit above and pivot items in one sweep
      emit<ipt, TB>(ck, tile_idx, always, prefix, tie_hi(prefix, mask), k_rem, c1, 0, k, out);
    }
    else if (compacts && c1 <= cap)
    {
      compact(ck, prefix, tie_hi(prefix, mask), out);
      __syncthreads();

      u32 cck[cpt];
      int cidx[cpt];
#pragma unroll
      for (int j = 0; j < cpt; ++j)
      {
        const int pos = tid * cpt + j;
        cand_word_t w;
        if constexpr (padded)
        {
          w = (pos < c1) ? storage.cands[tid * cand_stride + j] : cand::make(0u, 0);
        }
        else
        {
          // Unconditional load from a clamped address, validity is tested where the item is used
          w = storage.cands[(pos < cap) ? pos : (cap - 1)];
        }
        cck[j] = cand::ck(w);
        cidx[j] = cand::idx(w);
      }
      const int cbase = tid * cpt;
      const range_valid cvalid{static_cast<u32>(cbase), static_cast<u32>(c1)};
      const auto cidx_of = [&cidx](int j) {
        return cidx[j];
      };
      finish<cpt, TB>(cck, cidx_of, cvalid, k_rem, prefix, mask, s1, k - s1, out);
    }
    else
    {
      // The pivot bucket is larger than the candidate buffer: sweep the register tile instead
      finish<ipt, TB>(ck, tile_idx, always, k_rem, prefix, mask, 0, k, out);
    }
  }

private:
  TempStorage& storage;
  int tid;

  // Validity of slot j of a register-resident set: every slot (the tile) or slot j valid when
  // lhs(j) < rhs (the compacted set, lhs = thread base + j, rhs = candidate count)
  struct always_valid
  {
    static constexpr bool always = true;
    __device__ __forceinline__ bool operator()(int) const
    {
      return true;
    }
    __device__ __forceinline__ u32 lhs(int) const
    {
      return 0u;
    }
    __device__ __forceinline__ u32 rhs() const
    {
      return 1u;
    }
  };
  struct range_valid
  {
    static constexpr bool always = false;
    u32 base;
    u32 count;
    __device__ __forceinline__ bool operator()(int j) const
    {
      return base + static_cast<u32>(j) < count;
    }
    __device__ __forceinline__ u32 lhs(int j) const
    {
      return base + static_cast<u32>(j);
    }
    __device__ __forceinline__ u32 rhs() const
    {
      return count;
    }
  };

  // Largest ck that still matches the prefix under the mask (the tie class is [prefix, tie_hi])
  __device__ __forceinline__ static u32 tie_hi(u32 prefix, u32 mask)
  {
    return prefix | (~mask & key_mask);
  }

  template <int Bins = num_buckets>
  __device__ __forceinline__ void zero_histogram(u32 (&hist)[num_buckets])
  {
#pragma unroll
    for (int b = tid; b < Bins; b += Threads)
    {
      hist[b] = 0;
    }
  }

  // Exclusive scan over the buckets, the thread whose bucket contains the k_rem-th item records
  // the pass state. Exactly one thread qualifies because the bucket counts sum to at least k_rem.
  template <int Bins>
  __device__ __forceinline__ void scan_and_choose(const u32 (&hist)[num_buckets], int k_rem)
  {
    constexpr int items = (Bins + Threads - 1) / Threads;
    u32 counts[items];
    u32 excl[items];
#pragma unroll
    for (int i = 0; i < items; ++i)
    {
      const int b = tid * items + i;
      counts[i]   = (b < Bins) ? hist[b] : 0u;
    }
    scan_t(storage.scan).ExclusiveSum(counts, excl);
#pragma unroll
    for (int i = 0; i < items; ++i)
    {
      const int b   = tid * items + i;
      const u32 e   = excl[i];
      const u32 inc = e + counts[i];
      if (b < Bins && e < static_cast<u32>(k_rem) && inc >= static_cast<u32>(k_rem))
      {
        storage.state = make_uint2((e << first_bits) | static_cast<u32>(b), counts[i]);
      }
    }
  }

  // One radix pass over a register-resident set: histogram of digit `pass` over the valid items
  // matching the current prefix, scan and pivot choice, pass-state update. Returns the number of
  // candidates (items in the pivot bucket) and narrows k_rem, prefix and mask.
  template <int M, int Pass, class ValidF>
  __device__ __forceinline__ int
  run_pass(const u32 (&ck)[M], ValidF valid, ::cuda::std::integral_constant<int, Pass>, int& k_rem, u32& prefix, u32& mask)
  {
    constexpr int pass       = Pass;
    constexpr int shift      = pass_shift(pass);
    constexpr int bins       = 1 << pass_bits(pass);
    constexpr int next_bins  = (pass + 1 < num_passes) ? (1 << pass_bits(pass + 1 < num_passes ? pass + 1 : pass)) : 0;
    const u32 digit_mask     = static_cast<u32>(bins - 1);
    u32(&hist)[num_buckets]  = storage.histogram[pass & 1];
    if constexpr (predicated)
    {
      const u32 hist_addr = detail::smem_addr(hist);
#pragma unroll
      for (int j = 0; j < M; ++j)
      {
        const u32 bin = (ck[j] >> shift) & digit_mask;
        detail::red_inc_if_eq<ValidF::always>(hist_addr + bin * 4u, ck[j] & mask, prefix, valid.lhs(j), valid.rhs());
      }
    }
    else
    {
#pragma unroll
      for (int j = 0; j < M; ++j)
      {
        if (valid(j) && (ck[j] & mask) == prefix)
        {
          atomicAdd_block(&hist[(ck[j] >> shift) & digit_mask], 1u);
        }
      }
    }
    if constexpr (pass + 1 < num_passes)
    {
      zero_histogram<next_bins>(storage.histogram[(pass + 1) & 1]);
    }
    __syncthreads();

    scan_and_choose<bins>(hist, k_rem);
    __syncthreads();

    const uint2 st     = storage.state;
    const int bucket   = static_cast<int>(st.x & (num_buckets - 1));
    const int selected = static_cast<int>(st.x >> first_bits);
    k_rem -= selected;
    prefix |= static_cast<u32>(bucket) << shift;
    mask |= digit_mask << shift;
    return static_cast<int>(st.y);
  }

  // Remaining passes over a register-resident set, then the final emission. out_base output slots
  // were already written by the caller, k_set outputs come from this set.
  template <int M, tie_break TB, class IdxF, class ValidF>
  __device__ __forceinline__ void finish(
    const u32 (&ck)[M], IdxF idx_of, ValidF valid, int k_rem, u32 prefix, u32 mask, int out_base, int k_set, int* out)
  {
    int cands = 0;
    bool done = false; // block-uniform early exit (all remaining candidates are selected)
    const auto step = [&](auto pc) {
      if (!done)
      {
        cands = run_pass<M>(ck, valid, pc, k_rem, prefix, mask);
        done  = (cands == k_rem);
      }
    };
    step(::cuda::std::integral_constant<int, 1>{});
    if constexpr (num_passes > 2)
    {
      step(::cuda::std::integral_constant<int, 2>{});
    }
    if constexpr (num_passes > 3)
    {
      step(::cuda::std::integral_constant<int, 3>{});
    }
    static_assert(num_passes <= 4, "extend the pass sequence for more than four passes");
    emit<M, TB>(ck, idx_of, valid, prefix, tie_hi(prefix, mask), k_rem, cands, out_base, k_set, out);
  }

  // After the first pass: indices of the items above the pivot bucket go to out in index order,
  // the pivot bucket's items are compacted into shared memory in index order.
  __device__ __forceinline__ void compact(const u32 (&ck)[ipt], u32 lo, u32 hi_incl, int* out)
  {
    u32 n_above = 0;
    u32 n_le    = 0;
#pragma unroll
    for (int i = 0; i < ipt; ++i)
    {
      n_above += (ck[i] < lo) ? 1u : 0u;
      n_le += (ck[i] <= hi_incl) ? 1u : 0u;
    }
    u32 excl;
    scan_t(storage.scan).ExclusiveSum((n_above << 16) | (n_le - n_above), excl);
    u32 pos_above  = excl >> 16;
    u32 pos_eq     = excl & 0xFFFFu;
    const int base = tid * ipt;
    if constexpr (predicated)
    {
      const u32 out_addr  = detail::smem_addr(out);
      const u32 cand_addr = detail::smem_addr(storage.cands);
#pragma unroll
      for (int i = 0; i < ipt; ++i)
      {
        const u32 idxv  = static_cast<u32>(base + i);
        const bool above = ck[i] < lo;
        const bool eq    = !above && (ck[i] <= hi_incl);
        const u32 word  = padded ? pos_eq + pos_eq / static_cast<u32>(cpt) : pos_eq;
        detail::sts_if_lt<true>(out_addr + pos_above * 4u, idxv, ck[i], lo, 0u, 1u);
        if constexpr (sizeof(cand_word_t) == 8)
        {
          detail::sts64_if_in(cand_addr + word * 8u, ck[i], idxv, ck[i], lo, hi_incl);
        }
        else
        {
          detail::sts32_if_in(cand_addr + word * 4u, cand::make(ck[i], base + i), ck[i], lo, hi_incl);
        }
        pos_above += above ? 1u : 0u;
        pos_eq += eq ? 1u : 0u;
      }
    }
    else if constexpr (lean)
    {
#pragma unroll
      for (int i = 0; i < ipt; ++i)
      {
        const int idxv   = base + i;
        const bool above = ck[i] < lo;
        const bool eq    = !above && (ck[i] <= hi_incl);
        const u32 word   = padded ? pos_eq + pos_eq / static_cast<u32>(cpt) : pos_eq;
        int* dst_above   = out + pos_above;
        cand_word_t* dst = storage.cands + word;
        if (above)
        {
          *dst_above = idxv;
        }
        if (eq)
        {
          *dst = cand::make(ck[i], idxv);
        }
        pos_above += above ? 1u : 0u;
        pos_eq += eq ? 1u : 0u;
      }
    }
    else
    {
#pragma unroll
      for (int i = 0; i < ipt; ++i)
      {
        const bool above = ck[i] < lo;
        const bool eq    = !above && (ck[i] <= hi_incl);
        if (above)
        {
          out[pos_above++] = base + i;
        }
        if (eq)
        {
          const u32 word = padded ? pos_eq + pos_eq / static_cast<u32>(cpt) : pos_eq;
          storage.cands[word] = cand::make(ck[i], base + i);
          ++pos_eq;
        }
      }
    }
  }

  // Final emission over a register-resident set: items with (ck & mask) < prefix are selected,
  // items with (ck & mask) == prefix are ties of which need_ties are taken in index order.
  template <int M, tie_break TB, class IdxF, class ValidF>
  __device__ __forceinline__ void emit(
    const u32 (&ck)[M],
    IdxF idx_of,
    ValidF valid,
    u32 lo,
    u32 hi_incl,
    int need_ties,
    int total_ties,
    int out_base,
    int k_set,
    int* out)
  {
    u32 n_above = 0;
    u32 n_le    = 0;
#pragma unroll
    for (int j = 0; j < M; ++j)
    {
      const bool v = valid(j);
      n_above += (v && ck[j] < lo) ? 1u : 0u;
      n_le += (v && ck[j] <= hi_incl) ? 1u : 0u;
    }
    u32 excl;
    scan_t(storage.scan).ExclusiveSum((n_above << 16) | (n_le - n_above), excl);
    int pos_above       = static_cast<int>(excl >> 16);
    int pos_tie         = static_cast<int>(excl & 0xFFFFu);
    const int sel_total = k_set - need_ties;
    int* out_above      = out + out_base;
    int* out_tie        = out + out_base + sel_total;
    if constexpr (predicated)
    {
      const u32 above_addr = detail::smem_addr(out_above);
      const u32 tie_addr   = detail::smem_addr(out_tie);
#pragma unroll
      for (int j = 0; j < M; ++j)
      {
        const bool v     = valid(j);
        const bool above = v && (ck[j] < lo);
        const bool tie   = v && !above && (ck[j] <= hi_incl);
        const int rank   = (TB == tie_break::larger_index) ? (total_ties - 1 - pos_tie) : pos_tie;
        const u32 idxv   = static_cast<u32>(idx_of(j));
        detail::sts_if_lt<ValidF::always>(above_addr + static_cast<u32>(pos_above) * 4u, idxv, ck[j], lo, valid.lhs(j), valid.rhs());
        detail::sts_if_in<ValidF::always>(
          tie_addr + static_cast<u32>(rank) * 4u,
          idxv,
          ck[j],
          lo,
          hi_incl,
          static_cast<u32>(rank),
          static_cast<u32>(need_ties),
          valid.lhs(j),
          valid.rhs());
        pos_above += above ? 1 : 0;
        pos_tie += tie ? 1 : 0;
      }
    }
    else if constexpr (lean)
    {
#pragma unroll
      for (int j = 0; j < M; ++j)
      {
        const bool v     = valid(j);
        const bool above = v && (ck[j] < lo);
        const bool tie   = v && !above && (ck[j] <= hi_incl);
        const int rank   = (TB == tie_break::larger_index) ? (total_ties - 1 - pos_tie) : pos_tie;
        const int idxv   = idx_of(j);
        int* dst_above   = out_above + pos_above;
        int* dst_tie     = out_tie + rank;
        if (above)
        {
          *dst_above = idxv;
        }
        if (tie && rank < need_ties)
        {
          *dst_tie = idxv;
        }
        pos_above += above ? 1 : 0;
        pos_tie += tie ? 1 : 0;
      }
    }
    else
    {
#pragma unroll
      for (int j = 0; j < M; ++j)
      {
        const bool v     = valid(j);
        const bool above = v && (ck[j] < lo);
        const bool tie   = v && !above && (ck[j] <= hi_incl);
        if (above)
        {
          out_above[pos_above++] = idx_of(j);
        }
        if (tie)
        {
          const int rank = (TB == tie_break::larger_index) ? (total_ties - 1 - pos_tie) : pos_tie;
          if (rank < need_ties)
          {
            out_tie[rank] = idx_of(j);
          }
          ++pos_tie;
        }
      }
    }
  }
};

} // namespace b3
