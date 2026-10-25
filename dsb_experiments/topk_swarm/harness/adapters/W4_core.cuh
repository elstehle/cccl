// W4 core: k-truncated merge of sorted runs (approach W4). Shared by the block chunk-merge
// adapters (W4_chunk4, W4_chunk8) and the warp truncated merge sort (W4_wmerge). This header
// has no CUDA dependency beyond the intrinsics behind W4_FN, so approaches/W4/host_test.cpp
// compiles it as plain C++ and checks the merge tree against std::sort.
//
// Words. Every item is one unsigned word whose integer order is (key descending, index
// ascending): fp32 keys become a 64-bit word (order preserving twiddle of the key in the high
// half, the bit-inverted source index in the low half), bf16 keys a 32-bit word (16 twiddled key
// bits above the inverted 16-bit index). Minus zero is folded onto plus zero before the twiddle.
// The all-zero word is below every real word and serves as the absent-item filler.
//
// Merge tree. The input is R sorted runs of length RunLen (the per-chunk top-k results of a
// batched kernel, or the per-lane sorted runs of the warp sort). Level l merges adjacent pairs
// of runs into runs of out_len = min(K, 2 x in_len) items, so the tree never carries more than
// k items per run, and the last level emits the k largest of the union, sorted. Threads split a
// pair's output evenly: thread j of a pair owns output positions [j x OPT, (j + 1) x OPT). It
// finds the merge-path split of its end diagonal with a static-count binary search (T from
// reference/WARP_MERGE_SORT_RESULTS.md section 4: a fixed trip count keeps the group converged),
// takes its start split from its predecessor (shared memory exchange at block scope, shfl_up at
// warp scope), so it knows exactly how many items it consumes from each run, and merges them in
// chunks of C: load up to C items from each side (loads beyond the consumed count predicated
// off, the register preset to the filler), take the C largest with one half-cleaner stage and a
// bitonic merge of log2(C) stages (all static register indices), and advance both heads by the
// counted wins. Runs live in shared memory as 32-bit arrays (fp32: key half and index half apart, the
// level 0 runs of the chunk merge carry no index array since their index is the position) in a
// layout swizzled on aligned groups of 32 words of the absolute position: threads of a pair stride
// through a run at about OPT / 2 words per thread and runs start at multiples of 32, which
// unswizzled maps 16 threads or 16 runs onto the same banks.
#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__CUDACC__)
#  define W4_FN __device__ __forceinline__
#  define W4_HD __host__ __device__
#  define W4_UNROLL _Pragma("unroll")
#  define W4_NOUNROLL _Pragma("unroll 1")
#else
#  define W4_FN inline
#  define W4_HD
#  define W4_UNROLL
#  define W4_NOUNROLL
#endif

namespace w4
{

W4_HD inline constexpr int cmin(int a, int b)
{
  return a < b ? a : b;
}
W4_HD inline constexpr int cmax(int a, int b)
{
  return a > b ? a : b;
}
W4_HD inline constexpr bool is_pow2(int x)
{
  return x > 0 && (x & (x - 1)) == 0;
}
W4_HD inline constexpr int log2_ceil(int x)
{
  int r = 0;
  while ((1 << r) < x)
  {
    ++r;
  }
  return r;
}
W4_HD inline constexpr int round_up(int x, int m)
{
  return (x + m - 1) / m * m;
}

// Widest chunk of the register merge (two chunks of CHUNK words live per thread).
inline constexpr int CHUNK = 16;

// ------------------------------------------------------------------------------------------------
// Twiddles and words
// ------------------------------------------------------------------------------------------------
W4_FN uint32_t twiddle_f32(float f)
{
#if defined(__CUDA_ARCH__)
  const uint32_t u = __float_as_uint(__fadd_rn(f, 0.0f));
#else
  const float g = f + 0.0f;
  uint32_t u;
  memcpy(&u, &g, sizeof(u));
#endif
  return u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
}

// bf16 given as its 16 raw bits.
W4_FN uint32_t twiddle_bf16(uint32_t b)
{
  const uint32_t u = (b == 0x8000u) ? 0u : b;
  return u ^ 0x8000u ^ ((u >> 15) * 0x7fffu);
}

template <typename W>
struct word_ops;

template <>
struct word_ops<uint64_t>
{
  W4_FN static uint64_t make(uint32_t tw, int idx)
  {
    return (static_cast<uint64_t>(tw) << 32) | static_cast<uint32_t>(~idx);
  }
  W4_FN static uint64_t from_parts(uint32_t hi, uint32_t lo)
  {
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }
  W4_FN static uint32_t hi(uint64_t w)
  {
    return static_cast<uint32_t>(w >> 32);
  }
  W4_FN static uint32_t lo(uint64_t w)
  {
    return static_cast<uint32_t>(w);
  }
  W4_FN static int index(uint64_t w)
  {
    return static_cast<int>(~static_cast<uint32_t>(w));
  }
};

template <>
struct word_ops<uint32_t>
{
  W4_FN static uint32_t make(uint32_t tw, int idx)
  {
    return (tw << 16) | (static_cast<uint32_t>(~idx) & 0xffffu);
  }
  W4_FN static int index(uint32_t w)
  {
    return static_cast<int>((~w) & 0xffffu);
  }
};

// Every run buffer is an array of 32-bit words (bf16: the whole word, fp32: the key half and the
// index half in two arrays), so the bank is the word position mod 32 and the swizzle group is 32
// words. The swizzle is applied to the absolute position in the buffer: the low five bits are
// xored with the next five, a permutation inside every aligned group of 32, so runs that start
// at multiples of 32 keep their extent and two runs at the same relative position do not share
// a bank.
inline constexpr int G = 32;
W4_FN int sw(int i)
{
  return i ^ ((i >> 5) & 31);
}

// Descending compare-exchange: two VIMNMX on 32-bit words (T42), compare and select on 64-bit.
template <typename W>
W4_FN void cas(W& hi, W& lo)
{
  const bool gt = hi > lo;
  const W a     = gt ? hi : lo;
  const W b     = gt ? lo : hi;
  hi            = a;
  lo            = b;
}

// ------------------------------------------------------------------------------------------------
// Run buffer access policies. key_at(p) is what the merge path compares (strict greater means
// "precedes"), word_at(p, idx) the full word for the chunk network, store(p, w) the write.
//   run_words:       one array of W words (bf16), key_at is the word itself
//   run_split<Imp>:  fp32, key array plus index-half array, Imp: the index half is implicit from
//                    the item position (level 0 of the chunk merge, whose runs are in index order)
// ------------------------------------------------------------------------------------------------
// Layout: Skew false is the xor swizzle on the absolute position (block form: the lanes of one
// warp stride through one run), Skew true is linear addressing with an odd run stride (warp
// form: the lanes read different runs at the same relative position, the stride spreads the runs
// over the banks and every access is one instruction with an immediate offset).
template <bool Skew>
W4_FN int at(int p)
{
  if constexpr (Skew)
  {
    return p;
  }
  else
  {
    return sw(p);
  }
}

template <typename W, bool Skew>
struct run_words
{
  W* w;
  W4_FN W key_at(int p) const
  {
    return w[at<Skew>(p)];
  }
  W4_FN W word_at(int p, int) const
  {
    return w[at<Skew>(p)];
  }
  W4_FN void store(int p, W x) const
  {
    w[at<Skew>(p)] = x;
  }
  W4_FN void store_pad(int p) const
  {
    w[at<Skew>(p)] = 0;
  }
};

template <bool Implicit, bool Skew>
struct run_split
{
  uint32_t* key;
  uint32_t* lo;
  W4_FN uint32_t key_at(int p) const
  {
    return key[at<Skew>(p)];
  }
  W4_FN uint64_t word_at(int p, int idx) const
  {
    if constexpr (Implicit)
    {
      return word_ops<uint64_t>::from_parts(key[at<Skew>(p)], static_cast<uint32_t>(~idx));
    }
    else
    {
      return word_ops<uint64_t>::from_parts(key[at<Skew>(p)], lo[at<Skew>(p)]);
    }
  }
  W4_FN void store(int p, uint64_t x) const
  {
    key[at<Skew>(p)] = word_ops<uint64_t>::hi(x);
    lo[at<Skew>(p)]  = word_ops<uint64_t>::lo(x);
  }
  // a pad word: key zero is below every real key, the index half does not matter
  W4_FN void store_pad(int p) const
  {
    key[at<Skew>(p)] = 0;
  }
};

// ------------------------------------------------------------------------------------------------
// Batcher odd-even merge sort of a register array, descending, static indices only (as B2)
// ------------------------------------------------------------------------------------------------
template <int IPT, int Lo, int Nn, int R, typename W>
W4_FN void oe_merge(W (&v)[IPT])
{
  constexpr int M = R * 2;
  if constexpr (M < Nn)
  {
    oe_merge<IPT, Lo, Nn, M>(v);
    oe_merge<IPT, Lo + R, Nn, M>(v);
    W4_UNROLL
    for (int i = Lo + R; i + R < Lo + Nn; i += M)
    {
      cas(v[i], v[i + R]);
    }
  }
  else
  {
    cas(v[Lo], v[Lo + R]);
  }
}

template <int IPT, int Lo, int Nn, typename W>
W4_FN void oe_sort(W (&v)[IPT])
{
  if constexpr (Nn > 1)
  {
    constexpr int M = Nn / 2;
    oe_sort<IPT, Lo, M>(v);
    oe_sort<IPT, Lo + M, M>(v);
    oe_merge<IPT, Lo, Nn, 1>(v);
  }
}

template <int IPT, typename W>
W4_FN void sort_desc(W (&v)[IPT])
{
  oe_sort<IPT, 0, IPT>(v);
}

// ------------------------------------------------------------------------------------------------
// Merge path on two descending runs at absolute buffer positions a_base and b_base, the first
// run wins ties (its indices are smaller, so this is the word order). Returns the number of
// items of A among the first d of the merge. Steps is a compile-time bound on
// ceil(log2(range + 1)), the loop runs it fully with a predicated body. The probe of a finished
// search (lo == hi) stays inside the buffer: A's position is at most len_a (the next run or the
// trailing pad group), B's is clamped at 0.
// ------------------------------------------------------------------------------------------------
template <int Steps, typename IO>
W4_FN int merge_path(const IO& A, int a_base, const IO& B, int b_base, int len_a, int len_b, int d)
{
  int lo = cmax(d - len_b, 0);
  int hi = cmin(d, len_a);
  W4_UNROLL
  for (int s = 0; s < Steps; ++s)
  {
    const int mid      = (lo + hi) >> 1;
    const bool go      = lo < hi;
    const auto a       = A.key_at(a_base + mid);
    const auto b       = B.key_at(b_base + cmax(d - 1 - mid, 0));
    const bool b_first = b > a;
    hi                 = (go & b_first) ? mid : hi;
    lo                 = (go & !b_first) ? mid + 1 : lo;
  }
  return lo;
}

// 4-ary form: three probes per step at the quarter points, six independent loads in flight per
// step, the range shrinks by four, Steps is a bound on ceil(log4(range + 1)). Same result.
template <int Steps, typename IO>
W4_FN int merge_path4(const IO& A, int a_base, const IO& B, int b_base, int len_a, int len_b, int d)
{
  int lo = cmax(d - len_b, 0);
  int hi = cmin(d, len_a);
  W4_UNROLL
  for (int s = 0; s < Steps; ++s)
  {
    const int r   = hi - lo;
    const int m1  = lo + (r >> 2);
    const int m2  = lo + (r >> 1);
    const int m3  = lo + ((3 * r) >> 2);
    const bool go = lo < hi;
    const auto a1 = A.key_at(a_base + m1);
    const auto b1 = B.key_at(b_base + cmax(d - 1 - m1, 0));
    const auto a2 = A.key_at(a_base + m2);
    const auto b2 = B.key_at(b_base + cmax(d - 1 - m2, 0));
    const auto a3 = A.key_at(a_base + m3);
    const auto b3 = B.key_at(b_base + cmax(d - 1 - m3, 0));
    const bool p1 = b1 > a1;
    const bool p2 = b2 > a2;
    const bool p3 = b3 > a3;
    // first true among p1, p2, p3 bounds the range from above, the last false from below
    const int nhi = p1 ? m1 : (p2 ? m2 : (p3 ? m3 : hi));
    const int nlo = p1 ? lo : (p2 ? m1 + 1 : (p3 ? m2 + 1 : m3 + 1));
    hi            = go ? nhi : hi;
    lo            = go ? nlo : lo;
  }
  return lo;
}

// ------------------------------------------------------------------------------------------------
// Chunked register merge. a[] and b[] hold up to C items each of two descending runs (absent
// items are 0). o[] receives the C largest, descending. Returns how many came from a[].
// ------------------------------------------------------------------------------------------------
// Sum of C flags as a balanced tree (the serial form is a C-deep dependent chain on the path to
// the next chunk's addresses).
template <int C>
W4_FN int tree_sum(const int (&f)[C])
{
  if constexpr (C == 1)
  {
    return f[0];
  }
  else
  {
    int h[C / 2];
    W4_UNROLL
    for (int i = 0; i < C / 2; ++i)
    {
      h[i] = f[2 * i] + f[2 * i + 1];
    }
    return tree_sum<C / 2>(h);
  }
}

template <int C, typename W>
W4_FN int top_chunk(const W (&a)[C], const W (&b)[C], W (&o)[C])
{
  int aw[C];
  W4_UNROLL
  for (int i = 0; i < C; ++i)
  {
    const W x  = a[i];
    const W y  = b[C - 1 - i];
    const bool g = x > y;
    o[i]       = g ? x : y;
    aw[i]      = g ? 1 : 0;
  }
  W4_UNROLL
  for (int s = C / 2; s >= 1; s >>= 1)
  {
    W4_UNROLL
    for (int i = 0; i < C; ++i)
    {
      if ((i & s) == 0)
      {
        cas(o[i], o[i + s]);
      }
    }
  }
  return tree_sum<C>(aw);
}

// One chunk: load C items from each run head, keep the C largest, store, advance the heads.
// Exact: loads beyond the consumed count are predicated off (the register preset to the filler),
// else the loads are unconditional and read the run's zero pad beyond its end.
template <int C, bool ToIndices, bool Exact, typename IO, typename OutIO>
W4_FN void merge_chunk(const IO& in, int a_base, int a_idx0, int b_base, int b_idx0, int& a_head, int& b_head,
                       int a_end, int b_end, const OutIO& out, int out_at, int* out_idx, int idx_at)
{
  using W = decltype(in.word_at(0, 0));
  W a[C], b[C], o[C];
  W4_UNROLL
  for (int i = 0; i < C; ++i)
  {
    if constexpr (Exact)
    {
      a[i] = 0;
      b[i] = 0;
      if (a_head + i < a_end)
      {
        a[i] = in.word_at(a_base + a_head + i, a_idx0 + a_head + i);
      }
      if (b_head + i < b_end)
      {
        b[i] = in.word_at(b_base + b_head + i, b_idx0 + b_head + i);
      }
    }
    else
    {
      a[i] = in.word_at(a_base + a_head + i, a_idx0 + a_head + i);
      b[i] = in.word_at(b_base + b_head + i, b_idx0 + b_head + i);
    }
  }
  const int wins = top_chunk<C>(a, b, o);
  W4_UNROLL
  for (int i = 0; i < C; ++i)
  {
    if constexpr (ToIndices)
    {
      out_idx[idx_at + i] = word_ops<W>::index(o[i]);
    }
    else
    {
      out.store(out_at + i, o[i]);
    }
  }
  a_head += wins;
  b_head += C - wins;
}

// Merge exactly [a_beg, a_end) of A with [b_beg, b_end) of B, OPT = (a_end - a_beg) +
// (b_end - b_beg) outputs in chunks of C. a_idx0 and b_idx0 are the first item indices of the
// runs (used by the implicit-index policy only). ToIndices: write source indices to out_idx at
// out_pos, else words to the output buffer at absolute position out_base + out_pos. The chunk
// loop is rolled when there are several chunks: the chunks are a dependent chain anyway (the
// next chunk's addresses need this chunk's win count) and one body keeps the code inside the
// instruction cache (T48), every register index inside the body is static.
template <int C, int OPT, bool ToIndices, bool Exact, typename IO, typename OutIO>
W4_FN void merge_top(const IO& in, int a_base, int a_idx0, int b_base, int b_idx0, int a_beg, int a_end, int b_beg,
                     int b_end, const OutIO& out, int out_base, int out_pos, int* out_idx)
{
  int a_head = a_beg;
  int b_head = b_beg;
  if constexpr (OPT == C)
  {
    merge_chunk<C, ToIndices, Exact>(in, a_base, a_idx0, b_base, b_idx0, a_head, b_head, a_end, b_end, out,
                                     out_base + out_pos, out_idx, out_pos);
  }
  else
  {
    W4_NOUNROLL
    for (int c = 0; c < OPT / C; ++c)
    {
      merge_chunk<C, ToIndices, Exact>(in, a_base, a_idx0, b_base, b_idx0, a_head, b_head, a_end, b_end, out,
                                       out_base + out_pos + c * C, out_idx, out_pos + c * C);
    }
  }
}

// Shape check shared by merge_tree::valid() and the adapters' supported() (which may not
// instantiate the tree): every level must split evenly over the threads with a power-of-two
// or CHUNK-multiple output count per thread, and the last level must emit exactly k.
W4_HD inline constexpr bool tree_valid(int threads, int runs, int run_len, int k)
{
  if (!is_pow2(runs) || runs < 2 || !is_pow2(run_len) || !is_pow2(k) || !is_pow2(threads))
  {
    return false;
  }
  if (k < 1 || 2 * k > runs * run_len)
  {
    return false;
  }
  const int levels = log2_ceil(runs);
  int in           = run_len;
  int out          = 0;
  for (int l = 0; l < levels; ++l)
  {
    out             = cmin(k, 2 * in);
    const int pairs = (runs >> l) / 2;
    if (pairs > threads || threads % pairs != 0)
    {
      return false;
    }
    const int tpp = threads / pairs;
    if (out % tpp != 0)
    {
      return false;
    }
    const int o = out / tpp;
    if (o < 1 || (o < CHUNK ? !is_pow2(o) : (o % CHUNK) != 0))
    {
      return false;
    }
    in = out;
  }
  return out == k;
}

// Temp storage bytes of a merge tree with the chunk-merge layout (exact counts, xor swizzle, no
// pad), computed without instantiating it, for the adapters' supported(): the two key buffers,
// the two index-half buffers of the split form (none at level 0 when the index is implicit) and
// the split exchange. Equal to sizeof(merge_tree<...>::TempStorage) for the chunk adapters
// (checked by approaches/W4/host_test.cpp).
W4_HD inline constexpr int chunk_temp_bytes(int threads, int runs, int run_len, int k, bool split, bool implicit0)
{
  const int levels = log2_ceil(runs);
  int words0 = 1, words1 = 1, lo0 = 1, lo1 = 1;
  int in     = run_len;
  for (int l = 0; l < levels; ++l)
  {
    const int w = (runs >> l) * round_up(in, G) + G;
    if (l % 2 == 0)
    {
      words0 = cmax(words0, w);
      if (split && (l >= 2 || !implicit0))
      {
        lo0 = cmax(lo0, w);
      }
    }
    else
    {
      words1 = cmax(words1, w);
      if (split)
      {
        lo1 = cmax(lo1, w);
      }
    }
    in = cmin(k, 2 * in);
  }
  const int bytes = 4 * (round_up(words0, 4) + round_up(words1, 4) + round_up(lo0, 4) + round_up(lo1, 4)) + 4 * threads;
  return round_up(bytes, 16);
}

// ------------------------------------------------------------------------------------------------
// The merge tree: Runs sorted runs of RunLen words, Threads cooperating threads, k largest out.
// Level l reads the buffers of parity l % 2 and writes the other parity (the last level writes
// indices). Split: fp32 key and index halves in separate arrays. Implicit0: the level 0 runs are
// in index order and carry no index array (the chunk merge).
// ------------------------------------------------------------------------------------------------
// Exact: every thread learns its end split from its successor's search (one exchange) and
// predicates its loads, no pads (block form). Not exact: every thread searches its own start
// diagonal only, loads unconditionally and the runs carry a zero pad of CHUNK words (warp form).
template <typename W, bool Split, bool Implicit0, int Threads, int Runs, int RunLen, int K, bool Quad = false,
          bool Exact = true, bool Skew = false>
struct merge_tree
{
  static constexpr int levels = log2_ceil(Runs);
  static constexpr int PAD    = Exact ? 0 : CHUNK;

  W4_HD static constexpr int in_len(int l)
  {
    return l == 0 ? RunLen : cmin(K, 2 * in_len(l - 1));
  }
  W4_HD static constexpr int out_len(int l)
  {
    return cmin(K, 2 * in_len(l));
  }
  W4_HD static constexpr int runs_at(int l)
  {
    return Runs >> l;
  }
  W4_HD static constexpr int pairs_at(int l)
  {
    return runs_at(l) / 2;
  }
  W4_HD static constexpr int tpp(int l)
  {
    return Threads / pairs_at(l);
  }
  W4_HD static constexpr int opt(int l)
  {
    return out_len(l) / tpp(l);
  }
  W4_HD static constexpr int chunk(int l)
  {
    return cmin(CHUNK, opt(l));
  }
  // run stride in words of level l's input buffer: the run plus its pad, in whole groups of 32
  // for the swizzled layout, one word more than the run and its pad for the skewed one (odd
  // stride, consecutive runs one bank apart)
  W4_HD static constexpr int stride(int l)
  {
    return Skew ? in_len(l) + PAD + 1 : round_up(in_len(l) + PAD, G);
  }
  // words of level l's input buffer, one trailing group for the finished-search probe
  W4_HD static constexpr int in_words(int l)
  {
    return runs_at(l) * stride(l) + G;
  }
  W4_HD static constexpr int buf_words(int parity, int from)
  {
    int m = 1;
    for (int l = parity; l < levels; l += 2)
    {
      if (l >= from)
      {
        m = cmax(m, in_words(l));
      }
    }
    return m;
  }
  static constexpr int words0 = buf_words(0, 0);
  static constexpr int words1 = buf_words(1, 0);
  static constexpr int lo0    = Split ? buf_words(0, Implicit0 ? 2 : 0) : 1;
  static constexpr int lo1    = Split ? words1 : 1;

  struct TempStorage
  {
    alignas(16) uint32_t k0[words0];
    alignas(16) uint32_t k1[words1];
    alignas(16) uint32_t l0[lo0];
    alignas(16) uint32_t l1[lo1];
    int xs[Threads];
  };

  W4_HD static constexpr bool valid()
  {
    return tree_valid(Threads, Runs, RunLen, K);
  }

  template <int Lv>
  struct level
  {
    static constexpr int IN    = in_len(Lv);
    static constexpr int OUT   = out_len(Lv);
    static constexpr int TPP   = tpp(Lv);
    static constexpr int OPT   = opt(Lv);
    static constexpr int C     = chunk(Lv);
    static constexpr int S     = stride(Lv);
    static constexpr int OS    = stride(Lv + 1 < levels ? Lv + 1 : Lv);
    static constexpr int Steps = Quad ? (log2_ceil(IN + 1) + 1) / 2 : log2_ceil(IN + 1);
    static constexpr bool last = (Lv + 1 == levels);
    static constexpr bool imp  = Implicit0 && (Lv == 0);

    using in_io  = std::conditional_t<Split, run_split<imp, Skew>, run_words<W, Skew>>;
    using out_io = std::conditional_t<Split, run_split<false, Skew>, run_words<W, Skew>>;

    W4_FN static in_io input(TempStorage& ts)
    {
      if constexpr (Split)
      {
        return in_io{(Lv % 2 == 0) ? ts.k0 : ts.k1, (Lv % 2 == 0) ? ts.l0 : ts.l1};
      }
      else
      {
        return in_io{reinterpret_cast<W*>((Lv % 2 == 0) ? ts.k0 : ts.k1)};
      }
    }
    W4_FN static out_io output(TempStorage& ts)
    {
      if constexpr (Split)
      {
        return out_io{(Lv % 2 == 0) ? ts.k1 : ts.k0, (Lv % 2 == 0) ? ts.l1 : ts.l0};
      }
      else
      {
        return out_io{reinterpret_cast<W*>((Lv % 2 == 0) ? ts.k1 : ts.k0)};
      }
    }

    // the merge-path split of one of this thread's diagonals: its end (Exact) or its start
    W4_FN static int split(int t, TempStorage& ts, int diag)
    {
      const int p   = t / TPP;
      const in_io A = input(ts);
      if constexpr (Quad)
      {
        return merge_path4<Steps>(A, (2 * p) * S, A, (2 * p + 1) * S, IN, IN, diag);
      }
      else
      {
        return merge_path<Steps>(A, (2 * p) * S, A, (2 * p + 1) * S, IN, IN, diag);
      }
    }
    // Exact phase 1: the end split (trivial when one thread owns a whole untruncated pair)
    W4_FN static int split_end(int t, TempStorage& ts)
    {
      if constexpr (TPP == 1 && OUT == 2 * IN)
      {
        return IN;
      }
      else
      {
        return split(t, ts, (t % TPP + 1) * OPT);
      }
    }
    // non-exact phase 1: the start split (zero for the first thread of a pair)
    W4_FN static int split_begin(int t, TempStorage& ts)
    {
      if constexpr (TPP == 1)
      {
        return 0;
      }
      else
      {
        return split(t, ts, (t % TPP) * OPT);
      }
    }

    // phase 2: merge from the start split a_beg (Exact: a_end is the successor's start split,
    // else ignored and the pads bound the loads). The last thread of a pair writes the output
    // run's pad.
    W4_FN static void merge(int t, TempStorage& ts, int* out_idx, int a_beg, int a_end)
    {
      const int p     = t / TPP;
      const int j     = t % TPP;
      const int d     = j * OPT;
      const int b_beg = d - a_beg;
      const int b_end = d + OPT - a_end;
      const out_io o  = output(ts);
      merge_top<C, OPT, last, Exact>(input(ts), (2 * p) * S, (2 * p) * RunLen, (2 * p + 1) * S, (2 * p + 1) * RunLen,
                                     a_beg, a_end, b_beg, b_end, o, p * OS, d, out_idx);
      if constexpr (!Exact && !last)
      {
        if (j == TPP - 1)
        {
          W4_UNROLL
          for (int i = 0; i < PAD; ++i)
          {
            o.store_pad(p * OS + OUT + i);
          }
        }
      }
    }
  };
};

// ------------------------------------------------------------------------------------------------
// Key to word. The device adapters add the __nv_bfloat16 specialization, the host test uses the
// raw 16-bit image.
// ------------------------------------------------------------------------------------------------
template <typename KeyT>
struct key_word;

template <>
struct key_word<float>
{
  using W                     = uint64_t;
  static constexpr bool split = true;
  W4_FN static uint32_t tw(float f)
  {
    return twiddle_f32(f);
  }
};

template <>
struct key_word<uint16_t>
{
  using W                     = uint32_t;
  static constexpr bool split = false;
  W4_FN static uint32_t tw(uint16_t b)
  {
    return twiddle_bf16(b);
  }
};

// Block chunk merge: the tile of N keys in blocked arrangement holds M sorted runs of N / M
// consecutive items, thread t stages its IPT items into the level 0 buffer (fp32: keys only,
// the index is the position).
template <typename KeyT, int Threads, int N, int K, int M, bool Quad = false>
struct chunk_core
{
  using kw                 = key_word<KeyT>;
  using W                  = typename kw::W;
  static constexpr int IPT = N / Threads;
  static constexpr int L   = N / M;
  using tree               = merge_tree<W, kw::split, kw::split, Threads, M, L, K, Quad, true, false>;
  using TempStorage        = typename tree::TempStorage;

  W4_FN static void stage(const KeyT (&keys)[IPT], int t, TempStorage& ts)
  {
    constexpr int S0 = tree::stride(0);
    const int first  = t * IPT;
    const int r      = first / L;
    const int pos    = first % L;
    const int base   = r * S0 + pos;
    W4_UNROLL
    for (int i = 0; i < IPT; ++i)
    {
      if constexpr (kw::split)
      {
        ts.k0[sw(base + i)] = kw::tw(keys[i]);
      }
      else
      {
        reinterpret_cast<W*>(ts.k0)[sw(base + i)] = word_ops<W>::make(kw::tw(keys[i]), first + i);
      }
    }
  }
};

// Warp truncated merge sort: lane l holds items [l x IPT, (l + 1) x IPT) (blocked), sorts them
// in registers and writes its run to the level 0 buffer, then the merge levels truncated at k.
// Halves 1: one sorted run of IPT per lane, 5 levels. Halves 2: the lane sorts its two halves
// one after the other (half the words live during the network) and writes two runs of IPT / 2,
// the first level then merges the lane's own two runs (no search, the split is trivial), 6 levels.
template <typename KeyT, int N, int K, int Halves = 1>
struct wmerge_core
{
  using kw                 = key_word<KeyT>;
  using W                  = typename kw::W;
  static constexpr int IPT = N / 32;
  static constexpr int RL  = IPT / Halves;
  using tree               = merge_tree<W, kw::split, false, 32, 32 * Halves, RL, K, false, false, true>;
  using TempStorage        = typename tree::TempStorage;

  W4_FN static void prepare(const KeyT (&keys)[IPT], int lane, TempStorage& ts)
  {
    constexpr int S0 = tree::stride(0);
    W4_UNROLL
    for (int h = 0; h < Halves; ++h)
    {
      W w[RL];
      W4_UNROLL
      for (int i = 0; i < RL; ++i)
      {
        w[i] = word_ops<W>::make(kw::tw(keys[h * RL + i]), lane * IPT + h * RL + i);
      }
      sort_desc<RL>(w);
      const int base = (lane * Halves + h) * S0;
      W4_UNROLL
      for (int i = 0; i < RL; ++i)
      {
        if constexpr (kw::split)
        {
          ts.k0[base + i] = word_ops<W>::hi(w[i]);
          ts.l0[base + i] = word_ops<W>::lo(w[i]);
        }
        else
        {
          reinterpret_cast<W*>(ts.k0)[base + i] = w[i];
        }
      }
      W4_UNROLL
      for (int i = 0; i < tree::PAD; ++i)
      {
        ts.k0[base + RL + i] = 0;
      }
    }
  }
};

} // namespace w4
