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
// counted wins. Runs live in shared memory in a layout swizzled on groups of 16 (64-bit words)
// or 32 (32-bit words) words: threads of a pair stride through a run at about OPT / 2 words per
// thread, which maps to two or four banks unswizzled.
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#  define W4_FN __device__ __forceinline__
#  define W4_HD __host__ __device__
#  define W4_UNROLL _Pragma("unroll")
#else
#  define W4_FN inline
#  define W4_HD
#  define W4_UNROLL
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
  static constexpr int group = 16; // words per swizzle group (16 x 8 bytes = 32 banks)
  W4_FN static uint64_t make(uint32_t tw, int idx)
  {
    return (static_cast<uint64_t>(tw) << 32) | static_cast<uint32_t>(~idx);
  }
  W4_FN static int index(uint64_t w)
  {
    return static_cast<int>(~static_cast<uint32_t>(w));
  }
};

template <>
struct word_ops<uint32_t>
{
  static constexpr int group = 32;
  W4_FN static uint32_t make(uint32_t tw, int idx)
  {
    return (tw << 16) | (static_cast<uint32_t>(~idx) & 0xffffu);
  }
  W4_FN static int index(uint32_t w)
  {
    return static_cast<int>((~w) & 0xffffu);
  }
};

// Swizzled word position inside a run buffer: the low log2(group) bits are xored with the next
// log2(group) bits, a permutation inside every aligned group, so run starts and extents that are
// multiples of `group` stay inside their run.
template <typename W>
W4_FN int sw(int i)
{
  constexpr int g = word_ops<W>::group;
  return i ^ ((i / g) & (g - 1));
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
// Merge path on two descending runs, the first run wins ties (its indices are smaller, so this
// is the word order). Returns the number of items of A among the first d of the merge. Steps is
// a compile-time bound on ceil(log2(range + 1)), the loop runs it fully with a predicated body.
// The probe of a finished search (lo == hi) is in range: A[mid] with mid <= len_a touches at
// most the first pad word after the run, B's position is clamped at 0.
// ------------------------------------------------------------------------------------------------
template <int Steps, typename W>
W4_FN int merge_path(const W* A, const W* B, int len_a, int len_b, int d)
{
  int lo = cmax(d - len_b, 0);
  int hi = cmin(d, len_a);
  W4_UNROLL
  for (int s = 0; s < Steps; ++s)
  {
    const int mid      = (lo + hi) >> 1;
    const bool go      = lo < hi;
    const W a          = A[sw<W>(mid)];
    const W b          = B[sw<W>(cmax(d - 1 - mid, 0))];
    const bool b_first = b > a;
    hi                 = (go & b_first) ? mid : hi;
    lo                 = (go & !b_first) ? mid + 1 : lo;
  }
  return lo;
}

// ------------------------------------------------------------------------------------------------
// Chunked register merge. a[] and b[] hold up to C items each of two descending runs (absent
// items are 0). o[] receives the C largest, descending. Returns how many came from a[].
// ------------------------------------------------------------------------------------------------
template <int C, typename W>
W4_FN int top_chunk(const W (&a)[C], const W (&b)[C], W (&o)[C])
{
  int wins = 0;
  W4_UNROLL
  for (int i = 0; i < C; ++i)
  {
    const W x     = a[i];
    const W y     = b[C - 1 - i];
    const bool aw = x > y;
    o[i]          = aw ? x : y;
    wins += aw ? 1 : 0;
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
  return wins;
}

// Merge exactly [a_beg, a_end) of A with [b_beg, b_end) of B, OPT = (a_end - a_beg) +
// (b_end - b_beg) outputs in chunks of C. ToIndices: write source indices to an int array,
// else words to a run buffer (swizzled, out_pos is the unswizzled start inside the buffer).
template <int C, int OPT, bool ToIndices, typename W, typename OutT>
W4_FN void merge_top(const W* A, const W* B, int a_beg, int a_end, int b_beg, int b_end, OutT* out, int out_pos)
{
  int a_head = a_beg;
  int b_head = b_beg;
  W4_UNROLL
  for (int c = 0; c < OPT / C; ++c)
  {
    W a[C], b[C], o[C];
    W4_UNROLL
    for (int i = 0; i < C; ++i)
    {
      a[i] = 0;
      b[i] = 0;
      if (a_head + i < a_end)
      {
        a[i] = A[sw<W>(a_head + i)];
      }
      if (b_head + i < b_end)
      {
        b[i] = B[sw<W>(b_head + i)];
      }
    }
    const int wins = top_chunk<C>(a, b, o);
    W4_UNROLL
    for (int i = 0; i < C; ++i)
    {
      if constexpr (ToIndices)
      {
        out[out_pos + c * C + i] = word_ops<W>::index(o[i]);
      }
      else
      {
        out[sw<W>(out_pos + c * C + i)] = o[i];
      }
    }
    a_head += wins;
    b_head += C - wins;
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

// ------------------------------------------------------------------------------------------------
// The merge tree: Runs sorted runs of RunLen words, Threads cooperating threads, k largest out.
// Level l reads buffer l % 2 and writes buffer (l + 1) % 2 (the last level writes indices).
// ------------------------------------------------------------------------------------------------
template <typename W, int Threads, int Runs, int RunLen, int K>
struct merge_tree
{
  static constexpr int G      = word_ops<W>::group;
  static constexpr int levels = log2_ceil(Runs);

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
  // threads per pair
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
  // run stride in words of level l's input buffer: whole swizzle groups plus one pad group
  W4_HD static constexpr int stride(int l)
  {
    return round_up(in_len(l), G) + G;
  }
  W4_HD static constexpr int in_words(int l)
  {
    return runs_at(l) * stride(l);
  }
  W4_HD static constexpr int buf_words(int parity)
  {
    int m = 1;
    for (int l = parity; l < levels; l += 2)
    {
      m = cmax(m, in_words(l));
    }
    return m;
  }
  static constexpr int words0 = buf_words(0);
  static constexpr int words1 = buf_words(1);

  W4_HD static constexpr bool valid()
  {
    return tree_valid(Threads, Runs, RunLen, K);
  }

  // One level for thread t. `xs` is a Threads-sized exchange array for the block form (the
  // caller synchronizes between phase 1 and phase 2), the warp form passes its shuffle result.
  template <int Lv>
  struct level
  {
    static constexpr int IN    = in_len(Lv);
    static constexpr int OUT   = out_len(Lv);
    static constexpr int TPP   = tpp(Lv);
    static constexpr int OPT   = opt(Lv);
    static constexpr int C     = chunk(Lv);
    static constexpr int S     = stride(Lv);
    static constexpr int OS    = stride(Lv + 1 < levels ? Lv + 1 : Lv); // output run stride
    static constexpr int Steps = log2_ceil(IN + 1);
    static constexpr bool last = (Lv + 1 == levels);

    // phase 1: the split of this thread's end diagonal
    W4_FN static int split_end(int t, const W* in)
    {
      const int p  = t / TPP;
      const int j  = t % TPP;
      const W* A   = in + (2 * p) * S;
      const W* B   = A + S;
      return merge_path<Steps>(A, B, IN, IN, (j + 1) * OPT);
    }

    // phase 2: merge with the start split a_beg (the predecessor's end split, 0 for j == 0)
    W4_FN static void merge(int t, const W* in, W* out, int* out_idx, int a_beg, int a_end)
    {
      const int p  = t / TPP;
      const int j  = t % TPP;
      const int d  = j * OPT;
      const W* A   = in + (2 * p) * S;
      const W* B   = A + S;
      const int b_beg = d - a_beg;
      const int b_end = d + OPT - a_end;
      if constexpr (last)
      {
        merge_top<C, OPT, true>(A, B, a_beg, a_end, b_beg, b_end, out_idx, d);
      }
      else
      {
        merge_top<C, OPT, false>(A, B, a_beg, a_end, b_beg, b_end, out + p * OS, d);
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
  using W = uint64_t;
  W4_FN static uint32_t tw(float f)
  {
    return twiddle_f32(f);
  }
};

template <>
struct key_word<uint16_t>
{
  using W = uint32_t;
  W4_FN static uint32_t tw(uint16_t b)
  {
    return twiddle_bf16(b);
  }
};

// Block chunk merge: the tile of N keys in blocked arrangement holds M sorted runs of N / M
// consecutive items, thread t stages its IPT items into run buffer 0.
template <typename KeyT, int Threads, int N, int K, int M>
struct chunk_core
{
  using kw                    = key_word<KeyT>;
  using W                     = typename kw::W;
  static constexpr int IPT    = N / Threads;
  static constexpr int L      = N / M;
  using tree                  = merge_tree<W, Threads, M, L, K>;

  W4_FN static void stage(const KeyT (&keys)[IPT], int t, W* buf0)
  {
    constexpr int S0 = tree::stride(0);
    const int first  = t * IPT;
    const int r      = first / L;
    const int pos    = first % L;
    W* dst           = buf0 + r * S0;
    W4_UNROLL
    for (int i = 0; i < IPT; ++i)
    {
      dst[sw<W>(pos + i)] = word_ops<W>::make(kw::tw(keys[i]), first + i);
    }
  }
};

// Warp truncated merge sort: lane l holds items [l x IPT, (l + 1) x IPT) (blocked), sorts them
// in registers and writes its run to buffer 0, then 5 merge levels truncated at k.
template <typename KeyT, int N, int K>
struct wmerge_core
{
  using kw                    = key_word<KeyT>;
  using W                     = typename kw::W;
  static constexpr int IPT    = N / 32;
  using tree                  = merge_tree<W, 32, 32, IPT, K>;

  W4_FN static void prepare(const KeyT (&keys)[IPT], int lane, W* buf0)
  {
    constexpr int S0 = tree::stride(0);
    W w[IPT];
    W4_UNROLL
    for (int i = 0; i < IPT; ++i)
    {
      w[i] = word_ops<W>::make(kw::tw(keys[i]), lane * IPT + i);
    }
    sort_desc<IPT>(w);
    W* dst = buf0 + lane * S0;
    W4_UNROLL
    for (int i = 0; i < IPT; ++i)
    {
      dst[sw<W>(i)] = w[i];
    }
  }
};

} // namespace w4
