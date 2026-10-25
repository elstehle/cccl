// Adapter N0_rows_naive: a harness test adapter for the rows-per-group path, not an approach. Two
// rows of bf16 keys per group in the packed_rows arrangement (row 0 in the low half of every word,
// row 1 in the high half), selected by k rounds of a shuffle arg-max per row with the smallest
// index winning ties, so both requirements come out exactly. Slow by design, it exists so the
// multi-row oracle and the row accounting of the harness can be exercised and so a two-row
// adapter author has a compiling example.
//
// Contract: scope warp (any logical warp size), rows 2, bf16 keys, k in [1, N], both requirements,
// indices only, no shared memory. NaN: never selected (comparisons false), never produced by
// focus-set inputs. -0.0 and +0.0 compare equal and the smaller index wins.
#pragma once

#include "../topk_harness.cuh"

struct N0_rows_naive_info
{
  static constexpr const char* id       = "N0_rows_naive";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract = "two bf16 rows per group, packed_rows, k shuffle arg-max rounds per row";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement)
  {
    return kt == topk::key_type::bf16 && threads <= 32 && topk::is_pow2(threads) && n % threads == 0 && k >= 1
        && k <= n;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N0_rows_naive
{
  static constexpr int rows                            = 2;
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::packed_rows;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = uint32_t[items_per_thread];
  struct TempStorage
  {
    int unused;
  };

  __device__ __forceinline__ static void
  run(key_storage& words, int (&idx)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    const int lane      = topk::group_lane<Threads>();
    const unsigned mask = topk::member_mask<Threads>();
#pragma unroll
    for (int r = 0; r < rows; ++r)
    {
      float v[items_per_thread];
#pragma unroll
      for (int j = 0; j < items_per_thread; ++j)
      {
        const uint16_t bits = static_cast<uint16_t>(words[j] >> (16 * r));
        v[j]                = topk::key_traits<KeyT>::to_float(topk::key_traits<KeyT>::from_bits(bits));
      }
      for (int round = 0; round < K; ++round)
      {
        float best = -INFINITY;
        int bi     = INT_MAX;
#pragma unroll
        for (int j = 0; j < items_per_thread; ++j)
        {
          if (v[j] > best || (v[j] == best && idx[j] < bi))
          {
            best = v[j];
            bi   = idx[j];
          }
        }
#pragma unroll
        for (int off = Threads / 2; off > 0; off >>= 1)
        {
          const float ob = __shfl_xor_sync(mask, best, off, Threads);
          const int oi   = __shfl_xor_sync(mask, bi, off, Threads);
          if (ob > best || (ob == best && oi < bi))
          {
            best = ob;
            bi   = oi;
          }
        }
        if (lane == 0)
        {
          out_idx[r * K + round] = bi;
        }
        // retire the winner: NaN never compares true again, a genuine -inf stays selectable
#pragma unroll
        for (int j = 0; j < items_per_thread; ++j)
        {
          if (idx[j] == bi)
          {
            v[j] = NAN;
          }
        }
      }
    }
  }
};
