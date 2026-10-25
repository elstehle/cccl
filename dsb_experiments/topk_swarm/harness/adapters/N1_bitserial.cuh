// Adapter N1_bitserial: bit-serial ballot select at warp scope, the 1-bit row of W3.
//
// One pass per key bit from the top: every lane ballots (candidate and bit set) for each of its
// items, the popcounts are summed, and the count c against the remaining r decides whether the
// k-th key has the bit set (candidates without it drop out) or not (the c items with the bit are
// selected, r shrinks by c). No histogram, no scan, no shared memory. 16 passes for bf16, 32 for
// fp32, fewer when a pass leaves exactly r candidates. Emission by ballot ranks in index order.
// Machinery in N6w_impl.cuh.
//
// Contract:
//   scope warp, 32 lanes, blocked input in registers, N a multiple of 32 up to 32768, k in
//   [1, N/2], keys bf16 or fp32 (minus zero folded onto plus zero), int32 indices from the
//   position. Both requirements run the same code. Output: k indices in index order in the
//   harness's shared output array. Temp storage none. Sync: ballots only. NaN: unsupported.
#pragma once

#include "N6w_impl.cuh"

struct N1_bitserial_info
{
  static constexpr const char* id       = "N1_bitserial";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "bit-serial ballot select, one ballot count per key bit, blocked registers in, indices to smem out, both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return n6w::supported_shape(kt, threads, n, k, req);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N1_bitserial
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr int IPT                             = items_per_thread;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[IPT];
  struct TempStorage
  {
    int unused;
  };

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage&, int* out_idx, KeyT*)
  {
    using namespace n6w;
    const int lane     = topk::group_lane<32>();
    const int idx_base = lane * IPT;
    const uint32_t lt  = lanemask_lt();

    uint32_t tw[IPT];
    load_images<KeyT, IPT>(keys, tw);
    range st{0u, FULL, K, N};
    bit_pass<KeyT, IPT, true>(tw, st, lane);
#pragma unroll 1
    while (st.n != st.r && __clz(st.tmin ^ st.tmax) < key_bits<KeyT>)
    {
      bit_pass<KeyT, IPT, false>(tw, st, lane);
    }
    emit_range<IPT, true>(tw, st, out_idx, idx_blocked{idx_base}, lt);
  }
};
