// Adapter B1_striped: block_topk_air from the striped branch (enh/block-topk-striped-rebased, commit
// c7cf73b09d, copied to approaches/B1/block_topk_air_striped.cuh) through its striped_to_striped pairs
// overload. Index-only output through the pairs path exactly as B1_reference: keys and int32 indices go
// in, the selected indices are written to the harness's output array, the selected keys are dropped.
// This is the B1 optimizer's baseline, the arrangement the batched agent on that branch feeds.
//
// Contract:
//   scope block, 256 threads (128 and 512 accepted), striped input in registers (thread t holds items
//   t + i * Threads), N = Threads x IPT with IPT >= 1, k in [1, N), keys fp32 or bf16, requirement
//   not_guaranteed only (ties at the k-th key are resolved by shared-memory atomic ticket order).
//   Output: k indices in the harness's shared output array, unsorted. Temp storage: the AIR TempStorage
//   (histogram double buffer, scan storage, exchange buffer of N keys or values). Sync: performs its own
//   __syncthreads inside, assumes the caller separates consecutive calls that reuse the temp storage by a
//   barrier (the harness does). NaN: twiddled as the largest key (radix order), never produced by focus-set
//   inputs. -0.0 is normalized to +0.0 for ranking.
#pragma once

#include "../../approaches/B1/block_topk_air_striped.cuh"
#include "../topk_harness.cuh"

struct B1_striped_info
{
  static constexpr const char* id       = "B1_striped";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "striped-branch block_topk_air pairs path, striped registers in, indices to smem out, not_guaranteed only";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 128) || (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % threads == 0) && (n / threads >= 1) && (k >= 1) && (k < n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed);
    // exchange buffer of n x max(sizeof(key), sizeof(int)) bytes must fit the opt-in smem limit
    const bool smem_ok = static_cast<long long>(n) * 4 + 4096 <= 227 * 1024;
    return key_ok && thread_ok && shape_ok && req_ok && smem_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct B1_striped
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];
  using topk_t      = topk_b1::block_topk_air_striped<KeyT, Threads, items_per_thread, int>;
  using TempStorage = typename topk_t::TempStorage;

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t(ts).template select_pairs_striped_to_striped<cub::detail::topk::select::max, true>(keys, idx, K, N);
    // The selected pairs occupy the tile's first K striped slots: item i of thread t is slot i * Threads + t
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < items_per_thread; ++i)
    {
      const int slot = i * Threads + tid;
      if (slot < K)
      {
        out_idx[slot] = idx[i];
      }
    }
  }
};
