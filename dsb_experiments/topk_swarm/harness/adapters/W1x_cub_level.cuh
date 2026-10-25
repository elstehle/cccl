// Adapter W1x_cub_level: W1x codegen experiment, the compare and select exchange of cub::detail::WarpBitonicTopK
// (two compares, PLOP3, select per shuffle stage) in the level-parallel tile order.
// Same packed bf16 word, same recursion and stage machinery as W1_level (approaches/W1/
// warp_level_topk.cuh), see approaches/W1/CODEGEN_EXPERIMENT.md. bf16 only in this experiment.
//
// Contract:
//   scope warp, 32 lanes, striped input in registers, N = 32 x IPT with N / round_up(k, 32) a power of two (bf16 N at most
//   65536), k in [1, N] with round_up(k, 32) <= N, bf16 keys twiddled to a 32-bit word with the
//   inverted index low, requirements not_guaranteed and prefer_smaller_index (same code). Output:
//   k indices in the harness's shared output array, sorted by (key descending, index ascending).
//   Temp storage none. Sync: shuffles only. NaN: ordered by its sign bit, never produced by
//   focus-set inputs. -0.0 is mapped to +0.0.
#pragma once

#include "../../approaches/W1/w1_adapters.cuh"

struct W1x_cub_level_info
{
  static constexpr const char* id       = "W1x_cub_level";
  static constexpr topk::scope scope    = topk::scope::warp;
  static constexpr bool data_dependent  = false;
  static constexpr const char* contract =
    "level-parallel bitonic top-k on the packed bf16 word with the CUB compare and select exchange, MaxK = round_up(k, 32), N / MaxK a power of two, both unsorted requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    return kt == topk::key_type::bf16 && w1::level_supported(kt, threads, n, k, req, false);
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct W1x_cub_level
    : w1::level_packed_adapter<KeyT,
                               Threads,
                               N,
                               K,
                               Req,
                               w1::shfl_split32,
                               w1::cub_ops<typename w1::packer<KeyT>::word_t>,
                               false>
{};
