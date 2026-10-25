// Adapter N3_stream: streaming block top-k with a running exact threshold over chunked input
// (approaches/N3/block_topk_stream.cuh). The harness hands the adapter the whole striped tile,
// the adapter feeds it to the primitive as N / 4096 chunks of 4096 items (16 items per thread at
// 256 threads, 8 at 512), one insert_chunk call per chunk, then finish. The primitive touches a
// chunk only inside its insert_chunk call, so the same code serves a row that arrives chunk by
// chunk from global memory. The chunk is 4096 because the first chunk's histogram has to put the
// k-th largest key of that chunk into a useful bin: with k at most half a chunk the threshold
// after chunk 0 is already that chunk's median, with k equal to the chunk it would be its minimum.
//
// Contract: block scope, 256 or 512 threads, striped input in registers, N a multiple of 4096 up
// to 32768 (16-bit index and count fields), k in [1, 2048] at run time (at most half a chunk),
// keys fp32 or bf16, requirements not_guaranteed and prefer_smaller_index (same code except the
// tie-class exit), output k source indices in the harness's shared array in unspecified order.
// Temp storage: 1024-bin histogram, a candidate buffer of 16384 entries (4 bytes per entry for
// bf16, 8 for fp32), a stage of Threads entries, scan storage, partials, dummy and control words.
// Sync: __syncthreads inside, the caller separates consecutive calls by a barrier. NaN: see the
// primitive (unsupported, never produced by focus-set inputs). -0.0 equals +0.0.
#pragma once

#include "../../approaches/N3/block_topk_stream.cuh"
#include "../topk_harness.cuh"

struct N3_stream_info
{
  static constexpr const char* id       = "N3_stream";
  static constexpr topk::scope scope    = topk::scope::block;
  static constexpr bool data_dependent  = true;
  static constexpr const char* contract =
    "streaming selection: chunks of 4096 filtered against a running threshold from a 1024-bin "
    "histogram of the candidate buffer, no merges, one classify pass and a rank finish at the end, "
    "both requirements";

  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req)
  {
    const bool key_ok    = (kt == topk::key_type::fp32) || (kt == topk::key_type::bf16);
    const bool thread_ok = (threads == 256) || (threads == 512);
    const bool shape_ok  = (n % 4096 == 0) && (n >= 4096) && (n <= 32768) && (k >= 1) && (k <= 2048) && (k < n);
    const bool req_ok    = (req == topk::requirement::not_guaranteed) || (req == topk::requirement::prefer_smaller_index);
    return key_ok && thread_ok && shape_ok && req_ok;
  }
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct N3_stream
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::striped;
  static constexpr topk::output_form output            = topk::output_form::indices;
  using key_storage                                    = KeyT[items_per_thread];

  static constexpr int chunk  = 4096;
  static constexpr int cpt    = chunk / Threads;
  static constexpr int chunks = N / chunk;
  static constexpr int cap    = 16384;
  static constexpr int bins   = 1024;
  static_assert(cap >= K + Threads + chunk, "the buffer must take a compacted set plus one chunk");

  using topk_t =
    n3::block_topk_stream<KeyT, Threads, chunk, N, cap, bins, Req == topk::requirement::prefer_smaller_index>;
  using TempStorage = typename topk_t::TempStorage;

  template <int J>
  __device__ __forceinline__ static void feed(topk_t& p, const key_storage& keys)
  {
    if constexpr (J < chunks)
    {
      KeyT ck[cpt];
#pragma unroll
      for (int i = 0; i < cpt; ++i)
      {
        ck[i] = keys[J * cpt + i];
      }
      p.template insert_chunk<J == 0>(ck, J * chunk, K);
      feed<J + 1>(p, keys);
    }
  }

  __device__ __forceinline__ static void
  run(key_storage& keys, int (&)[items_per_thread], TempStorage& ts, int* out_idx, KeyT*)
  {
    topk_t p(ts);
    p.begin();
    feed<0>(p, keys);
    p.finish(K, out_idx);
  }
};
