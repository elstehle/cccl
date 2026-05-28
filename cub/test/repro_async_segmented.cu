// Minimal reproducer for the async-keys multi-CTA all-large segmented topk failure.
#include <cub/cub.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <cuda/iterator>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bt = cub::detail::batched_topk;

int main(int argc, char** argv)
{
  using key_t = float;
  using val_t = int;

  const long long segment_size = (argc > 1) ? std::atoll(argv[1]) : 1024 * 1024;
  const long long k            = (argc > 2) ? std::atoll(argv[2]) : 32;
  const long long num_segments = (argc > 3) ? std::atoll(argv[3]) : 1;
  const auto direction         = cub::detail::topk::select::max;

  std::printf("repro: segment_size=%lld k=%lld num_segments=%lld\n",
              segment_size, k, num_segments);
  std::fflush(stdout);

  thrust::device_vector<key_t> keys_in(num_segments * segment_size);
  thrust::device_vector<key_t> keys_out(num_segments * k);
  thrust::device_vector<val_t> values_out(num_segments * k);

  // Match catch2: random fill (c2h::gen uses a PRNG with a per-test seed).
  // Use thrust::sequence first for determinism, then xor with a per-element
  // hash so the values look distributed.
  thrust::sequence(keys_in.begin(), keys_in.end());
  thrust::transform(keys_in.begin(), keys_in.end(),
                    thrust::make_counting_iterator(0),
                    keys_in.begin(),
                    [] __device__ (key_t v, int i) {
                      unsigned u = static_cast<unsigned>(i);
                      u ^= u >> 16; u *= 0x7feb352dU;
                      u ^= u >> 15; u *= 0x846ca68bU;
                      u ^= u >> 16;
                      return *reinterpret_cast<key_t*>(&u);
                    });

  auto d_keys_in_ptr    = thrust::raw_pointer_cast(keys_in.data());
  auto d_keys_out_ptr   = thrust::raw_pointer_cast(keys_out.data());
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out.data());

  auto d_keys_in    = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out   = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  auto d_values_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  auto d_values_out = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  // Use the same launch shape as cub.test.device.segmented_topk_pairs.lid_2:
  // capture the dispatch in a CUDA Graph and launch the graph.
  const bool use_graph = (argc > 4) ? std::atoll(argv[4]) != 0 : true;
  std::printf("use_graph = %d\n", (int)use_graph);
  std::fflush(stdout);

  auto invoke = [&](void* d_temp, size_t& temp_bytes, cudaStream_t stream) {
    return bt::dispatch(
        d_temp, temp_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        bt::segment_size_uniform<1024 * 1024, 1024 * 1024>{static_cast<cuda::std::int64_t>(segment_size)},
        bt::k_uniform<1, 1024 * 1024>{static_cast<cuda::std::int64_t>(k)},
        bt::select_direction_uniform{direction},
        bt::num_segments_uniform<>{static_cast<cuda::std::int64_t>(num_segments)},
        bt::total_num_items_guarantee{static_cast<cuda::std::int64_t>(num_segments * segment_size)},
        stream);
  };

  cudaStream_t stream{};
  cudaStreamCreate(&stream);

  // Size query.
  size_t temp_bytes = 0;
  auto err = invoke(nullptr, temp_bytes, stream);
  if (err != cudaSuccess) {
    std::printf("size-query dispatch returned %d: %s\n", (int)err, cudaGetErrorString(err));
    return 1;
  }
  std::printf("temp_bytes = %zu\n", temp_bytes);
  std::fflush(stdout);
  thrust::device_vector<char> temp_storage(temp_bytes);
  auto d_temp = thrust::raw_pointer_cast(temp_storage.data());

  if (use_graph) {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
      std::printf("BeginCapture: %s\n", cudaGetErrorString(err));
      return 3;
    }
    err = invoke(d_temp, temp_bytes, stream);
    cudaError_t capture_err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) { std::printf("dispatch (in capture): %s\n", cudaGetErrorString(err)); return 1; }
    if (capture_err != cudaSuccess) { std::printf("EndCapture: %s\n", cudaGetErrorString(capture_err)); return 3; }
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) { std::printf("Instantiate: %s\n", cudaGetErrorString(err)); return 3; }
    err = cudaGraphLaunch(exec, stream);
    if (err != cudaSuccess) { std::printf("Launch: %s\n", cudaGetErrorString(err)); return 3; }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { std::printf("Sync after graph: %s\n", cudaGetErrorString(err)); return 2; }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  } else {
    err = invoke(d_temp, temp_bytes, stream);
    if (err != cudaSuccess) { std::printf("dispatch: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { std::printf("Sync: %s\n", cudaGetErrorString(err)); return 2; }
  }

  cudaStreamDestroy(stream);
  std::printf("OK\n");
  return 0;
}
