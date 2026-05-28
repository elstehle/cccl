// Minimal reproducer for the async-keys multi-CTA all-large segmented topk failure.
#include <cub/cub.cuh>
#include <cub/device/device_batched_topk.cuh>

#include <cuda/iterator>
#include <cuda/std/cstdio>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bt = cub::detail::batched_topk;

int main(int argc, char** argv)
{
  using key_t = float;
  using val_t = int;
  using ssize_t = long long;
  using nseg_t  = long long;

  const ssize_t segment_size = (argc > 1) ? std::atoll(argv[1]) : 1024 * 1024;
  const ssize_t k            = (argc > 2) ? std::atoll(argv[2]) : 32;
  const nseg_t  num_segments = (argc > 3) ? std::atoll(argv[3]) : 1;
  const auto direction       = cub::detail::topk::select::max;

  std::printf("repro: segment_size=%lld k=%lld num_segments=%lld\n",
              static_cast<long long>(segment_size),
              static_cast<long long>(k),
              static_cast<long long>(num_segments));

  thrust::device_vector<key_t> keys_in(num_segments * segment_size);
  thrust::device_vector<key_t> keys_out(num_segments * k);
  thrust::device_vector<val_t> values_out(num_segments * k);

  // Fill input with a counting pattern (deterministic).
  thrust::sequence(keys_in.begin(), keys_in.end());

  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out.data());
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out.data());

  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);
  auto values_in_it   = cuda::make_counting_iterator(val_t{0});
  auto d_values_in    = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  auto d_values_out   = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  // Dispatch via the batched topk API (matches the failing test).
  auto err = cub::DeviceBatchedTopK::MaxPairs(
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      bt::segment_size_uniform<1024 * 1024, 1024 * 1024>{segment_size},
      bt::k_uniform<1, 1024 * 1024>{k},
      bt::select_direction_uniform{direction},
      bt::num_segments_uniform<>{num_segments},
      bt::total_num_items_guarantee{num_segments * segment_size});

  if (err != cudaSuccess) {
    std::printf("DeviceBatchedTopK::MaxPairs returned %d: %s\n",
                static_cast<int>(err), cudaGetErrorString(err));
    return 1;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("cudaDeviceSynchronize returned %d: %s\n",
                static_cast<int>(err), cudaGetErrorString(err));
    return 2;
  }

  std::printf("OK\n");
  return 0;
}
