// clang-format off
// nvcc -lineinfo -O3 -DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DTEST_LAUNCH=1 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DVAR_IDX=1 -D_CCCL_NO_SYSTEM_HEADER -Itest -I./ -I../libcudacxx/include -I../thrust --extended-lambda rif_repro.cu
// clang-format on

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

#include <iostream>
#include <vector>

struct RemovePred
{
  __device__ __forceinline__ bool operator()(const thrust::tuple<int16_t, uint32_t, uint32_t>& ele)
  {
    return (thrust::get<0>(ele) == 1);
  }
};

struct mod_op
{
  __device__ __forceinline__ int16_t operator()(const uint32_t val)
  {
    return (int16_t) (val % 2);
  }
};

struct mul_op
{
  __device__ __forceinline__ uint32_t operator()(const uint32_t val)
  {
    return val * 2;
  }
};

bool test_remove_if()
{
  // Input size
  constexpr int N = 10000 * 128 * 10;

  // Input (iota) iterator: 0, 1, 2, 3, ...
  auto counting_it = thrust::make_counting_iterator(int{});
  // Iterator to generate the flags of items to be removed: 0,
  auto remove_flags_it = thrust::make_transform_iterator(counting_it, mod_op{});

  // Prepare input arrays
  thrust::device_vector<int16_t> selection_flags(N);
  thrust::device_vector<uint32_t> data_a(N);
  thrust::device_vector<uint32_t> data_b(N);
  thrust::copy(remove_flags_it, remove_flags_it + N, selection_flags.begin());
  thrust::copy(counting_it, counting_it + N, data_a.begin());
  thrust::copy(counting_it, counting_it + N, data_b.begin());

  // This produces wrong results intermittently.
  auto zip_it       = thrust::make_zip_iterator(selection_flags.begin(), data_a.begin(), data_b.begin());
  auto end_it       = thrust::remove_if(zip_it, zip_it + N, RemovePred{});
  auto num_selected = end_it - zip_it;

  // Sanity checks to make sure remove_if did the right thing.
  auto expected_it = thrust::make_transform_iterator(counting_it, mul_op{});
  bool is_equal    = thrust::equal(expected_it, expected_it + num_selected, data_a.begin());
  is_equal         = is_equal && thrust::equal(expected_it, expected_it + num_selected, data_b.begin());

  if (!is_equal)
  {
    thrust::host_vector<uint32_t> h_data_a(data_a);
    thrust::host_vector<uint32_t> h_data_b(data_b);
    int c = 0;
    for (int i = 0; i < num_selected; i++)
    {
      if ((h_data_a[i] != expected_it[i] || h_data_b[i] != expected_it[i]) && c++ < 1000)
      {
        std::cout << i << ": " << h_data_a[i] << " != " << h_data_b[i] << "\n";
      }
    }
  }
  return (is_equal);
}

int main()
{
  for (int ii = 0; ii < 10000; ++ii)
  {
    bool bPass = test_remove_if();
    if (!bPass)
    {
      std::cout << "Test failed!\n";
      return 1;
    }
    else
    {
      if (ii % 1000 == 0)
      {
        std::cout << "Test passed for attempt = " << ii << std::endl;
      }
    }
  }

  return 0;
}
