#include "../../../src/common/compressed_iterator.h"
#include "../../../src/common/device_helpers.cuh"
#include "gtest/gtest.h"
#include <algorithm>
#include <thrust/device_vector.h>

namespace xgboost {
namespace common {

void TestCompressedIterator(){
  dh::safe_cuda(cudaSetDevice(0));
  std::vector<int> test_cases = {1, 3, 426, 21, 64, 256, 100000, INT32_MAX};
  int num_elements = 1000;
  int repetitions = 1000;
  srand(9);

  for (auto alphabet_size : test_cases) {
    for (int i = 0; i < repetitions; i++) {
      std::vector<int> input(num_elements);
      std::generate(input.begin(), input.end(), [=]() { return rand() % alphabet_size; });
      thrust::device_vector<int> input_d(input);

      thrust::device_vector<unsigned char> buffer_d(
          CompressedWriter::CalculateBufferSize(input.size(), alphabet_size));
      CompressedWriter writer(buffer_d.data().get(), alphabet_size);

      // write the data on device
      auto input_data_d = input_d.data().get();
      dh::LaunchN(input_d.size(),
                  [=] __device__(std::size_t idx) mutable { writer.Write(idx, input_data_d[idx]); });

      CompressedIterator iter(buffer_d.data().get(), alphabet_size);
      thrust::device_vector<int> output_d(input.size());
      auto output_data_d = output_d.data().get();
      dh::LaunchN(output_d.size(),
                  [=] __device__(std::size_t idx) { output_data_d[idx] = iter[idx]; });

      std::vector<int> output(output_d.size());
      thrust::copy(output_d.begin(), output_d.end(), output.begin());

      ASSERT_TRUE(input == output);
    }
  }
}

TEST(CompressedIterator, TestGPU) {
TestCompressedIterator();
}

}  // namespace common
}  // namespace xgboost
