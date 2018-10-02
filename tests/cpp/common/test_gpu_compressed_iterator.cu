#include "../../../src/common/compressed_iterator.h"
#include "../../../src/common/device_helpers.cuh"
#include "gtest/gtest.h"
#include <algorithm>
#include <thrust/device_vector.h>

namespace xgboost {
namespace common {

struct WriteSymbolFunction {
  CompressedBufferWriter cbw;
  unsigned char* buffer_data_d;
  int* input_data_d;
  WriteSymbolFunction(CompressedBufferWriter cbw, unsigned char* buffer_data_d,
                      int* input_data_d)
    : cbw(cbw), buffer_data_d(buffer_data_d), input_data_d(input_data_d) {}

  __device__ void operator()(size_t i) {
    cbw.AtomicWriteSymbol(buffer_data_d, input_data_d[i], i);
  }
};

struct ReadSymbolFunction {
  CompressedIterator<int> ci;
  int* output_data_d;
  ReadSymbolFunction(CompressedIterator<int> ci, int* output_data_d)
    : ci(ci), output_data_d(output_data_d) {}

  __device__ void operator()(size_t i) {
    output_data_d[i] = ci[i];
  }
};

TEST(CompressedIterator, TestGPU) {
  std::vector<int> test_cases = {1, 3, 426, 21, 64, 256, 100000, INT32_MAX};
  int num_elements = 1000;
  int repetitions = 1000;
  srand(9);

  for (auto alphabet_size : test_cases) {
    for (int i = 0; i < repetitions; i++) {
      std::vector<int> input(num_elements);
      std::generate(input.begin(), input.end(),
        [=]() { return rand() % alphabet_size; });
      CompressedBufferWriter cbw(alphabet_size);
      thrust::device_vector<int> input_d(input);

      thrust::device_vector<unsigned char> buffer_d(
        CompressedBufferWriter::CalculateBufferSize(input.size(),
          alphabet_size));

      // write the data on device
      auto input_data_d = input_d.data().get();
      auto buffer_data_d = buffer_d.data().get();
      dh::LaunchN(0, input_d.size(),
                        WriteSymbolFunction(cbw, buffer_data_d, input_data_d));

      // read the data on device
      CompressedIterator<int> ci(buffer_d.data().get(), alphabet_size);
      thrust::device_vector<int> output_d(input.size());
      auto output_data_d = output_d.data().get();
      dh::LaunchN(0, output_d.size(), ReadSymbolFunction(ci, output_data_d));

      std::vector<int> output(output_d.size());
      thrust::copy(output_d.begin(), output_d.end(), output.begin());

      ASSERT_TRUE(input == output);
    }
  }
}

}  // namespace common
}  // namespace xgboost
