#include "../../../src/common/compressed_iterator.h"
#include "gtest/gtest.h"
#include <algorithm>

namespace xgboost {
namespace common {

TEST(CompressedIterator, Test) {
  ASSERT_TRUE(detail::SymbolBits(256) == 8);
  ASSERT_TRUE(detail::SymbolBits(150) == 8);
  std::vector<int> test_cases = {1, 3, 426, 21, 64, 256, 100000, INT32_MAX};
  int num_elements = 1000;
  int repetitions = 1000;
  srand(9);

  for (auto alphabet_size : test_cases) {
    for (int i = 0; i < repetitions; i++) {
      std::vector<int> input(num_elements);
      std::generate(input.begin(), input.end(),
        [=]() { return rand() % alphabet_size; });

      // Test write entire array
      std::vector<unsigned char> buffer(
        CompressedWriter::CalculateBufferSize(input.size(),
          alphabet_size));
      CompressedWriter writer(buffer.data(), alphabet_size);
      CompressedIterator iter(buffer.data(), alphabet_size);

      for (size_t i = 0; i < input.size(); i++) {
        writer.Write(i,input[i]);
      }

      std::vector<int> output(input.size());
      for (size_t i = 0; i < input.size(); i++) {
        output[i] = iter[i];
      }

      ASSERT_TRUE(input == output);

    }
  }
}

}  // namespace common
}  // namespace xgboost
