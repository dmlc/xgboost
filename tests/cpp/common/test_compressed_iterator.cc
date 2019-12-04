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
      CompressedBufferWriter cbw(alphabet_size);

      // Test write entire array
      std::vector<unsigned char> buffer(
        CompressedBufferWriter::CalculateBufferSize(input.size(),
          alphabet_size));

      cbw.Write(buffer.data(), input.begin(), input.end());

      CompressedIterator<int> ci(buffer.data(), alphabet_size);
      std::vector<int> output(input.size());
      for (size_t i = 0; i < input.size(); i++) {
        output[i] = ci[i];
      }

      ASSERT_TRUE(input == output);

      // Test write Symbol
      std::vector<unsigned char> buffer2(
        CompressedBufferWriter::CalculateBufferSize(input.size(),
          alphabet_size));
      for (size_t i = 0; i < input.size(); i++) {
        cbw.WriteSymbol(buffer2.data(), input[i], i);
      }
      CompressedIterator<int> ci2(buffer.data(), alphabet_size);
      std::vector<int> output2(input.size());
      for (size_t i = 0; i < input.size(); i++) {
        output2[i] = ci2[i];
      }
      ASSERT_TRUE(input == output2);
    }
  }
}

TEST(CompressedIterator, CalculateMaxRows) {
  const size_t num_bytes = 12652838912;
  const size_t row_stride = 100;
  const size_t num_symbols = 256 * row_stride + 1;
  const size_t extra_bytes = 8;
  size_t num_rows =
      CompressedBufferWriter::CalculateMaxRows(num_bytes, num_symbols, row_stride, extra_bytes);
  EXPECT_EQ(num_rows, 64720403);

  // The calculated # rows should fit within the given number of bytes.
  size_t buffer = CompressedBufferWriter::CalculateBufferSize(num_rows * row_stride, num_symbols);
  size_t extras = extra_bytes * num_rows;
  EXPECT_LE(buffer + extras, num_bytes);

  // An extra row wouldn't fit.
  num_rows++;
  buffer = CompressedBufferWriter::CalculateBufferSize(num_rows * row_stride, num_symbols);
  extras = extra_bytes * num_rows;
  EXPECT_GT(buffer + extras, num_bytes);
}

}  // namespace common
}  // namespace xgboost
