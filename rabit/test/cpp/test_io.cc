/*!
 *  Copyright (c) 2019 by Contributors
 */
#include <gtest/gtest.h>
#include <rabit/internal/io.h>

#include <vector>

namespace rabit {
TEST(MemoryFixSizeBuffer, Seek) {
  size_t constexpr kSize { 64 };
  std::vector<int32_t> memory( kSize );
  utils::MemoryFixSizeBuffer buf(memory.data(), memory.size());
  buf.Seek(utils::MemoryFixSizeBuffer::SeekEnd);
  size_t end = buf.Tell();
  ASSERT_EQ(end, kSize);
}
}  // namespace rabit
