/*!
 * Copyright (c) by XGBoost Contributors 2019
 */
#include <gtest/gtest.h>
#include "../../../src/common/io.h"

namespace xgboost {
namespace common {
TEST(IO, FileExtension) {
  std::string filename {u8"model.json"};
  auto ext = FileExtension(filename);
  ASSERT_EQ(ext, u8"json");
}

TEST(IO, FixedSizeStream) {
  std::string buffer {"This is the content of stream"};
  {
    MemoryFixSizeBuffer stream((void *)buffer.c_str(), buffer.size());
    PeekableInStream peekable(&stream);
    FixedSizeStream fixed(&peekable);

    std::string out_buffer;
    fixed.Take(&out_buffer);
    ASSERT_EQ(buffer, out_buffer);
  }

  {
    std::string huge_buffer;
    for (size_t i = 0; i < 512; i++) {
      huge_buffer += buffer;
    }

    MemoryFixSizeBuffer stream((void *)huge_buffer.c_str(), huge_buffer.size());
    PeekableInStream peekable(&stream);
    FixedSizeStream fixed(&peekable);

    std::string out_buffer;
    fixed.Take(&out_buffer);
    ASSERT_EQ(huge_buffer, out_buffer);
  }
}
}
}  // namespace xgboost
