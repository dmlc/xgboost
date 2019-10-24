/*!
 * Copyright (c) by Contributors 2019
 */

#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include <mutex>

#include "../../../src/common/io.h"

namespace xgboost {
namespace common {

template <typename MutexT>
void TestFileLock(std::string const& read_or_write) {
  std::string lock_path {"test_file_lock"};
  {
    MutexT lock {lock_path};
    std::lock_guard<MutexT> guard(lock);
    std::ifstream fin {lock_path + ".xgboost." + read_or_write + ".lock"};
    ASSERT_TRUE(fin);

    MutexT lock_new {lock_path};
    ASSERT_FALSE(lock_new.try_lock());
  }

  std::ifstream fin {lock_path + ".xgboost." + read_or_write + ".lock"};
  ASSERT_FALSE(fin);
}

TEST(IO, FileLock) {
  TestFileLock<ReadFileLock>("read");
  TestFileLock<WriteFileLock>("write");
}
}  // namespace common
}  // namespace xgboost
