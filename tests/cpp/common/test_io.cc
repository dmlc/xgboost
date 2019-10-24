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

TEST(IO, FileLock) {
  std::string lock_path {"test_file_lock"};
  {
    FileLock lock {lock_path};
    std::lock_guard<FileLock> guard(lock);
    std::ifstream fin {lock_path + ".xgboost.lock"};
    ASSERT_TRUE(fin);

    FileLock lock_new {lock_path};
    ASSERT_FALSE(lock_new.try_lock());
  }

  std::ifstream fin {lock_path + ".xgboost.lock"};
  ASSERT_FALSE(fin);
}

}  // namespace common
}  // namespace xgboost
