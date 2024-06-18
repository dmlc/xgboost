#include <gtest/gtest.h>
#include <rabit/internal/utils.h>

TEST(Utils, Assert) {
  EXPECT_THROW({rabit::utils::Assert(false, "foo");}, dmlc::Error);
}
