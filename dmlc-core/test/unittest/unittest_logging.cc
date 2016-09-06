// Copyright by Contributors
#define DMLC_LOG_FATAL_THROW 0
#include <dmlc/logging.h>
#include <gtest/gtest.h>

using namespace std;

TEST(Logging, basics) {
  LOG(INFO) << "hello";
  LOG(ERROR) << "error";

  int x = 1, y = 1;
  CHECK_EQ(x, y);
  CHECK_GE(x, y);

  int *z = &x;
  CHECK_EQ(*CHECK_NOTNULL(z), x);

  ASSERT_DEATH(CHECK_NE(x, y), ".*");
}
