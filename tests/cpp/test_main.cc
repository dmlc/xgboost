// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/logging.h>
#include <string>
#include <memory>
#include <vector>

#include "helpers.h"

int main(int argc, char ** argv) {
  auto rmm_alloc = xgboost::SetUpRMMResource();
  xgboost::Args args {{"verbosity", "2"}};
  xgboost::ConsoleLogger::Configure(args);

  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
