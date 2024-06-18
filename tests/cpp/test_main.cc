// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/logging.h>
#include <string>
#include <memory>
#include <vector>

#include "helpers.h"

int main(int argc, char ** argv) {
  xgboost::Args args {{"verbosity", "2"}};
  xgboost::ConsoleLogger::Configure(args);

  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  auto rmm_alloc = xgboost::SetUpRMMResourceForCppTests(argc, argv);
  return RUN_ALL_TESTS();
}
