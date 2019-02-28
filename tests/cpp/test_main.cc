// Copyright by Contributors
#include <gtest/gtest.h>
#include <string>

#include "xgboost/logging.h"
int main(int argc, char ** argv) {
  // Just put here for future convenience.
  std::vector<std::pair<std::string, std::string>> args{{"verbosity", "1"}};
  xgboost::ConsoleLogger::Configure(args.begin(), args.end());

  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
