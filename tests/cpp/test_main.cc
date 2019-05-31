// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/logging.h>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
  std::vector<std::pair<std::string, std::string>> args {{"verbosity", "3"}};
  xgboost::ConsoleLogger::Configure(args.begin(), args.end());
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
