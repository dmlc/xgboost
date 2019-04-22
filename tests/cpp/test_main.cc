// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/logging.h>

int main(int argc, char ** argv) {
  std::vector<std::pair<std::string, std::string>> verbosity{{"verbosity", "3"}};
  xgboost::ConsoleLogger::Configure(verbosity.begin(), verbosity.end());

  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
