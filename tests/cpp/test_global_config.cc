#include <gtest/gtest.h>
#include <xgboost/logging.h>
#include <xgboost/global_config.h>

namespace xgboost {

TEST(GlobalConfiguration, Verbosity) {
  // Configure verbosity via global configuration
  GlobalConfiguration::SetArgs({{"verbosity", "0"}});
  // Now verbosity should be updated
  EXPECT_EQ(ConsoleLogger::GlobalVerbosity(), ConsoleLogger::LogVerbosity::kSilent);
  EXPECT_NE(ConsoleLogger::LogVerbosity::kSilent, ConsoleLogger::DefaultVerbosity());
  // GetArgs() should also return updated verbosity
  Args current_args = GlobalConfiguration::GetArgs();
  Args expected_args{ { "verbosity", "0" } };
  EXPECT_EQ(current_args, expected_args);
}

}  // namespace xgboost
