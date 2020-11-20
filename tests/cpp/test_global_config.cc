#include <gtest/gtest.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>
#include <xgboost/global_config.h>

namespace xgboost {

TEST(GlobalConfiguration, Verbosity) {
  // Configure verbosity via global configuration
  Json config{JsonObject()};
  config["verbosity"] = Integer(0);
  GlobalConfiguration::SetConfig(config);
  // Now verbosity should be updated
  EXPECT_EQ(ConsoleLogger::GlobalVerbosity(), ConsoleLogger::LogVerbosity::kSilent);
  EXPECT_NE(ConsoleLogger::LogVerbosity::kSilent, ConsoleLogger::DefaultVerbosity());
  // GetConfig() should also return updated verbosity
  Json current_config = GlobalConfiguration::GetConfig();
  EXPECT_EQ(get<Integer>(current_config["verbosity"]), 0);
}

}  // namespace xgboost
