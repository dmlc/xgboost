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
  // GetArgs() should also return updated verbosity
  Json current_config = GlobalConfiguration::GetConfig();
  Args current_config_args;
  for (auto const& kv : get<Object const>(current_config)) {
    current_config_args.emplace_back(kv.first, get<String const>(kv.second));
  }
  Args expected_config_args{ { "verbosity", "0" } };
  EXPECT_EQ(current_config_args, expected_config_args);
}

}  // namespace xgboost
