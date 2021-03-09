#include <gtest/gtest.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>
#include <xgboost/global_config.h>

namespace xgboost {

TEST(GlobalConfiguration, Verbosity) {
  // Configure verbosity via global configuration
  Json config{JsonObject()};
  config["verbosity"] = String("0");
  auto& global_config = *GlobalConfigThreadLocalStore::Get();
  FromJson(config, &global_config);
  // Now verbosity should be updated
  EXPECT_EQ(ConsoleLogger::GlobalVerbosity(), ConsoleLogger::LogVerbosity::kSilent);
  EXPECT_NE(ConsoleLogger::LogVerbosity::kSilent, ConsoleLogger::DefaultVerbosity());
  // GetConfig() should also return updated verbosity
  Json current_config { ToJson(*GlobalConfigThreadLocalStore::Get()) };
  EXPECT_EQ(get<String>(current_config["verbosity"]), "0");
}

TEST(GlobalConfiguration, UseRMM) {
  Json config{JsonObject()};
  config["use_rmm"] = String("true");
  auto& global_config = *GlobalConfigThreadLocalStore::Get();
  FromJson(config, &global_config);
  // GetConfig() should return updated use_rmm flag
  Json current_config { ToJson(*GlobalConfigThreadLocalStore::Get()) };
  EXPECT_EQ(get<String>(current_config["use_rmm"]), "1");
}

}  // namespace xgboost
