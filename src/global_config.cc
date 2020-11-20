/*!
 * Copyright 2020 by Contributors
 * \file global_config.cc
 * \brief Global configuration for XGBoost
 * \author Hyunsu Cho
 */

#include <dmlc/thread_local.h>
#include "xgboost/global_config.h"
#include "common/charconv.h"

namespace xgboost {

void GlobalConfiguration::SetConfig(Json const& config) {
  CHECK(IsA<Object>(config)) << "The JSON string in global config must be a JSON object";
  for (auto const& kv : get<Object const>(config)) {
    if (kv.first == "verbosity") {
      CHECK(IsA<Integer>(kv.second)) << "verbosity must be an integer";
      auto integer = get<Integer const>(kv.second);
      ConsoleLogger::Configure({{kv.first, std::to_string(integer)}});
    } else {
      LOG(FATAL) << "Unknown parameter in global configuration: '" << kv.first << "'";
    }
  }
}

Json GlobalConfiguration::GetConfig() {
  Object config;
  config["verbosity"] = static_cast<int>(ConsoleLogger::GlobalVerbosity());
  return Json(std::move(config));
}

using GlobalConfigurationThreadLocalStore
  = dmlc::ThreadLocalStore<GlobalConfigurationThreadLocalEntry>;

GlobalConfigurationThreadLocalEntry& GlobalConfiguration::GetThreadLocal() {
  return *GlobalConfigurationThreadLocalStore::Get();
}

}  // namespace xgboost
