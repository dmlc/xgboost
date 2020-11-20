/*!
 * Copyright 2020 by Contributors
 * \file global_config.cc
 * \brief Global configuration for XGBoost
 * \author Hyunsu Cho
 */

#include <dmlc/thread_local.h>
#include "xgboost/global_config.h"

namespace xgboost {

void GlobalConfiguration::SetArgs(const Args& args) {
  for (const auto& e : args) {
    if (e.first == "verbosity") {
      ConsoleLogger::Configure({e});
    } else {
      LOG(FATAL) << "Unknown global configuration: '" << e.first << "'";
    }
  }
}

Args GlobalConfiguration::GetArgs() {
  Args args{
    {"verbosity", std::to_string(static_cast<int>(ConsoleLogger::GlobalVerbosity()))}
  };
  return args;
}

using GlobalConfigurationThreadLocalStore
  = dmlc::ThreadLocalStore<GlobalConfigurationThreadLocalEntry>;

GlobalConfigurationThreadLocalEntry& GlobalConfiguration::GetThreadLocal() {
  return *GlobalConfigurationThreadLocalStore::Get();
}

}  // namespace xgboost
