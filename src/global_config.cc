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
  /* Convert JSON string into vector of string pairs */
  Args args;
  CHECK(IsA<Object>(config)) << "The JSON string in global config must be a JSON object";
  for (auto const& kv : get<Object const>(config)) {
    CHECK(!IsA<Array>(kv.second)) << "Global config cannot contain a value that's an array";
    CHECK(!IsA<Object>(kv.second)) << "Global config cannot contain a value that's an object";
    CHECK(!IsA<Null>(kv.second)) << "Global config cannot contain a null value";
    if (IsA<Number>(kv.second)) {
      auto number = get<Number const>(kv.second);
      char buf[NumericLimits<float>::kToCharsSize];
      auto ret = to_chars(buf, buf + NumericLimits<float>::kToCharsSize, number);
      CHECK(ret.ec == std::errc());

      args.emplace_back(kv.first,
                        std::string{buf, static_cast<size_t>(std::distance(buf, ret.ptr))});
    } else if (IsA<Integer>(kv.second)) {
      auto integer = get<Integer const>(kv.second);
      args.emplace_back(kv.first, std::to_string(integer));
    } else if (IsA<Boolean>(kv.second)) {
      auto boolean = get<Boolean const>(kv.second);
      args.emplace_back(kv.first, (boolean ? "1" : "0"));
    } else {
      CHECK(IsA<String>(kv.second));
      args.emplace_back(kv.first, get<String const>(kv.second));
    }
  }

  for (const auto& e : args) {
    if (e.first == "verbosity") {
      ConsoleLogger::Configure({e});
    } else {
      LOG(FATAL) << "Unknown global configuration: '" << e.first << "'";
    }
  }
}

Json GlobalConfiguration::GetConfig() {
  Object config;
  config["verbosity"] = std::to_string(static_cast<int>(ConsoleLogger::GlobalVerbosity()));
  return Json(std::move(config));
}

using GlobalConfigurationThreadLocalStore
  = dmlc::ThreadLocalStore<GlobalConfigurationThreadLocalEntry>;

GlobalConfigurationThreadLocalEntry& GlobalConfiguration::GetThreadLocal() {
  return *GlobalConfigurationThreadLocalStore::Get();
}

}  // namespace xgboost
