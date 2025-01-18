/**
 * Copyright 2020-2024, XGBoost Contributors
 * \file global_config.h
 * \brief Global configuration for XGBoost
 * \author Hyunsu Cho
 */
#ifndef XGBOOST_GLOBAL_CONFIG_H_
#define XGBOOST_GLOBAL_CONFIG_H_

#include <dmlc/thread_local.h>  // for ThreadLocalStore
#include <xgboost/parameter.h>  // for XGBoostParameter

#include <cstdint>  // for int32_t

namespace xgboost {
struct GlobalConfiguration : public XGBoostParameter<GlobalConfiguration> {
  std::int32_t verbosity{1};
  bool use_rmm{false};
  std::int32_t nthread{0};
  DMLC_DECLARE_PARAMETER(GlobalConfiguration) {
    DMLC_DECLARE_FIELD(verbosity)
        .set_range(0, 3)
        .set_default(1)  // shows only warning
        .describe("Flag to print out detailed breakdown of runtime.");
    DMLC_DECLARE_FIELD(use_rmm).set_default(false).describe(
        "Whether to use RAPIDS Memory Manager to allocate GPU memory in XGBoost");
    DMLC_DECLARE_FIELD(nthread).set_lower_bound(-1).set_default(0);
  }
};

using GlobalConfigThreadLocalStore = dmlc::ThreadLocalStore<GlobalConfiguration>;

struct InitNewThread {
  GlobalConfiguration config = *GlobalConfigThreadLocalStore::Get();

  void operator()() const;
};
}  // namespace xgboost

#endif  // XGBOOST_GLOBAL_CONFIG_H_
