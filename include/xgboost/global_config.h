/*!
 * Copyright 2020 by Contributors
 * \file global_config.h
 * \brief Global configuration for XGBoost
 * \author Hyunsu Cho
 */
#ifndef XGBOOST_GLOBAL_CONFIG_H_
#define XGBOOST_GLOBAL_CONFIG_H_

#include <xgboost/parameter.h>
#include <vector>
#include <string>

namespace xgboost {
class Json;

struct GlobalConfiguration : public XGBoostParameter<GlobalConfiguration> {
  int verbosity { 1 };
  DMLC_DECLARE_PARAMETER(GlobalConfiguration) {
    DMLC_DECLARE_FIELD(verbosity)
        .set_range(0, 3)
        .set_default(1)  // shows only warning
        .describe("Flag to print out detailed breakdown of runtime.");
  }
};

using GlobalConfigThreadLocalStore = dmlc::ThreadLocalStore<GlobalConfiguration>;
}  // namespace xgboost

#endif  // XGBOOST_GLOBAL_CONFIG_H_
