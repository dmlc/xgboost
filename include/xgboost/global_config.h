/*!
 * Copyright 2020 by Contributors
 * \file global_config.h
 * \brief Global configuration for XGBoost
 * \author Hyunsu Cho
 */
#ifndef XGBOOST_GLOBAL_CONFIG_H_
#define XGBOOST_GLOBAL_CONFIG_H_

#include <mutex>
#include <vector>
#include <string>
#include "xgboost/logging.h"

namespace xgboost {

struct GlobalConfigurationThreadLocalEntry {
  Args args;
  std::vector<const char*> names;
  std::vector<const char*> values;
};

class GlobalConfiguration {
 public:
  static void SetArgs(Args const& args);
  static Args GetArgs();

  /*! \brief Get thread local memory for returning data from GlobalConfiguration.
   *         Used in the C API. */
  static GlobalConfigurationThreadLocalEntry& GetThreadLocal();
};

}  // namespace xgboost

#endif  // XGBOOST_GLOBAL_CONFIG_H_
