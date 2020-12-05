/*!
 * Copyright 2020 by Contributors
 * \file global_config.cc
 * \brief Global configuration for XGBoost
 * \author Hyunsu Cho
 */

#include <dmlc/thread_local.h>
#include "xgboost/global_config.h"
#include "xgboost/json.h"

namespace xgboost {
DMLC_REGISTER_PARAMETER(GlobalConfiguration);
}  // namespace xgboost
