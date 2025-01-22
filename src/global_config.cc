/**
 * Copyright 2020-2024, XGBoost Contributors
 * \file global_config.cc
 * \brief Global configuration for XGBoost
 * \author Hyunsu Cho
 */

#include "xgboost/global_config.h"

#include <dmlc/thread_local.h>

namespace xgboost {
DMLC_REGISTER_PARAMETER(GlobalConfiguration);

void InitNewThread::operator()() const {
  *GlobalConfigThreadLocalStore::Get() = config;
  if (config.nthread > 0) {
    omp_set_num_threads(config.nthread);
  }
}
}  // namespace xgboost
