/**
 * Copyright 2020-2025, XGBoost Contributors
 * \file global_config.cc
 * \brief Global configuration for XGBoost
 * \author Hyunsu Cho
 */

#include "xgboost/global_config.h"

#include <dmlc/thread_local.h>

#include "common/cuda_rt_utils.h"  // for SetDevice

namespace xgboost {
DMLC_REGISTER_PARAMETER(GlobalConfiguration);

InitNewThread::InitNewThread()
    : config{*GlobalConfigThreadLocalStore::Get()}, device{curt::CurrentDevice(false)} {}

void InitNewThread::operator()() const {
  *GlobalConfigThreadLocalStore::Get() = config;
  if (config.nthread > 0) {
    omp_set_num_threads(config.nthread);
  }
  if (device >= 0) {
    curt::SetDevice(this->device);
  }
}
}  // namespace xgboost
