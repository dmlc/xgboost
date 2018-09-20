/*!
 * Copyright 2018 XGBoost contributors
 */
#include "common.h"

namespace xgboost {

int AllVisibleImpl::AllVisible() {
  int n_visgpus = 0;
  try {
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
  } catch(const std::exception& e) {
    return 0;
  }
  return n_visgpus;
}

}  // namespace xgboost
