/**
 * Copyright 2022 by XGBoost Contributors
 */
#include "common/cuda_context.cuh"  // CUDAContext
#include "xgboost/context.h"

namespace xgboost {
CUDAContext const* Context::CUDACtx() const {
  if (!cuctx_) {
    cuctx_.reset(new CUDAContext{});
  }
  return cuctx_.get();
}
}  // namespace xgboost
