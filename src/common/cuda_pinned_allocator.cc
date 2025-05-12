/**
 * Copyright 2025, XGBoost Contributors
 */
#include "cuda_pinned_allocator.h"

#if !defined(XGBOOST_USE_CUDA)
[[nodiscard]] MemPoolHdl CreateHostMemPool() {
  common::AssertGPUSupport();
  return nullptr;
}
#endif  // !defined(XGBOOST_USE_CUDA)
