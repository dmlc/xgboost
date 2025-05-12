/**
 * Copyright 2025, XGBoost Contributors
 */
#if !defined(XGBOOST_USE_CUDA)
[[nodiscard]] MemPoolHdl CreateHostMemPool() {
  common::AssertGPUSupport();
  return nullptr;
}
#endif  // !defined(XGBOOST_USE_CUDA)
