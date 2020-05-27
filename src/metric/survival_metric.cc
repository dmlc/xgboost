/*!
 * Copyright 2020 XGBoost contributors
 */
// Dummy file to keep the CUDA conditional compile trick.

#if !defined(XGBOOST_USE_CUDA)
#include "survival_metric.cu"
#endif  // !defined(XGBOOST_USE_CUDA)
