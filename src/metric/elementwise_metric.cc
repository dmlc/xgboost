/*!
 * Copyright 2018 XGBoost contributors
 */
// Dummy file to keep the CUDA conditional compile trick.

#if !defined(XGBOOST_USE_CUDA)
#include "elementwise_metric.cu"
#endif  // !defined(XGBOOST_USE_CUDA)
