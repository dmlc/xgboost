/**
 * Copyright 2021-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>

#include "test_prediction_cache.h"

namespace xgboost {
TEST_F(TestPredictionCache, Approx) {
  Context ctx;
  this->RunTest(&ctx, "grow_histmaker", "one_output_per_tree");
}

TEST_F(TestPredictionCache, Hist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", "one_output_per_tree");
}

TEST_F(TestPredictionCache, HistMulti) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", "multi_output_tree");
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestPredictionCache, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist", "one_output_per_tree");
}

TEST_F(TestPredictionCache, GpuApprox) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_approx", "one_output_per_tree");
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost