/**
 * Copyright 2020-2024 by XGBoost contributors
 */
#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "../tree/test_prediction_cache.h"
#pragma GCC diagnostic pop

namespace xgboost::sycl::tree {

class SyclPredictionCache : public xgboost::TestPredictionCache {};

TEST_F(SyclPredictionCache, Hist) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  this->RunTest(&ctx, "grow_quantile_histmaker_sycl", "one_output_per_tree");
}

}  // namespace xgboost::sycl::tree
