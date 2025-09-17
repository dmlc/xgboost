/**
 * Copyright 2016-2023, XGBoost Contributors
 */
#include "test_rank_metric.h"

#include <gtest/gtest.h>      // for Test, EXPECT_NEAR, ASSERT_STREQ
#include <xgboost/context.h>  // for Context
#include <xgboost/metric.h>   // for Metric

#include <memory>  // for unique_ptr

#include "../helpers.h"    // for GetMetricEval, CreateEmptyGe...
#include "xgboost/base.h"  // for bst_float, kRtEps

namespace xgboost::metric {
TEST(Metric, AMS) {
  auto ctx = MakeCUDACtx(GPUIDX);
  EXPECT_ANY_THROW(Metric::Create("ams", &ctx));
  std::unique_ptr<Metric> metric{Metric::Create("ams@0.5f", &ctx)};
  ASSERT_STREQ(metric->Name(), "ams@0.5");
  EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1}, {0, 1}), 0.311f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric.get(), {0.1f, 0.9f, 0.1f, 0.9f}, {0, 0, 1, 1}), 0.29710f,
              0.001f);

  metric.reset(Metric::Create("ams@0", &ctx));
  ASSERT_STREQ(metric->Name(), "ams@0");
  EXPECT_NEAR(GetMetricEval(metric.get(), {0, 1}, {0, 1}), 0.311f, 0.001f);
}
}  // namespace xgboost::metric
