// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"

TEST(Metric, UnknownMetric) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = nullptr;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name", &ctx));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("rmse", &ctx));
  if (metric) {
    delete metric;
  }
  metric = nullptr;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name@1", &ctx));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("error@0.5f", &ctx));
  if (metric) {
    delete metric;
  }
}
