// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"

TEST(Metric, UnknownMetric) {
  auto tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  xgboost::Metric * metric = nullptr;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name", &tparam));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("rmse", &tparam));
  if (metric) {
    delete metric;
  }
  metric = nullptr;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name@1", &tparam));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("error@0.5f", &tparam));
  if (metric) {
    delete metric;
  }
}
