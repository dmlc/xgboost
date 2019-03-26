// Copyright by Contributors
#include <xgboost/metric/metric.h>

#include "../helpers.h"

TEST(Metric, UnknownMetric) {
  xgboost::Metric * metric = nullptr;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name"));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("rmse"));
  if (metric) {
    delete metric;
  }
  metric = nullptr;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name@1"));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("error@0.5f"));
  if (metric) {
    delete metric;
  }
}
