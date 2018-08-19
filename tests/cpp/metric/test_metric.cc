// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"

TEST(Metric, UnknownMetric) {
  xgboost::Metric * metric;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name"));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("rmse"));
  delete metric;
  EXPECT_ANY_THROW(metric = xgboost::Metric::Create("unknown_name@1"));
  EXPECT_NO_THROW(metric = xgboost::Metric::Create("error@0.5f"));
  delete metric;
}
