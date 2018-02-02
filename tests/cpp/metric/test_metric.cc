// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"

TEST(Metric, UnknownMetric) {
  EXPECT_ANY_THROW(xgboost::Metric::Create("unknown_name"));
  EXPECT_NO_THROW(xgboost::Metric::Create("rmse"));
  EXPECT_ANY_THROW(xgboost::Metric::Create("unknown_name@1"));
  EXPECT_NO_THROW(xgboost::Metric::Create("error@0.5f"));
}
