// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"

TEST(Metric, RMSE) {
  xgboost::Metric * metric = xgboost::Metric::Create("rmse");
  ASSERT_STREQ(metric->Name(), "rmse");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.6403, 0.001);
}
