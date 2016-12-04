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

TEST(Metric, MAE) {
  xgboost::Metric * metric = xgboost::Metric::Create("mae");
  ASSERT_STREQ(metric->Name(), "mae");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.5, 0.001);
}

TEST(Metric, LogLoss) {
  xgboost::Metric * metric = xgboost::Metric::Create("logloss");
  ASSERT_STREQ(metric->Name(), "logloss");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              1.2039, 0.001);
}

TEST(Metric, Error) {
  xgboost::Metric * metric = xgboost::Metric::Create("error");
  ASSERT_STREQ(metric->Name(), "error");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.9, 0.1, 0.9},
                            {  0,   0,   1,   1}),
              0.5, 0.001);

  EXPECT_ANY_THROW(xgboost::Metric::Create("error@abc"));
  delete metric;
  metric = xgboost::Metric::Create("error@0.5");
  EXPECT_STREQ(metric->Name(), "error");

  delete metric;
  metric = xgboost::Metric::Create("error@0.1");
  ASSERT_STREQ(metric->Name(), "error@0.1");
  EXPECT_STREQ(metric->Name(), "error@0.1");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.2, 0.1, 0.2},
                            {  0,   0,   1,   1}),
              0.5, 0.001);
}

TEST(Metric, PoissionNegLogLik) {
  xgboost::Metric * metric = xgboost::Metric::Create("poisson-nloglik");
  ASSERT_STREQ(metric->Name(), "poisson-nloglik");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.5, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1, 0.2, 0.1, 0.2},
                            {  0,   0,   1,   1}),
              1.1280, 0.001);
}
