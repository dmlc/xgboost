/*!
 * Copyright 2018-2019 XGBoost contributors
 */
#include <xgboost/metric.h>
#include <map>
#include "../helpers.h"

TEST(Metric, DeclareUnifiedTest(RMSE)) {
  auto lparam = xgboost::CreateEmptyGenericParam(0, NGPUS);
  xgboost::Metric * metric = xgboost::Metric::Create("rmse", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "rmse");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1},
                            {  0,   1,   2,   3}),
              0.6403f, 0.001f);
  delete metric;
}

TEST(Metric, DeclareUnifiedTest(MAE)) {
  auto lparam = xgboost::CreateEmptyGenericParam(0, NGPUS);
  xgboost::Metric * metric = xgboost::Metric::Create("mae", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "mae");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);
  delete metric;
}

TEST(Metric, DeclareUnifiedTest(LogLoss)) {
  auto lparam = xgboost::CreateEmptyGenericParam(0, NGPUS);
  xgboost::Metric * metric = xgboost::Metric::Create("logloss", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "logloss");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              1.2039f, 0.001f);
  delete metric;
}

TEST(Metric, DeclareUnifiedTest(Error)) {
  auto lparam = xgboost::CreateEmptyGenericParam(0, NGPUS);
  xgboost::Metric * metric = xgboost::Metric::Create("error", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "error");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);

  EXPECT_ANY_THROW(xgboost::Metric::Create("error@abc", &lparam));
  delete metric;

  metric = xgboost::Metric::Create("error@0.5f", &lparam);
  metric->Configure({});
  EXPECT_STREQ(metric->Name(), "error");

  delete metric;

  metric = xgboost::Metric::Create("error@0.1", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "error@0.1");
  EXPECT_STREQ(metric->Name(), "error@0.1");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.2f, 0.1f, 0.2f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);
  delete metric;
}

TEST(Metric, DeclareUnifiedTest(PoissionNegLogLik)) {
  auto lparam = xgboost::CreateEmptyGenericParam(0, NGPUS);
  xgboost::Metric * metric = xgboost::Metric::Create("poisson-nloglik", &lparam);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "poisson-nloglik");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.5f, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.2f, 0.1f, 0.2f},
                            {  0,   0,   1,   1}),
              1.1280f, 0.001f);
  delete metric;
}

#if defined(XGBOOST_USE_NCCL) && defined(__CUDACC__)
TEST(Metric, MGPU_RMSE) {
  {
    auto lparam = xgboost::CreateEmptyGenericParam(0, -1);
    xgboost::Metric * metric = xgboost::Metric::Create("rmse", &lparam);
    metric->Configure({});
    ASSERT_STREQ(metric->Name(), "rmse");
    EXPECT_NEAR(GetMetricEval(metric, {0}, {0}), 0, 1e-10);
    EXPECT_NEAR(GetMetricEval(metric,
                              {0.1f, 0.9f, 0.1f, 0.9f},
                              {  0,   0,   1,   1}),
                0.6403f, 0.001f);
    delete metric;
  }

  {
    auto lparam = xgboost::CreateEmptyGenericParam(1, -1);
    xgboost::Metric * metric = xgboost::Metric::Create("rmse", &lparam);
    ASSERT_STREQ(metric->Name(), "rmse");
    EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
    EXPECT_NEAR(GetMetricEval(metric,
                              {0.1f, 0.9f, 0.1f, 0.9f},
                              {  0,   0,   1,   1}),
                0.6403f, 0.001f);
    delete metric;
  }
}
#endif
