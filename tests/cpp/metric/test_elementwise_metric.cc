/*!
 * Copyright 2018 XGBoost contributors
 */
#include <xgboost/metric.h>
#include <map>
#include "../helpers.h"

using Arg = std::pair<std::string, std::string>;

#if defined(__CUDACC__)
#define N_GPU() Arg{"n_gpus", "1"}
#else
#define N_GPU() Arg{"n_gpus", "0"}
#endif

TEST(Metric, DeclareUnifiedTest(RMSE)) {
  xgboost::Metric * metric = xgboost::Metric::Create("rmse");
  metric->Configure({N_GPU()});
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
  xgboost::Metric * metric = xgboost::Metric::Create("mae");
  metric->Configure({N_GPU()});
  ASSERT_STREQ(metric->Name(), "mae");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);
  delete metric;
}

TEST(Metric, DeclareUnifiedTest(LogLoss)) {
  xgboost::Metric * metric = xgboost::Metric::Create("logloss");
  metric->Configure({N_GPU()});
  ASSERT_STREQ(metric->Name(), "logloss");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              1.2039f, 0.001f);
  delete metric;
}

TEST(Metric, DeclareUnifiedTest(Error)) {
  xgboost::Metric * metric = xgboost::Metric::Create("error");
  metric->Configure({N_GPU()});
  ASSERT_STREQ(metric->Name(), "error");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.5f, 0.001f);

  EXPECT_ANY_THROW(xgboost::Metric::Create("error@abc"));
  delete metric;
  metric = xgboost::Metric::Create("error@0.5f");
  metric->Configure({N_GPU()});
  EXPECT_STREQ(metric->Name(), "error");

  delete metric;
  metric = xgboost::Metric::Create("error@0.1");
  metric->Configure({N_GPU()});
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
  xgboost::Metric * metric = xgboost::Metric::Create("poisson-nloglik");
  metric->Configure({N_GPU()});
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
    xgboost::Metric * metric = xgboost::Metric::Create("rmse");
    metric->Configure({Arg{"n_gpus", "-1"}});
    ASSERT_STREQ(metric->Name(), "rmse");
    EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0, 1e-10);
    EXPECT_NEAR(GetMetricEval(metric,
                              {0.1f, 0.9f, 0.1f, 0.9f},
                              {  0,   0,   1,   1}),
                0.6403f, 0.001f);
    delete metric;
  }

  {
    xgboost::Metric * metric = xgboost::Metric::Create("rmse");
    metric->Configure({Arg{"n_gpus", "-1"}, Arg{"gpu_id", "1"}});
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
