// Copyright by Contributors
#include <xgboost/metric.h>

#include "../helpers.h"

using Arg = std::pair<std::string, std::string>;

#if defined(__CUDACC__)
#define N_GPU() Arg{"n_gpus", "1"}
#else
#define N_GPU() Arg{"n_gpus", "0"}
#endif

TEST(Metric, DeclareUnifiedTest(MultiClassError)) {
  xgboost::Metric * metric = xgboost::Metric::Create("merror");
  metric->Configure({N_GPU()});
  ASSERT_STREQ(metric->Name(), "merror");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0}, {0, 0}));
  EXPECT_NEAR(GetMetricEval(
    metric, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 1, 2}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f},
                            {0, 1, 2}),
              0.666f, 0.001f);

  delete metric;
}

TEST(Metric, DeclareUnifiedTest(MultiClassLogLoss)) {
  xgboost::Metric * metric = xgboost::Metric::Create("mlogloss");
  metric->Configure({N_GPU()});
  ASSERT_STREQ(metric->Name(), "mlogloss");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0}, {0, 0}));
  EXPECT_NEAR(GetMetricEval(
    metric, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 1, 2}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f},
                            {0, 1, 2}),
              2.302f, 0.001f);

  delete metric;
}
