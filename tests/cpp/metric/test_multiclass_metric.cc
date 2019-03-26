// Copyright by Contributors
#include <xgboost/metric/metric.h>
#include <string>

#include "../helpers.h"

using Arg = std::pair<std::string, std::string>;

#if defined(__CUDACC__)
#define N_GPU() Arg{"n_gpus", "1"}
#else
#define N_GPU() Arg{"n_gpus", "0"}
#endif

inline void TestMultiClassError(std::vector<Arg> args) {
  xgboost::Metric * metric = xgboost::Metric::Create("merror");
  metric->Configure(args);
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

TEST(Metric, DeclareUnifiedTest(MultiClassError)) {
  TestMultiClassError({N_GPU()});
}

inline void TestMultiClassLogLoss(std::vector<Arg> args) {
  xgboost::Metric * metric = xgboost::Metric::Create("mlogloss");
  metric->Configure(args);
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

TEST(Metric, DeclareUnifiedTest(MultiClassLogLoss)) {
  TestMultiClassLogLoss({N_GPU()});
}

#if defined(XGBOOST_USE_NCCL) && defined(__CUDACC__)
TEST(Metric, MGPU_MultiClassError) {
  TestMultiClassError({Arg{"n_gpus", "-1"}});
  TestMultiClassError({Arg{"n_gpus", "-1"}, Arg{"gpu_id", "1"}});

  TestMultiClassLogLoss({Arg{"n_gpus", "-1"}});
  TestMultiClassLogLoss({Arg{"n_gpus", "-1"}, Arg{"gpu_id", "1"}});
}
#endif  // defined(XGBOOST_USE_NCCL)
