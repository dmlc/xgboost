// Copyright by Contributors
#include <xgboost/metric.h>
#include <string>

#include "../helpers.h"

inline void TestMultiClassError(xgboost::GPUSet devices) {
  auto lparam = xgboost::CreateEmptyGenericParam(0, NGPUS());
  lparam.gpu_id = *devices.begin();
  lparam.n_gpus = devices.Size();
  xgboost::Metric * metric = xgboost::Metric::Create("merror", &lparam);
  metric->Configure({});
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
  auto devices = xgboost::GPUSet::Range(0, NGPUS());
  TestMultiClassError(devices);
}

inline void TestMultiClassLogLoss(xgboost::GPUSet devices) {
  auto lparam = xgboost::CreateEmptyGenericParam(0, NGPUS());
  lparam.gpu_id = *devices.begin();
  lparam.n_gpus = devices.Size();
  xgboost::Metric * metric = xgboost::Metric::Create("mlogloss", &lparam);
  metric->Configure({});
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
  auto devices = xgboost::GPUSet::Range(0, NGPUS());
  TestMultiClassLogLoss(devices);
}

#if defined(XGBOOST_USE_NCCL) && defined(__CUDACC__)
TEST(Metric, MGPU_MultiClassError) {
  {
    auto devices = xgboost::GPUSet::All(0, -1);
    TestMultiClassError(devices);
  }
  {
    auto devices = xgboost::GPUSet::All(1, -1);
    TestMultiClassError(devices);
  }
  {
    auto devices = xgboost::GPUSet::All(0, -1);
    TestMultiClassLogLoss(devices);
  }
  {
    auto devices = xgboost::GPUSet::All(1, -1);
    TestMultiClassLogLoss(devices);
  }
}
#endif  // defined(XGBOOST_USE_NCCL)
