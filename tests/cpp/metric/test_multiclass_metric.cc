// Copyright by Contributors
#include <xgboost/metric.h>
#include <string>

#include "../helpers.h"

inline void TestMultiClassError(int device) {
  auto lparam = xgboost::CreateEmptyGenericParam(device);
  lparam.gpu_id = device;
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
  TestMultiClassError(GPUIDX);
}

inline void TestMultiClassLogLoss(int device) {
  auto lparam = xgboost::CreateEmptyGenericParam(device);
  lparam.gpu_id = device;
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
  TestMultiClassLogLoss(GPUIDX);
}

#if defined(XGBOOST_USE_NCCL) && defined(__CUDACC__)
namespace xgboost {
namespace common {
TEST(Metric, MGPU_MultiClassError) {
  if (AllVisibleGPUs() < 2) {
    LOG(WARNING) << "Not testing in multi-gpu environment.";
    return;
  }

  {
    TestMultiClassError(0);
  }
  {
    TestMultiClassError(1);
  }
  {
    TestMultiClassLogLoss(0);
  }
  {
    TestMultiClassLogLoss(1);
  }
}
}  // namespace common
}  // namespace xgboost
#endif  // defined(XGBOOST_USE_NCCL)
