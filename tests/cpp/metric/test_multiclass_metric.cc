// Copyright by Contributors
#include <xgboost/metric.h>
#include <string>

#include "../helpers.h"

namespace xgboost {
inline void CheckDeterministicMetricMultiClass(StringView name, int32_t device) {
  auto lparam = CreateEmptyGenericParam(device);
  std::unique_ptr<Metric> metric{Metric::Create(name.c_str(), &lparam)};

  HostDeviceVector<float> predts;
  MetaInfo info;
  auto &h_predts = predts.HostVector();

  SimpleLCG lcg;

  size_t n_samples = 2048, n_classes = 4;

  info.labels.Reshape(n_samples);
  auto &h_labels = info.labels.Data()->HostVector();
  h_predts.resize(n_samples * n_classes);

  {
    SimpleRealUniformDistribution<float> dist{0.0f, static_cast<float>(n_classes)};
    for (size_t i = 0; i < n_samples; ++i) {
      h_labels[i] = dist(&lcg);
    }
  }

  {
    SimpleRealUniformDistribution<float> dist{0.0f, 1.0f};
    for (size_t i = 0; i < n_samples * n_classes; ++i) {
      h_predts[i] = dist(&lcg);
    }
  }

  auto result = metric->Eval(predts, info, false);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(metric->Eval(predts, info, false), result);
  }
}
}  // namespace xgboost

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
  xgboost::CheckDeterministicMetricMultiClass(xgboost::StringView{"merror"}, GPUIDX);
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
  xgboost::CheckDeterministicMetricMultiClass(xgboost::StringView{"mlogloss"}, GPUIDX);
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
