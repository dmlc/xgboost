// Copyright by Contributors
#ifndef TESTS_CPP_METRIC_TEST_MULTICLASS_METRIC_H_
#define TESTS_CPP_METRIC_TEST_MULTICLASS_METRIC_H_

#include <xgboost/metric.h>

#include <cmath>
#include <string>

#include "../helpers.h"

namespace xgboost {
namespace metric {

inline void CheckDeterministicMetricMultiClass(StringView name, int32_t device) {
  auto ctx = MakeCUDACtx(device);
  std::unique_ptr<Metric> metric{Metric::Create(name.c_str(), &ctx)};

  HostDeviceVector<float> predts;
  auto p_fmat = EmptyDMatrix();
  MetaInfo &info = p_fmat->Info();
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

  auto result = metric->Evaluate(predts, p_fmat);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(metric->Evaluate(predts, p_fmat), result);
  }
}

inline void TestMultiClassError(DeviceOrd device) {
  auto ctx = MakeCUDACtx(device.ordinal);
  xgboost::Metric *metric = xgboost::Metric::Create("merror", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "merror");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0}, {0, 0}, {}, {}));
  EXPECT_NEAR(GetMetricEval(metric, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 1, 2}, {}, {}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f},
                            {0, 1, 2}, {}, {}),
              0.666f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric, {0.9f, 0.1f, 0.8f, 0.2f}, {0, 1}, {1, 3}, {}), 0.75f, 1e-6);
  EXPECT_THROW(GetMetricEval(metric, {0.9f, 0.1f}, {-1}, {}, {}), dmlc::Error);
  EXPECT_THROW(GetMetricEval(metric, {0.9f, 0.1f}, {2}, {}, {}), dmlc::Error);
  EXPECT_NEAR(GetMetricEval(metric, {0.9f, 0.1f}, {0}, {}, {}), 0.0f, 1e-6);
  delete metric;
}

inline void VerifyMultiClassError(DeviceOrd device) {
  TestMultiClassError(device);
  CheckDeterministicMetricMultiClass(StringView{"merror"}, device.ordinal);
}

inline void TestMultiClassLogLoss(DeviceOrd device) {
  auto ctx = MakeCUDACtx(device.ordinal);
  xgboost::Metric *metric = xgboost::Metric::Create("mlogloss", &ctx);
  metric->Configure({});
  ASSERT_STREQ(metric->Name(), "mlogloss");
  EXPECT_ANY_THROW(GetMetricEval(metric, {0}, {0, 0}, {}, {}));
  EXPECT_NEAR(GetMetricEval(metric, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 1, 2}, {}, {}), 0, 1e-10);
  EXPECT_NEAR(GetMetricEval(metric, {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f},
                            {0, 1, 2}, {}, {}),
              2.302f, 0.001f);

  EXPECT_NEAR(GetMetricEval(metric, {0.8f, 0.2f, 0.6f, 0.4f}, {0, 1}, {1, 3}, {}),
              (-std::log(0.8) - 3.0 * std::log(0.4)) / 4.0, 1e-6);
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0}, {}, {}), -std::log(1e-16), 1e-5);
  EXPECT_THROW(GetMetricEval(metric, {0.9f, 0.1f}, {-1}, {}, {}), dmlc::Error);
  EXPECT_THROW(GetMetricEval(metric, {0.9f, 0.1f}, {2}, {}, {}), dmlc::Error);
  EXPECT_NEAR(GetMetricEval(metric, {1, 0}, {0}, {}, {}), 0.0f, 1e-6);
  delete metric;
}

inline void VerifyMultiClassLogLoss(DeviceOrd device) {
  TestMultiClassLogLoss(device);
  CheckDeterministicMetricMultiClass(StringView{"mlogloss"}, device.ordinal);
}

}  // namespace metric
}  // namespace xgboost

#endif  // TESTS_CPP_METRIC_TEST_MULTICLASS_METRIC_H_
