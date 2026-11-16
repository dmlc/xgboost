/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>

#include <cmath>

#include "../../../src/collective/communicator-inl.h"
#include "../../../src/common/survival_util.h"
#include "../helpers.h"
#include "xgboost/metric.h"

namespace xgboost {
namespace common {
inline void CheckDeterministicMetricElementWise(StringView name, int32_t device) {
  auto ctx = MakeCUDACtx(device);
  std::unique_ptr<Metric> metric{Metric::Create(name.c_str(), &ctx)};
  metric->Configure(Args{});

  HostDeviceVector<float> predts;
  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  auto& h_predts = predts.HostVector();

  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist{0.0f, 1.0f};

  size_t n_samples = 2048;
  h_predts.resize(n_samples);

  for (size_t i = 0; i < n_samples; ++i) {
    h_predts[i] = dist(&lcg);
  }

  auto& h_upper = info.labels_upper_bound_.HostVector();
  auto& h_lower = info.labels_lower_bound_.HostVector();
  h_lower.resize(n_samples);
  h_upper.resize(n_samples);
  for (size_t i = 0; i < n_samples; ++i) {
    h_lower[i] = 1;
    h_upper[i] = 10;
  }

  auto result = metric->Evaluate(predts, p_fmat);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(metric->Evaluate(predts, p_fmat), result);
  }
}

template <typename Distribution>
void VerifyWeightedAFT(Metric* metric) {
  auto dmat = EmptyDMatrix();
  auto& info = dmat->Info();
  info.num_row_ = 4;
  info.labels_lower_bound_.HostVector() = {100.0f, 0.0f, 60.0f, 16.0f};
  info.labels_upper_bound_.HostVector() = {100.0f, 20.0f, std::numeric_limits<float>::infinity(),
                                           200.0f};
  info.weights_.HostVector() = {1, 2, 0, 3};
  HostDeviceVector<float> preds{1.0f, 2.0f, 3.0f, 4.0f};
  metric->Configure({{"aft_loss_distribution_scale", "2.0"}});
  double expected = 0;
  for (std::size_t i = 0; i < preds.Size(); ++i) {
    expected += AFTLoss<Distribution>::Loss(info.labels_lower_bound_.HostVector()[i],
                                            info.labels_upper_bound_.HostVector()[i],
                                            preds.HostVector()[i], 2.0) *
                info.weights_.HostVector()[i];
  }
  expected /= 6.0;
  EXPECT_NEAR(metric->Evaluate(preds, dmat), expected, 1e-6);
  if (collective::GetWorldSize() > 1) {
    if (collective::GetRank() == 0) {
      info.num_row_ = 0;
      info.labels_lower_bound_.Resize(0);
      info.labels_upper_bound_.Resize(0);
      info.weights_.Resize(0);
      preds.Resize(0);
    }
    EXPECT_NEAR(metric->Evaluate(preds, dmat), expected, 1e-6);
  }
  info.num_row_ = 0;
  info.labels_lower_bound_.Resize(0);
  info.labels_upper_bound_.Resize(0);
  info.weights_.Resize(0);
  preds.Resize(0);
  EXPECT_EQ(metric->Evaluate(preds, dmat), 0.0);
}

inline void VerifyAFTNegLogLik(DeviceOrd device) {
  auto ctx = MakeCUDACtx(device.ordinal);

  /**
   * Test aggregate output from the AFT metric over a small test data set.
   * This is unlike AFTLoss.* tests, which verify metric values over individual data points.
   **/
  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  info.num_row_ = 4;
  info.labels_lower_bound_.HostVector() = {100.0f, 0.0f, 60.0f, 16.0f};
  info.labels_upper_bound_.HostVector() = {100.0f, 20.0f,
                                           std::numeric_limits<bst_float>::infinity(), 200.0f};
  info.weights_.HostVector() = std::vector<bst_float>();
  HostDeviceVector<bst_float> preds(4, std::log(64));

  struct TestCase {
    std::string dist_type;
    bst_float reference_value;
  };
  for (const auto& test_case :
       std::vector<TestCase>{{"normal", 2.1508f}, {"logistic", 2.1804f}, {"extreme", 2.0706f}}) {
    std::unique_ptr<Metric> metric(Metric::Create("aft-nloglik", &ctx));
    metric->Configure(
        {{"aft_loss_distribution", test_case.dist_type}, {"aft_loss_distribution_scale", "1.0"}});
    EXPECT_NEAR(metric->Evaluate(preds, p_fmat), test_case.reference_value, 1e-4);
    if (test_case.dist_type == "normal") {
      VerifyWeightedAFT<NormalDistribution>(metric.get());
    } else if (test_case.dist_type == "logistic") {
      VerifyWeightedAFT<LogisticDistribution>(metric.get());
    } else {
      VerifyWeightedAFT<ExtremeDistribution>(metric.get());
    }
  }
}

inline void VerifyIntervalRegressionAccuracy(DeviceOrd device) {
  auto ctx = MakeCUDACtx(device.ordinal);

  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  info.num_row_ = 4;
  info.labels_lower_bound_.HostVector() = {20.0f, 0.0f, 60.0f, 16.0f};
  info.labels_upper_bound_.HostVector() = {80.0f, 20.0f, 80.0f, 200.0f};
  info.weights_.HostVector() = std::vector<bst_float>();
  HostDeviceVector<bst_float> preds(4, std::log(60.0f));

  std::unique_ptr<Metric> metric(Metric::Create("interval-regression-accuracy", &ctx));
  EXPECT_FLOAT_EQ(metric->Evaluate(preds, p_fmat), 0.75f);
  info.labels_lower_bound_.HostVector()[2] = 70.0f;
  EXPECT_FLOAT_EQ(metric->Evaluate(preds, p_fmat), 0.50f);
  info.labels_upper_bound_.HostVector()[2] = std::numeric_limits<bst_float>::infinity();
  EXPECT_FLOAT_EQ(metric->Evaluate(preds, p_fmat), 0.50f);
  info.labels_upper_bound_.HostVector()[3] = std::numeric_limits<bst_float>::infinity();
  EXPECT_FLOAT_EQ(metric->Evaluate(preds, p_fmat), 0.50f);
  info.labels_lower_bound_.HostVector()[0] = 70.0f;
  EXPECT_FLOAT_EQ(metric->Evaluate(preds, p_fmat), 0.25f);

  info.weights_.HostVector() = {1, 2, 3, 4};
  EXPECT_NEAR(metric->Evaluate(preds, p_fmat), 0.4, 1e-6);
  info.weights_.HostVector() = {0, 0, 0, 0};
  EXPECT_EQ(metric->Evaluate(preds, p_fmat), 0.0);
  info.num_row_ = 0;
  info.labels_lower_bound_.Resize(0);
  info.labels_upper_bound_.Resize(0);
  info.weights_.Resize(0);
  preds.Resize(0);
  EXPECT_EQ(metric->Evaluate(preds, p_fmat), 0.0);

  CheckDeterministicMetricElementWise(StringView{"interval-regression-accuracy"}, device.ordinal);
}
}  // namespace common
}  // namespace xgboost
