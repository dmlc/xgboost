/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>

#include <cmath>

#include "../../../src/common/survival_util.h"
#include "../helpers.h"
#include "xgboost/metric.h"

namespace xgboost {
namespace common {
inline void CheckDeterministicMetricElementWise(StringView name, int32_t device) {
  auto ctx = CreateEmptyGenericParam(device);
  std::unique_ptr<Metric> metric{Metric::Create(name.c_str(), &ctx)};
  metric->Configure(Args{});

  HostDeviceVector<float> predts;
  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  auto &h_predts = predts.HostVector();

  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist{0.0f, 1.0f};

  size_t n_samples = 2048;
  h_predts.resize(n_samples);

  for (size_t i = 0; i < n_samples; ++i) {
    h_predts[i] = dist(&lcg);
  }

  auto &h_upper = info.labels_upper_bound_.HostVector();
  auto &h_lower = info.labels_lower_bound_.HostVector();
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

inline void VerifyAFTNegLogLik(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);

  /**
   * Test aggregate output from the AFT metric over a small test data set.
   * This is unlike AFTLoss.* tests, which verify metric values over individual data points.
   **/
  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  info.num_row_ = 4;
  info.labels_lower_bound_.HostVector()
      = { 100.0f, 0.0f, 60.0f, 16.0f };
  info.labels_upper_bound_.HostVector()
      = { 100.0f, 20.0f, std::numeric_limits<bst_float>::infinity(), 200.0f };
  info.weights_.HostVector() = std::vector<bst_float>();
  info.data_split_mode = data_split_mode;
  HostDeviceVector<bst_float> preds(4, std::log(64));

  struct TestCase {
    std::string dist_type;
    bst_float reference_value;
  };
  for (const auto& test_case : std::vector<TestCase>{ {"normal", 2.1508f}, {"logistic", 2.1804f},
                                                     {"extreme", 2.0706f} }) {
    std::unique_ptr<Metric> metric(Metric::Create("aft-nloglik", &ctx));
    metric->Configure({ {"aft_loss_distribution", test_case.dist_type},
                       {"aft_loss_distribution_scale", "1.0"} });
    EXPECT_NEAR(metric->Evaluate(preds, p_fmat), test_case.reference_value, 1e-4);
  }
}

inline void VerifyIntervalRegressionAccuracy(DataSplitMode data_split_mode = DataSplitMode::kRow) {
  auto ctx = xgboost::CreateEmptyGenericParam(GPUIDX);

  auto p_fmat = EmptyDMatrix();
  MetaInfo& info = p_fmat->Info();
  info.num_row_ = 4;
  info.labels_lower_bound_.HostVector() = { 20.0f, 0.0f, 60.0f, 16.0f };
  info.labels_upper_bound_.HostVector() = { 80.0f, 20.0f, 80.0f, 200.0f };
  info.weights_.HostVector() = std::vector<bst_float>();
  info.data_split_mode = data_split_mode;
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

  CheckDeterministicMetricElementWise(StringView{"interval-regression-accuracy"}, GPUIDX);
}
}  // namespace common
}  // namespace xgboost
