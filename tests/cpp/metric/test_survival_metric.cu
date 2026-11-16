/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>

#include "test_survival_metric.h"
#include "xgboost/metric.h"

/** Tests for Survival metrics that should run both on CPU and GPU **/

namespace xgboost::common {
// Test configuration of AFT metric
TEST(AFTNegLogLikMetric, DeclareUnifiedTest(Configuration)) {
  auto ctx = MakeCUDACtx(GPUIDX);
  std::unique_ptr<Metric> metric(Metric::Create("aft-nloglik", &ctx));
  metric->Configure({{"aft_loss_distribution", "normal"}, {"aft_loss_distribution_scale", "10"}});

  // Configuration round-trip test
  Json j_obj{Object()};
  metric->SaveConfig(&j_obj);
  auto aft_param_json = j_obj["aft_loss_param"];
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution"]), "normal");
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution_scale"]), "10");

  CheckDeterministicMetricElementWise(StringView{"aft-nloglik"}, GPUIDX);
}

TEST(AFTNegLogLikMetric, DeclareUnifiedTest(LoadConfig)) {
  auto ctx = MakeCUDACtx(GPUIDX);
  auto dmat = EmptyDMatrix();
  auto& info = dmat->Info();
  info.num_row_ = 4;
  info.labels_lower_bound_.HostVector() = {100.0f, 0.0f, 60.0f, 16.0f};
  info.labels_upper_bound_.HostVector() = {100.0f, 20.0f, std::numeric_limits<float>::infinity(),
                                           200.0f};
  info.weights_.HostVector() = {1, 2, 3, 4};
  HostDeviceVector<float> preds{1.0f, 2.0f, 3.0f, 4.0f};
  for (auto distribution : {"normal", "logistic", "extreme"}) {
    std::unique_ptr<Metric> source{Metric::Create("aft-nloglik", &ctx)};
    source->Configure(
        {{"aft_loss_distribution", distribution}, {"aft_loss_distribution_scale", "2.5"}});
    auto expected = source->Evaluate(preds, dmat);
    Json config{Object{}};
    source->SaveConfig(&config);

    std::unique_ptr<Metric> restored{Metric::Create("aft-nloglik", &ctx)};
    restored->LoadConfig(config);
    EXPECT_NEAR(restored->Evaluate(preds, dmat), expected, 1e-6);

    restored->Configure(
        {{"aft_loss_distribution", "normal"}, {"aft_loss_distribution_scale", "1.0"}});
    restored->LoadConfig(config);
    EXPECT_NEAR(restored->Evaluate(preds, dmat), expected, 1e-6);
    Json reloaded{Object{}};
    restored->SaveConfig(&reloaded);
    EXPECT_EQ(config, reloaded);
  }
}
}  // namespace xgboost::common
