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
  Json j_obj{ Object() };
  metric->SaveConfig(&j_obj);
  auto aft_param_json = j_obj["aft_loss_param"];
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution"]), "normal");
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution_scale"]), "10");

  CheckDeterministicMetricElementWise(StringView{"aft-nloglik"}, GPUIDX);
}
}  // namespace xgboost::common
