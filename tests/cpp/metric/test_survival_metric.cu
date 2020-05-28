/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>
#include <cmath>
#include "xgboost/metric.h"
#include "../helpers.h"
#include "../../../src/common/survival_util.h"

/** Tests for Survival metrics that should run both on CPU and GPU **/

namespace xgboost {
namespace common {

TEST(Metric, DeclareUnifiedTest(AFTNegLogLik)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);

  /**
   * Test aggregate output from the AFT metric over a small test data set.
   * This is unlike AFTLoss.* tests, which verify metric values over individual data points.
   **/
  MetaInfo info;
  info.num_row_ = 4;
  info.labels_lower_bound_.HostVector()
    = { 100.0f, 0.0f, 60.0f, 16.0f };
  info.labels_upper_bound_.HostVector()
    = { 100.0f, 20.0f, std::numeric_limits<bst_float>::infinity(), 200.0f };
  info.weights_.HostVector() = std::vector<bst_float>();
  HostDeviceVector<bst_float> preds(4, std::log(64));

  struct TestCase {
    std::string dist_type;
    bst_float reference_value;
  };
  for (const auto& test_case : std::vector<TestCase>{ {"normal", 2.1508f}, {"logistic", 2.1804f},
                                                      {"extreme", 2.0706f} }) {
    std::unique_ptr<Metric> metric(Metric::Create("aft-nloglik", &lparam));
    metric->Configure({ {"aft_loss_distribution", test_case.dist_type},
                        {"aft_loss_distribution_scale", "1.0"} });
    EXPECT_NEAR(metric->Eval(preds, info, false), test_case.reference_value, 1e-4);
  }
}

TEST(Metric, DeclareUnifiedTest(IntervalRegressionAccuracy)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);

  MetaInfo info;
  info.num_row_ = 4;
  info.labels_lower_bound_.HostVector() = { 20.0f, 0.0f, 60.0f, 16.0f };
  info.labels_upper_bound_.HostVector() = { 80.0f, 20.0f, 80.0f, 200.0f };
  info.weights_.HostVector() = std::vector<bst_float>();
  HostDeviceVector<bst_float> preds(4, std::log(60.0f));

  std::unique_ptr<Metric> metric(Metric::Create("interval-regression-accuracy", &lparam));
  EXPECT_FLOAT_EQ(metric->Eval(preds, info, false), 0.75f);
  info.labels_lower_bound_.HostVector()[2] = 70.0f;
  EXPECT_FLOAT_EQ(metric->Eval(preds, info, false), 0.50f);
  info.labels_upper_bound_.HostVector()[2] = std::numeric_limits<bst_float>::infinity();
  EXPECT_FLOAT_EQ(metric->Eval(preds, info, false), 0.50f);
  info.labels_upper_bound_.HostVector()[3] = std::numeric_limits<bst_float>::infinity();
  EXPECT_FLOAT_EQ(metric->Eval(preds, info, false), 0.50f);
  info.labels_lower_bound_.HostVector()[0] = 70.0f;
  EXPECT_FLOAT_EQ(metric->Eval(preds, info, false), 0.25f);
}

// Test configuration of AFT metric
TEST(AFTNegLogLikMetric, DeclareUnifiedTest(Configuration)) {
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Metric> metric(Metric::Create("aft-nloglik", &lparam));
  metric->Configure({{"aft_loss_distribution", "normal"}, {"aft_loss_distribution_scale", "10"}});

  // Configuration round-trip test
  Json j_obj{ Object() };
  metric->SaveConfig(&j_obj);
  auto aft_param_json = j_obj["aft_loss_param"];
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution"]), "normal");
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution_scale"]), "10");
}

}  // namespace common
}  // namespace xgboost
