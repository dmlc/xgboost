/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <cmath>

#include "xgboost/metric.h"
#include "xgboost/logging.h"
#include "../helpers.h"
#include "../../../src/common/survival_util.h"

namespace xgboost {
namespace common {

TEST(AFTMetric, Configuration) {
  auto lparam = CreateEmptyGenericParam(-1);  // currently AFT metric is CPU only
  std::unique_ptr<Metric> metric(Metric::Create("aft-nloglik", &lparam));
  metric->Configure({ {"aft_loss_distribution", "normal"}, {"aft_loss_distribution_scale", "10"} });

  // Configuration round-trip test
  Json j_obj{ Object() };
  metric->SaveConfig(&j_obj);
  auto aft_param_json = j_obj["aft_loss_param"];
  ASSERT_EQ(get<String>(aft_param_json["aft_loss_distribution"]), "normal");
  ASSERT_EQ(get<String>(aft_param_json["aft_loss_distribution_scale"]), "10");
}

TEST(AFTLoss, Uncensored) {
  std::unique_ptr<AFTLoss> loss(new AFTLoss(AFTDistributionType::kNormal));

  // Given true label 100, compute the AFT loss for various prediction values
  const int num_point = 20;
  const double true_label = 100.0;
  std::vector<double> y_lower(num_point, true_label);
  std::vector<double> y_higher(y_lower);
  std::vector<double> y_pred(num_point);
  std::vector<double> loss_val(num_point);

  // Generate prediction value ranging from 2**1 to 2**15, using grid points in log scale
  const double log_y_low = 1.0;
  const double log_y_high = 15.0;
  for (int i = 0; i < num_point; ++i) {
    y_pred[i] = std::pow(2.0, i * (log_y_high - log_y_low) / (num_point - 1) + log_y_low);
    loss_val[i] = loss->Loss(y_lower[i], y_higher[i], std::log(y_pred[i]), 1.0);
    LOG(INFO) << y_pred[i] << ", " << loss_val[i];
  }
}

} // namespace common
}  // namespace xgboost
