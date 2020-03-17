/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>
#include <memory>

#include "xgboost/metric.h"
#include "xgboost/logging.h"
#include "../helpers.h"

namespace xgboost {
namespace common {

TEST(AFTMetric, Configuration) {
  auto lparam = CreateEmptyGenericParam(-1);  // currently AFT metric is CPU only
  std::unique_ptr<Metric> metric(Metric::Create("aft-nloglik", &lparam));
  metric->Configure({ {"aft_noise_distribution", "normal"}, {"aft_sigma", "10"} });

  // Configuration round-trip test
  Json j_obj{ Object() };
  metric->SaveConfig(&j_obj);
  auto aft_param_json = j_obj["aft_loss_param"];
  ASSERT_EQ(get<String>(aft_param_json["aft_noise_distribution"]), "normal");
  ASSERT_EQ(get<String>(aft_param_json["aft_sigma"]), "10");
}

} // namespace common
}  // namespace xgboost
