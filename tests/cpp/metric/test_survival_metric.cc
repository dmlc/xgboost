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
}

} // namespace common
}  // namespace xgboost
