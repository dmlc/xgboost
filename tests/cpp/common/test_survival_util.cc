/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>

#include "../../../src/common/survival_util.h"

namespace xgboost {
namespace common {

TEST(AFTLoss, RobustGradientPair) {  // Ensure that INF and NAN don't show up
  const double y_lower = 16.0;
  const double y_upper = 200.0;
  const double sigma = 2.0;

  for (auto dist_type : { ProbabilityDistributionType::kNormal,
                          ProbabilityDistributionType::kLogistic,
                          ProbabilityDistributionType::kExtreme }) {
    AFTLoss loss(dist_type);
    for (int i = 50; i >= -50; --i) {
      const double y_pred = std::pow(10.0, static_cast<double>(i));
      const double z = (std::log(y_lower) - std::log(y_pred)) / sigma;
      const double gradient = loss.Gradient(y_lower, y_upper, std::log(y_pred), sigma);
      const double hessian = loss.Hessian(y_lower, y_upper, std::log(y_pred), sigma);
      ASSERT_FALSE(std::isnan(gradient)) << "z = " << z << ", y_pred = " << y_pred
        << ", dist = " << static_cast<int>(dist_type);
      ASSERT_FALSE(std::isinf(gradient)) << "z = " << z << ", y_pred = " << y_pred
        << ", dist = " << static_cast<int>(dist_type);
      ASSERT_FALSE(std::isnan(hessian)) << "z = " << z << ", y_pred = " << y_pred
        << ", dist = " << static_cast<int>(dist_type);
      ASSERT_FALSE(std::isinf(hessian)) << "z = " << z << ", y_pred = " << y_pred
        << ", dist = " << static_cast<int>(dist_type);
    }
  }
}

}  // namespace common
}  // namespace xgboost
