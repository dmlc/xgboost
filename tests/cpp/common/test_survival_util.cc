/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>

#include "../../../src/common/survival_util.h"

namespace xgboost {
namespace common {

inline static void RobustTestSuite(ProbabilityDistributionType dist_type,
                                   double y_lower, double y_upper, double sigma) {
  AFTLoss loss(dist_type);
  for (int i = 50; i >= -50; --i) {
    const double y_pred = std::pow(10.0, static_cast<double>(i));
    const double z = (std::log(y_lower) - std::log(y_pred)) / sigma;
    const double gradient = loss.Gradient(y_lower, y_upper, std::log(y_pred), sigma);
    const double hessian = loss.Hessian(y_lower, y_upper, std::log(y_pred), sigma);
    ASSERT_FALSE(std::isnan(gradient)) << "z = " << z << ", y \\in ["
      << y_lower << ", " << y_upper << "], y_pred = " << y_pred
      << ", dist = " << static_cast<int>(dist_type);
    ASSERT_FALSE(std::isinf(gradient)) << "z = " << z << ", y \\in ["
      << y_lower << ", " << y_upper << "], y_pred = " << y_pred
      << ", dist = " << static_cast<int>(dist_type);
    ASSERT_FALSE(std::isnan(hessian)) << "z = " << z << ", y \\in ["
      << y_lower << ", " << y_upper << "], y_pred = " << y_pred
      << ", dist = " << static_cast<int>(dist_type);
    ASSERT_FALSE(std::isinf(hessian)) << "z = " << z << ", y \\in ["
      << y_lower << ", " << y_upper << "], y_pred = " << y_pred
      << ", dist = " << static_cast<int>(dist_type);
  }
}

TEST(AFTLoss, RobustGradientPair) {  // Ensure that INF and NAN don't show up in gradient pair
  RobustTestSuite(ProbabilityDistributionType::kNormal, 16.0, 200.0, 2.0);
  RobustTestSuite(ProbabilityDistributionType::kLogistic, 16.0, 200.0, 2.0);
  RobustTestSuite(ProbabilityDistributionType::kExtreme, 16.0, 200.0, 2.0);
  RobustTestSuite(ProbabilityDistributionType::kNormal, 100.0, 100.0, 2.0);
  RobustTestSuite(ProbabilityDistributionType::kLogistic, 100.0, 100.0, 2.0);
  RobustTestSuite(ProbabilityDistributionType::kExtreme, 100.0, 100.0, 2.0);
}

}  // namespace common
}  // namespace xgboost
