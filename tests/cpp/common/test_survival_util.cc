/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>

#include "../../../src/common/survival_util.h"

namespace xgboost {
namespace common {

template <typename Distribution>
inline static void RobustTestSuite(double y_lower, double y_upper, double sigma) {
  for (int i = 50; i >= -50; --i) {
    const double y_pred = std::pow(10.0, static_cast<double>(i));
    const double z = (std::log(y_lower) - std::log(y_pred)) / sigma;
    const double gradient
      = AFTLoss<Distribution>::Gradient(y_lower, y_upper, std::log(y_pred), sigma);
    const double hessian
      = AFTLoss<Distribution>::Hessian(y_lower, y_upper, std::log(y_pred), sigma);
    ASSERT_FALSE(std::isnan(gradient)) << "z = " << z << ", y \\in ["
      << y_lower << ", " << y_upper << "], y_pred = " << y_pred
      << ", dist = " << static_cast<int>(Distribution::Type());
    ASSERT_FALSE(std::isinf(gradient)) << "z = " << z << ", y \\in ["
      << y_lower << ", " << y_upper << "], y_pred = " << y_pred
      << ", dist = " << static_cast<int>(Distribution::Type());
    ASSERT_FALSE(std::isnan(hessian)) << "z = " << z << ", y \\in ["
      << y_lower << ", " << y_upper << "], y_pred = " << y_pred
      << ", dist = " << static_cast<int>(Distribution::Type());
    ASSERT_FALSE(std::isinf(hessian)) << "z = " << z << ", y \\in ["
      << y_lower << ", " << y_upper << "], y_pred = " << y_pred
      << ", dist = " << static_cast<int>(Distribution::Type());
  }
}

TEST(AFTLoss, RobustGradientPair) {  // Ensure that INF and NAN don't show up in gradient pair
  RobustTestSuite<NormalDistribution>(16.0, 200.0, 2.0);
  RobustTestSuite<LogisticDistribution>(16.0, 200.0, 2.0);
  RobustTestSuite<ExtremeDistribution>(16.0, 200.0, 2.0);
  RobustTestSuite<NormalDistribution>(100.0, 100.0, 2.0);
  RobustTestSuite<LogisticDistribution>(100.0, 100.0, 2.0);
  RobustTestSuite<ExtremeDistribution>(100.0, 100.0, 2.0);
}

}  // namespace common
}  // namespace xgboost
