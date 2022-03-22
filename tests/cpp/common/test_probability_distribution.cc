/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>
#include <memory>
#include <cmath>

#include "xgboost/logging.h"
#include "../../../src/common/probability_distribution.h"

namespace xgboost {
namespace common {

template <typename Distribution>
void RunDistributionGenericTest() {
  double integral_of_pdf = Distribution::CDF(-2.0);
  double integral_of_grad_pdf = Distribution::PDF(-2.0);
  double integral_of_hess_pdf = Distribution::GradPDF(-2.0);
  // Perform numerical differentiation and integration
  // Enumerate 4000 grid points in range [-2, 2]
  for (int i = 0; i <= 4000; ++i) {
    const double x = static_cast<double>(i) / 1000.0 - 2.0;
    // Numerical differentiation (p. 246, Numerical Analysis 2nd ed. by Timothy Sauer)
    EXPECT_NEAR((Distribution::CDF(x + 1e-5) - Distribution::CDF(x - 1e-5)) / 2e-5,
                Distribution::PDF(x), 6e-11);
    EXPECT_NEAR((Distribution::PDF(x + 1e-5) - Distribution::PDF(x - 1e-5)) / 2e-5,
                Distribution::GradPDF(x), 6e-11);
    EXPECT_NEAR((Distribution::GradPDF(x + 1e-5) - Distribution::GradPDF(x - 1e-5)) / 2e-5,
                Distribution::HessPDF(x), 6e-11);
    // Numerical integration using Trapezoid Rule (p. 257, Sauer)
    integral_of_pdf += 5e-4 * (Distribution::PDF(x - 1e-3) + Distribution::PDF(x));
    integral_of_grad_pdf += 5e-4 * (Distribution::GradPDF(x - 1e-3) + Distribution::GradPDF(x));
    integral_of_hess_pdf += 5e-4 * (Distribution::HessPDF(x - 1e-3) + Distribution::HessPDF(x));
    EXPECT_NEAR(integral_of_pdf, Distribution::CDF(x), 2e-4);
    EXPECT_NEAR(integral_of_grad_pdf, Distribution::PDF(x), 2e-4);
    EXPECT_NEAR(integral_of_hess_pdf, Distribution::GradPDF(x), 2e-4);
  }
}

TEST(ProbabilityDistribution, DistributionGeneric) {
  // Assert d/dx CDF = PDF, d/dx PDF = GradPDF, d/dx GradPDF = HessPDF
  // Do this for every distribution type
  RunDistributionGenericTest<NormalDistribution>();
  RunDistributionGenericTest<LogisticDistribution>();
  RunDistributionGenericTest<ExtremeDistribution>();
}

TEST(ProbabilityDistribution, NormalDist) {
  // "Three-sigma rule" (https://en.wikipedia.org/wiki/68–95–99.7_rule)
  //   68% of values are within 1 standard deviation away from the mean
  //   95% of values are within 2 standard deviation away from the mean
  // 99.7% of values are within 3 standard deviation away from the mean
  EXPECT_NEAR(NormalDistribution::CDF(0.5) - NormalDistribution::CDF(-0.5), 0.3829, 0.00005);
  EXPECT_NEAR(NormalDistribution::CDF(1.0) - NormalDistribution::CDF(-1.0), 0.6827, 0.00005);
  EXPECT_NEAR(NormalDistribution::CDF(1.5) - NormalDistribution::CDF(-1.5), 0.8664, 0.00005);
  EXPECT_NEAR(NormalDistribution::CDF(2.0) - NormalDistribution::CDF(-2.0), 0.9545, 0.00005);
  EXPECT_NEAR(NormalDistribution::CDF(2.5) - NormalDistribution::CDF(-2.5), 0.9876, 0.00005);
  EXPECT_NEAR(NormalDistribution::CDF(3.0) - NormalDistribution::CDF(-3.0), 0.9973, 0.00005);
  EXPECT_NEAR(NormalDistribution::CDF(3.5) - NormalDistribution::CDF(-3.5), 0.9995, 0.00005);
  EXPECT_NEAR(NormalDistribution::CDF(4.0) - NormalDistribution::CDF(-4.0), 0.9999, 0.00005);
}

TEST(ProbabilityDistribution, LogisticDist) {
  /**
   * Enforce known properties of the logistic distribution.
   * (https://en.wikipedia.org/wiki/Logistic_distribution)
   **/

  // Enumerate 4000 grid points in range [-2, 2]
  for (int i = 0; i <= 4000; ++i) {
    const double x = static_cast<double>(i) / 1000.0 - 2.0;
    // PDF = 1/4 * sech(x/2)**2
    const double sech_x = 1.0 / std::cosh(x * 0.5);  // hyperbolic secant at x/2
    EXPECT_NEAR(0.25 * sech_x * sech_x, LogisticDistribution::PDF(x), 1e-15);
    // CDF = 1/2 + 1/2 * tanh(x/2)
    EXPECT_NEAR(0.5 + 0.5 * std::tanh(x * 0.5), LogisticDistribution::CDF(x), 1e-15);
  }
}

TEST(ProbabilityDistribution, ExtremeDist) {
  /**
   * Enforce known properties of the extreme distribution (also known as Gumbel distribution).
   * The mean is the negative of the Euler-Mascheroni constant.
   * The variance is 1/6 * pi**2. (https://mathworld.wolfram.com/GumbelDistribution.html)
   **/

  // Enumerate 25000 grid points in range [-20, 5].
  // Compute the mean (expected value) of the distribution using numerical integration.
  // Nearly all mass of the extreme distribution is concentrated between -20 and 5,
  // so numerically integrating x*PDF(x) over [-20, 5] gives good estimate of the mean.
  double mean = 0.0;
  for (int i = 0; i <= 25000; ++i) {
    const double x = static_cast<double>(i) / 1000.0 - 20.0;
    // Numerical integration using Trapezoid Rule (p. 257, Sauer)
    mean +=
      5e-4 * ((x - 1e-3) * ExtremeDistribution::PDF(x - 1e-3) + x * ExtremeDistribution::PDF(x));
  }
  EXPECT_NEAR(mean, -kEulerMascheroni, 1e-7);

  // Enumerate 25000 grid points in range [-20, 5].
  // Compute the variance of the distribution using numerical integration.
  // Nearly all mass of the extreme distribution is concentrated between -20 and 5,
  // so numerically integrating (x-mean)*PDF(x) over [-20, 5] gives good estimate of the variance.
  double variance = 0.0;
  for (int i = 0; i <= 25000; ++i) {
    const double x = static_cast<double>(i) / 1000.0 - 20.0;
    // Numerical integration using Trapezoid Rule (p. 257, Sauer)
    variance += 5e-4 * ((x - 1e-3 - mean) * (x - 1e-3 - mean) * ExtremeDistribution::PDF(x - 1e-3)
                        + (x - mean) * (x - mean) * ExtremeDistribution::PDF(x));
  }
  EXPECT_NEAR(variance, kPI * kPI / 6.0, 1e-6);
}

} // namespace common
}  // namespace xgboost
