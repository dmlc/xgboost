/*!
 * Copyright 2019-2020 by Contributors
 * \file probability_distribution.h
 * \brief Implementation of a few useful probability distributions
 * \author Avinash Barnwal and Hyunsu Cho
 */

#ifndef XGBOOST_COMMON_PROBABILITY_DISTRIBUTION_H_
#define XGBOOST_COMMON_PROBABILITY_DISTRIBUTION_H_

#include <cmath>

namespace xgboost {
namespace common {

#ifndef __CUDACC__

using std::exp;
using std::sqrt;
using std::isinf;
using std::isnan;

#endif  // __CUDACC__

/*! \brief Constant PI */
constexpr double kPI = 3.14159265358979323846;
/*! \brief The Euler-Mascheroni_constant */
constexpr double kEulerMascheroni = 0.57721566490153286060651209008240243104215933593992;

/*! \brief Enum encoding possible choices of probability distribution */
enum class ProbabilityDistributionType : int {
  kNormal = 0, kLogistic = 1, kExtreme = 2
};

struct NormalDistribution {
  XGBOOST_DEVICE static double PDF(double z) {
    return exp(-z * z / 2.0) / sqrt(2.0 * kPI);
  }

  XGBOOST_DEVICE static double CDF(double z) {
    return 0.5 * (1 + erf(z / sqrt(2.0)));
  }

  XGBOOST_DEVICE static double GradPDF(double z) {
    return -z * PDF(z);
  }

  XGBOOST_DEVICE static double HessPDF(double z) {
    return (z * z - 1.0) * PDF(z);
  }

  XGBOOST_DEVICE static ProbabilityDistributionType Type() {
    return ProbabilityDistributionType::kNormal;
  }
};

struct LogisticDistribution {
  XGBOOST_DEVICE static double PDF(double z) {
    const double w = exp(z);
    const double sqrt_denominator = 1 + w;
    if (isinf(w) || isinf(w * w)) {
      return 0.0;
    } else {
      return w / (sqrt_denominator * sqrt_denominator);
    }
  }

  XGBOOST_DEVICE static double CDF(double z) {
    const double w = exp(z);
    return isinf(w) ? 1.0 : (w / (1 + w));
  }

  XGBOOST_DEVICE static double GradPDF(double z) {
    const double w = exp(z);
    return isinf(w) ? 0.0 : (PDF(z) * (1 - w) / (1 + w));
  }

  XGBOOST_DEVICE static double HessPDF(double z) {
    const double w = exp(z);
    if (isinf(w) || isinf(w * w)) {
      return 0.0;
    } else {
      return PDF(z) * (w * w - 4 * w + 1) / ((1 + w) * (1 + w));
    }
  }

  XGBOOST_DEVICE static ProbabilityDistributionType Type() {
    return ProbabilityDistributionType::kLogistic;
  }
};

struct ExtremeDistribution {
  XGBOOST_DEVICE static double PDF(double z) {
    const double w = exp(z);
    return isinf(w) ? 0.0 : (w * exp(-w));
  }

  XGBOOST_DEVICE static double CDF(double z) {
    const double w = exp(z);
    return 1 - exp(-w);
  }

  XGBOOST_DEVICE static double GradPDF(double z) {
    const double w = exp(z);
    return isinf(w) ? 0.0 : ((1 - w) * PDF(z));
  }

  XGBOOST_DEVICE static double HessPDF(double z) {
    const double w = exp(z);
    if (isinf(w) || isinf(w * w)) {
      return 0.0;
    } else {
      return (w * w - 3 * w + 1) * PDF(z);
    }
  }

  XGBOOST_DEVICE static ProbabilityDistributionType Type() {
    return ProbabilityDistributionType::kExtreme;
  }
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_PROBABILITY_DISTRIBUTION_H_
