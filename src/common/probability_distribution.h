/*!
 * Copyright 2020 by Contributors
 * \file probability_distribution.h
 * \brief Implementation of a few useful probability distributions
 * \author Avinash Barnwal and Hyunsu Cho
 */

#ifndef XGBOOST_COMMON_PROBABILITY_DISTRIBUTION_H_
#define XGBOOST_COMMON_PROBABILITY_DISTRIBUTION_H_

namespace xgboost {
namespace common {

namespace probability_constant {

/*! \brief Constant PI */
const double kPI = 3.14159265358979323846;
/*! \brief The Euler-Mascheroni_constant */
const double kEulerMascheroni = 0.57721566490153286060651209008240243104215933593992;

}  // namespace probability_constant

/*! \brief Enum encoding possible choices of probability distribution */
enum class ProbabilityDistributionType : int {
  kNormal = 0, kLogistic = 1, kExtreme = 2
};

struct NormalDistribution {
  XGBOOST_DEVICE inline static
  double PDF(double z) {
    return std::exp(-z * z / 2.0) / std::sqrt(2.0 * probability_constant::kPI);
  }

  XGBOOST_DEVICE inline static
  double CDF(double z) {
    return 0.5 * (1 + std::erf(z / std::sqrt(2.0)));
  }

  XGBOOST_DEVICE inline static
  double GradPDF(double z) {
    return -z * PDF(z);
  }

  XGBOOST_DEVICE inline static
  double HessPDF(double z) {
    return (z * z - 1.0) * PDF(z);
  }

  XGBOOST_DEVICE inline static
  ProbabilityDistributionType Type() {
    return ProbabilityDistributionType::kNormal;
  }
};

struct LogisticDistribution {
  XGBOOST_DEVICE inline static
  double PDF(double z) {
    const double w = std::exp(z);
    const double sqrt_denominator = 1 + w;
    if (std::isinf(w) || std::isinf(w * w)) {
      return 0.0;
    } else {
      return w / (sqrt_denominator * sqrt_denominator);
    }
  }

  XGBOOST_DEVICE inline static
  double CDF(double z) {
    const double w = std::exp(z);
    return std::isinf(w) ? 1.0 : (w / (1 + w));
  }

  XGBOOST_DEVICE inline static
  double GradPDF(double z) {
    const double w = std::exp(z);
    return std::isinf(w) ? 0.0 : (PDF(z) * (1 - w) / (1 + w));
  }

  XGBOOST_DEVICE inline static
  double HessPDF(double z) {
    const double w = std::exp(z);
    if (std::isinf(w) || std::isinf(w * w)) {
      return 0.0;
    } else {
      return PDF(z) * (w * w - 4 * w + 1) / ((1 + w) * (1 + w));
    }
  }

  XGBOOST_DEVICE inline static
  ProbabilityDistributionType Type() {
    return ProbabilityDistributionType::kLogistic;
  }
};

struct ExtremeDistribution {
  XGBOOST_DEVICE inline static
  double PDF(double z) {
    const double w = std::exp(z);
    return std::isinf(w) ? 0.0 : (w * std::exp(-w));
  }

  XGBOOST_DEVICE inline static
  double CDF(double z) {
    const double w = std::exp(z);
    return 1 - std::exp(-w);
  }

  XGBOOST_DEVICE inline static
  double GradPDF(double z) {
    const double w = std::exp(z);
    return std::isinf(w) ? 0.0 : ((1 - w) * PDF(z));
  }

  XGBOOST_DEVICE inline static
  double HessPDF(double z) {
    const double w = std::exp(z);
    if (std::isinf(w) || std::isinf(w * w)) {
      return 0.0;
    } else {
      return (w * w - 3 * w + 1) * PDF(z);
    }
  }

  XGBOOST_DEVICE inline static
  ProbabilityDistributionType Type() {
    return ProbabilityDistributionType::kExtreme;
  }
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_PROBABILITY_DISTRIBUTION_H_
