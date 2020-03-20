/*!
 * Copyright 2020 by Contributors
 * \file probability_distribution.cc
 * \brief Implementation of a few useful probability distributions
 * \author Avinash Barnwal and Hyunsu Cho
 */

#include <xgboost/logging.h>
#include <cmath>
#include "probability_distribution.h"

namespace xgboost {
namespace common {

ProbabilityDistribution* ProbabilityDistribution::Create(ProbabilityDistributionType dist) {
  switch (dist) {
    case ProbabilityDistributionType::kNormal:
      return new NormalDist;
    case ProbabilityDistributionType::kLogistic:
      return new LogisticDist;
    case ProbabilityDistributionType::kExtreme:
      return new ExtremeDist;
    default:
      LOG(FATAL) << "Unknown distribution";
  }
  return nullptr;
}

double NormalDist::PDF(double z) {
  const double pdf = std::exp(-z * z / 2) / std::sqrt(2 * probability_constant::kPI);
  return pdf;
}

double NormalDist::CDF(double z) {
  const double cdf = 0.5 * (1 + std::erf(z / std::sqrt(2)));
  return cdf;
}

double NormalDist::GradPDF(double z) {
  const double pdf = this->PDF(z);
  const double grad = -1 * z * pdf;
  return grad;
}

double NormalDist::HessPDF(double z) {
  const double pdf = this->PDF(z);
  const double hess = (z * z - 1) * pdf;
  return hess;
}

double LogisticDist::PDF(double z) {
  const double e_z = std::exp(z);
  const double sqrt_denominator = 1 + e_z;
  const double pdf = e_z / (sqrt_denominator * sqrt_denominator);
  return pdf;
}

double LogisticDist::CDF(double z) {
  const double e_z = std::exp(z);
  const double cdf = e_z / (1 + e_z);
  return cdf;
}

double LogisticDist::GradPDF(double z) {
  const double pdf = this->PDF(z);
  const double e_z = std::exp(z);
  const double grad = pdf * (1 - e_z) / (1 + e_z);
  return grad;
}

double LogisticDist::HessPDF(double z) {
  const double pdf = this->PDF(z);
  const double w = std::exp(z);
  const double hess = pdf * (w * w - 4 * w + 1) / ((1 + w) * (1 + w));
  return hess;
}

double ExtremeDist::PDF(double z) {
  const double w = std::exp(z);
  const double pdf = w * std::exp(-w);
  return pdf;
}

double ExtremeDist::CDF(double z) {
  const double w = std::exp(z);
  const double cdf = 1 - std::exp(-w);
  return cdf;
}

double ExtremeDist::GradPDF(double z) {
  const double pdf = this->PDF(z);
  const double w = std::exp(z);
  const double grad = (1 - w) * pdf;
  return grad;
}

double ExtremeDist::HessPDF(double z) {
  const double pdf = this->PDF(z);
  const double w = std::exp(z);
  const double hess = (w * w - 3 * w + 1) * pdf;
  return hess;
}

}  // namespace common
}  // namespace xgboost
