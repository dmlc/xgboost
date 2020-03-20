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
  double pdf;
  pdf = std::exp(-z * z / 2) / std::sqrt(2 * kPI);
  return pdf;
}

double NormalDist::CDF(double z) {
  double cdf;
  cdf = 0.5 * (1 + std::erf(z / std::sqrt(2)));
  return cdf;
}

double NormalDist::GradPDF(double z) {
  double pdf;
  double grad;
  pdf = this->PDF(z);
  grad = -1 * z * pdf;
  return grad;
}

double NormalDist::HessPDF(double z) {
  double pdf;
  double hess;
  pdf = this->PDF(z);
  hess = (z * z - 1) * pdf;
  return hess;
}

double LogisticDist::PDF(double z) {
  double pdf;
  double e_z;
  double sqrt_denominator;
  e_z = std::exp(z);
  sqrt_denominator = 1 + e_z;
  pdf = e_z / (sqrt_denominator * sqrt_denominator);
  return pdf;
}

double LogisticDist::CDF(double z) {
  double cdf;
  double e_z;
  e_z = std::exp(z);
  cdf = e_z / (1 + e_z);
  return cdf;
}

double LogisticDist::GradPDF(double z) {
  double pdf;
  double grad;
  double e_z;
  pdf = this->PDF(z);
  e_z = std::exp(z);
  grad = pdf * (1 - e_z) / (1 + e_z);
  return grad;
}

double LogisticDist::HessPDF(double z) {
  double pdf;
  double hess;
  double w;
  pdf = this->PDF(z);
  w = std::exp(z);
  hess = pdf * (w * w - 4 * w + 1) / ((1 + w) * (1 + w));
  return hess;
}

double ExtremeDist::PDF(double z) {
  double pdf;
  double w;
  w = std::exp(z);
  pdf = w * std::exp(-w);
  return pdf;
}

double ExtremeDist::CDF(double z) {
  double cdf;
  double w;
  w = std::exp(z);
  cdf = 1 - std::exp(-w);
  return cdf;
}

double ExtremeDist::GradPDF(double z) {
  double pdf;
  double w;
  double grad;

  pdf = this->PDF(z);
  w = std::exp(z);
  grad = (1 - w) * pdf;
  return grad;
}

double ExtremeDist::HessPDF(double z) {
  double pdf;
  double w;
  double hess;
  pdf = this->PDF(z);
  w = std::exp(z);
  hess = (w * w - 3 * w + 1) * pdf;
  return hess;
}

}  // namespace common
}  // namespace xgboost
