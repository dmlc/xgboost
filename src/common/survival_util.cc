/*!
 * Copyright 2019 by Contributors
 * \file survival_util.cc
 * \brief Utility functions, useful for implementing objective and metric functions for survival
 *        analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */

#include <dmlc/registry.h>
#include <algorithm>
#include <cmath>
#include "survival_util.h"

/*
- Formulas are motivated from document -
  http://members.cbio.mines-paristech.fr/~thocking/survival.pdf
- Detailed Derivation of Loss/Gradient/Hessian -
  https://github.com/avinashbarnwal/GSOC-2019/blob/master/doc/Accelerated_Failure_Time.pdf
*/

namespace {

// Allowable range for gradient and hessian. Used for regularization
constexpr double kMinGradient = -15.0;
constexpr double kMaxGradient = 15.0;
constexpr double kMinHessian = 1e-16;  // Ensure that no data point gets zero hessian
constexpr double kMaxHessian = 15.0;

constexpr double kEps = 1e-12;  // A denomitor in a fraction should not be too small

// Clip (limit) x to fit range [x_min, x_max].
// If x < x_min, return x_min; if x > x_max, return x_max; if x_min <= x <= x_max, return x.
// This function assumes x_min < x_max; behavior is undefined if this assumption does not hold.
inline double Clip(double x, double x_min, double x_max) {
  if (x < x_min) {
    return x_min;
  }
  if (x > x_max) {
    return x_max;
  }
  return x;
}

using xgboost::common::ProbabilityDistributionType;

enum class CensoringType : uint8_t {
  kUncensored, kRightCensored, kLeftCensored, kIntervalCensored
};

using xgboost::GradientPairPrecise;

inline GradientPairPrecise GetLimitAtInfPred(ProbabilityDistributionType dist_type,
                                             CensoringType censor_type,
                                             double sign, double sigma) {
  switch (censor_type) {
  case CensoringType::kUncensored:
    switch (dist_type) {
    case ProbabilityDistributionType::kNormal:
      return sign ? GradientPairPrecise{ kMinGradient, 1.0 / (sigma * sigma) }
                  : GradientPairPrecise{ kMaxGradient, 1.0 / (sigma * sigma) };
    case ProbabilityDistributionType::kLogistic:
      return sign ? GradientPairPrecise{ -1.0 / sigma, kMinHessian }
                  : GradientPairPrecise{ 1.0 / sigma, kMinHessian };
    case ProbabilityDistributionType::kExtreme:
      return sign ? GradientPairPrecise{ kMinGradient, kMaxHessian }
                  : GradientPairPrecise{ 1.0 / sigma, kMinHessian };
    default:
      LOG(FATAL) << "Unknown distribution type";
    }
  case CensoringType::kRightCensored:
    switch (dist_type) {
    case ProbabilityDistributionType::kNormal:
      return sign ? GradientPairPrecise{ kMinGradient, 1.0 / (sigma * sigma) }
                  : GradientPairPrecise{ 0.0, kMinHessian };
    case ProbabilityDistributionType::kLogistic:
      return sign ? GradientPairPrecise{ -1.0 / sigma, kMinHessian }
                  : GradientPairPrecise{ 0.0, kMinHessian };
    case ProbabilityDistributionType::kExtreme:
      return sign ? GradientPairPrecise{ kMinGradient, kMaxHessian }
                  : GradientPairPrecise{ 0.0, kMinHessian };
    default:
      LOG(FATAL) << "Unknown distribution type";
    }
  case CensoringType::kLeftCensored:
    switch (dist_type) {
    case ProbabilityDistributionType::kNormal:
      return sign ? GradientPairPrecise{ 0.0, kMinHessian }
                  : GradientPairPrecise{ kMaxGradient, 1.0 / (sigma * sigma) };
    case ProbabilityDistributionType::kLogistic:
      return sign ? GradientPairPrecise{ 0.0, kMinHessian }
                  : GradientPairPrecise{ 1.0 / sigma, kMinHessian };
    case ProbabilityDistributionType::kExtreme:
      return sign ? GradientPairPrecise{ 0.0, kMinHessian }
                  : GradientPairPrecise{ 1.0 / sigma, kMinHessian };
    default:
      LOG(FATAL) << "Unknown distribution type";
    }
  case CensoringType::kIntervalCensored:
    switch (dist_type) {
    case ProbabilityDistributionType::kNormal:
      return sign ? GradientPairPrecise{ kMinGradient, 1.0 / (sigma * sigma) }
                  : GradientPairPrecise{ kMaxGradient, 1.0 / (sigma * sigma) };
    case ProbabilityDistributionType::kLogistic:
      return sign ? GradientPairPrecise{ -1.0 / sigma, kMinHessian }
                  : GradientPairPrecise{ 1.0 / sigma, kMinHessian };
    case ProbabilityDistributionType::kExtreme:
      return sign ? GradientPairPrecise{ kMinGradient, kMaxHessian }
                  : GradientPairPrecise{ 1.0 / sigma, kMinHessian };
    default:
      LOG(FATAL) << "Unknown distribution type";
    }
  default:
    LOG(FATAL) << "Unknown censoring type";
  }

  return { 0.0, 0.0 };
}

}  // anonymous namespace

namespace xgboost {
namespace common {

DMLC_REGISTER_PARAMETER(AFTParam);

double AFTLoss::Loss(double y_lower, double y_upper, double y_pred, double sigma) {
  const double log_y_lower = std::log(y_lower);
  const double log_y_upper = std::log(y_upper);

  double cost;

  if (y_lower == y_upper) {  // uncensored
    const double z = (log_y_lower - y_pred) / sigma;
    const double pdf = dist_->PDF(z);
    // Regularize the denominator with eps, to avoid INF or NAN
    cost = -std::log(std::max(pdf / (sigma * y_lower), kEps));
  } else {  // censored; now check what type of censorship we have
    double z_u, z_l, cdf_u, cdf_l;
    if (std::isinf(y_upper)) {  // right-censored
      cdf_u = 1;
    } else {  // left-censored or interval-censored
      z_u = (log_y_upper - y_pred) / sigma;
      cdf_u = dist_->CDF(z_u);
    }
    if (std::isinf(y_lower)) {  // left-censored
      cdf_l = 0;
    } else {  // right-censored or interval-censored
      z_l = (log_y_lower - y_pred) / sigma;
      cdf_l = dist_->CDF(z_l);
    }
    // Regularize the denominator with eps, to avoid INF or NAN
    cost = -std::log(std::max(cdf_u - cdf_l, kEps));
  }

  return cost;
}

double AFTLoss::Gradient(double y_lower, double y_upper, double y_pred, double sigma) {
  const double log_y_lower = std::log(y_lower);
  const double log_y_upper = std::log(y_upper);
  double numerator, denominator, gradient;  // numerator and denominator of gradient
  CensoringType censor_type;
  bool z_sign;  // sign of z-score

  if (y_lower == y_upper) {  // uncensored
    const double z = (log_y_lower - y_pred) / sigma;
    const double pdf = dist_->PDF(z);
    const double grad_pdf = dist_->GradPDF(z);
    censor_type = CensoringType::kUncensored;
    numerator = grad_pdf;
    denominator = sigma * pdf;
    z_sign = (z > 0);
  } else {  // censored; now check what type of censorship we have
    double z_u = 0.0, z_l = 0.0, pdf_u, pdf_l, cdf_u, cdf_l;
    censor_type = CensoringType::kIntervalCensored;
    if (std::isinf(y_upper)) {  // right-censored
      pdf_u = 0;
      cdf_u = 1;
      censor_type = CensoringType::kRightCensored;
    } else {  // interval-censored or left-censored
      z_u = (log_y_upper - y_pred) / sigma;
      pdf_u = dist_->PDF(z_u);
      cdf_u = dist_->CDF(z_u);
    }
    if (std::isinf(y_lower)) {  // left-censored
      pdf_l = 0;
      cdf_l = 0;
      censor_type = CensoringType::kLeftCensored;
    } else {  // interval-censored or right-censored
      z_l = (log_y_lower - y_pred) / sigma;
      pdf_l = dist_->PDF(z_l);
      cdf_l = dist_->CDF(z_l);
    }
    z_sign = (z_u > 0 || z_l > 0);
    numerator = pdf_u - pdf_l;
    denominator = sigma * (cdf_u - cdf_l);
  }
  gradient = numerator / denominator;
  if (denominator < kEps && (std::isnan(gradient) || std::isinf(gradient))) {
    gradient = GetLimitAtInfPred(dist_type_, censor_type, z_sign, sigma).GetGrad();
  }

  return Clip(gradient, kMinGradient, kMaxGradient);
}

double AFTLoss::Hessian(double y_lower, double y_upper, double y_pred, double sigma) {
  const double log_y_lower = std::log(y_lower);
  const double log_y_upper = std::log(y_upper);
  double numerator, denominator, hessian;  // numerator and denominator of hessian
  CensoringType censor_type;
  bool z_sign;  // sign of z-score

  if (y_lower == y_upper) {  // uncensored
    const double z = (log_y_lower - y_pred) / sigma;
    const double pdf = dist_->PDF(z);
    const double grad_pdf = dist_->GradPDF(z);
    const double hess_pdf = dist_->HessPDF(z);
    censor_type = CensoringType::kUncensored;
    numerator = -(pdf * hess_pdf - grad_pdf * grad_pdf);
    denominator = sigma * sigma * pdf * pdf;
    z_sign = (z > 0);
  } else {  // censored; now check what type of censorship we have
    double z_u = 0.0, z_l = 0.0, grad_pdf_u, grad_pdf_l, pdf_u, pdf_l, cdf_u, cdf_l;
    censor_type = CensoringType::kIntervalCensored;
    if (std::isinf(y_upper)) {  // right-censored
      pdf_u = 0;
      cdf_u = 1;
      grad_pdf_u = 0;
      censor_type = CensoringType::kRightCensored;
    } else {  // interval-censored or left-censored
      z_u = (log_y_upper - y_pred) / sigma;
      pdf_u = dist_->PDF(z_u);
      cdf_u = dist_->CDF(z_u);
      grad_pdf_u = dist_->GradPDF(z_u);
    }
    if (std::isinf(y_lower)) {  // left-censored
      pdf_l = 0;
      cdf_l = 0;
      grad_pdf_l = 0;
      censor_type = CensoringType::kLeftCensored;
    } else {  // interval-censored or right-censored
      z_l = (log_y_lower - y_pred) / sigma;
      pdf_l = dist_->PDF(z_l);
      cdf_l = dist_->CDF(z_l);
      grad_pdf_l = dist_->GradPDF(z_l);
    }
    const double cdf_diff = cdf_u - cdf_l;
    const double pdf_diff = pdf_u - pdf_l;
    const double grad_diff = grad_pdf_u - grad_pdf_l;
    const double sqrt_denominator = sigma * cdf_diff;
    z_sign = (z_u > 0 || z_l > 0);
    numerator = -(cdf_diff * grad_diff - pdf_diff * pdf_diff);
    denominator = sqrt_denominator * sqrt_denominator;
  }
  hessian = numerator / denominator;
  if (denominator < kEps && (std::isnan(hessian) || std::isinf(hessian))) {
    hessian = GetLimitAtInfPred(dist_type_, censor_type, z_sign, sigma).GetHess();
  }

  return Clip(hessian, kMinHessian, kMaxHessian);
}

}  // namespace common
}  // namespace xgboost
