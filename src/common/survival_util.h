/*!
 * Copyright 2019-2020 by Contributors
 * \file survival_util.h
 * \brief Utility functions, useful for implementing objective and metric functions for survival
 *        analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */
#ifndef XGBOOST_COMMON_SURVIVAL_UTIL_H_
#define XGBOOST_COMMON_SURVIVAL_UTIL_H_

/*
 * For the derivation of the loss, gradient, and hessian for the Accelerated Failure Time model,
 * refer to the paper "Survival regression with accelerated failure time model in XGBoost"
 * at https://arxiv.org/abs/2006.04920.
 */

#include <xgboost/parameter.h>
#include <memory>
#include <algorithm>
#include <limits>
#include "probability_distribution.h"

DECLARE_FIELD_ENUM_CLASS(xgboost::common::ProbabilityDistributionType);

namespace xgboost {
namespace common {

#ifndef __CUDACC__

using std::log;
using std::fmax;

#endif  // __CUDACC__

enum class CensoringType : uint8_t {
  kUncensored, kRightCensored, kLeftCensored, kIntervalCensored
};

namespace aft {

// Allowable range for gradient and hessian. Used for regularization
constexpr double kMinGradient = -15.0;
constexpr double kMaxGradient = 15.0;
constexpr double kMinHessian = 1e-16;  // Ensure that no data point gets zero hessian
constexpr double kMaxHessian = 15.0;

constexpr double kEps = 1e-12;  // A denomitor in a fraction should not be too small

// Clip (limit) x to fit range [x_min, x_max].
// If x < x_min, return x_min; if x > x_max, return x_max; if x_min <= x <= x_max, return x.
// This function assumes x_min < x_max; behavior is undefined if this assumption does not hold.
XGBOOST_DEVICE
inline double Clip(double x, double x_min, double x_max) {
  if (x < x_min) {
    return x_min;
  }
  if (x > x_max) {
    return x_max;
  }
  return x;
}

template<typename Distribution>
XGBOOST_DEVICE inline double
GetLimitGradAtInfPred(CensoringType censor_type, bool sign, double sigma);

template<typename Distribution>
XGBOOST_DEVICE inline double
GetLimitHessAtInfPred(CensoringType censor_type, bool sign, double sigma);

}  // namespace aft

/*! \brief Parameter structure for AFT loss and metric */
struct AFTParam : public XGBoostParameter<AFTParam> {
  /*! \brief Choice of probability distribution for the noise term in AFT */
  ProbabilityDistributionType aft_loss_distribution;
  /*! \brief Scaling factor to be applied to the distribution */
  float aft_loss_distribution_scale;
  DMLC_DECLARE_PARAMETER(AFTParam) {
    DMLC_DECLARE_FIELD(aft_loss_distribution)
        .set_default(ProbabilityDistributionType::kNormal)
        .add_enum("normal", ProbabilityDistributionType::kNormal)
        .add_enum("logistic", ProbabilityDistributionType::kLogistic)
        .add_enum("extreme", ProbabilityDistributionType::kExtreme)
        .describe("Choice of distribution for the noise term in "
                  "Accelerated Failure Time model");
    DMLC_DECLARE_FIELD(aft_loss_distribution_scale)
        .set_default(1.0f)
        .describe("Scaling factor used to scale the distribution in "
                  "Accelerated Failure Time model");
  }
};

/*! \brief The AFT loss function */
template<typename Distribution>
struct AFTLoss {
  XGBOOST_DEVICE inline static
  double Loss(double y_lower, double y_upper, double y_pred, double sigma) {
    const double log_y_lower = log(y_lower);
    const double log_y_upper = log(y_upper);

    double cost;

    if (y_lower == y_upper) {  // uncensored
      const double z = (log_y_lower - y_pred) / sigma;
      const double pdf = Distribution::PDF(z);
      // Regularize the denominator with eps, to avoid INF or NAN
      cost = -log(fmax(pdf / (sigma * y_lower), aft::kEps));
    } else {  // censored; now check what type of censorship we have
      double z_u, z_l, cdf_u, cdf_l;
      if (isinf(y_upper)) {  // right-censored
        cdf_u = 1;
      } else {  // left-censored or interval-censored
        z_u = (log_y_upper - y_pred) / sigma;
        cdf_u = Distribution::CDF(z_u);
      }
      if (y_lower <= 0.0) {  // left-censored
        cdf_l = 0;
      } else {  // right-censored or interval-censored
        z_l = (log_y_lower - y_pred) / sigma;
        cdf_l = Distribution::CDF(z_l);
      }
      // Regularize the denominator with eps, to avoid INF or NAN
      cost = -log(fmax(cdf_u - cdf_l, aft::kEps));
    }

    return cost;
  }

  XGBOOST_DEVICE inline static
  double Gradient(double y_lower, double y_upper, double y_pred, double sigma) {
    const double log_y_lower = log(y_lower);
    const double log_y_upper = log(y_upper);
    double numerator, denominator, gradient;  // numerator and denominator of gradient
    CensoringType censor_type;
    bool z_sign;  // sign of z-score

    if (y_lower == y_upper) {  // uncensored
      const double z = (log_y_lower - y_pred) / sigma;
      const double pdf = Distribution::PDF(z);
      const double grad_pdf = Distribution::GradPDF(z);
      censor_type = CensoringType::kUncensored;
      numerator = grad_pdf;
      denominator = sigma * pdf;
      z_sign = (z > 0);
    } else {  // censored; now check what type of censorship we have
      double z_u = 0.0, z_l = 0.0, pdf_u, pdf_l, cdf_u, cdf_l;
      censor_type = CensoringType::kIntervalCensored;
      if (isinf(y_upper)) {  // right-censored
        pdf_u = 0;
        cdf_u = 1;
        censor_type = CensoringType::kRightCensored;
      } else {  // interval-censored or left-censored
        z_u = (log_y_upper - y_pred) / sigma;
        pdf_u = Distribution::PDF(z_u);
        cdf_u = Distribution::CDF(z_u);
      }
      if (y_lower <= 0.0) {  // left-censored
        pdf_l = 0;
        cdf_l = 0;
        censor_type = CensoringType::kLeftCensored;
      } else {  // interval-censored or right-censored
        z_l = (log_y_lower - y_pred) / sigma;
        pdf_l = Distribution::PDF(z_l);
        cdf_l = Distribution::CDF(z_l);
      }
      z_sign = (z_u > 0 || z_l > 0);
      numerator = pdf_u - pdf_l;
      denominator = sigma * (cdf_u - cdf_l);
    }
    gradient = numerator / denominator;
    if (denominator < aft::kEps && (isnan(gradient) || isinf(gradient))) {
      gradient = aft::GetLimitGradAtInfPred<Distribution>(censor_type, z_sign, sigma);
    }

    return aft::Clip(gradient, aft::kMinGradient, aft::kMaxGradient);
  }

  XGBOOST_DEVICE inline static
  double Hessian(double y_lower, double y_upper, double y_pred, double sigma) {
    const double log_y_lower = log(y_lower);
    const double log_y_upper = log(y_upper);
    double numerator, denominator, hessian;  // numerator and denominator of hessian
    CensoringType censor_type;
    bool z_sign;  // sign of z-score

    if (y_lower == y_upper) {  // uncensored
      const double z = (log_y_lower - y_pred) / sigma;
      const double pdf = Distribution::PDF(z);
      const double grad_pdf = Distribution::GradPDF(z);
      const double hess_pdf = Distribution::HessPDF(z);
      censor_type = CensoringType::kUncensored;
      numerator = -(pdf * hess_pdf - grad_pdf * grad_pdf);
      denominator = sigma * sigma * pdf * pdf;
      z_sign = (z > 0);
    } else {  // censored; now check what type of censorship we have
      double z_u = 0.0, z_l = 0.0, grad_pdf_u, grad_pdf_l, pdf_u, pdf_l, cdf_u, cdf_l;
      censor_type = CensoringType::kIntervalCensored;
      if (isinf(y_upper)) {  // right-censored
        pdf_u = 0;
        cdf_u = 1;
        grad_pdf_u = 0;
        censor_type = CensoringType::kRightCensored;
      } else {  // interval-censored or left-censored
        z_u = (log_y_upper - y_pred) / sigma;
        pdf_u = Distribution::PDF(z_u);
        cdf_u = Distribution::CDF(z_u);
        grad_pdf_u = Distribution::GradPDF(z_u);
      }
      if (y_lower <= 0.0) {  // left-censored
        pdf_l = 0;
        cdf_l = 0;
        grad_pdf_l = 0;
        censor_type = CensoringType::kLeftCensored;
      } else {  // interval-censored or right-censored
        z_l = (log_y_lower - y_pred) / sigma;
        pdf_l = Distribution::PDF(z_l);
        cdf_l = Distribution::CDF(z_l);
        grad_pdf_l = Distribution::GradPDF(z_l);
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
    if (denominator < aft::kEps && (isnan(hessian) || isinf(hessian))) {
      hessian = aft::GetLimitHessAtInfPred<Distribution>(censor_type, z_sign, sigma);
    }

    return aft::Clip(hessian, aft::kMinHessian, aft::kMaxHessian);
  }
};

namespace aft {

template <>
XGBOOST_DEVICE inline double
GetLimitGradAtInfPred<NormalDistribution>(CensoringType censor_type, bool sign, double sigma) {
  // Remove unused parameter compiler warning.
  (void) sigma;

  switch (censor_type) {
  case CensoringType::kUncensored:
    return sign ? kMinGradient : kMaxGradient;
  case CensoringType::kRightCensored:
    return sign ? kMinGradient : 0.0;
  case CensoringType::kLeftCensored:
    return sign ? 0.0 : kMaxGradient;
  case CensoringType::kIntervalCensored:
    return sign ? kMinGradient : kMaxGradient;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

template <>
XGBOOST_DEVICE inline double
GetLimitHessAtInfPred<NormalDistribution>(CensoringType censor_type, bool sign, double sigma) {
  switch (censor_type) {
  case CensoringType::kUncensored:
    return 1.0 / (sigma * sigma);
  case CensoringType::kRightCensored:
    return sign ? (1.0 / (sigma * sigma)) : kMinHessian;
  case CensoringType::kLeftCensored:
    return sign ? kMinHessian : (1.0 / (sigma * sigma));
  case CensoringType::kIntervalCensored:
    return 1.0 / (sigma * sigma);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

template <>
XGBOOST_DEVICE inline double
GetLimitGradAtInfPred<LogisticDistribution>(CensoringType censor_type, bool sign, double sigma) {
  switch (censor_type) {
  case CensoringType::kUncensored:
    return sign ? (-1.0 / sigma) : (1.0 / sigma);
  case CensoringType::kRightCensored:
    return sign ? (-1.0 / sigma) : 0.0;
  case CensoringType::kLeftCensored:
    return sign ? 0.0 : (1.0 / sigma);
  case CensoringType::kIntervalCensored:
    return sign ? (-1.0 / sigma) : (1.0 / sigma);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

template <>
XGBOOST_DEVICE inline double
GetLimitHessAtInfPred<LogisticDistribution>(CensoringType censor_type, bool sign, double sigma) {
  // Remove unused parameter compiler warning.
  (void) sign;
  (void) sigma;

  switch (censor_type) {
  case CensoringType::kUncensored:
  case CensoringType::kRightCensored:
  case CensoringType::kLeftCensored:
  case CensoringType::kIntervalCensored:
    return kMinHessian;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

template <>
XGBOOST_DEVICE inline double
GetLimitGradAtInfPred<ExtremeDistribution>(CensoringType censor_type, bool sign, double sigma) {
  switch (censor_type) {
  case CensoringType::kUncensored:
    return sign ? kMinGradient : (1.0 / sigma);
  case CensoringType::kRightCensored:
    return sign ? kMinGradient : 0.0;
  case CensoringType::kLeftCensored:
    return sign ? 0.0 : (1.0 / sigma);
  case CensoringType::kIntervalCensored:
    return sign ? kMinGradient : (1.0 / sigma);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

template <>
XGBOOST_DEVICE inline double
GetLimitHessAtInfPred<ExtremeDistribution>(CensoringType censor_type, bool sign, double sigma) {
  // Remove unused parameter compiler warning.
  (void) sigma;

  switch (censor_type) {
  case CensoringType::kUncensored:
  case CensoringType::kRightCensored:
    return sign ? kMaxHessian : kMinHessian;
  case CensoringType::kLeftCensored:
    return kMinHessian;
  case CensoringType::kIntervalCensored:
    return sign ? kMaxHessian : kMinHessian;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace aft

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_SURVIVAL_UTIL_H_
