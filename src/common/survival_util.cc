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
    cost = -std::log(pdf / (sigma * y_lower));
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
    cost = -std::log(cdf_u - cdf_l);
  }

  return cost;
}

double AFTLoss::Gradient(double y_lower, double y_upper, double y_pred, double sigma) {
  const double log_y_lower = std::log(y_lower);
  const double log_y_upper = std::log(y_upper);
  double gradient;

  if (y_lower == y_upper) {  // uncensored
    const double z = (log_y_lower - y_pred) / sigma;
    const double pdf = dist_->PDF(z);
    const double grad_pdf = dist_->GradPDF(z);
    gradient = grad_pdf / (sigma * pdf);
  } else {  // censored; now check what type of censorship we have
    double z_u, z_l, pdf_u, pdf_l, cdf_u, cdf_l;
    if (std::isinf(y_upper)) {  // right-censored
      pdf_u = 0;
      cdf_u = 1;
    } else {  // interval-censored or left-censored
      z_u = (log_y_upper - y_pred) / sigma;
      pdf_u = dist_->PDF(z_u);
      cdf_u = dist_->CDF(z_u);
    }
    if (std::isinf(y_lower)) {  // left-censored
      pdf_l = 0;
      cdf_l = 0;
    } else {  // interval-censored or right-censored
      z_l = (log_y_lower - y_pred) / sigma;
      pdf_l = dist_->PDF(z_l);
      cdf_l = dist_->CDF(z_l);
    }
    // Regularize the denominator with eps, so that gradient doesn't get too big
    const double eps = 1e-12;
    gradient = (pdf_u - pdf_l) / (sigma * std::max(cdf_u - cdf_l, eps));
  }

  return gradient;
}

double AFTLoss::Hessian(double y_lower, double y_upper, double y_pred, double sigma) {
  const double log_y_lower = std::log(y_lower);
  const double log_y_upper = std::log(y_upper);
  double hessian;

  if (y_lower == y_upper) {  // uncensored
    const double z = (log_y_lower - y_pred) / sigma;
    const double pdf = dist_->PDF(z);
    const double grad_pdf = dist_->GradPDF(z);
    const double hess_pdf = dist_->HessPDF(z);
    hessian = -(pdf * hess_pdf - std::pow(grad_pdf, 2)) / (std::pow(sigma, 2) * std::pow(pdf, 2));
  } else {  // censored; now check what type of censorship we have
    double z_u, z_l, grad_pdf_u, grad_pdf_l, pdf_u, pdf_l, cdf_u, cdf_l;
    if (std::isinf(y_upper)) {  // right-censored
      pdf_u = 0;
      cdf_u = 1;
      grad_pdf_u = 0;
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
    } else {  // interval-censored or right-censored
      z_l = (log_y_lower - y_pred) / sigma;
      pdf_l = dist_->PDF(z_l);
      cdf_l = dist_->CDF(z_l);
      grad_pdf_l = dist_->GradPDF(z_l);
    }
    const double cdf_diff = cdf_u - cdf_l;
    const double pdf_diff = pdf_u - pdf_l;
    const double grad_diff = grad_pdf_u - grad_pdf_l;
    // Regularize the denominator with eps, so that gradient doesn't get too big
    const double eps = 1e-12;
    const double cdf_diff_thresh = std::max(cdf_diff, eps);
    const double numerator = -(cdf_diff * grad_diff - pdf_diff * pdf_diff);
    const double sqrt_denominator = sigma * cdf_diff_thresh;
    const double denominator = sqrt_denominator * sqrt_denominator;
    hessian = numerator / denominator;
  }

  return hessian;
}

}  // namespace common
}  // namespace xgboost
