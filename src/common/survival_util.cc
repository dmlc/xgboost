/*!
 * Copyright 2019 by Contributors
 * \file survival_util.cc
 * \brief Utility functions, useful for implementing objective and metric functions for survival
 *        analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */

#include <dmlc/registry.h>
#include <xgboost/enum_class_param.h>
#include <algorithm>
#include <cstdlib>
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

AFTDistribution* AFTDistribution::Create(AFTDistributionType dist) {
  switch (dist) {
    case AFTDistributionType::kNormal:
      return new AFTNormal;
    case AFTDistributionType::kLogistic:
      return new AFTLogistic;
    case AFTDistributionType::kExtreme:
      return new AFTExtreme;
    default:
      LOG(FATAL) << "Unknown distribution";
  }
  return nullptr;
}

double AFTNormal::PDF(double z) {
  double pdf;
  pdf = std::exp(-z * z / 2) / std::sqrt(2 * kPI);
  return pdf;
}

double AFTNormal::CDF(double z) {
  double cdf;
  cdf = 0.5 * (1 + std::erf(z / std::sqrt(2)));
  return cdf;
}

double AFTNormal::GradPDF(double z) {
  double pdf;
  double grad;
  pdf = this->PDF(z);
  grad = -1 * z * pdf;
  return grad;
}

double AFTNormal::HessPDF(double z) {
  double pdf;
  double hess;
  pdf = this->PDF(z);
  hess = (z * z - 1) * pdf;
  return hess;
}

double AFTLogistic::PDF(double z) {
  double pdf;
  double e_z;
  double sqrt_denominator;
  e_z = std::exp(z);
  sqrt_denominator = 1 + e_z;
  pdf = e_z / (sqrt_denominator * sqrt_denominator);
  return pdf;
}

double AFTLogistic::CDF(double z) {
  double cdf;
  double e_z;
  e_z = std::exp(z);
  cdf = e_z / (1 + e_z);
  return cdf;
}

double AFTLogistic::GradPDF(double z) {
  double pdf;
  double grad;
  double e_z;
  pdf = this->PDF(z);
  e_z = std::exp(z);
  grad = pdf * (1 - e_z) / (1 + e_z);
  return grad;
}

double AFTLogistic::HessPDF(double z) {
  double pdf;
  double hess;
  double w;
  pdf = this->PDF(z);
  w = std::exp(z);
  hess = pdf * (w * w - 4 * w + 1) / ((1 + w) * (1 + w));
  return hess;
}

double AFTExtreme::PDF(double z) {
  double pdf;
  double w;
  w = std::exp(z);
  pdf = w * std::exp(-w);
  return pdf;
}

double AFTExtreme::CDF(double z) {
  double cdf;
  double w;
  w = std::exp(z);
  cdf = 1 - std::exp(-w);
  return cdf;
}

double AFTExtreme::GradPDF(double z) {
  double pdf;
  double w;
  double grad;

  pdf = this->PDF(z);
  w = std::exp(z);
  grad = (1 - w) * pdf;
  return grad;
}

double AFTExtreme::HessPDF(double z) {
  double pdf;
  double w;
  double hess;
  pdf = this->PDF(z);
  w = std::exp(z);
  hess = (w * w - 3 * w + 1) * pdf;
  return hess;
}


double AFTLoss::Loss(double y_lower, double y_higher, double y_pred, double sigma) {
  double pdf;
  double cdf_u, cdf_l, z_u, z_l;
  double cost;
  if (y_lower == y_higher) {  // uncensored
    z_l = (y_lower - y_pred) / sigma;
    pdf = dist_->PDF(z_l);
    cost = -std::log(pdf / (sigma * y_lower));
  } else {  // censored; now check what type of censorship we have
    if (std::isinf(y_higher)) {  // right-censored
      cdf_u = 1;
    } else {  // left-censored or interval-censored
      z_u = (y_higher - y_pred) / sigma;
      cdf_u = dist_->CDF(z_u);
    }
    if (std::isinf(y_lower)) {  // left-censored
      cdf_l = 0;
    } else {  // right-censored or interval-censored
      z_l = (y_lower - y_pred) / sigma;
      cdf_l = dist_->CDF(z_l);
    }
    cost = -std::log(cdf_u - cdf_l);
  }
  return cost;
}

double AFTLoss::Gradient(double y_lower, double y_higher, double y_pred, double sigma) {
  double pdf_l;
  double pdf_u;
  double pdf;
  double grad;
  double z;
  double z_u;
  double z_l;
  double cdf_u;
  double cdf_l;
  double gradient;
  const double eps = 1e-12f;

  if (y_lower == y_higher) {  // uncensored
    z = (y_lower - y_pred) / sigma;
    pdf = dist_->PDF(z);
    grad = dist_->GradPDF(z);
    gradient = grad / (sigma * pdf);
  } else {  // censored; now check what type of censorship we have
    if (std::isinf(y_higher)) {  // right-censored
      pdf_u = 0;
      cdf_u = 1;
    } else {  // interval-censored or left-censored
      z_u = (y_higher - y_pred) / sigma;
      pdf_u = dist_->PDF(z_u);
      cdf_u = dist_->CDF(z_u);
    }
    if (std::isinf(y_lower)) {  // left-censored
      pdf_l = 0;
      cdf_l = 0;
    } else {  // interval-censored or right-censored
      z_l = (y_lower - y_pred) / sigma;
      pdf_l = dist_->PDF(z_l);
      cdf_l = dist_->CDF(z_l);
    }
    gradient = (pdf_u - pdf_l) / (sigma * std::max(cdf_u - cdf_l, eps));
  }

  return gradient;
}

double AFTLoss::Hessian(double y_lower, double y_higher, double y_pred, double sigma) {
  double z;
  double z_u;
  double z_l;
  double pdf;
  double pdf_u;
  double pdf_l;
  double cdf;
  double cdf_u;
  double cdf_l;
  double grad_u;
  double grad_l;
  double grad;
  double cdf_diff;
  double pdf_diff;
  double grad_diff;
  double cdf_diff_thresh;
  double numerator;
  double sqrt_denominator;
  double denominator;
  double hessian;
  double hess_dist;
  const double eps = 1e-12f;

  if (y_lower == y_higher) {  // uncensored
    z = (y_lower - y_pred) / sigma;
    pdf = dist_->PDF(z);
    grad = dist_->GradPDF(z);
    hess_dist = dist_->HessPDF(z);
    hessian = -(pdf * hess_dist - std::pow(grad, 2)) / (std::pow(sigma, 2) * std::pow(pdf, 2));
  } else {  // censored; now check what type of censorship we have
    if (std::isinf(y_higher)) {  // right-censored
      pdf_u = 0;
      cdf_u = 1;
      grad_u = 0;
    } else {  // interval-censored or left-censored
      z_u = (y_higher - y_pred) / sigma;
      pdf_u = dist_->PDF(z_u);
      cdf_u = dist_->CDF(z_u);
      grad_u = dist_->GradPDF(z_u);
    }
    if (std::isinf(y_lower)) {  // left-censored
      pdf_l = 0;
      cdf_l = 0;
      grad_l = 0;
    } else {  // interval-censored or right-censored
      z_l = (y_lower - y_pred) / sigma;
      pdf_l = dist_->PDF(z_l);
      cdf_l = dist_->CDF(z_l);
      grad_l = dist_->GradPDF(z_l);
    }
    cdf_diff = cdf_u - cdf_l;
    pdf_diff = pdf_u - pdf_l;
    grad_diff = grad_u - grad_l;
    cdf_diff_thresh = std::max(cdf_diff, eps);
    numerator = -(cdf_diff * grad_diff - pdf_diff * pdf_diff);
    sqrt_denominator = sigma * cdf_diff_thresh;
    denominator = sqrt_denominator * sqrt_denominator;
    hessian = numerator / denominator;
  }
  return hessian;
}

}  // namespace common
}  // namespace xgboost
