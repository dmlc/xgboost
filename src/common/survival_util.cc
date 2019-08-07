#include <dmlc/registry.h>
#include <xgboost/enum_class_param.h>
#include "survival_util.h"

namespace xgboost {
namespace common {

DMLC_REGISTER_PARAMETER(AFTParam);

AFTDistribution* AFTDistribution::Create(AFTDistributionType dist) {
  switch (dist) {
   case AFTDistributionType::kNormal:
    return new AFTNormal;
   case AFTDistributionType::kLogistic:
    return new AFTLogistic;
   case AFTDistributionType::kWeibull:
    LOG(FATAL) << "Not implemented";
   default:
    LOG(FATAL) << "Unknown distribution";
  }
  return nullptr;
}

double AFTNormal::pdf(double x, double mu, double sd) {
  double pdf;
  pdf = (std::exp(-std::pow((x-mu)/(std::sqrt(2)*sd),2)))/std::sqrt(2*kPI*std::pow(sd,2));
  return pdf;
}

double AFTNormal::cdf(double x, double mu, double sd) {
  double cdf;
  cdf = 0.5*(1+std::erf((x-mu)/(sd*std::sqrt(2))));
  return cdf;
}

double AFTNormal::grad_pdf(double x, double mu, double sd) {
  double pdf;
  double z;
  double grad;
  pdf  = this->pdf(x,mu,sd);
  z = (x-mu)/sd;
  grad = -1*z*pdf;
  return grad;
}

double AFTNormal::hess_pdf(double x, double mu, double sd) {
  double pdf;
  double z;
  double hess;
  pdf     = this->pdf(x,mu,sd);
  z       = (x-mu)/sd;
  hess = (std::pow(z,2)-1)*pdf;
  return hess;
}

double AFTLogistic::pdf(double x, double mu, double sd) {
  double pdf;
  pdf = std::exp((x-mu)/sd)/(sd*std::pow((1+std::exp((x-mu)/sd)),2));
  return pdf;
}

double AFTLogistic::cdf(double x, double mu, double sd) {
  double cdf;
  cdf = std::exp((x-mu)/sd)/(1+std::exp((x-mu)/sd));
  return cdf;
}

double AFTLogistic::grad_pdf(double x, double mu, double sd) {
  double pdf;
  double z;
  double grad;
  pdf  = this->pdf(x, mu, sd);
  z    = (x-mu)/sd;
  grad = pdf*(1-std::pow(std::exp(1),z))/(1+std::pow(std::exp(1),z));
  return grad;
}

double AFTLogistic::hess_pdf(double x, double mu, double sd) {
  double pdf;
  double z;
  double hess;
  double w;
  pdf     = this->pdf(x,mu,sd);
  z       = (x-mu)/sd;
  w       = std::pow(std::exp(1),z);
  hess    = pdf*(std::pow(w,2)-4*w+1)/std::pow((1+w),2);
  return hess;
}

double AFTLoss::loss(double y_lower, double y_higher, double y_pred, double sigma) {
  double pdf;
  double cdf_u, cdf_l, z, z_u, z_l;
  double cost;
  if (y_lower == y_higher) {  // uncensored
    z    = (std::log(y_lower)-y_pred)/sigma;
    pdf  = dist_->pdf(z, 0, 1);
    cost = -std::log(pdf/(sigma*y_lower));
  } else {  // censored; now check what type of censorship we have
    if (std::isinf(y_higher)) {  // right-censored
      z    = (std::log(y_lower)-y_pred)/sigma;
      cdf_u = 1;
      cdf_l  = dist_->cdf(z, 0, 1);
    } else if (std::isinf(y_lower)) {  // left-censored
      z    = (std::log(y_higher)-y_pred)/sigma;
      cdf_u  = dist_->cdf(z, 0, 1);
      cdf_l = 0;
    } else if (!std::isinf(y_lower) && !std::isinf(y_higher)) {  // interval-censored
      z_u   = (std::log(y_higher) - y_pred)/sigma;
      z_l   = (std::log(y_lower) - y_pred)/sigma;
      cdf_u = dist_->cdf(z_u,0,1);
      cdf_l = dist_->cdf(z_l,0,1);
    } else {
      LOG(FATAL) << "AFTLoss: Could not determine event type: y_lower = " << y_lower
                 << ", y_higher = " << y_higher;
    }
    cost = -std::log(cdf_u - cdf_l);
  }
  return cost;
}

double AFTLoss::gradient(double y_lower, double y_higher, double y_pred, double sigma) {
  double pdf_l;
  double pdf_u;
  double pdf;
  double _grad;
  double z;
  double z_u;
  double z_l;
  double cdf_u;
  double cdf_l;
  double gradVar;

  const double eps = 1e-12f;

  if (y_lower == y_higher) {  // uncensored
    z    = (std::log(y_lower)-y_pred)/sigma;
    pdf  = dist_->pdf(z,0,1);
    _grad = dist_->grad_pdf(z,0,1);
    gradVar = _grad/(sigma*pdf);
  } else {  // censored; now check what type of censorship we have
    if (std::isinf(y_higher)) {  // right-censored
      z_l    = (std::log(y_lower)-y_pred)/sigma;
      pdf_u  = 0;
      pdf_l  = dist_->pdf(z_l,0,1);
      cdf_u  = 1;
      cdf_l  = dist_->cdf(z_l,0,1);
    } else if (std::isinf(y_lower)) {  // left-censored
      z_u    = (std::log(y_higher)-y_pred)/sigma;
      pdf_u  = dist_->pdf(z_u,0,1);
      pdf_l  = 0;
      cdf_u  = dist_->cdf(z_u,0,1);
      cdf_l  = 0;
    } else if (!std::isinf(y_lower) && !std::isinf(y_higher)) {  // interval-censored
      z_u    = (std::log(y_higher)-y_pred)/sigma;
      z_l    = (std::log(y_lower)-y_pred)/sigma;
      pdf_u  = dist_->pdf(z_u,0,1);
      pdf_l  = dist_->pdf(z_l,0,1);
      cdf_u  = dist_->cdf(z_u,0,1);
      cdf_l  = dist_->cdf(z_l,0,1);
    } else {
      LOG(FATAL) << "AFTLoss: Could not determine event type: y_lower = " << y_lower
                 << ", y_higher = " << y_higher;
    }
    gradVar  = (pdf_u-pdf_l)/(sigma*std::max(cdf_u-cdf_l,eps));
  }
  
  return gradVar;
}

double AFTLoss::hessian(double y_lower,double y_higher,double y_pred,double sigma){
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
  double gradVar;
  double hessVar;
  double hess_dist;

  const double eps = 1e-12f;
  if (y_lower == y_higher) {  // uncensored
    z    = (std::log(y_lower)-y_pred)/sigma;
    pdf       = dist_->pdf(z,0,1);
    gradVar   = dist_->cdf(z,0,1);
    hess_dist = dist_->hess_pdf(z,0,1);
    hessVar = -(pdf*hess_dist - std::pow(gradVar,2))/(std::pow(sigma,2)*std::pow(pdf,2));
  } else {  // censored; now check what type of censorship we have
    if (std::isinf(y_higher)) {  // right-censored
      z_l     = (std::log(y_lower)-y_pred)/sigma;
      pdf_u   = 0;
      pdf_l   = dist_->pdf(z_l,0,1);
      cdf_u   = 1;
      cdf_l   = dist_->cdf(z_l,0,1);
      grad_u  = 0;
      grad_l  = dist_->grad_pdf(z_l,0,1);
    } else if (std::isinf(y_lower)) {  // left-censored
      z_u    = (std::log(y_higher)-y_pred)/sigma;
      pdf_u  = dist_->pdf(z_u,0,1);
      pdf_l  = 0;
      cdf_u  = dist_->cdf(z_u,0,1);
      cdf_l  = 0;
      grad_u  = dist_->grad_pdf(z_u,0,1);
      grad_l  = 0;
    } else if (!std::isinf(y_lower) && !std::isinf(y_higher)) {  // interval-censored
      z_u    = (std::log(y_higher)-y_pred)/sigma;
      z_l    = (std::log(y_lower)-y_pred)/sigma;
      pdf_u  = dist_->pdf(z_u,0,1);
      pdf_l  = dist_->pdf(z_l,0,1);
      cdf_u  = dist_->cdf(z_u,0,1);
      cdf_l  = dist_->cdf(z_l,0,1);
      grad_u  = dist_->grad_pdf(z_u,0,1);
      grad_l  = dist_->grad_pdf(z_l,0,1);
    } else {
      LOG(FATAL) << "AFTLoss: Could not determine event type: y_lower = " << y_lower
                 << ", y_higher = " << y_higher;
    }
    hessVar = -((cdf_u-cdf_l)*(grad_u-grad_l)-std::pow((pdf_u-pdf_l),2))/(std::pow(sigma,2)*std::pow(std::max(cdf_u-cdf_l,eps),2));
  }
  return hessVar;
}

}  // namespace common
}  // namespace xgboost
