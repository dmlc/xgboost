#include <dmlc/registry.h>
#include <xgboost/enum_class_param.h>
#include "survival_util.h"

namespace xgboost {
namespace common {

DMLC_REGISTER_PARAMETER(AFTParam);

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

}  // namespace common
}  // namespace xgboost
