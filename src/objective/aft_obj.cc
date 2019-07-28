/*!
 * Copyright 2015 by Contributors
 * \file rank.cc
 * \brief Definition of aft loss.
 */
#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <math.h>
#include "../common/math.h"
#include "../common/random.h"

namespace xgboost {
namespace obj {

// Choice of distribution for the noise term in AFT
enum class AFTNoiseDistribution : int {
  kNormal = 0, kLogistic = 1, kWeibull = 2
};

// Type of Censorship
enum class AFTEventType : int {
  kUncensored = 0, kLeftCensored = 1, kRightCensored = 2, kIntervalCensored = 3
};

// Constant PI
const double kPI = 3.14159265358979323846;

}  // namespace obj
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::obj::AFTNoiseDistribution);

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(aft_obj);

struct AFTParam : public dmlc::Parameter<AFTParam> {
  AFTNoiseDistribution aft_noise_distribution;
  float aft_sigma;
  DMLC_DECLARE_PARAMETER(AFTParam) {
    DMLC_DECLARE_FIELD(aft_noise_distribution)
        .set_default(AFTNoiseDistribution::kNormal)
        .add_enum("normal", AFTNoiseDistribution::kNormal)
        .add_enum("logistic", AFTNoiseDistribution::kLogistic)
        .add_enum("weibull", AFTNoiseDistribution::kWeibull)
        .describe("Choice of distribution for the noise term in "
                  "Accelerated Failure Time model");
    DMLC_DECLARE_FIELD(aft_sigma)
        .set_default(1.0f)
        .describe("Scaling factor used to scale the distribution in "
                  "Accelerated Failure Time model");
  }
};

class AFTObj : public ObjFunction {
 public:
  double dlogis(double x, double mu , double sd){
    double pdf;
    pdf = std::exp((x-mu)/sd)/(sd*std::pow((1+std::exp((x-mu)/sd)),2));
    return pdf;
  }

  double dnorm(double x, double mu , double sd){
    double pdf;
    pdf = (std::exp(-std::pow((x-mu)/(std::sqrt(2)*sd),2)))/std::sqrt(2*kPI*std::pow(sd,2));
    return pdf;
  }


  double plogis(double x, double mu , double sd){
    double cdf;
    cdf = std::exp((x-mu)/sd)/(1+std::exp((x-mu)/sd));
    return cdf;
  }

  double pnorm(double x, double mu , double sd){
    double cdf;
    cdf = 0.5*(1+std::erf((x-mu)/(sd*std::sqrt(2))));
    return cdf;
  }

  double grad_logis(double x, double mu, double sd){
    double pdf;
    double z;
    double grad;
    pdf  = dlogis(x,mu,sd);
    z    = (x-mu)/sd;
    grad = pdf*(1-std::pow(std::exp(1),z))/(1+std::pow(std::exp(1),z));
    return grad;
  }

  double grad_norm(double x, double mu, double sd){
    double pdf;
    double z;
    double grad;
    pdf  = dnorm(x,mu,sd);
    z = (x-mu)/sd;
    grad = -1*z*pdf;
    return grad;
  }

  double hess_logis(double x, double mu, double sd){
    double pdf;
    double z;
    double hess;
    double w;
    pdf     = dlogis(x,mu,sd);
    z       = (x-mu)/sd;
    w       = std::pow(std::exp(1),z);
    hess    = pdf*(std::pow(w,2)-4*w+1)/std::pow((1+w),2);
    return hess;
  }

  double hess_norm(double x, double mu, double sd){
    double pdf;
    double z;
    double hess;
    pdf     = dnorm(x,mu,sd);
    z       = (x-mu)/sd;
    hess = (std::pow(z,2)-1)*pdf;
    return hess;
 }

  double loss_interval(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double z_u;
    double z_l;
    double cdf_u;
    double cdf_l;
    double cost;

    z_u   = (std::log(y_higher) - y_pred)/sigma;
    z_l   = (std::log(y_lower) - y_pred)/sigma;

    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      cdf_u = pnorm(z_u,0,1);
      cdf_l = pnorm(z_l,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      cdf_u = plogis(z_u,0,1);
      cdf_l = plogis(z_l,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    cost = -std::log(cdf_u - cdf_l);
    return cost;
  }

  double loss_left(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double z;
    double cdf;
    double cost;

    z    = (std::log(y_higher)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      cdf = pnorm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      cdf = plogis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    cost = -std::log(cdf);
    return cost;
  }

  double loss_right(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double z;
    double cdf;
    double cost;
    z    = (std::log(y_lower)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      cdf = pnorm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      cdf = plogis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    cost = -std::log(1-cdf);
    return cost;
  }

  double loss_uncensored(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double z;
    double pdf;
    double cost;
    z       = (std::log(y_lower)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf = dnorm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf = dlogis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    cost = -std::log(pdf/(sigma*y_lower));
    return cost;
  }

  double neg_grad_interval(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double pdf_l;
    double pdf_u;
    double z_u;
    double z_l;
    double cdf_u;
    double cdf_l;
    double neg_grad;
    z_u    = (std::log(y_higher)-y_pred)/sigma;
    z_l    = (std::log(y_lower)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf_u  = dnorm(z_u,0,1);
      pdf_l  = dnorm(z_l,0,1);
      cdf_u  = pnorm(z_u,0,1);
      cdf_l  = pnorm(z_l,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf_u  = dlogis(z_u,0,1);
      pdf_l  = dlogis(z_l,0,1);
      cdf_u  = plogis(z_u,0,1);
      cdf_l  = plogis(z_l,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    neg_grad  = -(pdf_u-pdf_l)/(sigma*(cdf_u-cdf_l));
    return neg_grad;
  }

  double neg_grad_left(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double pdf;
    double z;
    double cdf;
    double neg_grad;
    z    = (std::log(y_higher)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf  = dnorm(z,0,1);
      cdf  = pnorm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf  = dlogis(z,0,1);
      cdf  = plogis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    neg_grad = -pdf/(sigma*cdf);
    return neg_grad;
  }

  double neg_grad_right(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double pdf;
    double z;
    double cdf;
    double neg_grad;

    z    = (std::log(y_lower)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf  = dnorm(z,0,1);
      cdf  = pnorm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf  = dlogis(z,0,1);
      cdf  = plogis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }

    neg_grad = pdf/(sigma*(1-cdf));
    return neg_grad;
  }

  double neg_grad_uncensored(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double pdf;
    double z;
    double grad;
    double neg_grad;
    z    = (std::log(y_lower)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf  = dnorm(z,0,1);
      grad = grad_norm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf  = dlogis(z,0,1);
      grad = grad_logis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    neg_grad = -grad/(sigma*pdf);
    return neg_grad;
  }

  double hessian_interval(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double z_u;
    double z_l;
    double pdf_u;
    double pdf_l;
    double cdf_u;
    double cdf_l;
    double grad_u;
    double grad_l;
    double hess;
    z_u   = (std::log(y_higher) - y_pred)/sigma;
    z_l   = (std::log(y_lower) - y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf_u   = dnorm(z_u,0,1);
      pdf_l   = dnorm(z_l,0,1);
      cdf_u   = pnorm(z_u,0,1);
      cdf_l   = pnorm(z_l,0,1);
      grad_u  = grad_norm(z_u,0,1);
      grad_l  = grad_norm(z_l,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf_u = dlogis(z_u,0,1);
      pdf_l = dlogis(z_l,0,1);
      cdf_u = plogis(z_u,0,1);
      cdf_l = plogis(z_l,0,1);
      grad_u  = grad_logis(z_u,0,1);
      grad_l  = grad_logis(z_l,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    hess = -((cdf_u-cdf_l)*(grad_u-grad_l)-std::pow((pdf_u-pdf_l),2))/(std::pow(sigma,2)*std::pow((cdf_u-cdf_l),2));
    return hess;
  }

  double hessian_left(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double pdf;
    double cdf;
    double z;
    double grad;
    double hess;
    z    = (std::log(y_higher)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf       = dnorm(z,0,1);
      cdf       = pnorm(z,0,1);
      grad      = grad_norm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf  = dlogis(z,0,1);
      cdf  = plogis(z,0,1);
      grad = grad_logis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    hess = -(cdf*grad - std::pow(pdf,2))/(std::pow(sigma,2)*std::pow(cdf,2));
    return hess;
  }

  double hessian_right(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double pdf;
    double cdf;
    double z;
    double grad;
    double hess;
    z    = (std::log(y_lower)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf       = dnorm(z,0,1);
      cdf       = pnorm(z,0,1);
      grad      = grad_norm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf  = dlogis(z,0,1);
      cdf  = plogis(z,0,1);
      grad = grad_logis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    hess = ((1-cdf)*grad + std::pow(pdf,2))/(std::pow(sigma,2)*std::pow(1-cdf,2));
    return hess;
  }

  double hessian_uncensored(double y_lower,double y_higher,double y_pred,double sigma,AFTNoiseDistribution dist){
    double pdf;
    double z;
    double grad;
    double hess;
    double hess_dist;
    z    = (std::log(y_lower)-y_pred)/sigma;
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      pdf       = dnorm(z,0,1);
      grad      = grad_norm(z,0,1);
      hess_dist = hess_norm(z,0,1);
      break;
     case AFTNoiseDistribution::kLogistic:
      pdf  = dlogis(z,0,1);
      grad = grad_logis(z,0,1);
      hess_dist = hess_logis(z,0,1);
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unrecognized AFT noise distribution type";
    }
    hess = -(pdf*hess_dist - std::pow(grad,2))/(std::pow(sigma,2)*std::pow(pdf,2));
    return hess;
  }

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    /* Boilerplate */
    CHECK_EQ(preds.Size(), info.labels_.Size());
    const auto& yhat     = preds.HostVector();
    const auto& y_lower  = info.labels_lower_bound_.HostVector();
    const auto& y_higher = info.labels_upper_bound_.HostVector();
    AFTEventType event;

    out_gpair->Resize(yhat.size());
    std::vector<GradientPair>& gpair = out_gpair->HostVector();
    size_t nsize = yhat.size();
    double first_order_grad;
    double second_order_grad;

    for (int i = 0; i < nsize; ++i) {
      if (y_lower[i] == y_higher[i]) {
        event = AFTEventType::kUncensored;
        first_order_grad  = neg_grad_uncensored(y_lower[i], y_higher[i], yhat[i],
                                                param_.aft_sigma, param_.aft_noise_distribution);
        second_order_grad = hessian_uncensored(y_lower[i], y_higher[i], yhat[i],
                                               param_.aft_sigma, param_.aft_noise_distribution);
      } else if (!std::isinf(y_lower[i]) && !std::isinf(y_higher[i])) {
        event = AFTEventType::kIntervalCensored;
        first_order_grad  = neg_grad_interval(y_lower[i], y_higher[i], yhat[i],
                                              param_.aft_sigma, param_.aft_noise_distribution);
        second_order_grad = hessian_interval(y_lower[i], y_higher[i], yhat[i],
                                             param_.aft_sigma, param_.aft_noise_distribution);
      } else if (std::isinf(y_lower[i])){
        event = AFTEventType::kLeftCensored;
        first_order_grad  = neg_grad_left(y_lower[i], y_higher[i], yhat[i],
                                          param_.aft_sigma, param_.aft_noise_distribution);
        second_order_grad = hessian_left(y_lower[i], y_higher[i], yhat[i],
                                         param_.aft_sigma, param_.aft_noise_distribution);
      } else if (std::isinf(y_higher[i])) {
        event = AFTEventType::kRightCensored;
        first_order_grad  = neg_grad_right(y_lower[i], y_higher[i], yhat[i],
                                           param_.aft_sigma, param_.aft_noise_distribution);
        second_order_grad = hessian_right(y_lower[i], y_higher[i], yhat[i],
                                          param_.aft_sigma, param_.aft_noise_distribution);
      } else {
        LOG(FATAL) << "AFTObj: Could not determine event type: y_lower = " << y_lower[i]
                   << ", y_higher = " << y_higher[i];
      }
      gpair[i] = GradientPair(first_order_grad, second_order_grad);
    }
  }

  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }
  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
  //  PredTransform(io_preds);
  }

  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }

  const char* DefaultEvalMetric() const override {
    return "aft_obj";
  }

 private:
  AFTParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(AFTParam);

XGBOOST_REGISTER_OBJECTIVE(AFTObj, "aft:survival")
.describe("AFT loss function")
.set_body([]() { return new AFTObj(); });

}  // namespace obj
}  // namespace xgboost


