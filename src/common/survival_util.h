#include <xgboost/enum_class_param.h>

namespace xgboost {
namespace common {

// Choice of distribution for the noise term in AFT
enum class AFTNoiseDistribution : int {
  kNormal = 0, kLogistic = 1, kWeibull = 2
};

}  // namespace common
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::common::AFTNoiseDistribution);

namespace xgboost {
namespace common {

// Constant PI
const double kPI = 3.14159265358979323846;

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

class AFTDistribution {
 public:
  virtual double pdf(double x, double mu, double sd) = 0;
  virtual double cdf(double x, double mu, double sd) = 0;
  virtual double grad_pdf(double x, double mu, double sd) = 0;
  virtual double hess_pdf(double x, double mu, double sd) = 0;
};

class AFTNormal : public AFTDistribution {
 public:
  virtual double pdf(double x, double mu, double sd) override {
    double pdf;
    pdf = (std::exp(-std::pow((x-mu)/(std::sqrt(2)*sd),2)))/std::sqrt(2*kPI*std::pow(sd,2));
    return pdf;
  }
  virtual double cdf(double x, double mu, double sd) override {
    double cdf;
    cdf = 0.5*(1+std::erf((x-mu)/(sd*std::sqrt(2))));
    return cdf;
  }
  virtual double grad_pdf(double x, double mu, double sd) override {
    double pdf;
    double z;
    double grad;
    pdf  = this->pdf(x,mu,sd);
    z = (x-mu)/sd;
    grad = -1*z*pdf;
    return grad;
  }
  virtual double hess_pdf(double x, double mu, double sd) override {
		double pdf;
		double z;
		double hess;
		pdf     = this->pdf(x,mu,sd);
		z       = (x-mu)/sd;
		hess = (std::pow(z,2)-1)*pdf;
		return hess;
  }
};

class AFTLogistic : public AFTDistribution {
 public:
  virtual double pdf(double x, double mu, double sd) override {
    double pdf;
    pdf = std::exp((x-mu)/sd)/(sd*std::pow((1+std::exp((x-mu)/sd)),2));
    return pdf;
  }
  virtual double cdf(double x, double mu, double sd) override {
    double cdf;
    cdf = std::exp((x-mu)/sd)/(1+std::exp((x-mu)/sd));
    return cdf;
  }
  virtual double grad_pdf(double x, double mu, double sd) override {
    double pdf;
    double z;
    double grad;
    pdf  = this->pdf(x, mu, sd);
    z    = (x-mu)/sd;
    grad = pdf*(1-std::pow(std::exp(1),z))/(1+std::pow(std::exp(1),z));
    return grad;
  }
  virtual double hess_pdf(double x, double mu, double sd) override {
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
};

class AFTLoss {
 private:
  AFTDistribution* dist_;

 public:
  AFTLoss(AFTNoiseDistribution dist) {
    switch (dist) {
     case AFTNoiseDistribution::kNormal:
      dist_ = new AFTNormal;
      break;
     case AFTNoiseDistribution::kLogistic:
      dist_ = new AFTLogistic;
      break;
     case AFTNoiseDistribution::kWeibull:
      LOG(FATAL) << "Not implemented";
      break;
     default:
      LOG(FATAL) << "Unknown distribution";
    }
  }

 public:
    virtual double loss(double y_lower,double y_higher,double y_pred,double sigma) {
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

    virtual double gradient(double y_lower,double y_higher,double y_pred,double sigma) {
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

  virtual double hessian(double y_lower,double y_higher,double y_pred,double sigma){
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
};

}  // namespace common
}  // namespace xgboost
