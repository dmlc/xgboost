#include <xgboost/enum_class_param.h>

namespace xgboost {
namespace common {

// Choice of distribution for the noise term in AFT
enum class AFTNoiseDistribution : int {
  kNormal = 0, kLogistic = 1, kWeibull = 2
};

// Type of Censorship
enum class AFTEventType : int {
  kUncensored = 0, kLeftCensored = 1, kRightCensored = 2, kIntervalCensored = 3
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

namespace aft {

class AFTloss {

  public:
    virtual double grad(double x, double mu, double sd){
      double pdf;
      double z;
      double gradVar;
      pdf  = common::aft::dnorm(x,mu,sd);
      z = (x-mu)/sd;
      gradVar = -1*z*pdf;
      return gradVar;
    }
    
    virtual double hess(double x, double mu, double sd){
      double pdf;
      double z;
      double hessVar;
      pdf     = common::aft::dnorm(x,mu,sd);
      z       = (x-mu)/sd;
      hessVar = (std::pow(z,2)-1)*pdf;
      return hessVar;
    }

    virtual double grad_interval(double y_lower,double y_higher,double y_pred,double sigma,AFTEventType event){
      
      double pdf_l;
      double pdf_u;
      double z_u;
      double z_l;
      double cdf_u;
      double cdf_l;
      double gradVar;

      const  double eps = 1e-12f;

      if(event==1){
        //event = 1 -> Left
        z_u    = (std::log(y_higher)-y_pred)/sigma;
        pdf_u  = common::aft::dnorm(z_u,0,1);
        pdf_l  = 0
        cdf_u  = common::aft::pnorm(z_u,0,1);
        cdf_l  = 0;
      }
      if(event==2){
        //event = 2 -> Right
        z_l    = (std::log(y_lower)-y_pred)/sigma;
        pdf_u  = 0;
        pdf_l  = common::aft::dnorm(z_l,0,1);
        cdf_u  = 1;
        cdf_l  = common::aft::pnorm(z_l,0,1);
      }
      if(event==3){
        //event = 3 -> Interval
        z_u    = (std::log(y_higher)-y_pred)/sigma;
        z_l    = (std::log(y_lower)-y_pred)/sigma;
        pdf_u  = common::aft::dnorm(z_u,0,1);
        pdf_l  = common::aft::dnorm(z_l,0,1);
        cdf_u  = common::aft::pnorm(z_u,0,1);
        cdf_l  = common::aft::pnorm(z_l,0,1);
      }
      
      gradVar  = (pdf_u-pdf_l)/(sigma*std::max(cdf_u-cdf_l,eps));
      return gradVar;
    }

  virtual double grad_uncensored(double y_lower,double y_higher,double y_pred,double sigma,AFTEventType event){
    
    double pdf;
    double z;
    double _grad;
    double gradVar;
    z    = (std::log(y_lower)-y_pred)/sigma;
    pdf  = common::aft::dnorm(z,0,1);
    _grad = grad(z,0,1);
    gradVar = _grad/(sigma*pdf);
    return gradVar;

  }


  virtual double hessian_interval(double y_lower,double y_higher,double y_pred,double sigma,AFTEventType event){
    
    double z_u;
    double z_l;
    double pdf_u;
    double pdf_l;
    double cdf_u;
    double cdf_l;
    double grad_u;
    double grad_l;
    double hessVar;
    
    const  double eps = 1e-12f;
    if(event==1){
      //event = 1 -> Left
      z_u    = (std::log(y_higher)-y_pred)/sigma;
      pdf_u  = common::aft::dnorm(z_u,0,1);
      pdf_l  = 0
      cdf_u  = common::aft::pnorm(z_u,0,1);
      cdf_l  = 0;
      grad_u  = grad(z_u,0,1);
      grad_l  = 0;
    }
    if(event==2){
      //event = 2 -> Right
      z_l     = (std::log(y_lower)-y_pred)/sigma;
      pdf_u   = 0;
      pdf_l   = common::aft::dnorm(z_l,0,1);
      cdf_u   = 1;
      cdf_l   = common::aft::pnorm(z_l,0,1);
      grad_u  = 0;
      grad_l  = grad(z_l,0,1);
    }
    if(event==3){
      //event = 3 -> Interval
      z_u    = (std::log(y_higher)-y_pred)/sigma;
      z_l    = (std::log(y_lower)-y_pred)/sigma;
      pdf_u  = common::aft::dnorm(z_u,0,1);
      pdf_l  = common::aft::dnorm(z_l,0,1);
      cdf_u  = common::aft::pnorm(z_u,0,1);
      cdf_l  = common::aft::pnorm(z_l,0,1);
    }
    hessVar = -((cdf_u-cdf_l)*(grad_u-grad_l)-std::pow((pdf_u-pdf_l),2))/(std::pow(sigma,2)*std::pow(std::max(cdf_u-cdf_l,eps),2));
    return hessVar;
  }

  virtual double hessian_uncensored(double y_lower,double y_higher,double y_pred,double sigma,AFTEventType event){
    double pdf;
    double z;
    double gradVar;
    double hessVar;
    double hess_dist;
    z    = (std::log(y_lower)-y_pred)/sigma;
    pdf       = common::aft::dnorm(z,0,1);
    gradVar      = grad(z,0,1);
    hess_dist = hess(z,0,1);
    hessVar = -(pdf*hess_dist - std::pow(gradVar,2))/(std::pow(sigma,2)*std::pow(pdf,2));
    return hessVar;
  }

}

class AFTNormal: public AFTloss{

}

class AFTLogistic: public AFTloss{

  virtual double grad(double x, double mu, double sd){
      double pdf;
      double z;
      double gradVar;
      pdf  = common::aft::dlogis(x,mu,sd);
      z    = (x-mu)/sd;
      gradVar = pdf*(1-std::pow(std::exp(1),z))/(1+std::pow(std::exp(1),z));
      return gradVar;
    }


  virtual double hess(double x, double mu, double sd){
      double pdf;
      double z;
      double hessVar;
      double w;
      pdf     = common::aft::dlogis(x,mu,sd);
      z       = (x-mu)/sd;
      w       = std::pow(std::exp(1),z);
      hessVar    = pdf*(std::pow(w,2)-4*w+1)/std::pow((1+w),2);
      return hessVar;
    }

  virtual double grad_interval(double y_lower,double y_higher,double y_pred,double sigma,AFTEventType event){
      double pdf_l;
      double pdf_u;
      double z_u;
      double z_l;
      double cdf_u;
      double cdf_l;
      double gradVar;
      const  double eps = 1e-12f;
      if(event==1){
        //event = 1 -> Left
        z_u    = (std::log(y_higher)-y_pred)/sigma;
        pdf_u  = common::aft::dlogis(z_u,0,1);
        pdf_l  = 0
        cdf_u  = common::aft::plogis(z_u,0,1);
        cdf_l  = 0;
      }
      if(event==2){
        //event = 2 -> Right
        z_l    = (std::log(y_lower)-y_pred)/sigma;
        pdf_u  = 0;
        pdf_l  = common::aft::dlogis(z_l,0,1);
        cdf_u  = 1;
        cdf_l  = common::aft::plogis(z_l,0,1);
      }
      if(event==3){
        //event = 3 -> Interval
        z_u    = (std::log(y_higher)-y_pred)/sigma;
        z_l    = (std::log(y_lower)-y_pred)/sigma;
        pdf_u  = common::aft::dlogis(z_u,0,1);
        pdf_l  = common::aft::dlogis(z_l,0,1);
        cdf_u  = common::aft::plogis(z_u,0,1);
        cdf_l  = common::aft::plogis(z_l,0,1);
      }
      gradVar  = (pdf_u-pdf_l)/(sigma*std::max(cdf_u-cdf_l,eps));
      return gradVar;
    }

  virtual double grad_uncensored(double y_lower,double y_higher,double y_pred,double sigma,AFTEventType event){
    
    double pdf;
    double z;
    double _grad;
    double gradVar;
    z    = (std::log(y_lower)-y_pred)/sigma;
    pdf  = common::aft::dlogis(z,0,1);
    _grad = grad_logis(z,0,1);
    gradVar = _grad/(sigma*pdf);
    return gradVar;
  }
  
  virtual double hessian_interval(double y_lower,double y_higher,double y_pred,double sigma,AFTEventType event){
    
    double z_u;
    double z_l;
    double pdf_u;
    double pdf_l;
    double cdf_u;
    double cdf_l;
    double grad_u;
    double grad_l;
    double hessVar;
    
    const  double eps = 1e-12f;
    if(event==1){
      //event = 1 -> Left
      z_u    = (std::log(y_higher)-y_pred)/sigma;
      pdf_u  = common::aft::dlogis(z_u,0,1);
      pdf_l  = 0
      cdf_u  = common::aft::plogis(z_u,0,1);
      cdf_l  = 0;
      grad_u  = grad(z_u,0,1);
      grad_l  = 0;
    }
    if(event==2){
      //event = 2 -> Right
      z_l     = (std::log(y_lower)-y_pred)/sigma;
      pdf_u   = 0;
      pdf_l   = common::aft::dlogis(z_l,0,1);
      cdf_u   = 1;
      cdf_l   = common::aft::plogis(z_l,0,1);
      grad_u  = 0;
      grad_l  = grad(z_l,0,1);
    }
    if(event==3){
      //event = 3 -> Interval
      z_u    = (std::log(y_higher)-y_pred)/sigma;
      z_l    = (std::log(y_lower)-y_pred)/sigma;
      pdf_u  = common::aft::dlogis(z_u,0,1);
      pdf_l  = common::aft::dlogis(z_l,0,1);
      cdf_u  = common::aft::plogis(z_u,0,1);
      cdf_l  = common::aft::plogis(z_l,0,1);
    }
    hessVar = -((cdf_u-cdf_l)*(grad_u-grad_l)-std::pow((pdf_u-pdf_l),2))/(std::pow(sigma,2)*std::pow(std::max(cdf_u-cdf_l,eps),2));
    return hessVar;
  }

  virtual double hessian_uncensored(double y_lower,double y_higher,double y_pred,double sigma,AFTEventType event){
    double pdf;
    double z;
    double gradVar;
    double hessVar;
    double hess_dist;
    z    = (std::log(y_lower)-y_pred)/sigma;
    pdf       = common::aft::dlogis(z,0,1);
    gradVar      = grad(z,0,1);
    hess_dist    = hess(z,0,1);
    hessVar = -(pdf*hess_dist - std::pow(grad,2))/(std::pow(sigma,2)*std::pow(pdf,2));
    return hessVar;
  }
}

inline double dlogis(double x, double mu , double sd){
	double pdf;
	pdf = std::exp((x-mu)/sd)/(sd*std::pow((1+std::exp((x-mu)/sd)),2));
	return pdf;
}

inline double dnorm(double x, double mu , double sd){
	double pdf;
	pdf = (std::exp(-std::pow((x-mu)/(std::sqrt(2)*sd),2)))/std::sqrt(2*kPI*std::pow(sd,2));
	return pdf;
}

inline double plogis(double x, double mu , double sd){
	double cdf;
	cdf = std::exp((x-mu)/sd)/(1+std::exp((x-mu)/sd));
	return cdf;
}

inline double pnorm(double x, double mu , double sd){
	double cdf;
	cdf = 0.5*(1+std::erf((x-mu)/(sd*std::sqrt(2))));
	return cdf;
}

}  // namespace aft
}  // namespace common
}  // namespace xgboost
