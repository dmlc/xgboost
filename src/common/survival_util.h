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
