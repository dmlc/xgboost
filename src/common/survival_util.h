#include <xgboost/enum_class_param.h>
#include <memory>

namespace xgboost {
namespace common {

// Choice of distribution for the noise term in AFT
enum class AFTDistributionType : int {
  kNormal = 0, kLogistic = 1, kWeibull = 2
};

}  // namespace common
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::common::AFTDistributionType);

namespace xgboost {
namespace common {

// Constant PI
const double kPI = 3.14159265358979323846;

struct AFTParam : public dmlc::Parameter<AFTParam> {
  AFTDistributionType aft_noise_distribution;
  float aft_sigma;
  DMLC_DECLARE_PARAMETER(AFTParam) {
    DMLC_DECLARE_FIELD(aft_noise_distribution)
        .set_default(AFTDistributionType::kNormal)
        .add_enum("normal", AFTDistributionType::kNormal)
        .add_enum("logistic", AFTDistributionType::kLogistic)
        .add_enum("weibull", AFTDistributionType::kWeibull)
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

  static AFTDistribution* Create(AFTDistributionType dist);
};

class AFTNormal : public AFTDistribution {
 public:
  double pdf(double x, double mu, double sd) override;
  double cdf(double x, double mu, double sd) override;
  double grad_pdf(double x, double mu, double sd) override;
  double hess_pdf(double x, double mu, double sd) override;
};

class AFTLogistic : public AFTDistribution {
 public:
  double pdf(double x, double mu, double sd) override;
  double cdf(double x, double mu, double sd) override;
  double grad_pdf(double x, double mu, double sd) override;
  double hess_pdf(double x, double mu, double sd) override;
};

class AFTLoss {
 private:
   std::unique_ptr<AFTDistribution> dist_;

 public:
  AFTLoss(AFTDistributionType dist) {
    dist_.reset(AFTDistribution::Create(dist));
  }

 public:
  double loss(double y_lower, double y_higher, double y_pred, double sigma);
  double gradient(double y_lower, double y_higher, double y_pred, double sigma);
  double hessian(double y_lower, double y_higher, double y_pred, double sigma);
};

}  // namespace common
}  // namespace xgboost
