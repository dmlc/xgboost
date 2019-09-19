/*!
 * Copyright 2019 by Contributors
 * \file survival_util.h
 * \brief Utility functions, useful for implementing objective and metric functions for survival
 *        analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */
#ifndef XGBOOST_COMMON_SURVIVAL_UTIL_H_
#define XGBOOST_COMMON_SURVIVAL_UTIL_H_

#include <xgboost/enum_class_param.h>
#include <memory>

namespace xgboost {
namespace common {

// Choice of distribution for the noise term in AFT
enum class AFTDistributionType : int {
  kNormal = 0, kLogistic = 1, kExtreme = 2
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
        .add_enum("extreme", AFTDistributionType::kExtreme)
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
  virtual double PDF(double z) = 0;
  virtual double CDF(double z) = 0;
  virtual double GradPDF(double z) = 0;
  virtual double HessPDF(double z) = 0;

  static AFTDistribution* Create(AFTDistributionType dist);
};

class AFTNormal : public AFTDistribution {
 public:
  double PDF(double z) override;
  double CDF(double z) override;
  double GradPDF(double z) override;
  double HessPDF(double z) override;
};

class AFTLogistic : public AFTDistribution {
 public:
  double PDF(double z) override;
  double CDF(double z) override;
  double GradPDF(double z) override;
  double HessPDF(double z) override;
};

class AFTExtreme : public AFTDistribution {
 public:
  double PDF(double z) override;
  double CDF(double z) override;
  double GradPDF(double z) override;
  double HessPDF(double z) override;
};

class AFTLoss {
 private:
  std::unique_ptr<AFTDistribution> dist_;

 public:
  explicit AFTLoss(AFTDistributionType dist) {
    dist_.reset(AFTDistribution::Create(dist));
  }

 public:
  double Loss(double y_lower, double y_higher, double y_pred, double sigma);
  double Gradient(double y_lower, double y_higher, double y_pred, double sigma);
  double Hessian(double y_lower, double y_higher, double y_pred, double sigma);
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_SURVIVAL_UTIL_H_
