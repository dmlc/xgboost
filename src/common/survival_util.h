/*!
 * Copyright 2019 by Contributors
 * \file survival_util.h
 * \brief Utility functions, useful for implementing objective and metric functions for survival
 *        analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */
#ifndef XGBOOST_COMMON_SURVIVAL_UTIL_H_
#define XGBOOST_COMMON_SURVIVAL_UTIL_H_

#include <xgboost/parameter.h>
#include <memory>
#include "probability_distribution.h"

DECLARE_FIELD_ENUM_CLASS(xgboost::common::ProbabilityDistributionType);

namespace xgboost {
namespace common {

/*! \brief Parameter structure for AFT loss and metric */
struct AFTParam : public XGBoostParameter<AFTParam> {
  /*! \brief Choice of probability distribution for the noise term in AFT */
  ProbabilityDistributionType aft_loss_distribution;
  /*! \brief Scaling factor to be applied to the distribution */
  float aft_loss_distribution_scale;
  DMLC_DECLARE_PARAMETER(AFTParam) {
    DMLC_DECLARE_FIELD(aft_loss_distribution)
        .set_default(ProbabilityDistributionType::kNormal)
        .add_enum("normal", ProbabilityDistributionType::kNormal)
        .add_enum("logistic", ProbabilityDistributionType::kLogistic)
        .add_enum("extreme", ProbabilityDistributionType::kExtreme)
        .describe("Choice of distribution for the noise term in "
                  "Accelerated Failure Time model");
    DMLC_DECLARE_FIELD(aft_loss_distribution_scale)
        .set_default(1.0f)
        .describe("Scaling factor used to scale the distribution in "
                  "Accelerated Failure Time model");
  }
};

/*! \brief The AFT loss function */
class AFTLoss {
 private:
  std::unique_ptr<ProbabilityDistribution> dist_;
  ProbabilityDistributionType dist_type_;

 public:
  /*!
   * \brief Constructor for AFT loss function
   * \param dist_type Choice of probability distribution for the noise term in AFT
   */
  explicit AFTLoss(ProbabilityDistributionType dist_type)
    : dist_(ProbabilityDistribution::Create(dist_type)),
      dist_type_(dist_type) {}

 public:
  /*!
   * \brief Compute the AFT loss
   * \param y_lower Lower bound for the true label
   * \param y_upper Upper bound for the true label
   * \param y_pred Predicted label
   * \param sigma Scaling factor to be applied to the distribution of the noise term
   */
  double Loss(double y_lower, double y_upper, double y_pred, double sigma);
  /*!
   * \brief Compute the gradient of the AFT loss
   * \param y_lower Lower bound for the true label
   * \param y_upper Upper bound for the true label
   * \param y_pred Predicted label
   * \param sigma Scaling factor to be applied to the distribution of the noise term
   */
  double Gradient(double y_lower, double y_upper, double y_pred, double sigma);
  /*!
   * \brief Compute the hessian of the AFT loss
   * \param y_lower Lower bound for the true label
   * \param y_upper Upper bound for the true label
   * \param y_pred Predicted label
   * \param sigma Scaling factor to be applied to the distribution of the noise term
   */
  double Hessian(double y_lower, double y_upper, double y_pred, double sigma);
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_SURVIVAL_UTIL_H_
