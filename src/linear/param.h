/*!
 * Copyright 2018 by Contributors
 * \file param.h
 * \brief training parameters.
 */
#ifndef XGBOOST_LINEAR_PARAM_H_
#define XGBOOST_LINEAR_PARAM_H_
#include "xgboost/parameter.h"

namespace xgboost {
namespace linear {
/**
 * \brief A set of available FeatureSelector's
 */
enum FeatureSelectorEnum {
  kCyclic = 0,
  kShuffle,
  kThrifty,
  kGreedy,
  kRandom
};

struct LinearTrainParam : public XGBoostParameter<LinearTrainParam> {
  /*! \brief learning_rate */
  float learning_rate;
  /*! \brief regularization weight for L2 norm */
  float reg_lambda;
  /*! \brief regularization weight for L1 norm */
  float reg_alpha;
  int feature_selector;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LinearTrainParam) {
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(0.5f)
        .describe("Learning rate of each update.");
    DMLC_DECLARE_FIELD(reg_lambda)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("L2 regularization on weights.");
    DMLC_DECLARE_FIELD(reg_alpha)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("L1 regularization on weights.");
    DMLC_DECLARE_FIELD(feature_selector)
        .set_default(kCyclic)
        .add_enum("cyclic", kCyclic)
        .add_enum("shuffle", kShuffle)
        .add_enum("thrifty", kThrifty)
        .add_enum("greedy", kGreedy)
        .add_enum("random", kRandom)
        .describe("Feature selection or ordering method.");
    // alias of parameters
    DMLC_DECLARE_ALIAS(learning_rate, eta);
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
  }
  /*! \brief Denormalizes the regularization penalties - to be called at each update */
  void DenormalizePenalties(double sum_instance_weight) {
    reg_lambda_denorm = reg_lambda * sum_instance_weight;
    reg_alpha_denorm = reg_alpha * sum_instance_weight;
  }
  // denormalizated regularization penalties
  float reg_lambda_denorm;
  float reg_alpha_denorm;
};

}  // namespace linear
}  // namespace xgboost

#endif  // XGBOOST_LINEAR_PARAM_H_
