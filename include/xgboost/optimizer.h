/*!
 * Copyright by Contributors
 * \file optimizer.h
 */
#pragma once
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/gbm.h>

namespace xgboost {

class Optimizer {
 public:
  virtual ~Optimizer() {}
  static Optimizer* Create(std::string name);
  virtual void Init(
      const std::vector<std::pair<std::string, std::string>>& cfg) {}

  /**
   * \fn  virtual void Optimizer::OptimizeGradients(std::vector<bst_gpair>* gpair)
   *
   * \brief Optimize gradients. Used for applying momentum or averaging to gradients based on gradient information from previous iterations.
   */

  virtual void OptimizeGradients(std::vector<bst_gpair>* gpair) {}

  /**
   * \fn  virtual void Optimizer::OptimizePredictions(std::vector<float>* predictions,GradientBooster *gbm, DMatrix *dmatrix)
   *
   * \brief Modify predictions if necessary. Used for nesterov style optimizer that can look ahead.
   */

  virtual void OptimizePredictions(std::vector<float>* predictions,GradientBooster *gbm, DMatrix *dmatrix) {}
};

/*!
 * \brief Registry entry for optimizer.
 */
struct OptimizerReg
    : public dmlc::FunctionRegEntryBase<OptimizerReg,
                                        std::function<Optimizer*()>> {};

#define XGBOOST_REGISTER_OPTIMIZER(UniqueId, Name)      \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::OptimizerReg& \
      __make_##OptimizerReg##_##UniqueId##__ =          \
          ::dmlc::Registry<::xgboost::OptimizerReg>::Get()->__REGISTER__(Name)
}  // namespace xgboost
