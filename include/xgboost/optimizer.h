/*!
 * Copyright by Contributors
 * \file optimizer.h
 */
#pragma once
#include <xgboost/base.h>
#include <xgboost/data.h>

namespace xgboost {

class Optimizer {
 public:
  virtual ~Optimizer() {}
  static Optimizer* Create(std::string name);
  virtual void Init(
      const std::vector<std::pair<std::string, std::string>>& cfg) {}
  virtual void OptimizeGradients(std::vector<bst_gpair>* gpair) {}
  virtual void OptimizePredictions(std::vector<float>* predictions) {}
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
