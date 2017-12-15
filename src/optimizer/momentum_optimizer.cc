/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/parameter.h>
#include <xgboost/optimizer.h>

namespace xgboost {
namespace optimizer {

DMLC_REGISTRY_FILE_TAG(momentum_optimizer);

/*! \brief momentum parameters */
struct MomentumOptimizerParam : public dmlc::Parameter<MomentumOptimizerParam> {
  float momentum;
  // declare parameters
  DMLC_DECLARE_PARAMETER(MomentumOptimizerParam) {
    DMLC_DECLARE_FIELD(momentum)
        .set_range(0.0f, 1.0f)
        .set_default(0.0f)
        .describe(
            "Momentum coefficient controlling amount of gradient applied from previous iteration.");
  }
};
DMLC_REGISTER_PARAMETER(MomentumOptimizerParam);

class MomentumOptimizer : public Optimizer {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& cfg) override {
    param.InitAllowUnknown(cfg);
  }
  void OptimizeGradients(std::vector<bst_gpair>* gpair) override {
    if (param.momentum == 0.0f) {
      return;
    }
    if (!previous_gpair_.empty()) {
       //apply momentum
      for (size_t i = 0; i < gpair->size(); i++) {
        (*gpair)[i] =
            (*gpair)[i] +
            bst_gpair(previous_gpair_[i].GetGrad() * param.momentum, 0.0f);
      }
    }
    previous_gpair_ = *gpair;
  }

 protected:
  MomentumOptimizerParam param;
  std::vector<bst_gpair> previous_gpair_;
};

XGBOOST_REGISTER_OPTIMIZER(MomentumOptimizer, "momentum_optimizer")
    .describe("Use momentum to accelerate gradient descent.")
    .set_body([]() { return new MomentumOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
