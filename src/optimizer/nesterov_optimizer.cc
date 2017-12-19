/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/parameter.h>
#include <xgboost/optimizer.h>

namespace xgboost {
namespace optimizer {

DMLC_REGISTRY_FILE_TAG(nesterov_optimizer);

/*! \brief momentum parameters */
struct NesterovOptimizerParam : public dmlc::Parameter<NesterovOptimizerParam> {
  float momentum;
  // declare parameters
  DMLC_DECLARE_PARAMETER(NesterovOptimizerParam) {
    DMLC_DECLARE_FIELD(momentum)
        .set_range(0.0f, 1.0f)
        .set_default(0.0f)
        .describe(
            "Momentum coefficient controlling amount of gradient applied from "
            "previous iteration.");
  }
};
DMLC_REGISTER_PARAMETER(NesterovOptimizerParam);

class NesterovOptimizer : public Optimizer {
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
      // apply momentum
      for (size_t i = 0; i < gpair->size(); i++) {
        (*gpair)[i] =
            (*gpair)[i] +
            bst_gpair(previous_gpair_[i].GetGrad() * param.momentum, 0.0f);
      }
    }
    previous_gpair_ = *gpair;
  }

  void OptimizePredictions(std::vector<float>* predictions,
                           GradientBooster* gbm, DMatrix* dmatrix) override {
    gbm->NesterovPredict(dmatrix, &nesterov_predictions_);
    CHECK_EQ(predictions->size(), nesterov_predictions_.size());
    for(int i = 0; i < predictions->size(); i++)
    {
      (*predictions)[i] += nesterov_predictions_[i];
    }
  }

 protected:
  NesterovOptimizerParam param;
  std::vector<bst_gpair> previous_gpair_;
  std::vector<float> nesterov_predictions_;
};

XGBOOST_REGISTER_OPTIMIZER(NesterovOptimizer, "nesterov_optimizer")
    .describe("Use nesterov momentum to accelerate gradient descent.")
    .set_body([]() { return new NesterovOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
