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
    param_.InitAllowUnknown(cfg);
  }
  void OptimizeGradients(HostDeviceVector<GradientPair>* gpair) override {
    auto& host_gpair = gpair->HostVector();
    if (param_.momentum == 0.0f) {
      return;
    }
    if (!previous_gpair_.empty()) {
      // apply momentum
      for (size_t i = 0; i < host_gpair.size(); i++) {
        host_gpair[i] =
            host_gpair[i] +
            GradientPair(previous_gpair_[i].GetGrad() * param_.momentum, 0.0f);
      }
    }
    previous_gpair_ = host_gpair;
  }

  void OptimizePredictions(HostDeviceVector<float>* predictions,
                           GradientBooster* gbm, DMatrix* dmatrix) override {
    gbm->NesterovPredict(dmatrix, &nesterov_predictions_);
    CHECK_EQ(predictions->Size(), nesterov_predictions_.Size());
    auto& host_predictions = predictions->HostVector();
    auto& host_nesterov_predictions = nesterov_predictions_.HostVector();
    for (int i = 0; i < predictions->Size(); i++) {
      host_predictions[i] += host_nesterov_predictions[i] * param_.momentum;
    }
  }

 protected:
  NesterovOptimizerParam param_;
  std::vector<GradientPair> previous_gpair_;
  HostDeviceVector<float> nesterov_predictions_;
};

XGBOOST_REGISTER_OPTIMIZER(NesterovOptimizer, "nesterov_optimizer")
    .describe("Use nesterov momentum to accelerate gradient descent.")
    .set_body([]() { return new NesterovOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
