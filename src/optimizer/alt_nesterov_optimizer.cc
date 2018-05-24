/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/parameter.h>
#include <xgboost/optimizer.h>
#include <cmath>

namespace xgboost {
namespace optimizer {

DMLC_REGISTRY_FILE_TAG(alt_nesterov_optimizer);

/*! \brief momentum parameters */
struct AltNesterovOptimizerParam
    : public dmlc::Parameter<AltNesterovOptimizerParam> {
  float momentum;
  // declare parameters
  DMLC_DECLARE_PARAMETER(AltNesterovOptimizerParam) {
    DMLC_DECLARE_FIELD(momentum)
        .set_range(0.0f, 1.0f)
        .set_default(0.0f)
        .describe(
            "Momentum coefficient controlling amount of gradient applied from "
            "previous iteration.");
  }
};
DMLC_REGISTER_PARAMETER(AltNesterovOptimizerParam);

class AltNesterovOptimizer : public Optimizer {
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
    // Evaluates the recusive parameter lambda
    lambda_ = UpdateLambda(lambda_);
    // Updates the Gamma value given the current value of lambda
    gamma_ = UpdateGamma(lambda_);

    if (!previous_gpair_.empty()) {
      // apply momentum
      for (size_t i = 0; i < host_gpair.size(); i++) {
        host_gpair[i] = GradientPair(host_gpair[i].GetGrad() * (1 - gamma_) +
                                         (previous_gpair_[i].GetGrad() * gamma_),
                                     host_gpair[i].GetHess());
      }
    }
    previous_gpair_ = host_gpair;
  }

  float UpdateLambda(float lambda) {
    return ((1 + std::sqrt(1 + (4 * lambda * lambda)))) / 2;
  }

  float UpdateGamma(float lambda) {
    // Returns the gamma value as given by the current lambda value and the
    // future lambda
    return (1 - lambda) / UpdateLambda(lambda);
  }

 protected:
  AltNesterovOptimizerParam param_;
  float lambda_ = 0;
  float gamma_ = 0;
  std::vector<GradientPair> previous_gpair_;
  HostDeviceVector<float> nesterov_predictions_;
};

XGBOOST_REGISTER_OPTIMIZER(AltNesterovOptimizer, "alt_nesterov_optimizer")
    .describe(
        "Use alternative nesterov momentum to accelerate gradient descent.")
    .set_body([]() { return new AltNesterovOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
