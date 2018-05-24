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
    if (param_.momentum == 0.0f) {
      return;
    }

    auto& host_gpair = gpair->HostVector();
    g.resize(host_gpair.size());
    for (int i = 0; i < host_gpair.size(); i++) {
      g[i] = param_.momentum*g[i] + host_gpair[i].GetGrad();
      host_gpair[i] += GradientPair(param_.momentum*g[i], 0.0f);
    }
  }


 protected:
  NesterovOptimizerParam param_;
  std::vector<float> g;
};

XGBOOST_REGISTER_OPTIMIZER(NesterovOptimizer, "nesterov_optimizer")
    .describe("Use nesterov momentum to accelerate gradient descent.")
    .set_body([]() { return new NesterovOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
