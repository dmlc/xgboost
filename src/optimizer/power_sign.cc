/*!
 * Copyright by Contributors 2017
 */
#define _USE_MATH_DEFINES

#include <dmlc/parameter.h>
#include <xgboost/optimizer.h>
#include <cmath>

namespace xgboost {
namespace optimizer {

DMLC_REGISTRY_FILE_TAG(power_sign_optimizer);

/*! \brief Power Sign parameters */
struct PowerSignOptimizerParam
    : public dmlc::Parameter<PowerSignOptimizerParam> {
  float beta1;
  float base;
  float decay;
  // declare parameters
  DMLC_DECLARE_PARAMETER(PowerSignOptimizerParam) {
    DMLC_DECLARE_FIELD(beta1)
        .set_range(0.0f, 1.0f)
        .set_default(0.9f)
        .describe("Decay for calculating the moving average");
    DMLC_DECLARE_FIELD(base)
        .set_range(0.0f, 10.0f)
        .set_default(M_E)
        .describe("Base value used in optimiser, defaults to e");
    DMLC_DECLARE_FIELD(decay)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe(
            "Coefficent for calculating exponential decay, if 1 there is no "
            "decay.");
  }
};
DMLC_REGISTER_PARAMETER(PowerSignOptimizerParam);

class PowerSignOptimizer : public Optimizer {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& cfg) override {
    param_.InitAllowUnknown(cfg);
  }

  void OptimizeGradients(HostDeviceVector<GradientPair>* gpair) override {
    auto& host_gpair = gpair->HostVector();
    t_++;
    if (!previous_gpair_.empty()) {
      // apply power sign update
      for (size_t i = 0; i < host_gpair.size(); i++) {
        float g = host_gpair[i].GetGrad();
        m_[i] = param_.beta1 * m_[i] + (1 - param_.beta1) * g;
        float new_grad =
            pow(param_.base, (Exponential(t_) * Sign(g) * Sign(m_[i]))) * g;
        host_gpair[i] = GradientPair(new_grad, host_gpair[i].GetHess());
      }
    } else {
      m_.resize(host_gpair.size());
      for (size_t i = 0; i < host_gpair.size(); i++) {
        m_[i] = host_gpair[i].GetGrad();
      }
    }
    previous_gpair_ = host_gpair;
  }

  int Sign(float x) { return (x > 0) - (x < 0); }

  float Exponential(float t) { return pow(param_.decay, t); }

 protected:
  PowerSignOptimizerParam param_;
  int t_ = 0;
  std::vector<float> m_;
  std::vector<GradientPair> previous_gpair_;
};

XGBOOST_REGISTER_OPTIMIZER(PowerSignOptimizer, "power_sign_optimizer")
    .describe("Use power sign to accelerate gradient descent.")
    .set_body([]() { return new PowerSignOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
