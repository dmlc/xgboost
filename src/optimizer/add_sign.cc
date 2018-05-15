/*!
 * Copyright by Contributors 2017
 */
#define _USE_MATH_DEFINES

#include <dmlc/parameter.h>
#include <xgboost/optimizer.h>
#include <cmath>

namespace xgboost {
namespace optimizer {

DMLC_REGISTRY_FILE_TAG(add_sign_optimizer);

/*! \brief Add Sign parameters */
struct AddSignOptimizerParam : public dmlc::Parameter<AddSignOptimizerParam> {
  float beta1;
  float alpha;
  float decay;
  // declare parameters
  DMLC_DECLARE_PARAMETER(AddSignOptimizerParam) {
    DMLC_DECLARE_FIELD(beta1)
        .set_range(0.0f, 1.0f)
        .set_default(0.9f)
        .describe("Decay for calculating the moving average");
    DMLC_DECLARE_FIELD(alpha)
        .set_range(0.0f, 10.0f)
        .set_default(1.0f)
        .describe("Alpha used in optimizer.");
    DMLC_DECLARE_FIELD(decay)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe(
            "Coefficent for calculating exponential decay, if 1 there is no "
            "decay.");
  }
};
DMLC_REGISTER_PARAMETER(AddSignOptimizerParam);

class AddSignOptimizer : public Optimizer {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& cfg) override {
    param_.InitAllowUnknown(cfg);
  }

  void OptimizeGradients(HostDeviceVector<GradientPair>* gpair) override {
    auto& host_gpair = gpair->HostVector();
    t_++;
    if (!previous_gpair_.empty()) {
      // apply Add sign update
      for (size_t i = 0; i < host_gpair.size(); i++) {
        float g = host_gpair[i].GetGrad();
        m_[i] = param_.beta1 * m_[i] + (1 - param_.beta1) * g;
        float new_grad =
            (param_.alpha + Exponential(t_) * Sign(g) * Sign(m_[i])) * g;
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
  AddSignOptimizerParam param_;
  int t_ = 0;
  float base_ = M_E;
  std::vector<float> m_;
  std::vector<GradientPair> previous_gpair_;
};

XGBOOST_REGISTER_OPTIMIZER(AddSignOptimizer, "add_sign_optimizer")
    .describe("Use add sign to accelerate gradient descent.")
    .set_body([]() { return new AddSignOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
