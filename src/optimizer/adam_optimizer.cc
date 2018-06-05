/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/parameter.h>
#include <xgboost/optimizer.h>
#include <cmath>

namespace xgboost {
namespace optimizer {

DMLC_REGISTRY_FILE_TAG(adam_optimizer);

/*! \brief momentum parameters */
struct AdamOptimizerParam : public dmlc::Parameter<AdamOptimizerParam> {
  float alpha;
  float beta1;
  float beta2;
  float epsilon;
  // declare parameters
  DMLC_DECLARE_PARAMETER(AdamOptimizerParam) {
    DMLC_DECLARE_FIELD(alpha)
        .set_range(0.0f, 1.0f)
        .set_default(0.01f)
        .describe(
            "Effectively the learning rate, controls the size of weights "
            "applied suring training");
    DMLC_DECLARE_FIELD(beta1)
        .set_range(0.0f, 1.0f)
        .set_default(0.9f)
        .describe("The exponential decay rate for the first moment estimates");
    DMLC_DECLARE_FIELD(beta2)
        .set_range(0.0f, 1.0f)
        .set_default(0.999f)
        .describe("The exponential decay rate for the second-moment estimates");
    DMLC_DECLARE_FIELD(epsilon)
        .set_range(0.0f, 1.0f)
        .set_default(0.00000001f)
        .describe(
            "a very small number to prevent any division by zero in the "
            "implementation");
  }
};
DMLC_REGISTER_PARAMETER(AdamOptimizerParam);

class AdamOptimizer : public Optimizer {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& cfg) override {
    param_.InitAllowUnknown(cfg);
  }
  void OptimizeGradients(HostDeviceVector<GradientPair>* gpair) override {
    auto& host_gpair = gpair->HostVector();
    if (param_.alpha == 0.0f) {
      return;
    }
    t_ = t_ + 1; /* increment t. */
    if (!previous_gpair_.empty()) {
      // apply Adam
      for (size_t i = 0; i < host_gpair.size(); i++) {
        /* unmodified gradient of the current step. */
        float g = host_gpair[i].GetGrad();
        /* biased estimate of first moment. */
        m_[i] = param_.beta1 * m_[i] + (1 - param_.beta1) * g;
        /* biased estimate of second moment. */
        v_[i] = param_.beta2 * v_[i] + (1 - param_.beta2) * pow(g, 2);
        /* Unbiased estimate of first moment. */
        float mu = m_[i] / (1 - pow(param_.beta1, t_));
        /* Unbiased estimate of second moment. */
        float vu = v_[i] / (1 - pow(param_.beta2, t_));
        host_gpair[i] = GradientPair(mu / (sqrt(vu) + param_.epsilon),
                                     host_gpair[i].GetHess());
      }
    } else {
      m_.resize(host_gpair.size());
      v_.resize(host_gpair.size());
    }
    previous_gpair_ = host_gpair;
  }

 protected:
  AdamOptimizerParam param_;
  std::vector<GradientPair> previous_gpair_;
  int t_ = 0;
  std::vector<float> m_;
  std::vector<float> v_;
};

XGBOOST_REGISTER_OPTIMIZER(AdamOptimizer, "adam_optimizer")
    .describe("Use Adam optimiser to accelerate gradient descent.")
    .set_body([]() { return new AdamOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
