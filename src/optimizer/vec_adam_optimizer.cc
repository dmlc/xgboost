/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/parameter.h>
#include <xgboost/optimizer.h>
#include <math.h> 

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
            "Effectively the learning rate, controls the size of weights applied suring training");
    DMLC_DECLARE_FIELD(beta1)
        .set_range(0.0f, 1.0f)
        .set_default(0.9f)
        .describe(
            "The exponential decay rate for the first moment estimates");
    DMLC_DECLARE_FIELD(beta2)
        .set_range(0.0f, 1.0f)
        .set_default(0.999f)
        .describe(
            "The exponential decay rate for the second-moment estimates");
    DMLC_DECLARE_FIELD(epsilon)
        .set_range(0.0f, 1.0f)
        .set_default(0.00000001f)
        .describe(
            "a very small number to prevent any division by zero in the implementation");
  }
};
DMLC_REGISTER_PARAMETER(AdamOptimizerParam);

class AdamOptimizer : public Optimizer {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& cfg) override {
    param.InitAllowUnknown(cfg);
  }
  void OptimizeGradients(std::vector<bst_gpair>* gpair) override {
    if (param.alpha == 0.0f) {
      return;
    }
    t = t + 1;/* increment t. */
    if (!previous_gpair_.empty()) {  
      // apply Adam
      for (size_t i = 0; i < gpair->size(); i++) {
	float g = (*gpair)[i].GetGrad();/* unmodified gradient of the current step. */
      	m[i] = param.beta1 * m[i] + (1 - param.beta1) * g;/* biased estimate of first moment. */
      	v[i] = param.beta2 * v[i] + (1 - param.beta2) * pow(g,2);/* biased estimate of second moment. */
	float mu = m[i] / (1 - pow(param.beta1,t));/* Unbiased estimate of first moment. */
	float vu = v[i] / (1 - pow(param.beta2,t));/* Unbiased estimate of second moment. */
        (*gpair)[i] =
            bst_gpair(mu/(sqrt(vu)+param.epsilon), (*gpair)[i].GetHess());
      }
    }
    else{
	int len = gpair->size();
	m = std::vector<float>(len);
	v = std::vector<float>(len);
    }
    previous_gpair_ = *gpair;
  }

 protected:
  AdamOptimizerParam param;
  std::vector<bst_gpair> previous_gpair_;
  int t = 0;
  std::vector<float> m;
  std::vector<float> v;
};

XGBOOST_REGISTER_OPTIMIZER(AdamOptimizer, "adam_optimizer")
    .describe("Use Adam optimiser to accelerate gradient descent.")
    .set_body([]() { return new AdamOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
