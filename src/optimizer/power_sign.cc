/*!
 * Copyright by Contributors 2017
 */
#define _USE_MATH_DEFINES

#include <dmlc/parameter.h>
#include <xgboost/optimizer.h>
#include <math.h>

namespace xgboost {
namespace optimizer {

DMLC_REGISTRY_FILE_TAG(power_sign_optimizer);

/*! \brief Power Sign parameters */
struct PowerSignOptimizerParam : public dmlc::Parameter<PowerSignOptimizerParam> {
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
        .describe("Coefficent for calculating exponential decay, if 1 there is no decay.");
  }
};
DMLC_REGISTER_PARAMETER(PowerSignOptimizerParam);

class PowerSignOptimizer : public Optimizer {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& cfg) override {
    param.InitAllowUnknown(cfg);
  }

  void OptimizeGradients(std::vector<bst_gpair>* gpair) override {
	t++;
    if (!previous_gpair_.empty()) {
       //apply power sign update
      for (size_t i = 0; i < gpair->size(); i++) {
	float g = (*gpair)[i].GetGrad();
	m[i] = param.beta1 * m[i] + (1 - param.beta1) * g;
	float newGrad = pow(param.base, (exponential(t) * sign(g) * sign(m[i]))) * g;
        (*gpair)[i] = bst_gpair(newGrad, (*gpair)[i].GetHess());
      }
    }
    else{
	int len = gpair->size();
	m = std::vector<float>(len);
	for (size_t i = 0; i < gpair->size(); i++) {
		m[i] = (*gpair)[i].GetGrad();
      	}
    }
    previous_gpair_ = *gpair;
  }

  int sign(float x){
	return (x > 0) - (x < 0);
  }

  float exponential(float t){
	return pow(param.decay,t);
  }

 protected:
  PowerSignOptimizerParam param;
  int t = 0;
  std::vector<float> m;
  std::vector<bst_gpair> previous_gpair_;
};

XGBOOST_REGISTER_OPTIMIZER(PowerSignOptimizer, "power_sign_optimizer")
    .describe("Use power sign to accelerate gradient descent.")
    .set_body([]() { return new PowerSignOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
