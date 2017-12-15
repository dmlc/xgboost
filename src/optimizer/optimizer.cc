/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/registry.h>
#include <xgboost/optimizer.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::OptimizerReg);
}  // namespace dmlc

namespace xgboost {
Optimizer* Optimizer::Create(std::string name) {
  auto* e = ::dmlc::Registry<OptimizerReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown optimizer type " << name;
  }
  return (e->body)();
}

class DefaultOptimizer : public Optimizer {};
}  // namespace xgboost

namespace xgboost {
namespace optimizer {
XGBOOST_REGISTER_OPTIMIZER(DefaultOptimizer, "default_optimizer")
    .describe("Null optimizer.")
    .set_body([]() { return new DefaultOptimizer(); });
DMLC_REGISTRY_LINK_TAG(momentum_optimizer);
}  // namespace optimizer
}  // namespace xgboost
