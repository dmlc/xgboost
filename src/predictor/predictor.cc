#include <xgboost/predictor.h>
#include <dmlc/registry.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::PredictorReg);
}  // namespace dmlc
namespace xgboost {
Predictor* Predictor::Create(std::string name) {

  auto* e =
      ::dmlc::Registry<PredictorReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown predictor type " << name;
  }
  return (e->body)();
}
}  // namespace xgboost