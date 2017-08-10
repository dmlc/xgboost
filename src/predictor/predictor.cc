/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/registry.h>
#include <xgboost/predictor.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::PredictorReg);
}  // namespace dmlc
namespace xgboost {
void Predictor::InitCache(const std::vector<std::shared_ptr<DMatrix> >& cache) {
  for (const std::shared_ptr<DMatrix>& d : cache) {
    PredictionCacheEntry e;
    e.data = d;
    cache_[d.get()] = std::move(e);
  }
}
Predictor* Predictor::Create(std::string name) {
  auto* e = ::dmlc::Registry<PredictorReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown predictor type " << name;
  }
  return (e->body)();
}
}  // namespace xgboost
