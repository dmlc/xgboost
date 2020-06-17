/*!
 * Copyright 2017-2020 by Contributors
 */
#include <dmlc/registry.h>
#include <mutex>

#include "xgboost/predictor.h"
#include "xgboost/data.h"
#include "xgboost/generic_parameters.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::PredictorReg);
}  // namespace dmlc
namespace xgboost {

void PredictionContainer::ClearExpiredEntries() {
  std::vector<DMatrix*> expired;
  for (auto& kv : container_) {
    if (kv.second.ref.expired()) {
      expired.emplace_back(kv.first);
    }
  }
  for (auto const& ptr : expired) {
    container_.erase(ptr);
  }
}

PredictionCacheEntry &PredictionContainer::Cache(std::shared_ptr<DMatrix> m, int32_t device) {
  std::lock_guard<std::mutex> guard { cache_lock_ };
  this->ClearExpiredEntries();
  container_[m.get()].ref = m;
  if (device != GenericParameter::kCpuId) {
    container_[m.get()].predictions.SetDevice(device);
  }
  return container_[m.get()];
}

PredictionCacheEntry &PredictionContainer::Entry(DMatrix *m) {
  CHECK(container_.find(m) != container_.cend());
  CHECK(container_.at(m).ref.lock())
      << "[Internal error]: DMatrix: " << m << " has expired.";
  return container_.at(m);
}

decltype(PredictionContainer::container_) const& PredictionContainer::Container() {
  this->ClearExpiredEntries();
  return container_;
}

void Predictor::Configure(
    const std::vector<std::pair<std::string, std::string>>& cfg) {
}
Predictor* Predictor::Create(
    std::string const& name, GenericParameter const* generic_param) {
  auto* e = ::dmlc::Registry<PredictorReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown predictor type " << name;
  }
  auto p_predictor = (e->body)(generic_param);
  return p_predictor;
}
}  // namespace xgboost

namespace xgboost {
namespace predictor {
// List of files that will be force linked in static links.
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(gpu_predictor);
#endif  // XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(cpu_predictor);
}  // namespace predictor
}  // namespace xgboost
