/*!
 * Copyright 2017-2021 by Contributors
 */
#include <dmlc/registry.h>
#include <mutex>

#include "xgboost/predictor.h"
#include "xgboost/data.h"
#include "xgboost/generic_parameters.h"

#include "../gbm/gbtree.h"

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
    const std::vector<std::pair<std::string, std::string>>&) {
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

template <int32_t D>
void ValidateBaseMarginShape(linalg::Tensor<float, D> const& margin, bst_row_t n_samples,
                             bst_group_t n_groups) {
  // FIXME: Bindings other than Python doesn't have shape.
  std::string expected{"Invalid shape of base_margin. Expected: (" + std::to_string(n_samples) +
                       ", " + std::to_string(n_groups) + ")"};
  CHECK_EQ(margin.Shape(0), n_samples) << expected;
  CHECK_EQ(margin.Shape(1), n_groups) << expected;
}

void Predictor::InitOutPredictions(const MetaInfo& info, HostDeviceVector<bst_float>* out_preds,
                                   const gbm::GBTreeModel& model) const {
  CHECK_NE(model.learner_model_param->num_output_group, 0);
  size_t n_classes = model.learner_model_param->num_output_group;
  size_t n = n_classes * info.num_row_;
  const HostDeviceVector<bst_float>* base_margin = info.base_margin_.Data();
  if (ctx_->gpu_id >= 0) {
    out_preds->SetDevice(ctx_->gpu_id);
  }
  if (!base_margin->Empty()) {
    out_preds->Resize(n);
    ValidateBaseMarginShape(info.base_margin_, info.num_row_, n_classes);
    out_preds->Copy(*base_margin);
  } else {
    // cannot rely on the Resize to fill as it might skip if the size is already correct.
    out_preds->Resize(n);
    auto base_score = model.learner_model_param->BaseScore(Context::kCpuId)(0);
    out_preds->Fill(base_score);
  }
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
