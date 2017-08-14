/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/registry.h>
#include <xgboost/predictor.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::PredictorReg);
}  // namespace dmlc
namespace xgboost {
void Predictor::Init(
    const std::vector<std::pair<std::string, std::string>>& cfg,
    const std::vector<std::shared_ptr<DMatrix>>& cache) {
  for (const std::shared_ptr<DMatrix>& d : cache) {
    PredictionCacheEntry e;
    e.data = d;
    cache_[d.get()] = std::move(e);
  }
}
bool Predictor::PredictFromCache(DMatrix* dmat,
                                 std::vector<bst_float>* out_preds,
                                 const gbm::GBTreeModel& model,
                                 unsigned ntree_limit) {
  if (ntree_limit == 0 ||
      ntree_limit * model.param.num_output_group >= model.trees.size()) {
    auto it = cache_.find(dmat);
    if (it != cache_.end()) {
      std::vector<bst_float>& y = it->second.predictions;
      if (y.size() != 0) {
        out_preds->resize(y.size());
        std::copy(y.begin(), y.end(), out_preds->begin());
        return true;
      }
    }
  }

  return false;
}
void Predictor::InitOutPredictions(const MetaInfo& info,
                                   std::vector<bst_float>* out_preds,
                                   const gbm::GBTreeModel& model) const {
  size_t n = model.param.num_output_group * info.num_row;
  const std::vector<bst_float>& base_margin = info.base_margin;
  out_preds->resize(n);
  if (base_margin.size() != 0) {
    CHECK_EQ(out_preds->size(), n);
    std::copy(base_margin.begin(), base_margin.end(), out_preds->begin());
  } else {
    std::fill(out_preds->begin(), out_preds->end(), model.base_margin);
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

namespace xgboost {
namespace predictor {
// List of files that will be force linked in static links.
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(gpu_predictor);
#endif
DMLC_REGISTRY_LINK_TAG(cpu_predictor);
}  // namespace predictor
}  // namespace xgboost
