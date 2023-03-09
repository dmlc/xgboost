/**
 * Copyright 2017-2023 by Contributors
 */
#include "xgboost/predictor.h"

#include <dmlc/registry.h>               // for DMLC_REGISTRY_LINK_TAG

#include <cstdint>                       // for int32_t
#include <string>                        // for string, to_string

#include "../gbm/gbtree_model.h"         // for GBTreeModel
#include "xgboost/base.h"                // for bst_float, Args, bst_group_t, bst_row_t
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/learner.h"             // for LearnerModelParam
#include "xgboost/linalg.h"              // for Tensor, TensorView
#include "xgboost/logging.h"             // for CHECK_EQ, CHECK_NE, LOG

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::PredictorReg);
}  // namespace dmlc

namespace xgboost {
void Predictor::Configure(Args const&) {}

Predictor* Predictor::Create(std::string const& name, Context const* ctx) {
  auto* e = ::dmlc::Registry<PredictorReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown predictor type " << name;
  }
  auto p_predictor = (e->body)(ctx);
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
  std::size_t n{model.learner_model_param->OutputLength() * info.num_row_};

  const HostDeviceVector<bst_float>* base_margin = info.base_margin_.Data();
  if (ctx_->gpu_id >= 0) {
    out_preds->SetDevice(ctx_->gpu_id);
  }
  if (!base_margin->Empty()) {
    out_preds->Resize(n);
    ValidateBaseMarginShape(info.base_margin_, info.num_row_,
                            model.learner_model_param->OutputLength());
    out_preds->Copy(*base_margin);
  } else {
    // cannot rely on the Resize to fill as it might skip if the size is already correct.
    out_preds->Resize(n);
    auto base_score = model.learner_model_param->BaseScore(Context::kCpuId)(0);
    out_preds->Fill(base_score);
  }
}
}  // namespace xgboost

namespace xgboost::predictor {
// List of files that will be force linked in static links.
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(gpu_predictor);
#endif  // XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(cpu_predictor);
}  // namespace xgboost::predictor
