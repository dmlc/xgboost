/**
 * Copyright 2017-2025, XGBoost Contributors
 */
#include "xgboost/predictor.h"

#include <dmlc/registry.h>  // for DMLC_REGISTRY_LINK_TAG

#include <cstdint>  // for int32_t
#include <string>   // for string, to_string

#include "../gbm/gbtree_model.h"         // for GBTreeModel
#include "xgboost/base.h"                // for Args, bst_group_t, bst_idx_t
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
void ValidateBaseMarginShape(linalg::Tensor<float, D> const& margin, bst_idx_t n_samples,
                             bst_group_t n_groups) {
  // FIXME: Bindings other than Python and R don't have shape.
  std::string expected{"Invalid shape of base_margin. Expected: (" + std::to_string(n_samples) +
                       ", " + std::to_string(n_groups) + ")"};
  CHECK_EQ(margin.Shape(0), n_samples) << expected;
  CHECK_EQ(margin.Shape(1), n_groups) << expected;
}

namespace cuda_impl {
void InitOutPredictions(Context const* ctx, linalg::VectorView<float const> base_score,
                        linalg::MatrixView<float> predt);
}

void Predictor::InitOutPredictions(const MetaInfo& info, HostDeviceVector<float>* out_preds,
                                   gbm::GBTreeModel const& model) const {
  CHECK_NE(model.learner_model_param->num_output_group, 0);

  if (ctx_->Device().IsCUDA()) {
    out_preds->SetDevice(ctx_->Device());
  }

  // Cannot rely on the Resize to fill as it might skip if the size is already correct.
  auto n = static_cast<size_t>(model.learner_model_param->OutputLength() * info.num_row_);
  out_preds->Resize(n);

  HostDeviceVector<float> const* base_margin = info.base_margin_.Data();
  if (!base_margin->Empty()) {
    ValidateBaseMarginShape(info.base_margin_, info.num_row_,
                            model.learner_model_param->OutputLength());
    out_preds->Copy(*base_margin);
    return;
  }

  auto base_score = model.learner_model_param->BaseScore(this->ctx_->Device());
  if (base_score.Size() == 1) {
    // Fill a scalar
    out_preds->Fill(model.learner_model_param->BaseScore(DeviceOrd::CPU())(0));
    return;
  }

  // Handle multi-output models where base_score is a vector.
  auto predt = linalg::MakeTensorView(this->ctx_, out_preds, info.num_row_,
                                      model.learner_model_param->OutputLength());
  CHECK_EQ(predt.Size(), out_preds->Size());

  if (this->ctx_->IsCUDA()) {
#if defined(XGBOOST_USE_CUDA)
    cuda_impl::InitOutPredictions(this->ctx_, base_score, predt);
#else
    common::AssertGPUSupport();
#endif
  } else {
    common::ParallelFor(info.num_row_, this->ctx_->Threads(), [&](auto i) {
      for (std::size_t j = 0, m = predt.Shape(1); j < m; ++j) {
        predt(i, j) = base_score(j);
      }
    });
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
