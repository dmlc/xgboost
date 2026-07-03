/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cross_validate.h"

#include "./c_api/c_api_error.h"
#include "./c_api/c_api_utils.h"  // for CastDMatrixHandle
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"

namespace xgboost {
[[nodiscard]] auto GetGradient(Context const* ctx, MetaInfo const& info,
                               FoldInfoBatches const& finfo, std::int32_t iter,
                               std::vector<linalg::Matrix<GradientPair>>* p_gpairs) {
  CHECK(info.num_nonzero_) << "Missing data is not yet supported.";
  std::string obj_name = "reg:squarederror";  // fixme
  std::vector<std::unique_ptr<ObjFunction>> objs;

  auto k_folds = finfo.KFolds();
  auto& gpairs = *p_gpairs;
  if (gpairs.empty()) {
    gpairs.resize(k_folds);
  }
  CHECK_EQ(gpairs.size(), k_folds);

  for (std::size_t i = 0, n = finfo.Size(); i < n; ++i) {
    for (std::size_t k = 0; k < k_folds; ++k) {
      objs.emplace_back(ObjFunction::Create(obj_name, ctx));
      objs.back()->Configure(Args{});

      auto ridxs = finfo.batches.at(i).TrainingFold(k);
      constexpr std::size_t kNnz = 0;  // fixme
      auto fold_info = info.Slice(ctx, ridxs, kNnz);

      // Init
      if (gpairs.size() <= k) {
        gpairs.emplace_back();
        CHECK_EQ(gpairs.size(), k + 1);
      }

      HostDeviceVector<float> preds(ridxs.size(), 0.0f, ctx->Device());

      linalg::Matrix<GradientPair> batch_gpair;
      objs.back()->GetGradient(preds, fold_info, iter, &batch_gpair);

      auto& out_gpairs = gpairs.at(k);
      linalg::Stack(&out_gpairs, batch_gpair);
    }
  }
  return gpairs;
}

// The model part of the cross validation result, containing the trees and objectives.
//
// Tree updaters should not be part of it as they are considered "optimizers" and not part
// of the model.
class CvFolds {
  std::vector<std::unique_ptr<ObjFunction>> objs_;
  Context ctx_;

 public:
  explicit CvFolds(std::size_t k_folds) {
    CHECK_GT(k_folds, 0);
    std::string obj_name = "reg:squarederror";  // FIXME(jiamingy): Support more objs.
    ctx_.Init({{"device", "cuda"}});
    for (std::size_t i = 0; i < k_folds; ++i) {
      objs_.emplace_back(ObjFunction::Create(obj_name, &ctx_));
      objs_.back()->Configure(Args{});
    }
  }
  [[nodiscard]] auto KFolds() const noexcept(true) { return this->objs_.size(); }

  [[nodiscard]] Context const* Ctx() const { return &this->ctx_; }
  [[nodiscard]] ObjFunction* Objective(std::size_t fold_idx) const {
    CHECK_LT(fold_idx, this->objs_.size());
    return this->objs_[fold_idx].get();
  }
};

using CvFoldsHandle = void*;
}  // namespace xgboost

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvFoldsCreate(size_t k_folds, CvFoldsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new CvFolds{k_folds};
  API_END();
}

XGB_DLL int XGBCvFoldsFree(CvFoldsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<CvFolds*>(hdl);
  API_END();
}

XGB_DLL int FoldGpairsCreate(FoldGpairsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new FoldGpairs{};
  API_END();
}

XGB_DLL int FoldGpairsFree(FoldGpairsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<FoldGpairs*>(hdl);
  API_END();
}

XGB_DLL int XGBCvGetGradient(DMatrixHandle dtrain, FoldInfoBatchesHandle c_fold_info,
                             FoldGpairsHandle hdl, int iter) {
  API_BEGIN();
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto fold_info = static_cast<FoldInfoBatches*>(c_fold_info);
  auto const& info = p_fmat->Info();
  CHECK(!fold_info->batches.empty());

  auto fold_gpairs = static_cast<FoldGpairs*>(hdl);
  CHECK(fold_gpairs);
  auto gpairs = GetGradient(p_fmat->Ctx(), info, *fold_info, iter, &fold_gpairs->gpairs);

  API_END();
}
