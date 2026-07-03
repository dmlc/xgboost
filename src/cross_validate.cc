/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cross_validate.h"

#include "./c_api/c_api_error.h"
#include "./data/extmem_quantile_dmatrix.h"  // for ExtMemQuantileDMatrix
#include "cross_validate/kfolds.h"

using namespace xgboost;  // NOLINT

namespace xgboost {
CvFolds::CvFolds(std::size_t k_folds) {
  CHECK_GT(k_folds, 0);
  std::string obj_name = "reg:squarederror";  // FIXME(jiamingy): Support more objs.
  ctx_.Init({{"device", "cuda"}});
  for (std::size_t i = 0; i < k_folds; ++i) {
    objs_.emplace_back(ObjFunction::Create(obj_name, &ctx_));
    objs_.back()->Configure(Args{});
  }
}

std::size_t CvFolds::KFolds() const noexcept(true) { return this->objs_.size(); }

Context const* CvFolds::Ctx() const { return &this->ctx_; }

ObjFunction* CvFolds::Objective(std::size_t fold_idx) const {
  CHECK_LT(fold_idx, this->objs_.size());
  return this->objs_[fold_idx].get();
}

void CvFolds::InitPrediction(MetaInfo const& info, FoldInfoBatches const& finfo) {
  CHECK_EQ(this->KFolds(), finfo.KFolds());
  auto n_targets = info.labels.Shape(1);
  CHECK_GT(n_targets, 0);

  predts_.resize(this->KFolds());
  for (std::size_t k = 0; k < this->KFolds(); ++k) {
    auto n_samples = finfo.FoldSize(k);
    auto& predt = predts_.at(k);
    predt.SetDevice(ctx_.Device());
    if (predt.Size() != n_samples * n_targets) {
      predt.Resize(n_samples * n_targets);
      predt.Fill(0.0f);
    }
    CHECK_EQ(predt.Size(), n_samples * n_targets);
  }
}

HostDeviceVector<float> const& CvFolds::Prediction(std::size_t fold_idx) const {
  return predts_.at(fold_idx);
}
}  // namespace xgboost

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

XGB_DLL int XGBCvFoldGpairsCreate(FoldGpairsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new FoldGpairs{};
  API_END();
}

XGB_DLL int XGBCvFoldGpairsGet(FoldGpairsHandle hdl, size_t k, float const** out_data,
                               size_t const** out_shape, size_t* out_len) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out_shape);
  xgboost_CHECK_C_ARG_PTR(out_len);
  xgboost_CHECK_C_ARG_PTR(out_data);
  xgboost_CHECK_C_ARG_PTR(hdl);
  auto gpairs = static_cast<FoldGpairs*>(hdl);
  *out_shape = gpairs->gpairs.at(k).Shape().data();
  *out_len = gpairs->gpairs.at(k).Shape().size();
  *out_data = reinterpret_cast<float const*>(gpairs->gpairs.at(k).Data()->ConstDevicePointer());
  API_END();
}

XGB_DLL int XGBCvFoldGpairsFree(FoldGpairsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<FoldGpairs*>(hdl);
  API_END();
}

XGB_DLL int XGBCvFoldInfoBatchesCreate(DMatrixHandle dtrain, size_t k_folds,
                                       FoldInfoBatchesHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  CHECK_GT(k_folds, 0);

  auto p_fmat = CastDMatrixHandle(dtrain);
  auto p_ext_fmat = std::dynamic_pointer_cast<data::ExtMemQuantileDMatrix>(p_fmat);
  CHECK(p_ext_fmat) << "Fold info batches require an ExtMemQuantileDMatrix.";

  auto p_out = std::make_unique<FoldInfoBatches>();
  auto const& batch_ptr = p_ext_fmat->BatchPtr();
  auto const& info = p_ext_fmat->Info();

  for (std::size_t i = 1, n = batch_ptr.size(); i < n; ++i) {
    auto begin = batch_ptr[i - 1];
    auto end = batch_ptr.at(i);
    CHECK_LE(end, info.num_row_);
    p_out->batches.emplace_back();
    FoldInfo& batch = p_out->batches.back();
    for (std::size_t k = 0; k < k_folds; ++k) {
      batch.ridxs.emplace_back();
      cv::KFold(p_ext_fmat->Ctx(), k_folds, begin, end, k, &batch.ridxs.back());
    }
  }

  *out = p_out.release();
  API_END();
}

XGB_DLL int XGBCvFoldInfoBatchesFree(FoldInfoBatchesHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<FoldInfoBatches*>(hdl);
  API_END();
}
