/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cross_validate.h"

#include "../c_api/c_api_error.h"
#include "../common/error_msg.h"              // for MaxFeatureSize
#include "../data/extmem_quantile_dmatrix.h"  // for ExtMemQuantileDMatrix
#include "./kfolds.h"
#include "xgboost/predictor.h"  // for Predictor

namespace xgboost::cv {
namespace {
[[nodiscard]] bst_feature_t GetNumFeatures(MetaInfo const& info) {
  error::MaxFeatureSize(info.num_col_);
  auto n_features = static_cast<bst_feature_t>(info.num_col_);
  CHECK_NE(n_features, 0) << "0 feature is supplied.";
  return n_features;
}

[[nodiscard]] linalg::Vector<float> DefaultBaseScore(Context const* ctx, bst_target_t n_targets) {
  CHECK_GT(n_targets, 0);
  std::vector<float> h_base_score(n_targets, ObjFunction::DefaultBaseScore());
  std::size_t shape[] = {h_base_score.size()};
  return linalg::Vector<float>{h_base_score.cbegin(), h_base_score.cend(), shape, ctx->Device()};
}

[[nodiscard]] std::unique_ptr<Predictor> CreatePredictor(Context const* ctx) {
  CHECK(ctx->IsCUDA()) << "Fused cross-validation requires CUDA.";
  auto predictor = std::unique_ptr<Predictor>{Predictor::Create("gpu_predictor", ctx)};
  predictor->Configure(Args{});
  return predictor;
}
}  // namespace

CvFolds::CvFolds(std::size_t k_folds, std::shared_ptr<DMatrix> dtrain) {
  auto ctx = dtrain->Ctx();
  auto const& info = dtrain->Info();
  auto n_features = GetNumFeatures(info);

  CHECK_GT(k_folds, 0);
  objs_.reserve(k_folds);
  properties_.reserve(k_folds);
  models_.reserve(k_folds);
  predts_.resize(k_folds);

  std::string obj_name = "reg:squarederror";  // FIXME(jiamingy): Support more objs.
  for (std::size_t i = 0; i < k_folds; ++i) {
    objs_.emplace_back(ObjFunction::Create(obj_name, ctx));
    auto& obj = objs_.back();
    obj->Configure(Args{});

    auto n_targets = obj->Targets(info);
    auto multi_strategy = MultiStrategy::kMultiOutputTree;  // FIXME(jiamingy): Support scalar-leaf.
    properties_.emplace_back(n_features, DefaultBaseScore(ctx, n_targets), n_targets, n_targets,
                             multi_strategy, ctx, obj->Task());
    models_.emplace_back(std::make_unique<gbm::GBTreeModel>(&properties_.back(), ctx));
    models_.back()->Configure(Args{});
  }
  CHECK_EQ(objs_.size(), k_folds);
  CHECK_EQ(properties_.size(), k_folds);
  CHECK_EQ(models_.size(), k_folds);
  CHECK_EQ(predts_.size(), k_folds);
}

std::size_t CvFolds::KFolds() const noexcept(true) { return this->objs_.size(); }

bst_target_t CvFolds::OutputLength(std::size_t fold_idx) const {
  CHECK_LT(fold_idx, this->properties_.size());
  return this->properties_[fold_idx].OutputLength();
}

ObjFunction* CvFolds::Objective(std::size_t fold_idx) const {
  CHECK_LT(fold_idx, this->objs_.size());
  return this->objs_[fold_idx].get();
}

void CvFolds::InitPrediction(Context const* ctx, MetaInfo const& info,
                             FoldInfoBatches const& finfo) {
  CHECK_EQ(this->KFolds(), finfo.KFolds());
  CHECK_EQ(this->predts_.size(), this->KFolds());

  auto predictor = CreatePredictor(ctx);
  for (std::size_t k = 0; k < this->KFolds(); ++k) {
    auto output_length = this->OutputLength(k);
    CHECK_EQ(info.labels.Shape(1), output_length);

    // FIXME(jiamingy): Unify the code paths.
    MetaInfo fold_info;
    fold_info.num_row_ = finfo.FoldSize(k);
    fold_info.num_col_ = info.num_col_;

    auto& predt = predts_.at(k);
    predictor->InitOutPredictions(fold_info, &predt, *models_.at(k));
    CHECK_EQ(predt.Device(), ctx->Device());
    CHECK_EQ(predt.Size(), fold_info.num_row_ * output_length);
  }
}

void CvFolds::CommitModel(std::vector<gbm::TreesOneIter>&& new_trees) {
  CHECK_EQ(new_trees.size(), this->KFolds());
  CHECK_EQ(this->models_.size(), this->KFolds());

  for (std::size_t k = 0; k < this->KFolds(); ++k) {
    auto const& property = properties_.at(k);
    if (property.IsVectorLeaf()) {
      CHECK_EQ(new_trees[k].size(), 1);
    } else {
      CHECK_EQ(new_trees[k].size(), property.OutputLength());
    }
    models_.at(k)->CommitModel(std::move(new_trees[k]));
  }
}

void CvFolds::LoadModel(Json const& in) { CHECK(this->models_.empty()); }

void CvFolds::SaveModel(Json* out) const { *out = Null{}; }

HostDeviceVector<float> const& CvFolds::Prediction(std::size_t fold_idx) const {
  return predts_.at(fold_idx);
}
}  // namespace xgboost::cv

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvFoldsCreate(size_t k_folds, DMatrixHandle dtrain, CvFoldsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  auto p_fmat = CastDMatrixHandle(dtrain);
  *out = new cv::CvFolds{k_folds, p_fmat};
  API_END();
}

XGB_DLL int XGBCvFoldsFree(CvFoldsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<cv::CvFolds*>(hdl);
  API_END();
}

XGB_DLL int XGBCvFoldGpairsCreate(FoldGpairsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new cv::FoldGpairs{};
  API_END();
}

XGB_DLL int XGBCvFoldGpairsGet(FoldGpairsHandle hdl, size_t k, float const** out_data,
                               size_t const** out_shape, size_t* out_len) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out_shape);
  xgboost_CHECK_C_ARG_PTR(out_len);
  xgboost_CHECK_C_ARG_PTR(out_data);
  xgboost_CHECK_C_ARG_PTR(hdl);
  auto gpairs = static_cast<cv::FoldGpairs*>(hdl);
  CHECK_LT(k, gpairs->KFolds());
  *out_shape = gpairs->gpairs[k].Shape().data();
  *out_len = gpairs->gpairs[k].Shape().size();
  *out_data = reinterpret_cast<float const*>(gpairs->gpairs[k].Data()->ConstDevicePointer());
  API_END();
}

XGB_DLL int XGBCvFoldGpairsFree(FoldGpairsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<cv::FoldGpairs*>(hdl);
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

  auto p_out = std::make_unique<cv::FoldInfoBatches>();
  auto const& batch_ptr = p_ext_fmat->BatchPtr();
  auto const& info = p_ext_fmat->Info();

  for (std::size_t i = 1, n = batch_ptr.size(); i < n; ++i) {
    auto begin = batch_ptr[i - 1];
    auto end = batch_ptr.at(i);
    CHECK_LE(end, info.num_row_);
    p_out->batches.emplace_back();
    cv::FoldInfo& batch = p_out->batches.back();
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
  delete static_cast<cv::FoldInfoBatches*>(hdl);
  API_END();
}
