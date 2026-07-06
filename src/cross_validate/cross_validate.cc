/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cross_validate.h"

#include "../c_api/c_api_error.h"
#include "../common/error_msg.h"              // for MaxFeatureSize
#include "../common/version.h"                // for Version
#include "../data/extmem_quantile_dmatrix.h"  // for ExtMemQuantileDMatrix
#include "./kfolds.h"
#include "xgboost/json.h"       // for Json, Array, Object, String, get
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

[[nodiscard]] linalg::Vector<float> BaseScore(Context const* ctx, LearnerModelParamLegacy const& p,
                                              ObjFunction* obj) {
  std::vector<float> h_base_score{p.base_score.cbegin(), p.base_score.cend()};
  std::size_t shape[] = {h_base_score.size()};
  linalg::Vector<float> base_score{h_base_score.cbegin(), h_base_score.cend(), shape,
                                   ctx->Device()};
  obj->ProbToMargin(&base_score);
  return base_score;
}

[[nodiscard]] std::unique_ptr<Predictor> CreatePredictor(Context const* ctx) {
  CHECK(ctx->IsCUDA()) << "Fused cross-validation requires CUDA.";
  auto predictor = std::unique_ptr<Predictor>{Predictor::Create("gpu_predictor", ctx)};
  predictor->Configure(Args{});
  return predictor;
}
}  // namespace

void FoldModels::Resize(std::size_t k_folds) {
  model_params_.resize(k_folds);
  properties_.resize(k_folds);
  objs_.resize(k_folds);
  models_.resize(k_folds);
}

void FoldModels::InitFold(std::size_t fold_idx, std::unique_ptr<ObjFunction> obj) {
  CHECK_LT(fold_idx, this->model_params_.size());
  CHECK_LT(fold_idx, this->properties_.size());
  CHECK_LT(fold_idx, this->objs_.size());
  CHECK_LT(fold_idx, this->models_.size());
  CHECK(obj);

  auto& param = this->model_params_.at(fold_idx);
  param.HandleOldFormat();
  param.Validate(&ctx_);

  auto base_score = BaseScore(&ctx_, param, obj.get());
  this->properties_.at(fold_idx) = LearnerModelParam{&ctx_, param, std::move(base_score),
                                                     obj->Task(), MultiStrategy::kMultiOutputTree};
  this->objs_.at(fold_idx) = std::move(obj);
  this->models_.at(fold_idx) =
      std::make_unique<gbm::GBTreeModel>(&this->properties_.at(fold_idx), &ctx_);
  this->models_.at(fold_idx)->Configure(Args{});
}

FoldModels::FoldModels(std::size_t k_folds, std::shared_ptr<DMatrix> dtrain) {
  CHECK(dtrain);
  this->ctx_.FromJson(dtrain->Ctx()->ToJson());
  auto const& info = dtrain->Info();
  auto n_features = GetNumFeatures(info);

  CHECK_GT(k_folds, 0);
  this->Resize(k_folds);

  std::string obj_name = "reg:squarederror";  // FIXME(jiamingy): Support more objs.
  for (std::size_t i = 0; i < k_folds; ++i) {
    auto obj = std::unique_ptr<ObjFunction>{ObjFunction::Create(obj_name, &ctx_)};
    obj->Configure(Args{});

    auto n_targets = obj->Targets(info);
    auto& param = model_params_.at(i);
    param.num_feature = n_features;
    param.num_target = n_targets;
    param.boost_from_average = false;
    auto base_score = DefaultBaseScore(&ctx_, n_targets);
    param.base_score = base_score.Data()->ConstHostVector();
    this->InitFold(i, std::move(obj));
  }
  CHECK_EQ(objs_.size(), k_folds);
  CHECK_EQ(model_params_.size(), k_folds);
  CHECK_EQ(properties_.size(), k_folds);
  CHECK_EQ(models_.size(), k_folds);
}

std::size_t FoldModels::KFolds() const noexcept(true) { return this->objs_.size(); }

bst_target_t FoldModels::OutputLength(std::size_t fold_idx) const {
  CHECK_LT(fold_idx, this->properties_.size());
  return this->properties_[fold_idx].OutputLength();
}

ObjFunction* FoldModels::Objective(std::size_t fold_idx) const {
  CHECK_LT(fold_idx, this->objs_.size());
  return this->objs_[fold_idx].get();
}

void FoldModels::InitPrediction(Context const* ctx, MetaInfo const& info,
                                FoldInfoBatches const& finfo, FoldPredictions* out) const {
  CHECK(out);
  CHECK_EQ(this->KFolds(), finfo.KFolds());
  if (out->predts.empty()) {
    out->predts.resize(this->KFolds());
  }
  CHECK_EQ(out->predts.size(), this->KFolds());

  auto predictor = CreatePredictor(ctx);
  for (std::size_t k = 0; k < this->KFolds(); ++k) {
    auto output_length = this->OutputLength(k);
    CHECK_EQ(info.labels.Shape(1), output_length);

    // FIXME(jiamingy): Unify the code paths.
    MetaInfo fold_info;
    fold_info.num_row_ = finfo.FoldSize(k);
    fold_info.num_col_ = info.num_col_;

    auto& predt = out->predts.at(k);
    predictor->InitOutPredictions(fold_info, &predt, *models_.at(k));
    CHECK_EQ(predt.Device(), ctx->Device());
    CHECK_EQ(predt.Size(), fold_info.num_row_ * output_length);
  }
}

void FoldModels::CommitModel(std::vector<gbm::TreesOneIter>&& new_trees) {
  CHECK_EQ(new_trees.size(), this->KFolds());
  CHECK_EQ(this->model_params_.size(), this->KFolds());
  CHECK_EQ(this->properties_.size(), this->KFolds());
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

FoldModels FoldModels::LoadModel(Json const& in) {
  CHECK(IsA<Object>(in));
  Version::Load(in);

  auto const& j_folds = get<Array const>(in["cv_folds"]);
  FoldModels out;
  out.ctx_ = Context{};
  out.Resize(j_folds.size());

  for (std::size_t k = 0; k < j_folds.size(); ++k) {
    auto const& fold = j_folds.at(k);
    auto const& j_fold = get<Object const>(fold);

    auto& param = out.model_params_.at(k);
    param.FromJson(j_fold.at("learner_model_param"));

    auto const& objective = j_fold.at("objective");
    auto obj_name = get<String const>(objective["name"]);
    auto obj = std::unique_ptr<ObjFunction>{ObjFunction::Create(obj_name, &out.ctx_)};
    obj->LoadConfig(objective);
    out.InitFold(k, std::move(obj));

    auto const& booster = j_fold.at("gradient_booster");
    CHECK_EQ(get<String const>(booster["name"]), "gbtree");
    out.models_.at(k)->LoadModel(booster["model"]);
  }
  return out;
}

void FoldModels::SaveModel(Json* out) const {
  CHECK(out);
  CHECK_EQ(this->model_params_.size(), this->KFolds());
  CHECK_EQ(this->properties_.size(), this->KFolds());
  CHECK_EQ(this->models_.size(), this->KFolds());

  Version::Save(out);
  (*out)["cv_folds"] = Array{};
  auto& j_folds = get<Array>((*out)["cv_folds"]);
  j_folds.resize(this->KFolds());

  for (std::size_t k = 0, n_folds = this->KFolds(); k < n_folds; ++k) {
    CHECK(this->objs_.at(k));
    CHECK(this->models_.at(k));

    j_folds[k] = Object{};
    auto& fold = j_folds[k];
    fold["learner_model_param"] = this->model_params_.at(k).ToJson();

    fold["objective"] = Object{};
    this->objs_.at(k)->SaveConfig(&fold["objective"]);

    fold["gradient_booster"] = Object{};
    auto& booster = fold["gradient_booster"];
    booster["name"] = String{"gbtree"};
    booster["model"] = Object{};
    this->models_.at(k)->SaveModel(&booster["model"]);
  }
}

}  // namespace xgboost::cv

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvFoldsCreate(size_t k_folds, DMatrixHandle dtrain, FoldsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  auto p_fmat = CastDMatrixHandle(dtrain);
  *out = new cv::FoldModels{k_folds, p_fmat};
  API_END();
}

XGB_DLL int XGBCvFoldsFree(FoldsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<cv::FoldModels*>(hdl);
  API_END();
}

XGB_DLL int XGBCvFoldPredictionsCreate(FoldPredictionsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new cv::FoldPredictions{};
  API_END();
}

XGB_DLL int XGBCvInitPrediction(FoldsHandle c_cv_folds, DMatrixHandle dtrain,
                                FoldInfoBatchesHandle c_fold_info,
                                FoldPredictionsHandle c_predt) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(c_predt);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto cv_folds = static_cast<cv::FoldModels*>(c_cv_folds);
  auto fold_info = static_cast<cv::FoldInfoBatches*>(c_fold_info);
  auto predt = static_cast<cv::FoldPredictions*>(c_predt);
  cv_folds->InitPrediction(p_fmat->Ctx(), p_fmat->Info(), *fold_info, predt);
  API_END();
}

XGB_DLL int XGBCvFoldPredictionsFree(FoldPredictionsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<cv::FoldPredictions*>(hdl);
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
