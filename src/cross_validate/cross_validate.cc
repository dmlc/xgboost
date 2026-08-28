/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cross_validate.h"

#include <dmlc/thread_local.h>  // for ThreadLocalStore

#include "../c_api/c_api_error.h"
#include "../common/api_entry.h"              // for XGBAPIThreadLocalEntry
#include "../common/error_msg.h"              // for MaxFeatureSize
#include "../common/json_utils.h"             // for RequiredArg
#include "../common/version.h"                // for Version
#include "../data/extmem_quantile_dmatrix.h"  // for ExtMemQuantileDMatrix
#include "kfolds.h"                           // for FoldInfo
#include "xgboost/json.h"                     // for Json, Array, Object, String, get
#include "xgboost/predictor.h"                // for Predictor

namespace xgboost::cv {
namespace {
[[nodiscard]] bst_feature_t GetNumFeatures(MetaInfo const& info) {
  error::MaxFeatureSize(info.num_col_);
  auto n_features = static_cast<bst_feature_t>(info.num_col_);
  CHECK_NE(n_features, 0) << "0 feature is supplied.";
  return n_features;
}

// FIXME(jiamingy): Calculate intercepts.
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
// FIXME(jiamingy): Make predictor stateless.
[[nodiscard]] std::unique_ptr<Predictor> CreatePredictor(Context const* ctx) {
  CHECK(ctx->IsCUDA()) << "Fused cross-validation requires CUDA.";
  auto predictor = std::unique_ptr<Predictor>{Predictor::Create("gpu_predictor", ctx)};
  predictor->Configure(Args{});
  return predictor;
}
}  // namespace

void FoldModels::Resize(std::size_t n_units) {
  model_params_.resize(n_units);
  properties_.resize(n_units);
  objs_.resize(n_units);
  models_.resize(n_units);
}

void FoldModels::InitUnit(std::size_t unit_idx, std::unique_ptr<ObjFunction> obj) {
  CHECK_LT(unit_idx, this->model_params_.size());
  CHECK_LT(unit_idx, this->properties_.size());
  CHECK_LT(unit_idx, this->objs_.size());
  CHECK_LT(unit_idx, this->models_.size());
  CHECK(obj);

  auto& param = this->model_params_.at(unit_idx);
  param.HandleOldFormat();
  param.Validate(&ctx_);

  auto base_score = BaseScore(&ctx_, param, obj.get());
  this->properties_.at(unit_idx) = LearnerModelParam{&ctx_, param, std::move(base_score),
                                                     obj->Task(), MultiStrategy::kMultiOutputTree};
  this->objs_.at(unit_idx) = std::move(obj);
  this->models_.at(unit_idx) =
      std::make_unique<gbm::GBTreeModel>(&this->properties_.at(unit_idx), &ctx_);
  this->models_.at(unit_idx)->Configure(Args{});
}

FoldModels::FoldModels(std::size_t k_folds, std::shared_ptr<DMatrix> dtrain, bool refit) {
  CHECK(dtrain);
  this->ctx_.FromJson(dtrain->Ctx()->ToJson());
  auto const& info = dtrain->Info();
  auto n_features = GetNumFeatures(info);

  CHECK_GT(k_folds, 0);
  this->layout_ = UnitLayout{k_folds, refit};
  auto n_units = this->NumUnits();
  this->Resize(n_units);

  // The refit unit is configured exactly like a fold, which is what makes it comparable to
  // a model trained on the full dataset on its own.
  std::string obj_name = "reg:squarederror";  // FIXME(jiamingy): Support more objs.
  for (std::size_t u = 0; u < n_units; ++u) {
    auto obj = std::unique_ptr<ObjFunction>{ObjFunction::Create(obj_name, &ctx_)};
    obj->Configure(Args{});

    auto n_targets = obj->Targets(info);
    auto& param = model_params_.at(u);
    param.num_feature = n_features;
    param.num_target = n_targets;
    param.boost_from_average = false;
    auto base_score = DefaultBaseScore(&ctx_, n_targets);
    param.base_score = base_score.Data()->ConstHostVector();
    this->InitUnit(u, std::move(obj));
  }
  CHECK_EQ(objs_.size(), n_units);
  CHECK_EQ(model_params_.size(), n_units);
  CHECK_EQ(properties_.size(), n_units);
  CHECK_EQ(models_.size(), n_units);
}

[[nodiscard]] std::int32_t FoldModels::BoostedRounds() const {
  CHECK(!this->models_.empty());
  CHECK(this->models_.front());
  auto n_rounds = this->models_.front()->BoostedRounds();
  for (auto const& model : this->models_) {
    CHECK(model);
    CHECK_EQ(model->BoostedRounds(), n_rounds) << "CV models are not synchronized.";
  }
  return n_rounds;
}

[[nodiscard]] bst_target_t FoldModels::OutputLength(std::size_t unit_idx) const {
  CHECK_LT(unit_idx, this->properties_.size());
  return this->properties_[unit_idx].OutputLength();
}

[[nodiscard]] bst_target_t FoldModels::LeafLength(std::size_t unit_idx) const {
  CHECK_LT(unit_idx, this->properties_.size());
  return this->properties_[unit_idx].LeafLength();
}

[[nodiscard]] bst_feature_t FoldModels::NumFeatures(std::size_t unit_idx) const {
  CHECK_LT(unit_idx, this->properties_.size());
  return this->properties_[unit_idx].num_feature;
}

[[nodiscard]] ObjFunction* FoldModels::Objective(std::size_t unit_idx) const {
  CHECK_LT(unit_idx, this->objs_.size());
  return this->objs_[unit_idx].get();
}

void FoldModels::InitPrediction(Context const* ctx, MetaInfo const& info,
                                FoldInfoBatches const& finfo, FoldPredictions* out) const {
  CHECK(out);
  CHECK_EQ(this->KFolds(), finfo.KFolds());
  auto n_units = this->NumUnits();
  out->layout = this->layout_;
  out->train.resize(n_units);

  // Init validation prediction vector
  auto predictor = CreatePredictor(ctx);
  CHECK_GT(this->KFolds(), 0);
  auto output_length = this->OutputLength(0);
  predictor->InitOutPredictions(info, &out->valid.predictions, *models_.front());
  out->valid.Reset();
  CHECK_EQ(out->valid.predictions.Device(), ctx->Device());
  CHECK_EQ(out->valid.predictions.Size(), info.num_row_ * output_length);

  // Init training prediction vector. Like the validation cache, it's indexed by the
  // global row index. The rows held out by a fold are padding, `GetGradient` reads back
  // only the rows listed in the fold info. The refit unit holds nothing out, so every row
  // of its cache is used.
  for (std::size_t u = 0; u < n_units; ++u) {
    CHECK_EQ(this->OutputLength(u), output_length)
        << "All CV models must share the same number of outputs.";
    CHECK_EQ(info.labels.Shape(1), output_length);

    auto& predt = out->train.at(u);
    predt.Reset();
    predictor->InitOutPredictions(info, &predt.predictions, *models_.at(u));
    CHECK_EQ(predt.predictions.Device(), ctx->Device());
    CHECK_EQ(predt.predictions.Size(), info.num_row_ * output_length);
  }
  out->output_length = output_length;
}

void FoldModels::CommitModel(std::vector<gbm::TreesOneIter>&& new_trees) {
  auto n_units = this->NumUnits();
  CHECK_EQ(new_trees.size(), n_units);
  CHECK_EQ(this->model_params_.size(), n_units);
  CHECK_EQ(this->properties_.size(), n_units);
  CHECK_EQ(this->models_.size(), n_units);

  for (std::size_t u = 0; u < n_units; ++u) {
    auto const& property = properties_.at(u);
    if (property.IsVectorLeaf()) {
      CHECK_EQ(new_trees[u].size(), 1);
    } else {
      CHECK_EQ(new_trees[u].size(), property.OutputLength());
    }
    models_.at(u)->CommitModel(std::move(new_trees[u]));
  }
}

void FoldModels::LoadUnit(std::size_t unit_idx, Json const& in) {
  auto const& j_unit = get<Object const>(in);

  auto& param = this->model_params_.at(unit_idx);
  param.FromJson(j_unit.at("learner_model_param"));

  auto const& objective = j_unit.at("objective");
  auto obj_name = get<String const>(objective["name"]);
  auto obj = std::unique_ptr<ObjFunction>{ObjFunction::Create(obj_name, &this->ctx_)};
  obj->LoadConfig(objective);
  this->InitUnit(unit_idx, std::move(obj));

  auto const& booster = j_unit.at("gradient_booster");
  CHECK_EQ(get<String const>(booster["name"]), "gbtree");
  this->models_.at(unit_idx)->LoadModel(booster["model"]);
}

FoldModels FoldModels::LoadModel(Json const& in) {
  CHECK(IsA<Object>(in));
  Version::Load(in);

  auto const& j_folds = get<Array const>(in["cv_folds"]);
  auto const& j_in = get<Object const>(in);
  auto refit_it = j_in.find("refit");

  FoldModels out;
  out.ctx_ = Context{};
  out.layout_ = UnitLayout{j_folds.size(), refit_it != j_in.cend()};
  out.Resize(out.NumUnits());

  for (std::size_t k = 0; k < j_folds.size(); ++k) {
    out.LoadUnit(k, j_folds.at(k));
  }
  if (out.HasRefit()) {
    out.LoadUnit(out.RefitIdx(), refit_it->second);
  }
  return out;
}

void FoldModels::SaveUnit(std::size_t unit_idx, Json* out) const {
  CHECK(this->objs_.at(unit_idx));
  CHECK(this->models_.at(unit_idx));

  auto& unit = *out;
  unit["learner_model_param"] = this->model_params_.at(unit_idx).ToJson();

  unit["objective"] = Object{};
  this->objs_.at(unit_idx)->SaveConfig(&unit["objective"]);

  unit["gradient_booster"] = Object{};
  auto& booster = unit["gradient_booster"];
  booster["name"] = String{"gbtree"};
  booster["model"] = Object{};
  this->models_.at(unit_idx)->SaveModel(&booster["model"]);
}

void FoldModels::SaveModel(Json* out) const {
  CHECK(out);
  auto n_units = this->NumUnits();
  CHECK_EQ(this->model_params_.size(), n_units);
  CHECK_EQ(this->properties_.size(), n_units);
  CHECK_EQ(this->models_.size(), n_units);

  Version::Save(out);
  (*out)["cv_folds"] = Array{};
  auto& j_folds = get<Array>((*out)["cv_folds"]);
  j_folds.resize(this->KFolds());

  for (std::size_t k = 0, k_folds = this->KFolds(); k < k_folds; ++k) {
    j_folds[k] = Object{};
    this->SaveUnit(k, &j_folds[k]);
  }
  // The key is absent when the run has no refit model, which is how `LoadModel` recovers
  // the layout.
  if (this->HasRefit()) {
    (*out)["refit"] = Object{};
    this->SaveUnit(this->RefitIdx(), &(*out)["refit"]);
  }
}
}  // namespace xgboost::cv

using namespace xgboost;  // NOLINT

namespace {
using CvAPIThreadLocalStore = dmlc::ThreadLocalStore<XGBAPIThreadLocalEntry>;
}  // namespace

XGB_DLL int XGBCvFoldModelsCreate(size_t k_folds, DMatrixHandle dtrain, int refit,
                                  FoldModelsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  auto p_fmat = CastDMatrixHandle(dtrain);
  *out = new cv::FoldModels{k_folds, p_fmat, static_cast<bool>(refit)};
  API_END();
}

XGB_DLL int XGBCvFoldModelsBoostedRounds(FoldModelsHandle hdl, int* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  xgboost_CHECK_C_ARG_PTR(out);
  *out = static_cast<cv::FoldModels*>(hdl)->BoostedRounds();
  API_END();
}

XGB_DLL int XGBCvFoldModelsSaveModelToBuffer(FoldModelsHandle hdl, char const* json_config,
                                             bst_ulong* out_len, char const** out_dptr) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  xgboost_CHECK_C_ARG_PTR(json_config);
  xgboost_CHECK_C_ARG_PTR(out_len);
  xgboost_CHECK_C_ARG_PTR(out_dptr);

  auto config = Json::Load(StringView{json_config});
  auto format = RequiredArg<String>(config, "format", __func__);
  // `std::ios::out` dumps JSON text, `std::ios::binary` dumps UBJSON.
  std::ios::openmode mode = std::ios::out;
  if (format == "ubj") {
    mode = std::ios::binary;
  } else if (format != "json") {
    LOG(FATAL) << "Unknown model format: `" << format
               << "`. Expecting UBJSON (`ubj`) or JSON (`json`).";
  }

  auto& raw_char_vec = CvAPIThreadLocalStore::Get()->ret_char_vec;
  Json out{Object{}};
  static_cast<cv::FoldModels const*>(hdl)->SaveModel(&out);
  Json::Dump(out, &raw_char_vec, mode);
  *out_dptr = dmlc::BeginPtr(raw_char_vec);
  *out_len = static_cast<bst_ulong>(raw_char_vec.size());
  API_END();
}

XGB_DLL int XGBCvFoldModelsFree(FoldModelsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<cv::FoldModels*>(hdl);
  API_END();
}

XGB_DLL int XGBCvFoldModelsInitPrediction(FoldModelsHandle c_cv_folds, DMatrixHandle dtrain,
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

XGB_DLL int XGBCvFoldPredictionsCreate(FoldPredictionsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new cv::FoldPredictions;
  API_END();
}

namespace {
void ReadPredictionCache(cv::FoldPredictions const* predts, HostDeviceVector<float> const& predt,
                         float const** out_data, size_t* out_n_rows, size_t* out_n_columns) {
  CHECK_GT(predts->output_length, 0) << "The prediction cache is not initialized.";
  CHECK_EQ(predt.Size() % predts->output_length, 0);
  *out_n_columns = predts->output_length;
  *out_n_rows = predt.Size() / predts->output_length;
  *out_data = predt.ConstDevicePointer();
}
}  // namespace

XGB_DLL int XGBCvFoldPredictionsGet(FoldPredictionsHandle hdl, size_t k, float const** out_data,
                                    size_t* out_n_rows, size_t* out_n_columns) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  xgboost_CHECK_C_ARG_PTR(out_data);
  xgboost_CHECK_C_ARG_PTR(out_n_rows);
  xgboost_CHECK_C_ARG_PTR(out_n_columns);
  auto predts = static_cast<cv::FoldPredictions const*>(hdl);
  // Bound by the fold count, not the unit count: this getter must not reach the refit cache.
  CHECK_LT(k, predts->layout.k_folds);
  ReadPredictionCache(predts, predts->Training(k).predictions, out_data, out_n_rows, out_n_columns);
  API_END();
}

XGB_DLL int XGBCvFoldPredictionsGetRefit(FoldPredictionsHandle hdl, float const** out_data,
                                         size_t* out_n_rows, size_t* out_n_columns) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  xgboost_CHECK_C_ARG_PTR(out_data);
  xgboost_CHECK_C_ARG_PTR(out_n_rows);
  xgboost_CHECK_C_ARG_PTR(out_n_columns);
  auto predts = static_cast<cv::FoldPredictions const*>(hdl);
  ReadPredictionCache(predts, predts->Refit().predictions, out_data, out_n_rows, out_n_columns);
  API_END();
}

XGB_DLL int XGBCvFoldPredictionsGetValid(FoldPredictionsHandle hdl, float const** out_data,
                                         size_t* out_n_rows, size_t* out_n_columns) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  xgboost_CHECK_C_ARG_PTR(out_data);
  xgboost_CHECK_C_ARG_PTR(out_n_rows);
  xgboost_CHECK_C_ARG_PTR(out_n_columns);
  auto predts = static_cast<cv::FoldPredictions const*>(hdl);
  ReadPredictionCache(predts, predts->Validation().predictions, out_data, out_n_rows,
                      out_n_columns);
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

namespace {
void ReadGpairs(linalg::Matrix<GradientPair> const& gpair, float const** out_data,
                size_t const** out_shape, size_t* out_len) {
  *out_shape = gpair.Shape().data();
  *out_len = gpair.Shape().size();
  *out_data = reinterpret_cast<float const*>(gpair.Data()->ConstDevicePointer());
}
}  // namespace

XGB_DLL int XGBCvFoldGpairsGet(FoldGpairsHandle hdl, size_t k, float const** out_data,
                               size_t const** out_shape, size_t* out_len) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out_shape);
  xgboost_CHECK_C_ARG_PTR(out_len);
  xgboost_CHECK_C_ARG_PTR(out_data);
  xgboost_CHECK_C_ARG_PTR(hdl);
  auto gpairs = static_cast<cv::FoldGpairs const*>(hdl);
  // Bound by the fold count, not the unit count: this getter must not reach the refit
  // gradient.
  CHECK_LT(k, gpairs->layout.k_folds);
  ReadGpairs(gpairs->gpairs[k], out_data, out_shape, out_len);
  API_END();
}

XGB_DLL int XGBCvFoldGpairsGetRefit(FoldGpairsHandle hdl, float const** out_data,
                                    size_t const** out_shape, size_t* out_len) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out_shape);
  xgboost_CHECK_C_ARG_PTR(out_len);
  xgboost_CHECK_C_ARG_PTR(out_data);
  xgboost_CHECK_C_ARG_PTR(hdl);
  auto gpairs = static_cast<cv::FoldGpairs const*>(hdl);
  ReadGpairs(gpairs->Refit(), out_data, out_shape, out_len);
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
    auto end = batch_ptr[i];
    CHECK_LE(end, info.num_row_);
    p_out->batches.emplace_back();
    cv::FoldInfo& batch = p_out->batches.back();
    for (std::size_t k = 0; k < k_folds; ++k) {
      cv::KFold(p_ext_fmat->Ctx(), k_folds, begin, end, k, &batch);
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
