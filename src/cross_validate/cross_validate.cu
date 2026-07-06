/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>   // for make_shared, make_unique, unique_ptr
#include <sstream>  // for ostringstream
#include <utility>  // for move

#include "../c_api/c_api_error.h"
#include "../c_api/c_api_utils.h"        // for CastDMatrixHandle
#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/cuda_rt_utils.h"     // for SetDevice
#include "../common/linalg_op.cuh"       // for tcbegin, tcend, tbegin
#include "../tree/updater_gpu_hist.cuh"  // for HistBatch, InitBatchCuts
#include "cross_validate.h"
#include "xgboost/json.h"  // for Json

namespace xgboost::cv {
namespace {
[[nodiscard]] HostDeviceVector<bst_idx_t> GlobalTrainingRows(Context const* ctx,
                                                             FoldInfo const& batch,
                                                             std::size_t fold,
                                                             bst_idx_t batch_begin) {
  auto d_local = batch.ridxs.at(fold).ConstDeviceSpan();
  HostDeviceVector<bst_idx_t> d_global(d_local.size(), 0ul, ctx->Device());
  thrust::transform(ctx->CUDACtx()->CTP(), dh::tcbegin(d_local), dh::tcend(d_local),
                    dh::tbegin(d_global.DeviceSpan()),
                    [=] __device__(std::size_t i) { return i + batch_begin; });
  return d_global;
}

[[nodiscard]] HostDeviceVector<float> BatchPrediction(Context const* ctx,
                                                      HostDeviceVector<float> const& predt,
                                                      std::size_t begin, std::size_t size) {
  HostDeviceVector<float> out(size, 0.0f, ctx->Device());
  auto d_predt = predt.ConstDeviceSpan().subspan(begin, size);
  auto d_out = out.DeviceSpan();
  thrust::copy(ctx->CUDACtx()->CTP(), dh::tcbegin(d_predt), dh::tcend(d_predt), dh::tbegin(d_out));
  return out;
}

void CopyBatchGpair(Context const* ctx, linalg::Matrix<GradientPair> const& batch_gpair,
                    bst_idx_t begin, bst_idx_t end, linalg::Matrix<GradientPair>* out_gpairs) {
  CHECK_EQ(batch_gpair.Shape(0), end - begin);
  CHECK(batch_gpair.Shape(1) == out_gpairs->Shape(1) || out_gpairs->Shape(1) <= 1);

  if (out_gpairs->Shape(0) < end) {
    out_gpairs->Reshape(end, batch_gpair.Shape(1));
  }

  auto d_batch_gpair = batch_gpair.View(ctx->Device());
  auto d_out = out_gpairs->View(ctx->Device()).Slice(linalg::Range(begin, end), linalg::All());
  thrust::copy(ctx->CUDACtx()->CTP(), linalg::tcbegin(d_batch_gpair), linalg::tcend(d_batch_gpair),
               linalg::tbegin(d_out));
}

[[nodiscard]] Args JsonToArgs(Json const& config) {
  CHECK(config.GetValue().Type() == Value::ValueKind::kObject)
      << "CV tree method configuration must be a JSON object.";

  Args args;
  for (auto const& kv : get<Object const>(config)) {
    args.emplace_back(kv.first, JsonScalarToString(kv.second));
  }
  return args;
}

void CheckNoUnknownParams(Args const& unknown) {
  if (unknown.empty()) {
    return;
  }
  std::stringstream ss;
  ss << "Unknown CV tree method parameters: { ";
  for (std::size_t i = 0; i < unknown.size(); ++i) {
    ss << unknown[i].first;
    if (i + 1 != unknown.size()) {
      ss << ", ";
    }
  }
  ss << " }";
  LOG(FATAL) << ss.str();
}
}  // namespace

void FoldModels::GetGradient(Context const* ctx, MetaInfo const& info,
                             FoldPredictions const& predts, FoldInfoBatches const& finfo,
                             std::vector<bst_idx_t> const& batch_ptr, std::int32_t iter,
                             FoldGpairs* out) const {
  CHECK(!finfo.Empty());
  CHECK(out);
  CHECK_EQ(batch_ptr.size(), finfo.Size() + 1);

  auto k_folds = finfo.KFolds();
  CHECK_EQ(this->KFolds(), k_folds);
  CHECK_EQ(predts.KFolds(), k_folds);

  auto& gpairs = out->gpairs;
  if (gpairs.empty()) {
    gpairs.resize(k_folds);
  }
  CHECK_EQ(gpairs.size(), k_folds);

  std::vector<bst_idx_t> cursors(k_folds, 0ul);

  for (std::size_t i = 0, n = finfo.Size(); i < n; ++i) {
    auto const& batch = finfo.batches.at(i);
    CHECK_EQ(batch.KFolds(), k_folds);
    auto batch_begin = batch_ptr.at(i);
    CHECK_LE(batch_ptr.at(i + 1), info.num_row_);

    for (std::size_t k = 0; k < k_folds; ++k) {
      auto ridxs = GlobalTrainingRows(ctx, batch, k, batch_begin);

      constexpr std::size_t kNnz = 0;  // fixme
      auto fold_info = info.Slice(ctx, ridxs.ConstDeviceSpan(), kNnz);

      auto const& fold_preds = predts.Prediction(k);
      auto output_length = this->OutputLength(k);
      CHECK_EQ(fold_info.labels.Shape(1), output_length);
      auto n_predts = ridxs.Size() * output_length;
      CHECK_EQ(fold_info.labels.Size(), n_predts);
      auto pred_begin = cursors[k] * output_length;
      CHECK_LE(pred_begin + n_predts, fold_preds.Size());
      auto preds = BatchPrediction(ctx, fold_preds, pred_begin, n_predts);

      linalg::Matrix<GradientPair> batch_gpair;
      this->Objective(k)->GetGradient(preds, fold_info, iter, &batch_gpair);

      auto prev = cursors[k];
      cursors[k] += ridxs.Size();
      CopyBatchGpair(ctx, batch_gpair, prev, cursors[k], &gpairs.at(k));
    }
  }

  for (std::size_t k = 0; k < k_folds; ++k) {
    CHECK_EQ(finfo.FoldSize(k), out->gpairs.at(k).Shape(0));
  }
}

class TreeMethod {
  Context const* ctx_;
  tree::TrainParam param_;
  tree::HistMakerTrainParam hist_param_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  DMatrix* p_last_fmat_{nullptr};

 public:
  explicit TreeMethod(Context const* ctx)
      : ctx_{ctx}, column_sampler_{std::make_shared<common::ColumnSampler>()} {
    CHECK(ctx_);
  }

  void Configure(Args args) {
    CHECK(ctx_->IsCUDA()) << "CV tree method `hist` requires a CUDA device.";

    auto unknown = param_.UpdateAllowUnknown(args);
    unknown = hist_param_.UpdateAllowUnknown(unknown);
    CheckNoUnknownParams(unknown);
  }

  void InitDataOnce(DMatrix* p_fmat, std::size_t k_folds) {
    CHECK(ctx_->IsCUDA()) << "CV tree method `hist` requires a CUDA device.";
    CHECK(p_fmat);
    curt::SetDevice(ctx_->Ordinal());
    p_fmat->Info().feature_types.SetDevice(ctx_->Device());

    auto batch = tree::cuda_impl::HistBatch(param_);
    auto [cuts, dense_compressed] = tree::InitBatchCuts(ctx_, p_fmat, batch);
    auto batch_ptr = p_fmat->BatchPtr();

    p_last_fmat_ = p_fmat;
  }

  void Update(FoldModels* folds, DMatrix* p_fmat, FoldInfoBatches const& finfo,
              FoldGpairs const& gpairs) {
    CHECK(folds);
    CHECK(p_fmat);
    CHECK_EQ(folds->KFolds(), finfo.KFolds());
    CHECK_EQ(folds->KFolds(), gpairs.KFolds());

    if (p_last_fmat_ != p_fmat) {
      this->InitDataOnce(p_fmat, folds->KFolds());
    }

    std::vector<gbm::TreesOneIter> new_trees(folds->KFolds());
    std::vector<RegTree*> tree_ptrs;
    tree_ptrs.reserve(folds->KFolds());
    for (std::size_t k = 0; k < folds->KFolds(); ++k) {
      new_trees[k].resize(1);
      auto tree = std::make_unique<RegTree>(folds->LeafLength(k), folds->NumFeatures(k), true);
      tree_ptrs.push_back(tree.get());
      new_trees[k].front().push_back(std::move(tree));
    }

    for (auto* tree : tree_ptrs) {
      tree->GetMultiTargetTree()->SetLeaves();
      hist_param_.CheckTreesSynchronized(ctx_, tree);
    }

    folds->CommitModel(std::move(new_trees));
  }
};
}  // namespace xgboost::cv

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvFoldModelsGetGradient(FoldModelsHandle c_cv_folds, DMatrixHandle dtrain,
                                       FoldInfoBatchesHandle c_fold_info,
                                       FoldPredictionsHandle c_predt, FoldGpairsHandle hdl,
                                       int iter) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(c_predt);
  xgboost_CHECK_C_ARG_PTR(hdl);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto cv_folds = static_cast<cv::FoldModels*>(c_cv_folds);
  auto fold_info = static_cast<cv::FoldInfoBatches*>(c_fold_info);
  auto predt = static_cast<cv::FoldPredictions*>(c_predt);
  auto const& info = p_fmat->Info();
  auto const& batch_ptr = p_fmat->BatchPtr();
  CHECK(!fold_info->batches.empty());
  CHECK_EQ(cv_folds->KFolds(), fold_info->KFolds());

  auto fold_gpairs = static_cast<cv::FoldGpairs*>(hdl);
  cv_folds->GetGradient(p_fmat->Ctx(), info, *predt, *fold_info, batch_ptr, iter, fold_gpairs);

  API_END();
}

XGB_DLL int XGBCvTreeMethodCreate(FoldModelsHandle c_cv_folds, char const* c_config,
                                  TreeMethodHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_config);
  xgboost_CHECK_C_ARG_PTR(out);
  auto cv_folds = static_cast<cv::FoldModels*>(c_cv_folds);
  Json config{Json::Load(StringView{c_config})};
  auto args = cv::JsonToArgs(config);
  auto ptr = std::make_unique<cv::TreeMethod>(cv_folds->Ctx());
  ptr->Configure(std::move(args));
  *out = ptr.release();
  API_END();
}

XGB_DLL int XGBCvTreeMethodFree(TreeMethodHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<cv::TreeMethod*>(hdl);
  API_END();
}

XGB_DLL int XGBCvTreeMethodUpdate(TreeMethodHandle hdl, FoldModelsHandle c_cv_folds,
                                  DMatrixHandle dtrain, FoldInfoBatchesHandle c_fold_info,
                                  FoldGpairsHandle c_gpairs) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(c_gpairs);
  auto tree_method = static_cast<cv::TreeMethod*>(hdl);
  auto cv_folds = static_cast<cv::FoldModels*>(c_cv_folds);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto fold_info = static_cast<cv::FoldInfoBatches*>(c_fold_info);
  auto gpairs = static_cast<cv::FoldGpairs*>(c_gpairs);
  tree_method->Update(cv_folds, p_fmat.get(), *fold_info, *gpairs);
  API_END();
}
