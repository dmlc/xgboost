/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <limits>   // for numeric_limits
#include <memory>   // for make_shared, make_unique, unique_ptr
#include <sstream>  // for ostringstream
#include <utility>  // for move

#include "../c_api/c_api_error.h"
#include "../c_api/c_api_utils.h"        // for CastDMatrixHandle
#include "../common/cuda_context.cuh"    // for CUDAContext
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

void CalcRootSumFolds(Context const* ctx,
                      std::vector<linalg::MatrixView<GradientPairInt64>> const& d_gpair,
                      std::vector<common::Span<GradientPairInt64>> const& root_sum) {
  CHECK_EQ(d_gpair.size(), root_sum.size());
  for (std::size_t k = 0; k < d_gpair.size(); ++k) {
    tree::cuda_impl::CalcRootSum(ctx, d_gpair[k], root_sum[k]);
  }
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

class FoldTreeMethod {
  std::shared_ptr<DMatrix> p_fmat_;
  Context const* ctx_{nullptr};
  tree::TrainParam param_;
  tree::HistMakerTrainParam hist_param_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  std::shared_ptr<common::HistogramCuts const> cuts_;
  bool initialized_{false};
  std::vector<std::unique_ptr<tree::DeviceHistogramBuilder>> histogram_;
  std::vector<std::unique_ptr<tree::GradientQuantiserGroup>> quantizers_;
  std::vector<linalg::Matrix<GradientPairInt64>> quantized_gpairs_;

 public:
  explicit FoldTreeMethod(std::shared_ptr<DMatrix> p_fmat)
      : p_fmat_{std::move(p_fmat)}, column_sampler_{std::make_shared<common::ColumnSampler>()} {
    CHECK(p_fmat_);
    ctx_ = p_fmat_->Ctx();
    CHECK(ctx_);
  }

  void Configure(Args const& args) {
    CHECK(ctx_->IsCUDA()) << "CV tree method `hist` requires a CUDA device.";

    auto unknown = param_.UpdateAllowUnknown(args);
    unknown = hist_param_.UpdateAllowUnknown(unknown);
    CheckNoUnknownParams(unknown);
  }

  void InitDataOnce() {
    CHECK(ctx_->IsCUDA()) << "CV tree method `hist` requires a CUDA device.";
    auto* p_fmat = p_fmat_.get();
    CHECK(p_fmat);
    p_fmat->Info().feature_types.SetDevice(ctx_->Device());

    auto batch = tree::cuda_impl::HistBatch(param_);
    auto [cuts, dense_compressed] = tree::InitBatchCuts(ctx_, p_fmat, batch);
    auto batch_ptr = p_fmat->BatchPtr();
    this->cuts_ = std::move(cuts);

    initialized_ = true;
  }

  void Reset(Context const* ctx, FoldInfoBatches const& finfo, FoldGpairs const& gpairs) {
    CHECK(!collective::IsDistributed())
        << "Distributed training is not supported by the CV tree method.";
    CHECK(!finfo.Empty());
    CHECK_EQ(finfo.KFolds(), gpairs.KFolds());
    CHECK(cuts_);

    auto k_folds = finfo.KFolds();
    if (this->histogram_.empty()) {
      this->histogram_.resize(k_folds);
    }
    if (this->quantizers_.empty()) {
      this->quantizers_.resize(k_folds);
    }
    if (this->quantized_gpairs_.empty()) {
      this->quantized_gpairs_.resize(k_folds);
    }
    CHECK_EQ(this->histogram_.size(), k_folds);
    CHECK_EQ(this->quantizers_.size(), k_folds);
    CHECK_EQ(this->quantized_gpairs_.size(), k_folds);

    bst_target_t n_split_targets{0};
    for (std::size_t k = 0; k < k_folds; ++k) {
      auto const& fold_gpair = gpairs.gpairs.at(k);
      CHECK_EQ(finfo.FoldSize(k), fold_gpair.Shape(0));
      CHECK_GT(fold_gpair.Shape(0), 0) << "Empty training folds are not supported.";
      CHECK_GT(fold_gpair.Shape(1), 0);

      auto in_gpair = fold_gpair.View(ctx->Device());
      CHECK(in_gpair.CContiguous());
      if (k == 0) {
        n_split_targets = in_gpair.Shape(1);
      }
      CHECK_EQ(n_split_targets, in_gpair.Shape(1));

      this->quantizers_[k] = std::make_unique<tree::GradientQuantiserGroup>(ctx, in_gpair);
      tree::CalcQuantizedGpairs(ctx, in_gpair, this->quantizers_[k]->DeviceSpan(),
                                &this->quantized_gpairs_[k]);

      auto n_total_bins = static_cast<bst_idx_t>(this->cuts_->TotalBins()) * n_split_targets;
      CHECK_LT(n_total_bins, std::numeric_limits<bst_bin_t>::max())
          << "Too many histogram bins: n_total_bins = total_bins * n_targets";
      bool force_global = false;
      if (!this->histogram_[k]) {
        this->histogram_[k] = std::make_unique<tree::DeviceHistogramBuilder>();
      }
      this->histogram_[k]->Reset(ctx, this->hist_param_.MaxCachedHistNodes(ctx->Device()),
                                 n_total_bins, force_global);
    }
  }

  void InitRoot(std::vector<RegTree*> const& trees) {
    auto k_folds = trees.size();
    CHECK_GT(k_folds, 0);
    CHECK_EQ(this->quantizers_.size(), k_folds);
    CHECK_EQ(this->quantized_gpairs_.size(), k_folds);

    auto n_targets = this->quantized_gpairs_.front().Shape(1);
    auto root_sums = linalg::Constant(ctx_, GradientPairInt64{}, k_folds, n_targets);
    auto d_root_sums = root_sums.View(ctx_->Device());

    std::vector<linalg::MatrixView<GradientPairInt64>> d_gpairs;
    std::vector<common::Span<GradientPairInt64>> d_root_sum_spans;
    d_gpairs.reserve(k_folds);
    d_root_sum_spans.reserve(k_folds);
    for (std::size_t k = 0; k < k_folds; ++k) {
      CHECK(trees.at(k));
      auto d_gpair = this->quantized_gpairs_.at(k).View(ctx_->Device());
      CHECK_EQ(d_gpair.Shape(1), n_targets);
      CHECK_EQ(trees[k]->NumTargets(), n_targets);
      d_gpairs.emplace_back(d_gpair);
      d_root_sum_spans.emplace_back(d_root_sums.Values().subspan(k * n_targets, n_targets));
    }
    CalcRootSumFolds(ctx_, d_gpairs, d_root_sum_spans);

    std::vector<common::Span<tree::GradientQuantiser const>> h_quantizers;
    h_quantizers.reserve(k_folds);
    for (std::size_t k = 0; k < k_folds; ++k) {
      CHECK_EQ(this->quantizers_[k]->Size(), n_targets);
      h_quantizers.emplace_back(this->quantizers_[k]->DeviceSpan());
    }
    dh::device_vector<common::Span<tree::GradientQuantiser const>> d_quantizers{h_quantizers};
    auto quantizers = dh::ToSpan(d_quantizers);

    auto root_weights = linalg::Empty<float>(ctx_, k_folds, n_targets);
    auto d_root_weights = root_weights.View(ctx_->Device());
    auto root_sum_hess = linalg::Constant(ctx_, 0.0f, k_folds);
    auto d_root_sum_hess = root_sum_hess.View(ctx_->Device());
    tree::EvalParam param{this->param_};
    auto eta = this->param_.learning_rate;
    dh::LaunchN(root_sums.Size(), ctx_->CUDACtx()->Stream(), [=] __device__(std::size_t i) mutable {
      auto k = i / n_targets;
      auto t = i % n_targets;
      auto sum = quantizers[k][t].ToFloatingPoint(d_root_sums(k, t));
      d_root_weights(k, t) = tree::CalcWeight(param, sum) * eta;
      atomicAdd(&d_root_sum_hess(k), static_cast<float>(sum.GetHess()));
    });

    auto h_root_sum_hess = root_sum_hess.HostView();
    for (std::size_t k = 0; k < k_folds; ++k) {
      trees[k]->SetRoot(d_root_weights.Slice(k, linalg::All()), h_root_sum_hess(k));
    }
  }

  void Update(FoldModels* folds, DMatrix* p_fmat, FoldInfoBatches const& finfo,
              FoldGpairs const& gpairs) {
    CHECK(folds);
    CHECK(p_fmat);
    CHECK_EQ(p_fmat, p_fmat_.get())
        << "CV tree method update must use the training DMatrix supplied at construction.";
    CHECK_EQ(folds->KFolds(), finfo.KFolds());
    CHECK_EQ(folds->KFolds(), gpairs.KFolds());

    if (!initialized_) {
      this->InitDataOnce();
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

    this->Reset(ctx_, finfo, gpairs);
    this->InitRoot(tree_ptrs);

    for (std::size_t k = 0, k_folds = folds->KFolds(); k < k_folds; ++k) {
      auto* tree = tree_ptrs.at(k);
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

XGB_DLL int XGBCvFoldTreeMethodCreate(FoldModelsHandle c_cv_folds, DMatrixHandle dtrain,
                                      char const* c_config, TreeMethodHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(dtrain);
  xgboost_CHECK_C_ARG_PTR(c_config);
  xgboost_CHECK_C_ARG_PTR(out);
  auto p_fmat = CastDMatrixHandle(dtrain);
  Json config{Json::Load(StringView{c_config})};
  auto args = cv::JsonToArgs(config);
  auto ptr = std::make_unique<cv::FoldTreeMethod>(std::move(p_fmat));
  ptr->Configure(std::move(args));
  *out = ptr.release();
  API_END();
}

XGB_DLL int XGBCvFoldTreeMethodFree(TreeMethodHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<cv::FoldTreeMethod*>(hdl);
  API_END();
}

XGB_DLL int XGBCvFoldTreeMethodUpdate(TreeMethodHandle hdl, FoldModelsHandle c_cv_folds,
                                      DMatrixHandle dtrain, FoldInfoBatchesHandle c_fold_info,
                                      FoldGpairsHandle c_gpairs) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(c_gpairs);
  auto tree_method = static_cast<cv::FoldTreeMethod*>(hdl);
  auto cv_folds = static_cast<cv::FoldModels*>(c_cv_folds);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto fold_info = static_cast<cv::FoldInfoBatches*>(c_fold_info);
  auto gpairs = static_cast<cv::FoldGpairs*>(c_gpairs);
  tree_method->Update(cv_folds, p_fmat.get(), *fold_info, *gpairs);
  API_END();
}
