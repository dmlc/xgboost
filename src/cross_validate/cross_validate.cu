/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../c_api/c_api_error.h"
#include "../c_api/c_api_utils.h"      // for CastDMatrixHandle
#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../common/linalg_op.cuh"     // for tcbegin, tcend, tbegin
#include "cross_validate.h"

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
}  // namespace

void GetGradient(Context const* ctx, MetaInfo const& info, FoldModels const& cv_folds,
                 FoldInfoBatches const& finfo, std::vector<bst_idx_t> const& batch_ptr,
                 std::int32_t iter, std::vector<linalg::Matrix<GradientPair>>* p_gpairs) {
  CHECK(!finfo.Empty());
  CHECK_EQ(batch_ptr.size(), finfo.Size() + 1);

  auto k_folds = finfo.KFolds();
  CHECK_EQ(cv_folds.KFolds(), k_folds);

  auto& gpairs = *p_gpairs;
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

      auto const& fold_preds = cv_folds.Prediction(k);
      auto output_length = cv_folds.OutputLength(k);
      CHECK_EQ(fold_info.labels.Shape(1), output_length);
      auto n_predts = ridxs.Size() * output_length;
      CHECK_EQ(fold_info.labels.Size(), n_predts);
      auto pred_begin = cursors[k] * output_length;
      CHECK_LE(pred_begin + n_predts, fold_preds.Size());
      auto preds = BatchPrediction(ctx, fold_preds, pred_begin, n_predts);

      linalg::Matrix<GradientPair> batch_gpair;
      cv_folds.Objective(k)->GetGradient(preds, fold_info, iter, &batch_gpair);

      auto prev = cursors[k];
      cursors[k] += ridxs.Size();
      CopyBatchGpair(ctx, batch_gpair, prev, cursors[k], &gpairs.at(k));
    }
  }

  for (std::size_t k = 0; k < k_folds; ++k) {
    CHECK_EQ(finfo.FoldSize(k), p_gpairs->at(k).Shape(0));
  }
}
}  // namespace xgboost::cv

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvGetGradient(FoldsHandle c_cv_folds, DMatrixHandle dtrain,
                             FoldInfoBatchesHandle c_fold_info, FoldGpairsHandle hdl, int iter) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(hdl);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto cv_folds = static_cast<cv::FoldModels*>(c_cv_folds);
  auto fold_info = static_cast<cv::FoldInfoBatches*>(c_fold_info);
  auto const& info = p_fmat->Info();
  auto const& batch_ptr = p_fmat->BatchPtr();
  CHECK(!fold_info->batches.empty());
  CHECK_EQ(cv_folds->KFolds(), fold_info->KFolds());
  cv_folds->InitPrediction(p_fmat->Ctx(), info, *fold_info);

  auto fold_gpairs = static_cast<cv::FoldGpairs*>(hdl);
  cv::GetGradient(p_fmat->Ctx(), info, *cv_folds, *fold_info, batch_ptr, iter,
                  &fold_gpairs->gpairs);

  API_END();
}
