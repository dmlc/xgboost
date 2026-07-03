/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <memory>  // for make_unique

#include "./c_api/c_api_error.h"
#include "./c_api/c_api_utils.h"    // for CastDMatrixHandle
#include "common/cuda_context.cuh"  // for CUDAContext
#include "common/linalg_op.cuh"     // for tcbegin, tcend, tbegin
#include "cross_validate.h"

namespace xgboost {
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
}  // namespace

void GetGradient(Context const* ctx, MetaInfo const& info, CvFolds const& cv_folds,
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

      HostDeviceVector<float> preds(ridxs.Size(), 0.0f, ctx->Device());  // fixme

      linalg::Matrix<GradientPair> batch_gpair;
      cv_folds.Objective(k)->GetGradient(preds, fold_info, iter, &batch_gpair);

      auto& out_gpairs = gpairs.at(k);
      auto prev = cursors[k];
      cursors[k] += ridxs.Size();
      CHECK_EQ(ridxs.Size(), batch_gpair.Shape(0));
      CHECK(batch_gpair.Shape(1) == out_gpairs.Shape(1) || out_gpairs.Shape(1) <= 1);

      if (out_gpairs.Shape(0) < cursors[k]) {
        out_gpairs.Reshape(cursors[k], batch_gpair.Shape(1));
      }
      auto d_batch_gpair = batch_gpair.View(ctx->Device());
      auto d_out =
          out_gpairs.View(ctx->Device()).Slice(linalg::Range(prev, cursors[k]), linalg::All());
      thrust::copy(ctx->CUDACtx()->CTP(), linalg::tcbegin(d_batch_gpair),
                   linalg::tcend(d_batch_gpair), linalg::tbegin(d_out));
    }
  }

  for (std::size_t k = 0; k < k_folds; ++k) {
    CHECK_EQ(finfo.FoldSize(k), p_gpairs->at(k).Shape(0));
  }
}
}  // namespace xgboost

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvGetGradient(DMatrixHandle dtrain, CvFoldsHandle c_cv_folds,
                             FoldInfoBatchesHandle c_fold_info, FoldGpairsHandle hdl, int iter) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_cv_folds);
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(hdl);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto cv_folds = static_cast<CvFolds*>(c_cv_folds);
  auto fold_info = static_cast<FoldInfoBatches*>(c_fold_info);
  auto const& info = p_fmat->Info();
  auto const& batch_ptr = p_fmat->BatchPtr();
  CHECK(!fold_info->batches.empty());
  CHECK_EQ(cv_folds->KFolds(), fold_info->KFolds());

  auto fold_gpairs = static_cast<FoldGpairs*>(hdl);
  GetGradient(p_fmat->Ctx(), info, *cv_folds, *fold_info, batch_ptr, iter, &fold_gpairs->gpairs);

  API_END();
}
