/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <memory>  // for make_unique

#include "./c_api/c_api_error.h"
#include "./c_api/c_api_utils.h"             // for CastDMatrixHandle
#include "./data/extmem_quantile_dmatrix.h"  // for ExtMemQuantileDMatrix
#include "common/cuda_context.cuh"           // for CUDAContext
#include "common/linalg_op.cuh"              // for tcbegin, tcend, tbegin
#include "cross_validate.h"
#include "cross_validate/kfolds.h"
#include "xgboost/objective.h"

namespace xgboost {
namespace {
[[nodiscard]] HostDeviceVector<bst_idx_t> GlobalTrainingRows(Context const* ctx,
                                                             FoldInfo const& batch,
                                                             std::size_t fold,
                                                             bst_idx_t batch_begin) {
  auto const& h_local = batch.ridxs.at(fold).ConstHostVector();
  std::vector<bst_idx_t> h_global(h_local.size());
  for (std::size_t i = 0; i < h_local.size(); ++i) {
    h_global[i] = batch_begin + h_local[i];
  }
  return HostDeviceVector<bst_idx_t>{h_global, ctx->Device()};
}
}  // namespace

void GetGradient(Context const* ctx, MetaInfo const& info, FoldInfoBatches const& finfo,
                 std::vector<bst_idx_t> const& batch_ptr, std::int32_t iter,
                 std::vector<linalg::Matrix<GradientPair>>* p_gpairs) {
  CHECK(!finfo.Empty());
  CHECK_EQ(batch_ptr.size(), finfo.Size() + 1);

  std::string obj_name = "reg:squarederror";  // fixme
  std::vector<std::unique_ptr<ObjFunction>> objs;

  auto k_folds = finfo.KFolds();
  // fixme
  for (std::size_t k = 0; k < k_folds; ++k) {
    objs.emplace_back(ObjFunction::Create(obj_name, ctx));
    objs.back()->Configure(Args{});
  }

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
      auto fold_info =
          info.Slice(ctx, ctx->IsCUDA() ? ridxs.ConstDeviceSpan() : ridxs.ConstHostSpan(), kNnz);

      HostDeviceVector<float> preds(ridxs.Size(), 0.0f, ctx->Device());

      linalg::Matrix<GradientPair> batch_gpair;
      objs.at(k)->GetGradient(preds, fold_info, iter, &batch_gpair);

      auto& out_gpairs = gpairs.at(k);
      auto prev = cursors[k];
      cursors[k] += ridxs.Size();
      CHECK_EQ(ridxs.Size(), batch_gpair.Shape(0));
      CHECK(batch_gpair.Shape(1) == out_gpairs.Shape(1) || out_gpairs.Shape(1) <= 1);

      if (out_gpairs.Shape(0) < cursors[k]) {
        out_gpairs.Reshape(cursors[k], batch_gpair.Shape(1));
      }
      auto d_batch_gpair = batch_gpair.View(ctx->Device());
      auto d_out = out_gpairs.Slice(linalg::Range(prev, cursors[k]), linalg::All());
      thrust::copy(ctx->CUDACtx()->CTP(), linalg::tcbegin(d_batch_gpair),
                   linalg::tcend(d_batch_gpair), linalg::tbegin(d_out));
    }
  }

  for (std::size_t k = 0; k < k_folds; ++k) {
    CHECK_EQ(finfo.FoldSize(k), p_gpairs->at(k).Size());
  }
}
}  // namespace xgboost

using namespace xgboost;  // NOLINT

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

XGB_DLL int XGBCvGetGradient(DMatrixHandle dtrain, FoldInfoBatchesHandle c_fold_info,
                             FoldGpairsHandle hdl, int iter) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_fold_info);
  xgboost_CHECK_C_ARG_PTR(hdl);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto fold_info = static_cast<FoldInfoBatches*>(c_fold_info);
  auto const& info = p_fmat->Info();
  auto const& batch_ptr = p_fmat->BatchPtr();
  CHECK(!fold_info->batches.empty());

  auto fold_gpairs = static_cast<FoldGpairs*>(hdl);
  GetGradient(p_fmat->Ctx(), info, *fold_info, batch_ptr, iter, &fold_gpairs->gpairs);

  API_END();
}
