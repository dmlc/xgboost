/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <memory>  // for make_unique

#include "./c_api/c_api_error.h"
#include "./c_api/c_api_utils.h"             // for CastDMatrixHandle
#include "./data/extmem_quantile_dmatrix.h"  // for ExtMemQuantileDMatrix
#include "cross_validate.h"
#include "cross_validate/kfolds.h"

namespace xgboost {}  // namespace xgboost

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
