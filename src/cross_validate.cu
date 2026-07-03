/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "./c_api/c_api_error.h"
#include "./c_api/c_api_utils.h"  // for CastDMatrixHandle
#include "./data/ellpack_page.h"
#include "./data/extmem_quantile_dmatrix.h"  // for ExtMemQuantileDMatrix
#include "cross_validate.h"

namespace xgboost {}  // namespace xgboost

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvFoldInfoCreate(DMatrixHandle dtrain, FoldInfoHandle* out) {
  API_BEGIN();
  auto p_fmat = CastDMatrixHandle(dtrain);
  CHECK(std::dynamic_pointer_cast<std::shared_ptr<data::ExtMemQuantileDMatrix>>(p_fmat));
  for (auto const& page :
       p_fmat->GetBatches<EllpackPage>(p_fmat->Ctx(), cuda_impl::StaticBatch(true))) {
  }
  API_END();
}

XGB_DLL int XGBCvFoldInfoFree(FoldInfoHandle hdl) {
  API_BEGIN();
  delete static_cast<FoldInfo*>(hdl);
  API_END();
}
