/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../common/algorithm.cuh"
#include "kfolds.h"

namespace xgboost::cv {
template <typename RowIndex>
void MembershipView::Select(Context const* ctx, FoldId fold, bool validation,
                            common::Span<RowIndex> out) const {
  if (out.empty()) {
    return;
  }
  auto view = *this;
  auto first = thrust::make_counting_iterator(base_rowid);
  common::CopyIf(
      ctx->CUDACtx(), first, first + ids.size(), out.data(),
      [=] XGBOOST_DEVICE(bst_idx_t row) { return view.IsValidation(fold, row) == validation; });
}
}  // namespace xgboost::cv
