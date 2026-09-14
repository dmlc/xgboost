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
  // NVCC 12.9 can crash when copying a host/device lambda from this member template.
  common::CopyIf(
      ctx->CUDACtx(), first, first + ids.size(), out.data(),
      [=] __device__(bst_idx_t row) { return view.IsValidation(fold, row) == validation; });
}
}  // namespace xgboost::cv
