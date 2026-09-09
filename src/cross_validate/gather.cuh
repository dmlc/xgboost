/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>  // for size_t

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for LaunchN
#include "xgboost/base.h"                // for bst_idx_t
#include "xgboost/context.h"             // for Context
#include "xgboost/span.h"                // for Span

namespace xgboost::cv {
/**
 * @brief Copy the rows listed in `ridx` out of `src` into `dst` at `row_offset`.
 *
 * Row-major with `n_columns` columns, `src` indexed by global row. An empty `ridx` is a
 * no-op: a batch can hold no row of a fold at all.
 */
template <typename T>
void GatherRows(Context const* ctx, common::Span<T const> src, common::Span<bst_idx_t const> ridx,
                bst_idx_t row_offset, std::size_t n_columns, common::Span<T> dst) {
  if (ridx.empty()) {
    return;
  }
  auto n = ridx.size() * n_columns;
  CHECK_LE((row_offset + ridx.size()) * n_columns, dst.size());
  auto d_dst = dst.subspan(row_offset * n_columns, n);
  dh::LaunchN(n, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    auto r = ridx[i / n_columns];
    d_dst[i] = src[r * n_columns + (i % n_columns)];
  });
}
}  // namespace xgboost::cv
