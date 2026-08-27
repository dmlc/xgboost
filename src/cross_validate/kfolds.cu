/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstddef>  // for size_t

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for LaunchN
#include "kfolds.h"
#include "xgboost/context.h"  // for Context
#include "xgboost/logging.h"  // for CHECK

namespace xgboost::cv {
void KFold(Context const* ctx, std::size_t k_folds, bst_idx_t begin, bst_idx_t end, std::int32_t k,
           FoldInfo* out) {
  CHECK(ctx->IsCUDA());
  CHECK_LT(static_cast<std::size_t>(k), k_folds);

  auto const fold = static_cast<std::size_t>(k);
  CHECK_GE(end, begin);
  auto const n_rows = end - begin;
  auto const n_per_fold = n_rows / k_folds;
  auto const remainder = n_rows % k_folds;
  auto const n_valid = n_per_fold + static_cast<bst_idx_t>(fold < remainder);
  auto const valid_begin =
      fold * n_per_fold + static_cast<bst_idx_t>(fold < remainder ? fold : remainder);
  auto const n_train = n_rows - n_valid;

  out->ridxs.emplace_back();
  auto& tr_idx = out->ridxs.back();
  tr_idx.SetDevice(ctx->Device());
  tr_idx.Resize(n_train);

  out->valid_ridxs.emplace_back();
  auto& valid_idx = out->valid_ridxs.back();
  valid_idx.SetDevice(ctx->Device());
  valid_idx.Resize(n_valid);

  auto d_tr_idx = tr_idx.DeviceSpan();
  dh::LaunchN(n_train, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    auto ridx = static_cast<bst_idx_t>(i);
    // Before and after the validation window, shifted into the global index space.
    d_tr_idx[i] = begin + (ridx < valid_begin ? ridx : ridx + n_valid);
  });

  auto d_valid_idx = valid_idx.DeviceSpan();
  dh::LaunchN(n_valid, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    auto ridx = static_cast<bst_idx_t>(i);
    d_valid_idx[i] = begin + valid_begin + ridx;
  });
}
}  // namespace xgboost::cv
