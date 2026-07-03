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
void KFold(Context const* ctx, std::size_t k_folds, MetaInfo const& info, std::int32_t k,
           HostDeviceVector<bst_idx_t>* out) {
  CHECK(ctx->IsCUDA());
  CHECK(out);
  CHECK_GT(k_folds, 0);
  CHECK_GE(k, 0);
  CHECK_LT(static_cast<std::size_t>(k), k_folds);

  auto const fold = static_cast<std::size_t>(k);
  auto const n_rows = info.num_row_;
  auto const n_per_fold = n_rows / k_folds;
  auto const remainder = n_rows % k_folds;
  auto const valid_size = n_per_fold + static_cast<bst_idx_t>(fold < remainder);
  auto const valid_begin =
      fold * n_per_fold + static_cast<bst_idx_t>(fold < remainder ? fold : remainder);
  auto const n_train = n_rows - valid_size;

  *out = HostDeviceVector<bst_idx_t>(n_train, bst_idx_t{0}, ctx->Device());
  if (n_train == 0) {
    return;
  }

  auto d_out = out->DeviceSpan();
  dh::LaunchN(n_train, ctx->CUDACtx()->Stream(), [=] XGBOOST_DEVICE(std::size_t i) {
    auto ridx = static_cast<bst_idx_t>(i);
    d_out[i] = ridx < valid_begin ? ridx : ridx + valid_size;
  });
}
}  // namespace xgboost::cv
