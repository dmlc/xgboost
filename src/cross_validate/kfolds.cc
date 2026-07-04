/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kfolds.h"

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"
#endif  // !defined(XGBOOST_USE_CUDA)

namespace xgboost::cv {
#if !defined(XGBOOST_USE_CUDA)
void KFold(Context const*, std::size_t, bst_idx_t, bst_idx_t, std::int32_t,
           HostDeviceVector<bst_idx_t>*) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::cv
