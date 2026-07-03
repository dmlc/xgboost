/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kfolds.h"

namespace xgboost::cv {
void KFold(Context const* ctx, std::size_t k_folds, MetaInfo const& info, std::int32_t k,
           HostDeviceVector<bst_idx_t>* out) {}
}  // namespace xgboost::cv
