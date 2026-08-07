/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>  // for size_t

#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::cv {
/**
 * @brief k-fold split of the row range `[begin, end)`.
 *
 * @param out The training rows of the k^th fold, as indices into the full dataset.
 */
void KFold(Context const* ctx, ::size_t k_folds, bst_idx_t begin, bst_idx_t end, std::int32_t k,
           HostDeviceVector<bst_idx_t>* out);
}  // namespace xgboost::cv
