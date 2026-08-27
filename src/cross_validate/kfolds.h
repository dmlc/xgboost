/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>  // for size_t

#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::cv {
// Training and validation rows of each fold within a single data batch. All row indices
// are global, namely indices into the full dataset instead of the batch that owns them.
struct FoldInfo {
  std::vector<HostDeviceVector<bst_idx_t>> ridxs;
  std::vector<HostDeviceVector<bst_idx_t>> valid_ridxs;

 public:
  [[nodiscard]] auto TrainingFold(std::size_t k) const { return ridxs.at(k).ConstDeviceSpan(); }
  [[nodiscard]] auto ValidationFold(std::size_t k) const {
    return valid_ridxs.at(k).ConstDeviceSpan();
  }
  [[nodiscard]] auto KFolds() const noexcept(true) { return this->ridxs.size(); }
};

/**
 * @brief k-fold split of the row range `[begin, end)`.
 *
 * @param out The training and validation rows of the k^th fold, as indices into the full
 *            dataset.
 */
void KFold(Context const* ctx, ::size_t k_folds, bst_idx_t begin, bst_idx_t end, std::int32_t k,
           FoldInfo* out);
}  // namespace xgboost::cv
