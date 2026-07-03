/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost {
struct FoldInfo {
  std::vector<HostDeviceVector<std::size_t>> ridxs;

 public:
  [[nodiscard]] auto TrainingFold(std::size_t k) const { return ridxs.at(k).ConstDeviceSpan(); }
  [[nodiscard]] auto KFolds() const noexcept(true) { return this->ridxs.size(); }
};
}  // namespace xgboost

using FoldInfoHandle = void*;
