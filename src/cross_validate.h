/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "xgboost/base.h"                // for GradientPair
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix
#include "xgboost/logging.h"

namespace xgboost {
struct FoldInfo {
  std::vector<HostDeviceVector<std::size_t>> ridxs;

 public:
  [[nodiscard]] auto TrainingFold(std::size_t k) const { return ridxs.at(k).ConstDeviceSpan(); }
  [[nodiscard]] auto KFolds() const noexcept(true) { return this->ridxs.size(); }
};

struct FoldInfoBatches {
  std::vector<FoldInfo> batches;

  [[nodiscard]] std::size_t Size() const { return batches.size(); }
  [[nodiscard]] bool Empty() const { return batches.empty(); }
  [[nodiscard]] auto KFolds() const noexcept(true) {
    CHECK(!this->Empty());
    return this->batches.front().KFolds();
  }
};

struct FoldGpairs {
  std::vector<linalg::Matrix<GradientPair>> gpairs;
};
}  // namespace xgboost

using FoldInfoBatchesHandle = void*;
using FoldGpairsHandle = void*;
