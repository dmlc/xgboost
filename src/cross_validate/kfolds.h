/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/span.h"

namespace xgboost::cv {
using FoldId = std::uint32_t;

struct RowRange {
  bst_idx_t begin, end;
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t Size() const { return end - begin; }
};

// A borrowed GPU view. Queries take global row IDs, even for a nonzero range.
struct MembershipView {
  common::Span<FoldId const> ids;
  bst_idx_t base_rowid;

  // Select global row IDs directly into the caller's GPU working buffer.
  template <typename RowIndex>
  void Select(Context const* ctx, FoldId fold, bool validation, common::Span<RowIndex> out) const;

  [[nodiscard]] XGBOOST_DEVICE bool IsTraining(std::size_t fold, bst_idx_t row) const {
    return ids[row - base_rowid] != fold;
  }
  [[nodiscard]] XGBOOST_DEVICE bool IsValidation(std::size_t fold, bst_idx_t row) const {
    return ids[row - base_rowid] == fold;
  }
};

// An owning, immutable assignment in global row order, independent of feature storage.
class FoldAssignment {
  // Constructed on CUDA and exposed only as a device span; no host mirror is read.
  HostDeviceVector<FoldId> ids_;
  std::vector<bst_idx_t> counts_;

 public:
  FoldAssignment(Context const* ctx, std::size_t k_folds, common::Span<std::int64_t const> ids);
  [[nodiscard]] auto Ids() const { return ids_.ConstDeviceSpan(); }
  [[nodiscard]] auto Device() const { return ids_.Device(); }
  [[nodiscard]] auto const& Counts() const { return counts_; }
  [[nodiscard]] auto KFolds() const { return counts_.size(); }
  [[nodiscard]] auto Size() const { return ids_.Size(); }
  [[nodiscard]] auto TrainFoldSize(std::size_t k) const { return this->Size() - counts_.at(k); }
  [[nodiscard]] auto ValidFoldSize(std::size_t k) const { return counts_.at(k); }
  // Count on the GPU; return only O(K) sizes needed by host-side buffer allocation.
  // The caller owns any physical page layout.
  [[nodiscard]] std::vector<bst_idx_t> CountValidation(Context const* ctx, RowRange rows) const;
  // Borrow a device subspan without copying. The shared assignment owns its lifetime.
  [[nodiscard]] MembershipView ReadRows(RowRange rows) const;
};

using FoldAssignmentPtr = std::shared_ptr<FoldAssignment const>;
}  // namespace xgboost::cv

using FoldAssignmentHandle = void*;
