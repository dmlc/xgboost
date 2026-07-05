/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "../gbm/gbtree_model.h"
#include "xgboost/base.h"                // for GradientPair
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix
#include "xgboost/logging.h"
#include "xgboost/objective.h"

namespace xgboost::cv {
struct FoldInfoBatches;

// The model part of the cross validation result, containing the trees and objectives.
//
// Tree updaters should not be part of it as they are considered "optimizers" and not part
// of the model.
class CvFolds : public Model {
  std::vector<std::unique_ptr<ObjFunction>> objs_;
  std::vector<std::unique_ptr<gbm::GBTreeModel>> models_;
  std::vector<HostDeviceVector<float>> predts_;
  Context ctx_;

 public:
  explicit CvFolds(std::size_t k_folds);
  [[nodiscard]] std::size_t KFolds() const noexcept(true);
  [[nodiscard]] Context const* Ctx() const;
  [[nodiscard]] ObjFunction* Objective(std::size_t fold_idx) const;
  void InitPrediction(MetaInfo const& info, FoldInfoBatches const& finfo);
  [[nodiscard]] HostDeviceVector<float> const& Prediction(std::size_t fold_idx) const;

  void CommitModel(std::vector<gbm::TreesOneIter>&& new_trees);

  void LoadModel(Json const& in) final;
  void SaveModel(Json* out) const final;
};

struct FoldInfo {
  std::vector<HostDeviceVector<bst_idx_t>> ridxs;

 public:
  [[nodiscard]] auto TrainingFold(std::size_t k) const { return ridxs.at(k).ConstDeviceSpan(); }
  [[nodiscard]] auto KFolds() const noexcept(true) { return this->ridxs.size(); }
};

struct FoldInfoBatches {
  std::vector<FoldInfo> batches;

  [[nodiscard]] std::size_t Size() const { return batches.size(); }
  [[nodiscard]] std::size_t FoldSize(std::size_t k) const {
    std::size_t acc = 0;
    for (auto const& batch : this->batches) {
      acc += batch.TrainingFold(k).size();
    }
    return acc;
  }
  [[nodiscard]] bool Empty() const { return batches.empty(); }
  [[nodiscard]] auto KFolds() const noexcept(true) {
    CHECK(!this->Empty());
    return this->batches.front().KFolds();
  }
};

struct FoldGpairs {
  std::vector<linalg::Matrix<GradientPair>> gpairs;

  [[nodiscard]] auto KFolds() const noexcept(true) { return this->gpairs.size(); }
};
}  // namespace xgboost::cv

using CvFoldsHandle = void*;
using FoldInfoBatchesHandle = void*;
using FoldGpairsHandle = void*;
