/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "../gbm/gbtree_model.h"
#include "../learner_model_param_legacy.h"
#include "xgboost/base.h"                // for GradientPair
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix
#include "xgboost/logging.h"
#include "xgboost/objective.h"

namespace xgboost::cv {
struct FoldInfoBatches;
struct FoldPredictions;
struct FoldGpairs;

// The model part of the cross validation result, containing the trees and objectives.
//
// Tree updaters should not be part of it as they are considered "optimizers" and not part
// of the model.
class FoldModels {
  Context ctx_;  // FIXME(jiamingy): Remove ctx reference from obj.
  std::vector<LearnerModelParamLegacy> model_params_;
  std::vector<LearnerModelParam> properties_;
  std::vector<std::unique_ptr<ObjFunction>> objs_;
  std::vector<std::unique_ptr<gbm::GBTreeModel>> models_;

  void Resize(std::size_t k_folds);
  void InitFold(std::size_t fold_idx, std::unique_ptr<ObjFunction> obj);
  FoldModels() = default;

 public:
  explicit FoldModels(std::size_t k_folds, std::shared_ptr<DMatrix> dtrain);
  [[nodiscard]] std::size_t KFolds() const noexcept(true);
  [[nodiscard]] bst_target_t OutputLength(std::size_t fold_idx) const;
  [[nodiscard]] ObjFunction* Objective(std::size_t fold_idx) const;
  void InitPrediction(Context const* ctx, MetaInfo const& info, FoldInfoBatches const& finfo,
                      FoldPredictions* out) const;
  void GetGradient(Context const* ctx, MetaInfo const& info, FoldPredictions const& predts,
                   FoldInfoBatches const& finfo, std::vector<bst_idx_t> const& batch_ptr,
                   std::int32_t iter, FoldGpairs* out) const;

  void CommitModel(std::vector<gbm::TreesOneIter>&& new_trees);

  [[nodiscard]] static FoldModels LoadModel(Json const& in);
  void SaveModel(Json* out) const;
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
      acc += batch.ridxs.at(k).Size();
    }
    return acc;
  }
  [[nodiscard]] bool Empty() const { return batches.empty(); }
  [[nodiscard]] auto KFolds() const noexcept(true) {
    CHECK(!this->Empty());
    return this->batches.front().KFolds();
  }
};

struct FoldPredictions {
  std::vector<HostDeviceVector<float>> predts;

  [[nodiscard]] auto KFolds() const noexcept(true) { return this->predts.size(); }
  [[nodiscard]] HostDeviceVector<float> const& Prediction(std::size_t fold_idx) const {
    return predts.at(fold_idx);
  }
};

struct FoldGpairs {
  std::vector<linalg::Matrix<GradientPair>> gpairs;

  [[nodiscard]] auto KFolds() const noexcept(true) { return this->gpairs.size(); }
};
}  // namespace xgboost::cv

using FoldsHandle = void*;
using FoldInfoBatchesHandle = void*;
using FoldPredictionsHandle = void*;
using FoldGpairsHandle = void*;
