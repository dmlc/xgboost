/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "../gbm/gbtree.h"  // for PredictionCacheEntry
#include "../gbm/gbtree_model.h"
#include "../learner_model_param_legacy.h"
#include "kfolds.h"                      // for FoldInfo
#include "xgboost/base.h"                // for GradientPair
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix
#include "xgboost/logging.h"
#include "xgboost/objective.h"
#include "xgboost/predictor.h"  // for PredictionCacheEntry

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
  [[nodiscard]] std::int32_t BoostedRounds() const;
  [[nodiscard]] bst_target_t OutputLength(std::size_t fold_idx) const;
  [[nodiscard]] bst_target_t LeafLength(std::size_t fold_idx) const;
  [[nodiscard]] bst_feature_t NumFeatures(std::size_t fold_idx) const;
  [[nodiscard]] ObjFunction* Objective(std::size_t fold_idx) const;
  void InitPrediction(Context const* ctx, MetaInfo const& info, FoldInfoBatches const& finfo,
                      FoldPredictions* out) const;
  void GetGradient(Context const* ctx, MetaInfo const& info, FoldPredictions const& predts,
                   FoldInfoBatches const& finfo, std::int32_t iter, FoldGpairs* out) const;

  void CommitModel(std::vector<gbm::TreesOneIter>&& new_trees);

  [[nodiscard]] static FoldModels LoadModel(Json const& in);
  void SaveModel(Json* out) const;
};

struct FoldInfoBatches {
  std::vector<FoldInfo> batches;

  [[nodiscard]] std::size_t Size() const { return batches.size(); }
  // Number of training rows in the k^th fold.
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

// Prediction caches for all folds, both indexed by the global row index. `train` holds one
// cache per fold, in which the rows held out by that fold are unused padding. `valid` is a
// single cache holding the out-of-fold prediction of every row.
struct FoldPredictions {
  std::vector<gbm::PredictionCacheEntry> train;
  gbm::PredictionCacheEntry valid;
  // Number of columns in each cache, shared by all folds. A `PredictionCacheEntry` is flat,
  // so this is what makes the buffers self-describing.
  bst_target_t output_length{0};

  [[nodiscard]] auto KFolds() const noexcept(true) { return this->train.size(); }
  [[nodiscard]] gbm::PredictionCacheEntry& Training(std::size_t fold_idx) {
    return train.at(fold_idx);
  }
  [[nodiscard]] gbm::PredictionCacheEntry const& Training(std::size_t fold_idx) const {
    return train.at(fold_idx);
  }
  [[nodiscard]] gbm::PredictionCacheEntry& Validation() { return valid; }
  [[nodiscard]] gbm::PredictionCacheEntry const& Validation() const { return valid; }
  [[nodiscard]] HostDeviceVector<float> const& Prediction(std::size_t fold_idx) const {
    return this->Training(fold_idx).predictions;
  }
};

// Gradient of each fold, indexed by the global row index. The rows held out by a fold are
// zeroed rather than left as padding: this buffer is consumed whole, so a stale value would
// leak into the fold's root sum and histograms.
struct FoldGpairs {
  std::vector<linalg::Matrix<GradientPair>> gpairs;

  [[nodiscard]] auto KFolds() const noexcept(true) { return this->gpairs.size(); }
};
}  // namespace xgboost::cv

using FoldModelsHandle = void*;
using FoldInfoBatchesHandle = void*;
using FoldPredictionsHandle = void*;
using FoldGpairsHandle = void*;
using TreeMethodHandle = void*;
