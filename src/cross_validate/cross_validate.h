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
#include "xgboost/predictor.h"    // for PredictionCacheEntry
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::cv {
struct FoldInfoBatches;
struct FoldPredictions;
struct FoldGpairs;

// The set of models grown together in one fused page loop: one per CV fold, plus an
// optional full-data refit model. The refit model is the last unit, so a unit index below
// `k_folds` is always a fold.
struct UnitLayout {
  std::size_t k_folds{0};
  bool refit{false};

  [[nodiscard]] std::size_t NumUnits() const noexcept(true) {
    return this->k_folds + static_cast<std::size_t>(this->refit);
  }
  [[nodiscard]] std::size_t RefitIdx() const {
    CHECK(this->refit) << "No refit model in this cross-validation run.";
    return this->k_folds;
  }
  [[nodiscard]] bool IsRefit(std::size_t unit_idx) const noexcept(true) {
    return this->refit && unit_idx == this->k_folds;
  }
  friend bool operator==(UnitLayout const& lhs, UnitLayout const& rhs) {
    return lhs.k_folds == rhs.k_folds && lhs.refit == rhs.refit;
  }
};

// All the buffers of one CV run must describe the same set of training units. A mismatch
// means buffers from two runs were mixed, or a step of the round was skipped.
inline void CheckLayout(UnitLayout const& expect, UnitLayout const& got, StringView name) {
  CHECK(expect == got) << "The " << name << " of this CV run describe " << got.NumUnits()
                       << " training units, the models describe " << expect.NumUnits() << ".";
}

// The model part of the cross validation result, containing the trees and objectives.
//
// Tree updaters should not be part of it as they are considered "optimizers" and not part
// of the model.
class FoldModels {
  Context ctx_;  // FIXME(jiamingy): Remove ctx reference from obj.
  UnitLayout layout_;
  // Indexed by unit: `[0, KFolds())` are the folds, `RefitIdx()` is the full-data model.
  std::vector<LearnerModelParamLegacy> model_params_;
  std::vector<LearnerModelParam> properties_;
  std::vector<std::unique_ptr<ObjFunction>> objs_;
  std::vector<std::unique_ptr<gbm::GBTreeModel>> models_;

  void Resize(std::size_t n_units);
  void InitUnit(std::size_t unit_idx, std::unique_ptr<ObjFunction> obj);
  void SaveUnit(std::size_t unit_idx, Json* out) const;
  void LoadUnit(std::size_t unit_idx, Json const& in);
  FoldModels() = default;

 public:
  explicit FoldModels(std::size_t k_folds, std::shared_ptr<DMatrix> dtrain, bool refit);
  [[nodiscard]] UnitLayout const& Layout() const noexcept(true) { return this->layout_; }
  [[nodiscard]] std::size_t KFolds() const noexcept(true) { return this->layout_.k_folds; }
  [[nodiscard]] std::size_t NumUnits() const noexcept(true) { return this->layout_.NumUnits(); }
  [[nodiscard]] bool HasRefit() const noexcept(true) { return this->layout_.refit; }
  [[nodiscard]] std::size_t RefitIdx() const { return this->layout_.RefitIdx(); }
  [[nodiscard]] std::int32_t BoostedRounds() const;
  [[nodiscard]] bst_target_t OutputLength(std::size_t unit_idx) const;
  [[nodiscard]] bst_target_t LeafLength(std::size_t unit_idx) const;
  [[nodiscard]] bst_feature_t NumFeatures(std::size_t unit_idx) const;
  [[nodiscard]] ObjFunction* Objective(std::size_t unit_idx) const;
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

// Prediction caches for all training units, both indexed by the global row index. `train`
// holds one cache per unit, in which the rows held out by that fold are unused padding; the
// refit unit holds nothing out, so its cache has no padding. `valid` is a single cache
// holding the out-of-fold prediction of every row, written by the folds alone.
struct FoldPredictions {
  std::vector<gbm::PredictionCacheEntry> train;
  gbm::PredictionCacheEntry valid;
  // Number of columns in each cache, shared by all units. A `PredictionCacheEntry` is flat,
  // so this is what makes the buffers self-describing.
  bst_target_t output_length{0};
  UnitLayout layout;

  [[nodiscard]] gbm::PredictionCacheEntry& Training(std::size_t unit_idx) {
    return train.at(unit_idx);
  }
  [[nodiscard]] gbm::PredictionCacheEntry const& Training(std::size_t unit_idx) const {
    return train.at(unit_idx);
  }
  [[nodiscard]] gbm::PredictionCacheEntry const& Refit() const {
    return this->Training(layout.RefitIdx());
  }
  [[nodiscard]] gbm::PredictionCacheEntry& Validation() { return valid; }
  [[nodiscard]] gbm::PredictionCacheEntry const& Validation() const { return valid; }
  [[nodiscard]] HostDeviceVector<float> const& Prediction(std::size_t unit_idx) const {
    return this->Training(unit_idx).predictions;
  }
};

// Gradient of each training unit, indexed by the global row index. The rows held out by a
// fold are zeroed rather than left as padding: this buffer is consumed whole, so a stale
// value would leak into the fold's root sum and histograms. The refit unit has a gradient
// for every row.
struct FoldGpairs {
  std::vector<linalg::Matrix<GradientPair>> gpairs;
  UnitLayout layout;

  [[nodiscard]] linalg::Matrix<GradientPair> const& Refit() const {
    return gpairs.at(layout.RefitIdx());
  }
};
}  // namespace xgboost::cv

using FoldModelsHandle = void*;
using FoldInfoBatchesHandle = void*;
using FoldPredictionsHandle = void*;
using FoldGpairsHandle = void*;
using TreeMethodHandle = void*;
