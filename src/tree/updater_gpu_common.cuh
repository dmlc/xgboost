/**
 * Copyright 2017-2026, XGBoost contributors
 */
#pragma once
#include <limits>  // for numeric_limits

#include "../data/batch_utils.h"  // for DftPrefetchBatches, StaticBatch
#include "param.h"                // for TrainParam
#include "xgboost/base.h"         // for bst_bin_t, GradientPairInt64
#include "xgboost/span.h"         // for Span
#include "xgboost/task.h"         // for ObjInfo

namespace xgboost::tree {
/**
 * @brief Default direction to be followed in case of missing values
 */
enum DefaultDirection {
  /** move to left child */
  kLeftDir = 0,
  /** move to right child */
  kRightDir
};

/**
 * @brief Best split found for a node, shared by the single-target and the multi-target
 *        evaluation.
 */
struct DeviceSplitCandidate {
  float loss_chg{-std::numeric_limits<float>::max()};
  DefaultDirection dir{kLeftDir};
  int findex{-1};
  // Split value for numerical splits, NaN for categorical splits.
  float fvalue{0};
  // categorical split, either it's the split category for OHE or the threshold for partition-based
  // split.
  bst_cat_t thresh{-1};

  bool is_cat{false};
  // Gradient sum of the child without missing values for each target. It's the left child
  // when missing values go right, and vice versa.
  common::Span<GradientPairInt64 const> child_sum;

  [[nodiscard]] XGBOOST_DEV_INLINE bool Update(float new_loss_chg, bst_feature_t fidx,
                                               DefaultDirection direction, bool is_categorical,
                                               common::Span<GradientPairInt64 const> sum) {
    if (new_loss_chg > this->loss_chg) {
      this->loss_chg = new_loss_chg;
      this->dir = direction;
      this->findex = fidx;
      this->child_sum = sum;
      this->is_cat = is_categorical;
      return true;
    }
    return false;
  }
};

namespace cuda_impl {
inline BatchParam HistBatch(TrainParam const& param) {
  auto p = BatchParam{param.max_bin, TrainParam::DftSparseThreshold()};
  p.prefetch_copy = true;
  p.n_prefetch_batches = ::xgboost::cuda_impl::DftPrefetchBatches();
  return p;
}

inline BatchParam ApproxBatch(TrainParam const& p, common::Span<float const> hess,
                              ObjInfo const& task) {
  auto batch = BatchParam{p.max_bin, hess, !task.const_hess};
  batch.prefetch_copy = true;
  batch.n_prefetch_batches = ::xgboost::cuda_impl::DftPrefetchBatches();
  return batch;
}
}  // namespace cuda_impl

template <typename T>
struct SumCallbackOp {
  // Running prefix
  T running_total{T{}};

  SumCallbackOp() = default;
  XGBOOST_DEVICE T operator()(T block_aggregate) {
    T old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};
}  // namespace xgboost::tree
