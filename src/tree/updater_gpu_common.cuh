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

  XGBOOST_DEVICE void Update(float loss_chg_in, DefaultDirection dir_in, float fvalue_in,
                             int findex_in, common::Span<GradientPairInt64 const> child_sum_in) {
    if (loss_chg_in > loss_chg) {
      loss_chg = loss_chg_in;
      dir = dir_in;
      fvalue = fvalue_in;
      is_cat = false;
      child_sum = child_sum_in;
      findex = findex_in;
    }
  }

  // Update for categorical splits.
  XGBOOST_DEVICE void UpdateCat(float loss_chg_in, DefaultDirection dir_in, bst_cat_t thresh_in,
                                bst_feature_t findex_in,
                                common::Span<GradientPairInt64 const> child_sum_in) {
    if (loss_chg_in > loss_chg) {
      loss_chg = loss_chg_in;
      dir = dir_in;
      fvalue = std::numeric_limits<float>::quiet_NaN();
      thresh = thresh_in;
      is_cat = true;
      child_sum = child_sum_in;
      findex = findex_in;
    }
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
