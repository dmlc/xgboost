/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef EXPAND_ENTRY_CUH_
#define EXPAND_ENTRY_CUH_

#include <limits>   // for numeric_limits
#include <utility>  // for move

#include "../param.h"                 // for TrainParam
#include "../updater_gpu_common.cuh"  // for DeviceSplitCandidate
#include "xgboost/base.h"             // for bst_node_t

namespace xgboost::tree {
struct GPUExpandEntry {
  bst_node_t nidx;
  bst_node_t depth;
  DeviceSplitCandidate split;

  float base_weight{std::numeric_limits<float>::quiet_NaN()};
  float left_weight{std::numeric_limits<float>::quiet_NaN()};
  float right_weight{std::numeric_limits<float>::quiet_NaN()};

  GPUExpandEntry() = default;
  XGBOOST_DEVICE GPUExpandEntry(bst_node_t nid, bst_node_t depth, DeviceSplitCandidate split,
                                float base, float left, float right)
      : nidx(nid),
        depth(depth),
        split(std::move(split)),
        base_weight{base},
        left_weight{left},
        right_weight{right} {}
  [[nodiscard]] bool IsValid(TrainParam const& param, bst_node_t num_leaves) const {
    if (split.loss_chg <= kRtEps) {
      return false;
    }
    if (split.left_sum.GetQuantisedHess() == 0 || split.right_sum.GetQuantisedHess() == 0) {
      return false;
    }
    if (split.loss_chg < param.min_split_loss) {
      return false;
    }
    if (param.max_depth > 0 && depth == param.max_depth) {
      return false;
    }
    if (param.max_leaves > 0 && num_leaves == param.max_leaves) {
      return false;
    }
    return true;
  }

  [[nodiscard]] float GetLossChange() const { return split.loss_chg; }

  [[nodiscard]] bst_node_t GetNodeId() const { return nidx; }

  [[nodiscard]] bst_node_t GetDepth() const { return depth; }

  friend std::ostream& operator<<(std::ostream& os, const GPUExpandEntry& e) {
    os << "GPUExpandEntry: \n";
    os << "nidx: " << e.nidx << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "left_sum: " << e.split.left_sum << "\n";
    os << "right_sum: " << e.split.right_sum << "\n";
    return os;
  }
};

namespace cuda_impl {
struct MultiExpandEntry {
  bst_node_t nidx{0};
  bst_node_t depth{0};
  MultiSplitCandidate split;

  common::Span<float> base_weight;
  // Sum of hessians across all targets for left/right children.
  double left_sum{0};
  double right_sum{0};

  MultiExpandEntry() = default;

  [[nodiscard]] float GetLossChange() const { return split.loss_chg; }

  [[nodiscard]] bst_node_t GetNodeId() const { return nidx; }

  [[nodiscard]] bst_node_t GetDepth() const { return depth; }

  [[nodiscard]] bool IsValid(TrainParam const& param, bst_node_t n_leaves) const {
    // The split evaluator handles the zero Hessian case. It returns an expand entry with
    // -inf loss_chg if the Hessian is invalid.
    if (split.loss_chg <= kRtEps) {
      return false;
    }
    if (base_weight.empty()) {
      return false;
    }
    if (split.loss_chg < param.min_split_loss) {
      return false;
    }
    if (param.max_depth > 0 && depth == param.max_depth) {
      return false;
    }
    if (param.max_leaves > 0 && n_leaves == param.max_leaves) {
      return false;
    }
    return true;
  }

  /**
   * @brief Update hessian statistics.
   * @param left_hess  Sum of hessians across all targets for left child.
   * @param right_hess Sum of hessians across all targets for right child.
   */
  __device__ void UpdateHessian(double left_hess, double right_hess) {
    this->left_sum = left_hess;
    this->right_sum = right_hess;
  }

  friend std::ostream& operator<<(std::ostream& os, MultiExpandEntry const& entry);
};
}  // namespace cuda_impl
}  // namespace xgboost::tree

#endif  // EXPAND_ENTRY_CUH_
