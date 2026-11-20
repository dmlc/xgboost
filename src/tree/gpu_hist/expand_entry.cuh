/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef EXPAND_ENTRY_CUH_
#define EXPAND_ENTRY_CUH_

#include <ostream>  // for ostream

#include "../updater_gpu_common.cuh"  // for DeviceSplitCandidate
#include "xgboost/base.h"             // for bst_node_t

namespace xgboost::tree {
/**
 * @brief Split entry for both scalar and vector leaves. Node weights and gradient sums are
 *        stored in the evaluator, indexed by the node ID.
 */
struct GPUExpandEntry {
  bst_node_t nidx{0};
  bst_node_t depth{0};
  DeviceSplitCandidate split;
  // Sum of hessians across all targets for the left and right children.
  double left_hess{0};
  double right_hess{0};

  [[nodiscard]] float GetLossChange() const { return split.loss_chg; }
  [[nodiscard]] bst_node_t GetNodeId() const { return nidx; }

  friend std::ostream& operator<<(std::ostream& os, GPUExpandEntry const& e) {
    os << "GPUExpandEntry: nidx: " << e.nidx << ", depth: " << e.depth
       << ", loss: " << e.split.loss_chg << ", findex: " << e.split.findex
       << ", left_hess: " << e.left_hess << ", right_hess: " << e.right_hess;
    return os;
  }
};
}  // namespace xgboost::tree

#endif  // EXPAND_ENTRY_CUH_
