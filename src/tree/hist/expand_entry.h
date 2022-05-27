/*!
 * Copyright 2021 XGBoost contributors
 */
#ifndef XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
#define XGBOOST_TREE_HIST_EXPAND_ENTRY_H_

#include <utility>
#include "../param.h"

namespace xgboost {
namespace tree {

struct CPUExpandEntry {
  int nid;
  int depth;
  SplitEntry split;
  CPUExpandEntry() = default;
  XGBOOST_DEVICE
  CPUExpandEntry(int nid, int depth, SplitEntry split)
      : nid(nid), depth(depth), split(std::move(split)) {}
  CPUExpandEntry(int nid, int depth, float loss_chg)
      : nid(nid), depth(depth)  {
    split.loss_chg = loss_chg;
  }

  bool IsValid(const TrainParam& param, int num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    if (split.left_sum.GetHess() == 0 || split.right_sum.GetHess() == 0) {
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

  float GetLossChange() const { return split.loss_chg; }
  bst_node_t GetNodeId() const { return nid; }

  static bool ChildIsValid(const TrainParam& param, int depth, int num_leaves) {
    if (param.max_depth > 0 && depth >= param.max_depth) return false;
    if (param.max_leaves > 0 && num_leaves >= param.max_leaves) return false;
    return true;
  }

  friend std::ostream& operator<<(std::ostream& os, const CPUExpandEntry& e) {
    os << "ExpandEntry:\n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "split:\n" << e.split << std::endl;
    return os;
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
