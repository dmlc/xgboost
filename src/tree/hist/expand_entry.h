/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
#define XGBOOST_TREE_HIST_EXPAND_ENTRY_H_

#include <ostream>  // for ostream
#include <vector>   // for vector

#include "../param.h"      // for SplitEntryContainer
#include "xgboost/base.h"  // for GradientPairPrecise, bst_node_t

namespace xgboost::tree {
/**
 * @brief Structure for storing tree split candidate.
 */
struct CPUExpandEntry {
  bst_node_t nid{0};
  bst_node_t depth{0};
  SplitEntryContainer<std::vector<GradientPairPrecise>> split;

  [[nodiscard]] float GetLossChange() const { return split.loss_chg; }
  [[nodiscard]] bst_node_t GetNodeId() const { return nid; }

  CPUExpandEntry() = default;
  CPUExpandEntry(bst_node_t nidx, bst_node_t depth) : nid{nidx}, depth{depth} {}

  friend std::ostream& operator<<(std::ostream& os, CPUExpandEntry const& e) {
    os << "ExpandEntry: \n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "split cond:" << e.split.split_value << "\n";
    os << "split ind:" << e.split.SplitIndex() << "\n";
    os << "left_sum: [";
    for (auto v : e.split.left_sum) {
      os << v << ", ";
    }
    os << "]\n";

    os << "right_sum: [";
    for (auto v : e.split.right_sum) {
      os << v << ", ";
    }
    os << "]\n";
    return os;
  }
};
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
