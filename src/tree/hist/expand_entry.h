/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
#define XGBOOST_TREE_HIST_EXPAND_ENTRY_H_

#include <ostream>  // for ostream
#include <utility>  // for move
#include <vector>   // for vector

#include "../param.h"      // for SplitEntry, SplitEntryContainer
#include "xgboost/base.h"  // for GradientPairPrecise, bst_node_t

namespace xgboost::tree {
/**
 * @brief Structure for storing tree split candidate.
 */
template <typename Impl>
struct ExpandEntryImpl {
  bst_node_t nid{0};
  bst_node_t depth{0};

  [[nodiscard]] float GetLossChange() const {
    return static_cast<Impl const*>(this)->split.loss_chg;
  }
  [[nodiscard]] bst_node_t GetNodeId() const { return nid; }
};

struct CPUExpandEntry : public ExpandEntryImpl<CPUExpandEntry> {
  SplitEntry split;

  CPUExpandEntry() = default;
  CPUExpandEntry(bst_node_t nidx, bst_node_t depth, SplitEntry split)
      : ExpandEntryImpl{nidx, depth}, split(std::move(split)) {}
  CPUExpandEntry(bst_node_t nidx, bst_node_t depth) : ExpandEntryImpl{nidx, depth} {}

  friend std::ostream& operator<<(std::ostream& os, CPUExpandEntry const& e) {
    os << "ExpandEntry:\n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "split:\n" << e.split << std::endl;
    return os;
  }
};

struct MultiExpandEntry : public ExpandEntryImpl<MultiExpandEntry> {
  SplitEntryContainer<std::vector<GradientPairPrecise>> split;

  MultiExpandEntry() = default;
  MultiExpandEntry(bst_node_t nidx, bst_node_t depth) : ExpandEntryImpl{nidx, depth} {}

  friend std::ostream& operator<<(std::ostream& os, MultiExpandEntry const& e) {
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
