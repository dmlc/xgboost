/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#ifndef PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_
#define PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../../src/tree/constraints.h"
#pragma GCC diagnostic pop
#include "../../src/tree/param.h"
#include "../../src/tree/tree_view.h"

namespace xgboost {
namespace sycl {
namespace tree {
/* tree growing policies */
struct ExpandEntry {
  static constexpr bst_node_t kRootNid = 0;

  bst_node_t nid{0};
  bst_node_t depth{0};
  xgboost::tree::SplitEntry split;

  ExpandEntry(int nid, int depth) : nid{nid}, depth{depth} {}

  [[nodiscard]] float GetLossChange() const { return split.loss_chg; }
  [[nodiscard]] bst_node_t GetNodeId() const { return nid; }

  bst_node_t GetSiblingId(::xgboost::tree::ScalarTreeView const& tree) const {
    CHECK_EQ(tree.IsRoot(nid), false);
    const size_t parent_id = tree.Parent(nid);
    return GetSiblingId(tree, parent_id);
  }

  bst_node_t GetSiblingId(::xgboost::tree::ScalarTreeView const& tree, size_t parent_id) const {
    return tree.IsLeftChild(nid) ? tree.RightChild(parent_id) : tree.LeftChild(parent_id);
  }
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_
