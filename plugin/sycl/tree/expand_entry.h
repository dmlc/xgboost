/**
 * Copyright 2017-2025, XGBoost Contributors
 */
#ifndef PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_
#define PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../../src/tree/constraints.h"
#pragma GCC diagnostic pop
#include "../../src/tree/hist/expand_entry.h"
#include "../../src/tree/tree_view.h"

namespace xgboost {
namespace sycl {
namespace tree {
/* tree growing policies */
struct ExpandEntry : public xgboost::tree::ExpandEntryImpl<ExpandEntry> {
  static constexpr bst_node_t kRootNid  = 0;

  xgboost::tree::SplitEntry split;

  ExpandEntry(int nid, int depth) : ExpandEntryImpl{nid, depth} {}

  bst_node_t GetSiblingId(::xgboost::tree::ScalarTreeView const& tree) const {
    CHECK_EQ(tree.IsRoot(nid), false);
    const size_t parent_id = tree.Parent(nid);
    return GetSiblingId(tree, parent_id);
  }

  bst_node_t GetSiblingId(::xgboost::tree::ScalarTreeView const& tree, size_t parent_id) const {
    return tree.IsLeftChild(nid) ? tree.RightChild(parent_id) : tree.LeftChild(parent_id);
  }

  bool IsValidImpl(xgboost::tree::TrainParam const &param, int32_t num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    if (split.loss_chg < param.min_split_loss) return false;
    if (param.max_depth > 0 && depth == param.max_depth) return false;
    if (param.max_leaves > 0 && num_leaves == param.max_leaves) return false;

    return true;
  }
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_
