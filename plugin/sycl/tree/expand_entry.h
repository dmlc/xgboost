/*!
 * Copyright 2017-2024 by Contributors
 * \file expand_entry.h
 */
#ifndef PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_
#define PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../../src/tree/constraints.h"
#pragma GCC diagnostic pop
#include "../../src/tree/hist/expand_entry.h"

namespace xgboost {
namespace sycl {
namespace tree {
/* tree growing policies */
struct ExpandEntry : public xgboost::tree::ExpandEntryImpl<ExpandEntry> {
  static constexpr bst_node_t kRootNid  = 0;

  xgboost::tree::SplitEntry split;

  ExpandEntry(int nid, int depth) : ExpandEntryImpl{nid, depth} {}

  inline bst_node_t GetSiblingId(const xgboost::RegTree* p_tree) const {
    CHECK_EQ((*p_tree)[nid].IsRoot(), false);
    const size_t parent_id = (*p_tree)[nid].Parent();
    return GetSiblingId(p_tree, parent_id);
  }

  inline bst_node_t GetSiblingId(const xgboost::RegTree* p_tree, size_t parent_id) const {
    return p_tree->IsLeftChild(nid) ? p_tree->RightChild(parent_id)
                                    : p_tree->LeftChild(parent_id);
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
