/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICT_FN_H_
#define XGBOOST_PREDICTOR_PREDICT_FN_H_

#include <memory>  // for unique_ptr
#include <vector>  // for vector

#include "../common/categorical.h"  // for IsCat, Decision
#include "xgboost/tree_model.h"     // for RegTree

namespace xgboost::predictor {
/** @brief Whether it should traverse to the left branch of a tree. */
template <bool has_categorical, typename TreeView>
XGBOOST_DEVICE bool GetDecision(TreeView const &sc_tree, bst_node_t nid, float fvalue,
                                RegTree::CategoricalSplitMatrix const &cats) {
  if (has_categorical && common::IsCat(cats.split_type, nid)) {
    auto node_categories = cats.categories.subspan(cats.node_ptr[nid].beg, cats.node_ptr[nid].size);
    return common::Decision(node_categories, fvalue);
  } else {
    return fvalue < sc_tree.SplitCond(nid);
  }
}

template <bool has_missing, bool has_categorical, typename TreeView>
XGBOOST_DEVICE bst_node_t GetNextNode(TreeView const &sc_tree, const bst_node_t nid, float fvalue,
                                      bool is_missing,
                                      RegTree::CategoricalSplitMatrix const &cats) {
  if (has_missing && is_missing) {
    return sc_tree.DefaultChild(nid);
  } else {
    return sc_tree.LeftChild(nid) + !GetDecision<has_categorical>(sc_tree, nid, fvalue, cats);
  }
}

/**
 * @brief Some old prediction methods accept the ntree_limit parameter and they use 0 to
 *        indicate no limit.
 */
inline bst_tree_t GetTreeLimit(std::vector<std::unique_ptr<RegTree>> const &trees,
                               bst_tree_t ntree_limit) {
  auto n_trees = static_cast<bst_tree_t>(trees.size());
  if (ntree_limit == 0 || ntree_limit > n_trees) {
    ntree_limit = n_trees;
  }
  return ntree_limit;
}
}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_PREDICT_FN_H_
