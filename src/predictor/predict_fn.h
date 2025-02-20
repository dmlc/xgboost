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
template <bool has_categorical>
XGBOOST_DEVICE bool GetDecision(RegTree::Node const &node, bst_node_t nid, float fvalue,
                                RegTree::CategoricalSplitMatrix const &cats) {
  if (has_categorical && common::IsCat(cats.split_type, nid)) {
    auto node_categories = cats.categories.subspan(cats.node_ptr[nid].beg, cats.node_ptr[nid].size);
    return common::Decision(node_categories, fvalue);
  } else {
    return fvalue < node.SplitCond();
  }
}

template <bool has_missing, bool has_categorical>
XGBOOST_DEVICE bst_node_t GetNextNode(const RegTree::Node &node, const bst_node_t nid, float fvalue,
                                      bool is_missing,
                                      RegTree::CategoricalSplitMatrix const &cats) {
  if (has_missing && is_missing) {
    return node.DefaultChild();
  } else {
    return node.LeftChild() + !GetDecision<has_categorical>(node, nid, fvalue, cats);
  }
}

template <bool has_missing, bool has_categorical>
XGBOOST_DEVICE bst_node_t GetNextNodeMulti(MultiTargetTree const &tree, bst_node_t const nidx,
                                           float fvalue, bool is_missing,
                                           RegTree::CategoricalSplitMatrix const &cats) {
  if (has_missing && is_missing) {
    return tree.DefaultChild(nidx);
  } else {
    if (has_categorical && common::IsCat(cats.split_type, nidx)) {
      auto node_categories =
          cats.categories.subspan(cats.node_ptr[nidx].beg, cats.node_ptr[nidx].size);
      return common::Decision(node_categories, fvalue) ? tree.LeftChild(nidx)
                                                       : tree.RightChild(nidx);
    } else {
      return tree.LeftChild(nidx) + !(fvalue < tree.SplitCond(nidx));
    }
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
