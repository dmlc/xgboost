/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICT_FN_H_
#define XGBOOST_PREDICTOR_PREDICT_FN_H_

#include <memory>  // for unique_ptr
#include <vector>  // for vector

#include "../common/categorical.h"  // for IsCat, Decision
#include "../common/math.h"         // for CheckNAN
#include "xgboost/tree_model.h"     // for RegTree

namespace xgboost::predictor {
/** @brief Whether it should traverse to the left branch of a tree. */
template <bool has_categorical, typename TreeView>
XGBOOST_DEVICE bool GetDecision(TreeView const &tree, bst_node_t nid, float fvalue,
                                RegTree::CategoricalSplitMatrix const &cats) {
  if (has_categorical && common::IsCat(cats.split_type, nid)) {
    auto node_categories = cats.categories.subspan(cats.node_ptr[nid].beg, cats.node_ptr[nid].size);
    return common::Decision(node_categories, fvalue);
  } else {
    return fvalue < tree.SplitCond(nid);
  }
}

/**
 * @brief Select the next node using a feature value, with NaN representing missing data.
 *
 * @tparam has_missing Whether feature values may be NaN. When false, callers must
 *                     provide non-NaN values and the missing-value check is omitted.
 * @tparam has_categorical Whether the tree may contain categorical splits.
 * @param fvalue Feature value for the split. With has_missing enabled, NaN selects
 *               the default child before numerical or categorical split evaluation.
 *               Any user-defined missing-value sentinel must already be converted
 *               to NaN by the input loader.
 */
template <bool has_missing, bool has_categorical, typename TreeView>
XGBOOST_DEVICE bst_node_t GetNextNode(TreeView const &tree, const bst_node_t nid, float fvalue,
                                      RegTree::CategoricalSplitMatrix const &cats) {
  if constexpr (has_missing) {
    if (common::CheckNAN(fvalue)) {
      return tree.DefaultChild(nid);
    }
  }
  return tree.LeftChild(nid) + !GetDecision<has_categorical>(tree, nid, fvalue, cats);
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
