/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICT_FN_H_
#define XGBOOST_PREDICTOR_PREDICT_FN_H_

#include <cstdint>  // for uint32_t
#include <cstring>  // for memcpy
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "../common/categorical.h"  // for IsCat, Decision
#include "../common/math.h"         // for CheckNAN
#include "xgboost/tree_model.h"     // for RegTree

namespace xgboost::predictor {
/**
 * @brief Same as common::CheckNAN for float, but always inlined.
 *
 * MSVC lowers std::isnan to an out-of-line CRT call, which is significant in the traversal
 * and when it runs for every input element.
 */
inline bool IsNaN(float v) {
#if defined(_MSC_VER)
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return (bits & 0x7fffffffu) > 0x7f800000u;
#else
  return common::CheckNAN(v);
#endif  // defined(_MSC_VER)
}

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

template <bool has_missing, bool has_categorical, typename TreeView>
XGBOOST_DEVICE bst_node_t GetNextNode(TreeView const &tree, const bst_node_t nid, float fvalue,
                                      bool is_missing,
                                      RegTree::CategoricalSplitMatrix const &cats) {
  if (has_missing && is_missing) {
    return tree.DefaultChild(nid);
  } else {
    return tree.LeftChild(nid) + !GetDecision<has_categorical>(tree, nid, fvalue, cats);
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
