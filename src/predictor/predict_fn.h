/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICT_FN_H_
#define XGBOOST_PREDICTOR_PREDICT_FN_H_
#include "../common/categorical.h"
#include "xgboost/tree_model.h"

namespace xgboost::predictor {
template <bool has_missing, bool has_categorical>
inline XGBOOST_DEVICE bst_node_t GetNextNode(const RegTree::Node &node, const bst_node_t nid,
                                             float fvalue, bool is_missing,
                                             RegTree::CategoricalSplitMatrix const &cats) {
  if (has_missing && is_missing) {
    return node.DefaultChild();
  } else {
    if (has_categorical && common::IsCat(cats.split_type, nid)) {
      auto node_categories =
          cats.categories.subspan(cats.node_ptr[nid].beg, cats.node_ptr[nid].size);
      return common::Decision(node_categories, fvalue) ? node.LeftChild() : node.RightChild();
    } else {
      return node.LeftChild() + !(fvalue < node.SplitCond());
    }
  }
}

template <bool has_missing, bool has_categorical>
inline XGBOOST_DEVICE bst_node_t GetNextNodeMulti(MultiTargetTree const &tree,
                                                  bst_node_t const nidx, float fvalue,
                                                  bool is_missing,
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

}  // namespace xgboost::predictor
#endif  // XGBOOST_PREDICTOR_PREDICT_FN_H_
