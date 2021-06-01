/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_PREDICTOR_PREDICT_FN_H_
#define XGBOOST_PREDICTOR_PREDICT_FN_H_
#include "xgboost/tree_model.h"
#include "../common/categorical.h"

namespace xgboost {
namespace predictor {
template <bool has_missing, bool has_categorical>
inline XGBOOST_DEVICE bst_node_t GetNextNode(
    common::Span<RegTree::Node const> tree, bst_node_t nid, float fvalue,
    bool is_missing, RegTree::CategoricalSplitMatrix const& cats) {
  if (has_missing && is_missing) {
    nid = tree[nid].DefaultChild();
  } else {
    bool go_left = true;
    if (has_categorical && common::IsCat(cats.split_type, nid)) {
      auto node_categories = cats.categories.subspan(cats.node_ptr[nid].beg,
                                                     cats.node_ptr[nid].size);
      go_left = Decision(node_categories, common::AsCat(fvalue));
    } else {
      go_left = fvalue < tree[nid].SplitCond();
    }
    if (go_left) {
      nid = tree[nid].LeftChild();
    } else {
      nid = tree[nid].RightChild();
    }
  }
  return nid;
}
}  // namespace predictor
}  // namespace xgboost
#endif  // XGBOOST_PREDICTOR_PREDICT_FN_H_
