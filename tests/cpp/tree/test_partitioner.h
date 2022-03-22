/*!
 * Copyright 2021-2022, XGBoost contributors.
 */
#include <xgboost/tree_model.h>
#include <vector>
#include "../../../src/tree/hist/expand_entry.h"

namespace xgboost {
namespace tree {
inline void GetSplit(RegTree *tree, float split_value, std::vector<CPUExpandEntry> *candidates) {
  tree->ExpandNode(
      /*nid=*/RegTree::kRoot, /*split_index=*/0, /*split_value=*/split_value,
      /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      /*left_sum=*/0.0f,
      /*right_sum=*/0.0f);
  candidates->front().split.split_value = split_value;
  candidates->front().split.sindex = 0;
  candidates->front().split.sindex |= (1U << 31);
}
}  // namespace tree
}  // namespace xgboost
