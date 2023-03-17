/**
 * Copyright 2021-2023 by XGBoost contributors.
 */
#ifndef XGBOOST_TESTS_CPP_TREE_TEST_PARTITIONER_H_
#define XGBOOST_TESTS_CPP_TREE_TEST_PARTITIONER_H_
#include <xgboost/context.h>                      // for Context
#include <xgboost/linalg.h>                       // for Constant, Vector
#include <xgboost/logging.h>                      // for CHECK
#include <xgboost/tree_model.h>                   // for RegTree

#include <vector>                                 // for vector

#include "../../../src/tree/hist/expand_entry.h"  // for CPUExpandEntry, MultiExpandEntry

namespace xgboost::tree {
inline void GetSplit(RegTree *tree, float split_value, std::vector<CPUExpandEntry> *candidates) {
  CHECK(!tree->IsMultiTarget());
  tree->ExpandNode(
      /*nid=*/RegTree::kRoot, /*split_index=*/0, /*split_value=*/split_value,
      /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      /*left_sum=*/0.0f,
      /*right_sum=*/0.0f);
  candidates->front().split.split_value = split_value;
  candidates->front().split.sindex = 0;
  candidates->front().split.sindex |= (1U << 31);
}

inline void GetMultiSplitForTest(RegTree *tree, float split_value,
                                 std::vector<MultiExpandEntry> *candidates) {
  CHECK(tree->IsMultiTarget());
  auto n_targets = tree->NumTargets();
  Context ctx;
  linalg::Vector<float> base_weight{linalg::Constant(&ctx, 0.0f, n_targets)};
  linalg::Vector<float> left_weight{linalg::Constant(&ctx, 0.0f, n_targets)};
  linalg::Vector<float> right_weight{linalg::Constant(&ctx, 0.0f, n_targets)};

  tree->ExpandNode(/*nidx=*/RegTree::kRoot, /*split_index=*/0, /*split_value=*/split_value,
                   /*default_left=*/true, base_weight.HostView(), left_weight.HostView(),
                   right_weight.HostView());
  candidates->front().split.split_value = split_value;
  candidates->front().split.sindex = 0;
  candidates->front().split.sindex |= (1U << 31);
}
}  // namespace xgboost::tree
#endif  // XGBOOST_TESTS_CPP_TREE_TEST_PARTITIONER_H_
