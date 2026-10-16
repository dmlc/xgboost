/**
 * Copyright 2021-2026, XGBoost contributors.
 */
#pragma once

#include <xgboost/context.h>     // for Context
#include <xgboost/linalg.h>      // for Constant, Vector
#include <xgboost/logging.h>     // for CHECK
#include <xgboost/tree_model.h>  // for RegTree

#include <vector>  // for vector

#include "../../../src/tree/hist/expand_entry.h"  // for CPUExpandEntry, MultiExpandEntry

namespace xgboost::tree {
inline void GetSplit(RegTree *tree, float split_value, std::vector<CPUExpandEntry> *candidates) {
  CHECK(!tree->IsMultiTarget());
  tree->Expand(
      {{RegTree::kRoot, 0, split_value, true}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, 0.0f});
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
  tree->SetRoot(base_weight.HostView(), /*sum_hess=*/0.0f);
  ExpandBatch batch{{SplitInfo{RegTree::kRoot, 0, split_value, true},
                     {base_weight.HostView().Values(), 0.0f},
                     {left_weight.HostView().Values(), 0.0f},
                     {right_weight.HostView().Values(), 0.0f},
                     0.0f}};
  tree->Expand(&ctx, batch);
  candidates->front().split.split_value = split_value;
  candidates->front().split.sindex = 0;
  candidates->front().split.sindex |= (1U << 31);
  tree->FinalizeLeaves(1.0f);
}
}  // namespace xgboost::tree
