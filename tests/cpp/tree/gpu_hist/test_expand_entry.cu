/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>
#include <xgboost/tree_model.h>  // for RegTree

#include "../../../../src/tree/gpu_hist/expand_entry.cuh"

namespace xgboost::tree {
TEST(ExpandEntry, IOGPU) {
  DeviceSplitCandidate split;
  GPUExpandEntry entry{RegTree::kRoot, 0, split, 3.0, 1.0, 2.0};

  Json je{Object{}};
  entry.Save(&je);

  GPUExpandEntry loaded;
  loaded.Load(je);

  ASSERT_EQ(entry.base_weight, loaded.base_weight);
  ASSERT_EQ(entry.left_weight, loaded.left_weight);
  ASSERT_EQ(entry.right_weight, loaded.right_weight);

  ASSERT_EQ(entry.GetDepth(), loaded.GetDepth());
  ASSERT_EQ(entry.GetLossChange(), loaded.GetLossChange());
}
}  // namespace xgboost::tree
