/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>     // for Context
#include <xgboost/multi_target_tree_model.h>
#include <xgboost/tree_model.h>  // for RegTree

namespace xgboost {
TEST(MultiTargetTree, JsonIO) {
  bst_target_t n_targets{3};
  bst_feature_t n_features{4};
  RegTree tree{n_targets, n_features};
  ASSERT_TRUE(tree.IsMultiTarget());
  linalg::Vector<float> base_weight{{1.0f, 2.0f, 3.0f}, {3ul}, Context::kCpuId};
  linalg::Vector<float> left_weight{{2.0f, 3.0f, 4.0f}, {3ul}, Context::kCpuId};
  linalg::Vector<float> right_weight{{3.0f, 4.0f, 5.0f}, {3ul}, Context::kCpuId};
  tree.ExpandNode(RegTree::kRoot, /*split_idx=*/1, 0.5f, true, base_weight.HostView(),
                  left_weight.HostView(), right_weight.HostView());
  ASSERT_EQ(tree.NumNodes(), 3);
  ASSERT_EQ(tree.NumTargets(), 3);
  ASSERT_EQ(tree.GetMultiTargetTree()->Size(), 3);
  ASSERT_EQ(tree.Size(), 3);

  Json jtree{Object{}};
  tree.SaveModel(&jtree);

  auto check_jtree = [](Json jtree, RegTree const& tree) {
    ASSERT_EQ(get<String const>(jtree["tree_param"]["num_nodes"]), std::to_string(tree.NumNodes()));
    ASSERT_EQ(get<F32Array const>(jtree["base_weights"]).size(),
              tree.NumNodes() * tree.NumTargets());
    ASSERT_EQ(get<I32Array const>(jtree["parents"]).size(), tree.NumNodes());
    ASSERT_EQ(get<I32Array const>(jtree["left_children"]).size(), tree.NumNodes());
    ASSERT_EQ(get<I32Array const>(jtree["right_children"]).size(), tree.NumNodes());
  };
  check_jtree(jtree, tree);

  RegTree loaded;
  loaded.LoadModel(jtree);
  ASSERT_TRUE(loaded.IsMultiTarget());
  ASSERT_EQ(loaded.NumNodes(), 3);

  Json jtree1{Object{}};
  loaded.SaveModel(&jtree1);
  check_jtree(jtree1, tree);
}
}  // namespace xgboost
