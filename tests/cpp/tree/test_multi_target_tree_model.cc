/**
 * Copyright 2023-2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context
#include <xgboost/multi_target_tree_model.h>
#include <xgboost/tree_model.h>  // for RegTree

#include <numeric>  // for iota

#include "../../../src/tree/tree_view.h"

namespace xgboost {
namespace {
auto MakeTreeForTest() {
  bst_target_t n_targets{3};
  bst_feature_t n_features{4};
  std::unique_ptr<RegTree> tree{std::make_unique<RegTree>(n_targets, n_features)};
  CHECK(tree->IsMultiTarget());
  linalg::Vector<float> base_weight{{1.0f, 2.0f, 3.0f}, {3ul}, DeviceOrd::CPU()};
  tree->SetRoot(base_weight.HostView());
  linalg::Vector<float> left_weight{{2.0f, 3.0f, 4.0f}, {3ul}, DeviceOrd::CPU()};
  linalg::Vector<float> right_weight{{3.0f, 4.0f, 5.0f}, {3ul}, DeviceOrd::CPU()};
  tree->ExpandNode(RegTree::kRoot, /*split_idx=*/1, 0.5f, true, base_weight.HostView(),
                   left_weight.HostView(), right_weight.HostView());
  tree->GetMultiTargetTree()->SetLeaves();
  return tree;
}
}  // namespace

TEST(MultiTargetTree, JsonIO) {
  auto tree = MakeTreeForTest();
  ASSERT_EQ(tree->NumNodes(), 3);
  ASSERT_EQ(tree->NumTargets(), 3);
  ASSERT_EQ(tree->GetMultiTargetTree()->Size(), 3);
  ASSERT_EQ(tree->Size(), 3);

  Json jtree{Object{}};
  tree->SaveModel(&jtree);

  auto check_jtree = [](Json jtree, RegTree const& tree) {
    ASSERT_EQ(get<String const>(jtree["tree_param"]["num_nodes"]), std::to_string(tree.NumNodes()));
    ASSERT_EQ(get<F32Array const>(jtree["base_weights"]).size(),
              tree.NumNodes() * tree.NumTargets());
    ASSERT_EQ(get<I32Array const>(jtree["parents"]).size(), tree.NumNodes());
    ASSERT_EQ(get<I32Array const>(jtree["left_children"]).size(), tree.NumNodes());
    ASSERT_EQ(get<I32Array const>(jtree["right_children"]).size(), tree.NumNodes());
  };
  check_jtree(jtree, *tree);
  Context ctx;

  RegTree loaded;
  loaded.LoadModel(jtree);
  ASSERT_TRUE(loaded.IsMultiTarget());
  ASSERT_EQ(loaded.NumNodes(), 3);
  ASSERT_EQ(loaded.GetMultiTargetTree()->LeafWeights(ctx.Device()),
            tree->GetMultiTargetTree()->LeafWeights(ctx.Device()));

  Json jtree1{Object{}};
  loaded.SaveModel(&jtree1);
  check_jtree(jtree1, *tree);

  RegTree loaded1;
  loaded1.LoadModel(jtree1);
  ASSERT_EQ(loaded1.GetMultiTargetTree()->LeafWeights(ctx.Device()),
            tree->GetMultiTargetTree()->LeafWeights(ctx.Device()));

  Json jtree2{Object{}};
  loaded1.SaveModel(&jtree2);
  ASSERT_EQ(Json::Dump(jtree1), Json::Dump(jtree2));
}

namespace {
void TestTreeDump(std::string format, std::string leaf_key) {
  auto tree = MakeTreeForTest();
  auto n_features = tree->NumFeatures();
  FeatureMap fmap;
  for (bst_feature_t f = 0; f < n_features; ++f) {
    auto name = "feat_" + std::to_string(f);
    fmap.PushBack(f, name.c_str(), "q");
  }
  {
    auto str = tree->DumpModel(fmap, false, format);
    ASSERT_NE(str.find(leaf_key + "[2, 3, 4]"), std::string::npos);
    ASSERT_NE(str.find(leaf_key + "[3, 4, 5]"), std::string::npos);
  }

  {
    // Test the "..."
    bst_target_t n_targets{4};
    RegTree tree{n_targets, n_features};
    linalg::Vector<float> weight{{1.0f, 2.0f, 3.0f, 4.0f}, {4ul}, DeviceOrd::CPU()};
    tree.SetRoot(weight.HostView());
    tree.ExpandNode(RegTree::kRoot, /*split_idx=*/1, 0.5f, true, weight.HostView(),
                    weight.HostView(), weight.HostView());
    tree.GetMultiTargetTree()->SetLeaves();
    auto str = tree.DumpModel(fmap, false, format);
    ASSERT_NE(str.find(leaf_key + "[1, 2, ..., 4]"), std::string::npos);
  }
}
}  // namespace

TEST(MultiTargetTree, DotDump) { TestTreeDump("dot", "leaf="); }

TEST(MultiTargetTree, TextDump) { TestTreeDump("text", "leaf="); }

TEST(MultiTargetTree, JsonDump) { TestTreeDump("json", "\"leaf\": "); }

TEST(MultiTargetTree, View) {
  auto tree = MakeTreeForTest();
  auto v = tree->HostMtView();
  ASSERT_EQ(v.NumTargets(), 3);
  ASSERT_EQ(v.Size(), 3);
  ASSERT_EQ(v.LeftChild(0), 1);
  ASSERT_EQ(v.RightChild(0), 2);
}

TEST(MultiTargetTree, SetLeaves) {
  bst_target_t n_targets{5};
  bst_feature_t n_features{4};
  std::unique_ptr<RegTree> tree{std::make_unique<RegTree>(n_targets, n_features)};
  CHECK(tree->IsMultiTarget());
  // Reduce to 2 targets
  linalg::Vector<float> base_weight{{1.0f, 2.0f}, {2ul}, DeviceOrd::CPU()};
  tree->SetRoot(base_weight.HostView());
  ASSERT_EQ(tree->GetMultiTargetTree()->NumSplitTargets(), 2);

  linalg::Vector<float> left_weight{{2.0f, 3.0f}, {2ul}, DeviceOrd::CPU()};
  linalg::Vector<float> right_weight{{3.0f, 4.0f}, {2ul}, DeviceOrd::CPU()};
  tree->ExpandNode(RegTree::kRoot, /*split_idx=*/1, 0.5f, true, base_weight.HostView(),
                   left_weight.HostView(), right_weight.HostView());

  std::vector<float> leaf_weights(n_targets * 2);
  std::iota(leaf_weights.begin(), leaf_weights.end(), 0);
  tree->SetLeaves({1, 2}, {leaf_weights});
  ASSERT_TRUE(tree->HostMtView().IsLeaf(1));
  ASSERT_TRUE(tree->HostMtView().IsLeaf(2));
  auto mt_tree = tree->HostMtView();
  auto n_leaves = tree->GetMultiTargetTree()->NumLeaves();
  ASSERT_EQ(tree->GetNumLeaves(), n_leaves);
  ASSERT_EQ(2, n_leaves);
  ASSERT_EQ(mt_tree.leaf_weights.Shape(0), n_leaves);
  ASSERT_EQ(mt_tree.leaf_weights.Shape(1), n_targets);
  auto leaves = mt_tree.leaf_weights;
  for (std::size_t i = 0; i < leaves.Size(); ++i) {
    ASSERT_EQ(leaves.Values()[i], i);
  }
  auto left = mt_tree.LeafValue(1);
  for (std::size_t i = 0; i < left.Size(); ++i) {
    ASSERT_EQ(left.Values()[i], i);
  }
  auto right = mt_tree.LeafValue(2);
  for (std::size_t i = 0; i < right.Size(); ++i) {
    ASSERT_EQ(right.Values()[i], i + left.Size());
  }
}
}  // namespace xgboost
