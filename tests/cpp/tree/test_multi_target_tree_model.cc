/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#include "test_multi_target_tree_model.h"

#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context
#include <xgboost/linalg.h>   // for Vector
#include <xgboost/multi_target_tree_model.h>
#include <xgboost/tree_model.h>  // for RegTree

#include <memory>   // for unique_ptr
#include <numeric>  // for iota

#include "../../../src/tree/tree_view.h"

namespace xgboost {
std::unique_ptr<RegTree> MakeMtTreeForTest(bst_target_t n_targets) {
  bst_feature_t n_features{4};
  std::unique_ptr<RegTree> tree{std::make_unique<RegTree>(n_targets, n_features)};
  CHECK(tree->IsMultiTarget());

  auto iota_weights = [&](float init, HostDeviceVector<float>* data,
                          common::Span<std::size_t> shape) {
    shape[0] = n_targets;
    auto& h_data = data->HostVector();
    h_data.resize(n_targets);
    std::iota(h_data.begin(), h_data.end(), init);
  };

  linalg::Vector<float> base_weight;
  base_weight.ModifyInplace([&](HostDeviceVector<float>* data, common::Span<std::size_t> shape) {
    iota_weights(1.0f, data, shape);
  });
  tree->SetRoot(base_weight.HostView(), /*sum_hess=*/1.0f);

  linalg::Vector<float> left_weight;
  left_weight.ModifyInplace([&](HostDeviceVector<float>* data, common::Span<std::size_t> shape) {
    iota_weights(2.0f, data, shape);
  });
  linalg::Vector<float> right_weight;
  right_weight.ModifyInplace([&](HostDeviceVector<float>* data, common::Span<std::size_t> shape) {
    iota_weights(3.0f, data, shape);
  });

  tree->ExpandNode(RegTree::kRoot, /*split_idx=*/1, 0.5f, true, base_weight.HostView(),
                   left_weight.HostView(), right_weight.HostView(), /*gain=*/0.5f,
                   /*sum_hess=*/1.0f, /*left_sum=*/0.6f, /*right_sum=*/0.4f);
  tree->GetMultiTargetTree()->SetLeaves();
  return tree;
}

TEST(MultiTargetTree, JsonIO) {
  auto tree = MakeMtTreeForTest(3);
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
  auto tree = MakeMtTreeForTest(3);
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
    tree.SetRoot(weight.HostView(), /*sum_hess=*/1.0f);
    tree.ExpandNode(RegTree::kRoot, /*split_idx=*/1, 0.5f, true, weight.HostView(),
                    weight.HostView(), weight.HostView(), /*gain=*/0.5f, /*sum_hess=*/1.0f,
                    /*left_sum=*/0.6f, /*right_sum=*/0.4f);
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
  auto tree = MakeMtTreeForTest(3);
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
  tree->SetRoot(base_weight.HostView(), /*sum_hess=*/1.0f);
  ASSERT_EQ(tree->GetMultiTargetTree()->NumSplitTargets(), 2);

  linalg::Vector<float> left_weight{{2.0f, 3.0f}, {2ul}, DeviceOrd::CPU()};
  linalg::Vector<float> right_weight{{3.0f, 4.0f}, {2ul}, DeviceOrd::CPU()};
  tree->ExpandNode(RegTree::kRoot, /*split_idx=*/1, 0.5f, true, base_weight.HostView(),
                   left_weight.HostView(), right_weight.HostView(), /*gain=*/0.5f,
                   /*sum_hess=*/1.0f, /*left_sum=*/0.6f, /*right_sum=*/0.4f);

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

TEST(MultiTargetTree, Statistics) {
  // Test that gain and sum_hess are serialized and deserialized correctly
  auto tree = MakeMtTreeForTest(3);
  // Following values are defined by the `MakeMtTreeForTest.
  auto view = tree->HostMtView();
  // Gain and sum_hess stored at the parent (split node)
  ASSERT_FLOAT_EQ(view.LossChg(0), 0.5f);
  ASSERT_FLOAT_EQ(view.SumHess(0), 1.0f);
  // Child nodes have their sum_hess values
  ASSERT_FLOAT_EQ(view.LossChg(1), 0.0f);  // Leaves have no gain
  ASSERT_FLOAT_EQ(view.SumHess(1), 0.6f);  // Left child
  ASSERT_FLOAT_EQ(view.LossChg(2), 0.0f);
  ASSERT_FLOAT_EQ(view.SumHess(2), 0.4f);  // Right child

  // Test serialization round-trip
  Json jtree{Object{}};
  tree->SaveModel(&jtree);

  // Check that statistics are in the JSON
  auto const& obj = get<Object const>(jtree);
  ASSERT_TRUE(obj.find("loss_changes") != obj.end());
  ASSERT_TRUE(obj.find("sum_hessian") != obj.end());
  auto const& gains = get<F32Array const>(jtree["loss_changes"]);
  ASSERT_EQ(gains.size(), tree->NumNodes());
  ASSERT_FLOAT_EQ(gains[0], 0.5f);

  // Load and verify statistics are preserved
  RegTree loaded;
  loaded.LoadModel(jtree);
  auto loaded_view = loaded.HostMtView();
  ASSERT_FLOAT_EQ(loaded_view.LossChg(0), 0.5f);
  ASSERT_FLOAT_EQ(loaded_view.SumHess(0), 1.0f);
  ASSERT_FLOAT_EQ(loaded_view.SumHess(1), 0.6f);
  ASSERT_FLOAT_EQ(loaded_view.SumHess(2), 0.4f);
}
}  // namespace xgboost
