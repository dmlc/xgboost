/**
 * Copyright 2019-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include "../../../src/tree/param.h"  // for TrainParam
#include "../helpers.h"
#include "test_column_split.h"  // for GenerateCatDMatrix

namespace xgboost::tree {
TEST(GrowHistMaker, InteractionConstraint) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;
  auto p_dmat = GenerateCatDMatrix(kRows, kCols, 0.0, false);
  Context ctx;

  linalg::Matrix<GradientPair> gpair({kRows}, ctx.Device());
  gpair.Data()->Copy(GenerateRandomGradients(kRows));

  ObjInfo task{ObjInfo::kRegression};
  {
    // With constraints
    RegTree tree{1, kCols};

    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_histmaker", &ctx, &task)};
    TrainParam param;
    param.UpdateAllowUnknown(
        Args{{"interaction_constraints", "[[0, 1]]"}, {"num_feature", std::to_string(kCols)}});
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    updater->Configure(Args{});
    updater->Update(&param, &gpair, p_dmat.get(), position, {&tree});

    ASSERT_EQ(tree.NumExtraNodes(), 4);
    ASSERT_EQ(tree[0].SplitIndex(), 1);

    ASSERT_EQ(tree[tree[0].LeftChild()].SplitIndex(), 0);
    ASSERT_EQ(tree[tree[0].RightChild()].SplitIndex(), 0);
  }
  {
    // Without constraints
    RegTree tree{1u, kCols};

    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_histmaker", &ctx, &task)};
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    TrainParam param;
    param.Init(Args{});
    updater->Configure(Args{});
    updater->Update(&param, &gpair, p_dmat.get(), position, {&tree});

    ASSERT_EQ(tree.NumExtraNodes(), 10);
    ASSERT_EQ(tree[0].SplitIndex(), 1);

    ASSERT_NE(tree[tree[0].LeftChild()].SplitIndex(), 0);
    ASSERT_NE(tree[tree[0].RightChild()].SplitIndex(), 0);
  }
}
}  // namespace xgboost::tree
