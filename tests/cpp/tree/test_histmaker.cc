/**
 * Copyright 2019-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include "../../../src/tree/param.h"  // for TrainParam
#include "../helpers.h"

namespace xgboost::tree {
std::shared_ptr<DMatrix> GenerateDMatrix(std::size_t rows, std::size_t cols,
                                         bool categorical = false) {
  if (categorical) {
    std::vector<FeatureType> ft(cols);
    for (size_t i = 0; i < ft.size(); ++i) {
      ft[i] = (i % 3 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
    }
    return RandomDataGenerator(rows, cols, 0.6f).Seed(3).Type(ft).MaxCategory(17).GenerateDMatrix();
  } else {
    return RandomDataGenerator{rows, cols, 0.6f}.Seed(3).GenerateDMatrix();
  }
}

TEST(GrowHistMaker, InteractionConstraint) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;
  auto p_dmat = GenerateDMatrix(kRows, kCols);
  auto p_gradients = GenerateGradients(kRows);

  Context ctx;
  ObjInfo task{ObjInfo::kRegression};
  {
    // With constraints
    RegTree tree{1, kCols};

    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_histmaker", &ctx, &task)};
    TrainParam param;
    param.UpdateAllowUnknown(
        Args{{"interaction_constraints", "[[0, 1]]"}, {"num_feature", std::to_string(kCols)}});
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    updater->Update(&param, p_gradients.get(), p_dmat.get(), position, {&tree});

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
    updater->Update(&param, p_gradients.get(), p_dmat.get(), position, {&tree});

    ASSERT_EQ(tree.NumExtraNodes(), 10);
    ASSERT_EQ(tree[0].SplitIndex(), 1);

    ASSERT_NE(tree[tree[0].LeftChild()].SplitIndex(), 0);
    ASSERT_NE(tree[tree[0].RightChild()].SplitIndex(), 0);
  }
}

namespace {
void VerifyColumnSplit(int32_t rows, bst_feature_t cols, bool categorical,
                       RegTree const& expected_tree) {
  auto p_dmat = GenerateDMatrix(rows, cols, categorical);
  auto p_gradients = GenerateGradients(rows);
  Context ctx;
  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_histmaker", &ctx, &task)};
  std::vector<HostDeviceVector<bst_node_t>> position(1);

  std::unique_ptr<DMatrix> sliced{
      p_dmat->SliceCol(collective::GetWorldSize(), collective::GetRank())};

  RegTree tree{1u, cols};
  TrainParam param;
  param.Init(Args{});
  updater->Update(&param, p_gradients.get(), sliced.get(), position, {&tree});

  Json json{Object{}};
  tree.SaveModel(&json);
  Json expected_json{Object{}};
  expected_tree.SaveModel(&expected_json);
  ASSERT_EQ(json, expected_json);
}

void TestColumnSplit(bool categorical) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;

  RegTree expected_tree{1u, kCols};
  ObjInfo task{ObjInfo::kRegression};
  {
    auto p_dmat = GenerateDMatrix(kRows, kCols, categorical);
    auto p_gradients = GenerateGradients(kRows);
    Context ctx;
    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_histmaker", &ctx, &task)};
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    TrainParam param;
    param.Init(Args{});
    updater->Update(&param, p_gradients.get(), p_dmat.get(), position, {&expected_tree});
  }

  auto constexpr kWorldSize = 2;
  RunWithInMemoryCommunicator(kWorldSize, VerifyColumnSplit, kRows, kCols, categorical,
                              std::cref(expected_tree));
}
}  // anonymous namespace

TEST(GrowHistMaker, ColumnSplitNumerical) { TestColumnSplit(false); }

TEST(GrowHistMaker, ColumnSplitCategorical) { TestColumnSplit(true); }
}  // namespace xgboost::tree
