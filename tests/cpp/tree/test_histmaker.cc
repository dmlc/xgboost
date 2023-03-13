/**
 * Copyright 2019-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include "../../../src/tree/param.h"  // for TrainParam
#include "../helpers.h"

namespace xgboost::tree {
std::shared_ptr<DMatrix> GenerateDMatrix(std::size_t rows, std::size_t cols){
  return RandomDataGenerator{rows, cols, 0.6f}.Seed(3).GenerateDMatrix();
}

std::unique_ptr<HostDeviceVector<GradientPair>> GenerateGradients(std::size_t rows) {
  auto p_gradients = std::make_unique<HostDeviceVector<GradientPair>>(rows);
  auto& h_gradients = p_gradients->HostVector();

  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(0.0f, 1.0f);

  for (std::size_t i = 0; i < rows; ++i) {
    auto grad = dist(&gen);
    auto hess = dist(&gen);
    h_gradients[i] = GradientPair{grad, hess};
  }

  return p_gradients;
}

TEST(GrowHistMaker, InteractionConstraint)
{
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
void TestColumnSplit(int32_t rows, bst_feature_t cols, RegTree const& expected_tree) {
  auto p_dmat = GenerateDMatrix(rows, cols);
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

  EXPECT_EQ(tree.NumExtraNodes(), 10);
  EXPECT_EQ(tree[0].SplitIndex(), 1);

  EXPECT_NE(tree[tree[0].LeftChild()].SplitIndex(), 0);
  EXPECT_NE(tree[tree[0].RightChild()].SplitIndex(), 0);

  EXPECT_EQ(tree, expected_tree);
}
}  // anonymous namespace

TEST(GrowHistMaker, ColumnSplit) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;

  RegTree expected_tree{1u, kCols};
  ObjInfo task{ObjInfo::kRegression};
  {
    auto p_dmat = GenerateDMatrix(kRows, kCols);
    auto p_gradients = GenerateGradients(kRows);
    Context ctx;
    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_histmaker", &ctx, &task)};
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    TrainParam param;
    param.Init(Args{});
    updater->Update(&param, p_gradients.get(), p_dmat.get(), position, {&expected_tree});
  }

  auto constexpr kWorldSize = 2;
  RunWithInMemoryCommunicator(kWorldSize, TestColumnSplit, kRows, kCols, std::cref(expected_tree));
}
}  // namespace xgboost::tree
