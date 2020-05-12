#include <gtest/gtest.h>

#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include "../helpers.h"

namespace xgboost {
namespace tree {

TEST(GrowHistMaker, InteractionConstraint) {
  size_t constexpr kRows = 32;
  size_t constexpr kCols = 16;

  GenericParameter param;
  param.UpdateAllowUnknown(Args{{"gpu_id", "0"}});

  auto p_dmat = RandomDataGenerator{kRows, kCols, 0.6f}.Seed(3).GenerateDMatrix();

  HostDeviceVector<GradientPair> gradients (kRows);
  std::vector<GradientPair>& h_gradients = gradients.HostVector();

  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(0.0f, 1.0f);

  for (size_t i = 0; i < kRows; ++i) {
    bst_float grad = dist(&gen);
    bst_float hess = dist(&gen);
    h_gradients[i] = GradientPair(grad, hess);
  }

  {
    // With constraints
    RegTree tree;
    tree.param.num_feature = kCols;

    std::unique_ptr<TreeUpdater> updater { TreeUpdater::Create("grow_histmaker", &param) };
    updater->Configure(Args{
        {"interaction_constraints", "[[0, 1]]"},
        {"num_feature", std::to_string(kCols)}});
    updater->Update(&gradients, p_dmat.get(), {&tree});

    ASSERT_EQ(tree.NumExtraNodes(), 4);
    ASSERT_EQ(tree[0].SplitIndex(), 1);

    ASSERT_EQ(tree[tree[0].LeftChild()].SplitIndex(), 0);
    ASSERT_EQ(tree[tree[0].RightChild()].SplitIndex(), 0);
  }
  {
    // Without constraints
    RegTree tree;
    tree.param.num_feature = kCols;

    std::unique_ptr<TreeUpdater> updater { TreeUpdater::Create("grow_histmaker", &param) };
    updater->Configure(Args{{"num_feature", std::to_string(kCols)}});
    updater->Update(&gradients, p_dmat.get(), {&tree});

    ASSERT_EQ(tree.NumExtraNodes(), 10);
    ASSERT_EQ(tree[0].SplitIndex(), 1);

    ASSERT_NE(tree[tree[0].LeftChild()].SplitIndex(), 0);
    ASSERT_NE(tree[tree[0].RightChild()].SplitIndex(), 0);
  }
}

}  // namespace tree
}  // namespace xgboost
