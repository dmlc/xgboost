#include <gtest/gtest.h>

#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include "../helpers.h"

namespace xgboost {
namespace tree {

TEST(GrowHistMaker, InteractionConstraint) {
  size_t constexpr kRows = 32;
  size_t constexpr kCols = 16;

  Context ctx;

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
  auto mparam = MakeMP(kCols, 0.5, 1);
  {
    // With constraints
    RegTree tree{&mparam};
    tree.param.num_feature = kCols;

    std::unique_ptr<TreeUpdater> updater{
        TreeUpdater::Create("grow_histmaker", &ctx, ObjInfo{ObjInfo::kRegression})};
    updater->Configure(Args{
        {"interaction_constraints", "[[0, 1]]"},
        {"num_feature", std::to_string(kCols)}});
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    updater->Update(&gradients, p_dmat.get(), position, {&tree});

    ASSERT_EQ(tree.NumExtraNodes(), 4);
    ASSERT_EQ(tree[0].SplitIndex(), 1);

    ASSERT_EQ(tree[tree[0].LeftChild()].SplitIndex(), 0);
    ASSERT_EQ(tree[tree[0].RightChild()].SplitIndex(), 0);
  }
  {
    // Without constraints
    RegTree tree{&mparam};
    tree.param.num_feature = kCols;

    std::unique_ptr<TreeUpdater> updater{
        TreeUpdater::Create("grow_histmaker", &ctx, ObjInfo{ObjInfo::kRegression})};
    updater->Configure(Args{{"num_feature", std::to_string(kCols)}});
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    updater->Update(&gradients, p_dmat.get(), position, {&tree});

    ASSERT_EQ(tree.NumExtraNodes(), 10);
    ASSERT_EQ(tree[0].SplitIndex(), 1);

    ASSERT_NE(tree[tree[0].LeftChild()].SplitIndex(), 0);
    ASSERT_NE(tree[tree[0].RightChild()].SplitIndex(), 0);
  }
}

}  // namespace tree
}  // namespace xgboost
