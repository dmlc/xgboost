/*!
 * Copyright 2021 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/tree_model.h>
#include "../helpers.h"

namespace xgboost {
class TestGrowPolicy : public ::testing::Test {
 protected:
  std::shared_ptr<DMatrix> Xy_;
  size_t n_samples_ = 4096, n_features_ = 13;
  float sparsity_ = 0.5;

 protected:
  void SetUp() override {
    Xy_ =
        RandomDataGenerator{n_samples_, n_features_, sparsity_}.GenerateDMatrix(
            true);
  }

  void TestTreeGrowPolicy(std::string tree_method, std::string policy) {
    {
      std::unique_ptr<Learner> learner{Learner::Create({this->Xy_})};
      learner->SetParam("tree_method", tree_method);
      learner->SetParam("max_leaves", "16");
      learner->SetParam("grow_policy", policy);
      learner->Configure();

      learner->UpdateOneIter(0, Xy_);
      Json model{Object{}};
      learner->SaveModel(&model);

      auto j_tree = model["learner"]["gradient_booster"]["model"]["trees"][0];
      RegTree tree;
      tree.LoadModel(j_tree);
      ASSERT_EQ(tree.GetNumLeaves(), 16);
    }
    {
      std::unique_ptr<Learner> learner{Learner::Create({this->Xy_})};
      learner->SetParam("tree_method", tree_method);
      learner->SetParam("max_depth", "3");
      learner->SetParam("grow_policy", policy);
      learner->Configure();

      learner->UpdateOneIter(0, Xy_);
      Json model{Object{}};
      learner->SaveModel(&model);

      auto j_tree = model["learner"]["gradient_booster"]["model"]["trees"][0];
      RegTree tree;
      tree.LoadModel(j_tree);
      bst_node_t depth = 0;
      tree.WalkTree([&](bst_node_t nidx) {
        depth = std::max(tree.GetDepth(nidx), depth);
        return true;
      });
      ASSERT_EQ(depth, 3);
    }
  }
};

TEST_F(TestGrowPolicy, DISABLED_Approx) {
  this->TestTreeGrowPolicy("approx", "depthwise");
  this->TestTreeGrowPolicy("approx", "lossguide");
}

TEST_F(TestGrowPolicy, Hist) {
  this->TestTreeGrowPolicy("hist", "depthwise");
  this->TestTreeGrowPolicy("hist", "lossguide");
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestGrowPolicy, GpuHist) {
  this->TestTreeGrowPolicy("gpu_hist", "depthwise");
  this->TestTreeGrowPolicy("gpu_hist", "lossguide");
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
