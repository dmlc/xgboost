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

  std::unique_ptr<Learner> TrainOneIter(std::string tree_method, std::string policy,
                                        int32_t max_leaves, int32_t max_depth) {
    std::unique_ptr<Learner> learner{Learner::Create({this->Xy_})};
    learner->SetParam("tree_method", tree_method);
    if (max_leaves >= 0) {
      learner->SetParam("max_leaves", std::to_string(max_leaves));
    }
    if (max_depth >= 0) {
      learner->SetParam("max_depth", std::to_string(max_depth));
    }
    learner->SetParam("grow_policy", policy);

    auto check_max_leave = [&]() {
      Json model{Object{}};
      learner->SaveModel(&model);
      auto j_tree = model["learner"]["gradient_booster"]["model"]["trees"][0];
      RegTree tree;
      tree.LoadModel(j_tree);
      CHECK_LE(tree.GetNumLeaves(), max_leaves);
    };

    auto check_max_depth = [&](int32_t sol) {
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
      if (sol > -1) {
        CHECK_EQ(depth, sol);
      } else {
        CHECK_EQ(depth, max_depth) << "tree method: " << tree_method << " policy: " << policy
                                   << " leaves:" << max_leaves << ", depth:" << max_depth;
      }
    };

    if (max_leaves == 0 && max_depth == 0) {
      // unconstrainted
      if (tree_method != "gpu_hist") {
        // GPU pre-allocates for all nodes.
        learner->UpdateOneIter(0, Xy_);
      }
    } else if (max_leaves > 0 && max_depth == 0) {
      learner->UpdateOneIter(0, Xy_);
      check_max_leave();
    } else if (max_leaves == 0 && max_depth > 0) {
      learner->UpdateOneIter(0, Xy_);
      check_max_depth(-1);
    } else if (max_leaves > 0 && max_depth > 0) {
      learner->UpdateOneIter(0, Xy_);
      check_max_leave();
      check_max_depth(2);
    } else if (max_leaves == -1 && max_depth == 0) {
      // default max_leaves is 0, so both of them are now 0
    } else {
      // default parameters
      learner->UpdateOneIter(0, Xy_);
    }
    return learner;
  }

  void TestCombination(std::string tree_method) {
    for (auto policy : {"depthwise", "lossguide"}) {
      // -1 means default
      for (auto leaves : {-1, 0, 3}) {
        for (auto depth : {-1, 0, 3}) {
          this->TrainOneIter(tree_method, policy, leaves, depth);
        }
      }
    }
  }

  void TestTreeGrowPolicy(std::string tree_method, std::string policy) {
    {
      /**
       *  max_leaves
       */
      auto learner = this->TrainOneIter(tree_method, policy, 16, -1);
      Json model{Object{}};
      learner->SaveModel(&model);

      auto j_tree = model["learner"]["gradient_booster"]["model"]["trees"][0];
      RegTree tree;
      tree.LoadModel(j_tree);
      ASSERT_EQ(tree.GetNumLeaves(), 16);
    }
    {
      /**
       *  max_depth
       */
      auto learner = this->TrainOneIter(tree_method, policy, -1, 3);
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

TEST_F(TestGrowPolicy, Approx) {
  this->TestTreeGrowPolicy("approx", "depthwise");
  this->TestTreeGrowPolicy("approx", "lossguide");

  this->TestCombination("approx");
}

TEST_F(TestGrowPolicy, Hist) {
  this->TestTreeGrowPolicy("hist", "depthwise");
  this->TestTreeGrowPolicy("hist", "lossguide");

  this->TestCombination("hist");
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestGrowPolicy, GpuHist) {
  this->TestTreeGrowPolicy("gpu_hist", "depthwise");
  this->TestTreeGrowPolicy("gpu_hist", "lossguide");

  this->TestCombination("gpu_hist");
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
