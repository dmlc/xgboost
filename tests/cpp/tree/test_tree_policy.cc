/**
 * Copyright 2021-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>     // for bst_node_t
#include <xgboost/context.h>  // for Context
#include <xgboost/tree_model.h>

#include <memory>  // for unique_ptr
#include <string>  // for string

#include "../../../src/tree/tree_view.h"  // for WalkTree
#include "../helpers.h"

namespace xgboost {
class TestGrowPolicy : public ::testing::Test {
 protected:
  bst_idx_t n_samples_ = 4096, n_features_ = 13;
  float sparsity_ = 0.5;

 protected:
  std::unique_ptr<Learner> TrainOneIter(Context const* ctx, bst_target_t n_targets,
                                        std::string tree_method, std::string policy,
                                        bst_node_t max_leaves, bst_node_t max_depth) {
    auto Xy =
        RandomDataGenerator{n_samples_, n_features_, sparsity_}.Targets(n_targets).GenerateDMatrix(
            true);

    std::unique_ptr<Learner> learner{Learner::Create({Xy})};
    learner->SetParam("tree_method", tree_method);
    learner->SetParam("device", ctx->DeviceName());
    if (max_leaves >= 0) {
      learner->SetParam("max_leaves", std::to_string(max_leaves));
    }
    if (max_depth >= 0) {
      learner->SetParam("max_depth", std::to_string(max_depth));
    }
    learner->SetParam("grow_policy", policy);
    if (n_targets > 1) {
      learner->SetParam("multi_strategy", "multi_output_tree");
    }

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
      tree::WalkTree(tree, [&](auto const& tree, bst_node_t nidx) {
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
      // unconstrained
      if (ctx->IsCPU()) {
        // GPU pre-allocates for all nodes.
        learner->UpdateOneIter(0, Xy);
      }
    } else if (max_leaves > 0 && max_depth == 0) {
      learner->UpdateOneIter(0, Xy);
      check_max_leave();
    } else if (max_leaves == 0 && max_depth > 0) {
      learner->UpdateOneIter(0, Xy);
      check_max_depth(-1);
    } else if (max_leaves > 0 && max_depth > 0) {
      learner->UpdateOneIter(0, Xy);
      check_max_leave();
      check_max_depth(2);
    } else if (max_leaves == -1 && max_depth == 0) {
      // default max_leaves is 0, so both of them are now 0
    } else {
      // default parameters
      learner->UpdateOneIter(0, Xy);
    }
    return learner;
  }

  void TestCombination(Context const* ctx, bst_target_t n_targets, std::string tree_method) {
    for (auto policy : {"depthwise", "lossguide"}) {
      // -1 means default
      for (auto leaves : {-1, 0, 3}) {
        for (auto depth : {-1, 0, 3}) {
          this->TrainOneIter(ctx, n_targets, tree_method, policy, leaves, depth);
        }
      }
    }
  }

  void TestTreeGrowPolicy(Context const* ctx, bst_target_t n_targets, std::string tree_method,
                          std::string policy) {
    {
      /**
       *  max_leaves
       */
      auto learner = this->TrainOneIter(ctx, n_targets, tree_method, policy, 16, -1);
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
      auto learner = this->TrainOneIter(ctx, n_targets, tree_method, policy, -1, 3);
      Json model{Object{}};
      learner->SaveModel(&model);

      auto j_tree = model["learner"]["gradient_booster"]["model"]["trees"][0];
      RegTree tree;
      tree.LoadModel(j_tree);
      bst_node_t depth = 0;
      tree::WalkTree(tree, [&](auto const& tree, bst_node_t nidx) {
        depth = std::max(tree.GetDepth(nidx), depth);
        return true;
      });
      ASSERT_EQ(depth, 3);
    }
  }
};

TEST_F(TestGrowPolicy, Approx) {
  Context ctx;
  bst_target_t n_targets = 1;
  this->TestTreeGrowPolicy(&ctx, n_targets, "approx", "depthwise");
  this->TestTreeGrowPolicy(&ctx, n_targets, "approx", "lossguide");

  this->TestCombination(&ctx, n_targets, "approx");
}

TEST_F(TestGrowPolicy, Hist) {
  Context ctx;
  bst_target_t n_targets = 1;
  this->TestTreeGrowPolicy(&ctx, n_targets, "hist", "depthwise");
  this->TestTreeGrowPolicy(&ctx, n_targets, "hist", "lossguide");

  this->TestCombination(&ctx, n_targets, "hist");
}

TEST_F(TestGrowPolicy, MultiHist) {
  Context ctx;
  bst_target_t n_targets = 3;
  this->TestTreeGrowPolicy(&ctx, n_targets, "hist", "depthwise");
  this->TestTreeGrowPolicy(&ctx, n_targets, "hist", "lossguide");

  this->TestCombination(&ctx, n_targets, "hist");
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestGrowPolicy, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  bst_target_t n_targets = 1;
  this->TestTreeGrowPolicy(&ctx, n_targets, "hist", "depthwise");
  this->TestTreeGrowPolicy(&ctx, n_targets, "hist", "lossguide");

  this->TestCombination(&ctx, n_targets, "hist");
}

TEST_F(TestGrowPolicy, GpuMultiHist) {
  auto ctx = MakeCUDACtx(0);
  bst_target_t n_targets = 3;
  this->TestTreeGrowPolicy(&ctx, n_targets, "hist", "depthwise");
  this->TestTreeGrowPolicy(&ctx, n_targets, "hist", "lossguide");

  this->TestCombination(&ctx, n_targets, "hist");
}

TEST_F(TestGrowPolicy, GpuApprox) {
  auto ctx = MakeCUDACtx(0);
  bst_target_t n_targets = 1;
  this->TestTreeGrowPolicy(&ctx, n_targets, "approx", "depthwise");
  this->TestTreeGrowPolicy(&ctx, n_targets, "approx", "lossguide");

  this->TestCombination(&ctx, n_targets, "approx");
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
