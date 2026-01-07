/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>       // for Context
#include <xgboost/gradient.h>      // for GradientContainer
#include <xgboost/task.h>          // for ObjInfo
#include <xgboost/tree_model.h>    // for RegTree
#include <xgboost/tree_updater.h>  // for TreeUpdater

#include <memory>  // for unique_ptr

#include "../../../src/tree/io_utils.h"   // for DftBadValue
#include "../../../src/tree/param.h"      // for TrainParam
#include "../../../src/tree/tree_view.h"  // for WalkTree
#include "../helpers.h"

namespace xgboost {
/**
 * @brief Test the tree statistic (like sum Hessian) is correct.
 */
class UpdaterTreeStatTest : public ::testing::Test {
 protected:
  std::shared_ptr<DMatrix> p_dmat_;
  GradientContainer gpairs_;
  size_t constexpr static kRows = 10;
  size_t constexpr static kCols = 10;

 protected:
  void SetUp() override {
    p_dmat_ = RandomDataGenerator(kRows, kCols, .5f).GenerateDMatrix(true);
    Context ctx;
    gpairs_ = GenerateRandomGradients(&ctx, kRows, 1);
  }

  void RunTest(Context const* ctx, std::string updater) {
    tree::TrainParam param;
    ObjInfo task{ObjInfo::kRegression};
    param.Init(Args{});

    auto up = std::unique_ptr<TreeUpdater>{TreeUpdater::Create(updater, ctx, &task)};
    up->Configure(Args{});
    RegTree tree{1u, kCols};
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    up->Update(&param, &gpairs_, p_dmat_.get(), position, {&tree});

    auto sc_tree = tree.HostScView();
    sc_tree.WalkTree([&sc_tree](bst_node_t nidx) {
      if (sc_tree.IsLeaf(nidx)) {
        // 1.0 is the default `min_child_weight`.
        CHECK_GE(sc_tree.Stat(nidx).sum_hess, 1.0);
      }
      return true;
    });
  }
};

#if defined(XGBOOST_USE_CUDA)
TEST_F(UpdaterTreeStatTest, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist");
}

TEST_F(UpdaterTreeStatTest, GpuApprox) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_approx");
}
#endif  // defined(XGBOOST_USE_CUDA)

TEST_F(UpdaterTreeStatTest, Hist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker");
}

TEST_F(UpdaterTreeStatTest, Exact) {
  Context ctx;
  this->RunTest(&ctx, "grow_colmaker");
}

TEST_F(UpdaterTreeStatTest, Approx) {
  Context ctx;
  this->RunTest(&ctx, "grow_histmaker");
}

namespace {
void BuildTree(Context const* ctx, DMatrix* p_fmat, GradientContainer* grad,
               std::string const& name, Args const& args, RegTree* p_tree) {
  tree::TrainParam param;
  param.Init(args);
  ObjInfo task{ObjInfo::kRegression};
  auto up = std::unique_ptr<TreeUpdater>{TreeUpdater::Create(name, ctx, &task)};
  up->Configure({});
  std::vector<HostDeviceVector<bst_node_t>> position(1);
  up->Update(&param, grad, p_fmat, common::Span{position}, {p_tree});
}
}  // namespace

/**
 * @brief Test changing learning rate doesn't change internal splits.
 */
class TestSplitWithEta : public ::testing::Test {
 protected:
  void Run(Context const* ctx, bst_target_t n_targets, std::string name) {
    auto Xy = RandomDataGenerator{512, 64, 0.2}.Targets(n_targets).GenerateDMatrix(true);

    auto gen_tree = [&](float eta) {
      auto tree =
          std::make_unique<RegTree>(n_targets, static_cast<bst_feature_t>(Xy->Info().num_col_));
      std::vector<RegTree*> trees{tree.get()};

      auto grad = GenerateRandomGradients(ctx, Xy->Info().num_row_, n_targets);
      CHECK_EQ(grad.gpair.Shape(1), n_targets);
      auto args = Args{{"learning_rate", std::to_string(eta)}};

      BuildTree(ctx, Xy.get(), &grad, name, args, tree.get());

      CHECK_EQ(tree->NumTargets(), n_targets);
      if (n_targets > 1) {
        CHECK(tree->IsMultiTarget());
      }
      return tree;
    };

    auto eta_ratio = 8.0f;
    auto p_tree0 = gen_tree(0.1f);
    auto p_tree1 = gen_tree(0.1f * eta_ratio);
    // Just to make sure we are not testing a stump.
    CHECK_GE(p_tree0->NumExtraNodes(), 32);

    bst_node_t n_nodes{0};
    tree::WalkTree(
        *p_tree0,
        [&](auto const& tree0, auto const& tree1, bst_node_t nidx) {
          if (tree0.IsLeaf(nidx)) {
            CHECK(tree1.IsLeaf(nidx));
            if (p_tree0->IsMultiTarget()) {
              CHECK(p_tree1->IsMultiTarget());
              auto leaf_0 = p_tree0->GetMultiTargetTree()->LeafValue(nidx);
              auto leaf_1 = p_tree1->GetMultiTargetTree()->LeafValue(nidx);
              CHECK_EQ(leaf_0.Size(), leaf_1.Size());
              for (std::size_t i = 0; i < leaf_0.Size(); ++i) {
                CHECK_EQ(leaf_0(i) * eta_ratio, leaf_1(i));
              }
              CHECK_EQ(DftBadValue(), tree0.SplitCond(nidx));
              CHECK_EQ(DftBadValue(), tree1.SplitCond(nidx));
            } else {
              // NON-mt tree reuses split cond for leaf value.
              auto leaf_0 = tree0.SplitCond(nidx);
              auto leaf_1 = tree1.SplitCond(nidx);
              CHECK_EQ(leaf_0 * eta_ratio, leaf_1);
            }
          } else {
            CHECK(!tree1.IsLeaf(nidx));
            CHECK_EQ(tree0.SplitCond(nidx), tree1.SplitCond(nidx));
          }
          n_nodes++;
          return true;
        },
        *p_tree1);
    ASSERT_EQ(n_nodes, p_tree0->NumExtraNodes() + 1);
  }
};

TEST_F(TestSplitWithEta, MultiHist) {
  Context ctx;
  bst_target_t n_targets{3};
  this->Run(&ctx, n_targets, "grow_quantile_histmaker");
}

TEST_F(TestSplitWithEta, Hist) {
  Context ctx;
  bst_target_t n_targets{1};
  this->Run(&ctx, n_targets, "grow_quantile_histmaker");
}

TEST_F(TestSplitWithEta, Approx) {
  Context ctx;
  bst_target_t n_targets{1};
  this->Run(&ctx, n_targets, "grow_histmaker");
}

TEST_F(TestSplitWithEta, Exact) {
  Context ctx;
  bst_target_t n_targets{1};
  this->Run(&ctx, n_targets, "grow_colmaker");
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestSplitWithEta, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  bst_target_t n_targets{1};
  this->Run(&ctx, n_targets, "grow_gpu_hist");
}

TEST_F(TestSplitWithEta, GpuMultiHist) {
  auto ctx = MakeCUDACtx(0);
  bst_target_t n_targets{3};
  this->Run(&ctx, n_targets, "grow_gpu_hist");
}

TEST_F(TestSplitWithEta, GpuApprox) {
  auto ctx = MakeCUDACtx(0);
  bst_target_t n_targets{1};
  this->Run(&ctx, n_targets, "grow_gpu_approx");
}
#endif  // defined(XGBOOST_USE_CUDA)

class TestMinSplitLoss : public ::testing::Test {
  std::shared_ptr<DMatrix> p_fmat_;
  GradientContainer gpair_;

  void SynthesizeData(bst_target_t n_targets) {
    constexpr size_t kRows = 32;
    constexpr size_t kCols = 16;
    constexpr float kSparsity = 0.6;
    p_fmat_ =
        RandomDataGenerator(kRows, kCols, kSparsity).Seed(3).Targets(n_targets).GenerateDMatrix();
    Context ctx;
    gpair_ = GenerateRandomGradients(&ctx, kRows, n_targets);
  }

  bst_node_t Update(Context const* ctx, std::string updater, float gamma) {
    Args args{{"max_depth", "1"},
              {"max_leaves", "0"},

              // Disable all other parameters.
              {"colsample_bynode", "1"},
              {"colsample_bylevel", "1"},
              {"colsample_bytree", "1"},
              {"min_child_weight", "0.01"},
              {"reg_alpha", "0"},
              {"reg_lambda", "0"},
              {"max_delta_step", "0"},

              // test gamma
              {"gamma", std::to_string(gamma)}};

    RegTree tree{static_cast<bst_target_t>(this->gpair_.gpair.Shape(1)),
                 static_cast<bst_target_t>(this->p_fmat_->Info().num_col_)};

    BuildTree(ctx, p_fmat_.get(), &gpair_, updater, args, &tree);
    auto n_nodes = tree.NumExtraNodes();
    return n_nodes;
  }

 public:
  void RunTest(Context const* ctx, std::string updater, bst_target_t n_targets) {
    this->SynthesizeData(n_targets);

    {
      bst_node_t n_nodes = this->Update(ctx, updater, 0.01);
      // This is not strictly verified, meaning the number `2` is whatever GPU_Hist retured
      // when writing this test, and only used for testing larger gamma (below) does prevent
      // building tree.
      ASSERT_EQ(n_nodes, 2);
    }
    {
      bst_node_t n_nodes = this->Update(ctx, updater, 100.0);
      // No new nodes with gamma == 100.
      ASSERT_EQ(n_nodes, static_cast<decltype(n_nodes)>(0));
    }
  }
};

/* Exact tree method requires a pruner as an additional updater, so not tested here. */

TEST_F(TestMinSplitLoss, Approx) {
  Context ctx;
  this->RunTest(&ctx, "grow_histmaker", 1u);
}

TEST_F(TestMinSplitLoss, Hist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", 1u);
}

TEST_F(TestMinSplitLoss, MultiHist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", 2u);
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestMinSplitLoss, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist", 1u);
}

TEST_F(TestMinSplitLoss, GpuMultiHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist", 2u);
}

TEST_F(TestMinSplitLoss, GpuApprox) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_approx", 1u);
}
#endif  // defined(XGBOOST_USE_CUDA)

class TestRegularization : public ::testing::Test {
 public:
  void Run(Context const* ctx, std::string const& updater, std::string p, bst_target_t n_targets) {
    bst_idx_t n_samples = 4096;
    bst_feature_t n_features = 32;
    auto p_fmat = RandomDataGenerator(n_samples, n_features, .0f)
                      .Seed(3)
                      .Targets(n_targets)
                      .GenerateDMatrix(true);
    auto gpairs = GenerateRandomGradients(ctx, n_samples, n_targets);

    RegTree tree_0{static_cast<bst_target_t>(gpairs.gpair.Shape(1)),
                   static_cast<bst_target_t>(p_fmat->Info().num_col_)};
    BuildTree(ctx, p_fmat.get(), &gpairs, updater, Args{{p, "0.0"}}, &tree_0);
    // not exact, just checking the tree can be built
    if (n_targets > 1) {
      ASSERT_GE(tree_0.NumNodes(), 40);
    } else {
      ASSERT_GE(tree_0.NumNodes(), 50);
    }

    RegTree tree_1{static_cast<bst_target_t>(gpairs.gpair.Shape(1)),
                   static_cast<bst_target_t>(p_fmat->Info().num_col_)};
    BuildTree(ctx, p_fmat.get(), &gpairs, updater, Args{{p, "1024.0"}}, &tree_1);
    ASSERT_EQ(tree_1.NumNodes(), 1);
  }
};

class TestLambda : public TestRegularization {
 public:
  void RunTest(Context const* ctx, std::string const& updater, bst_target_t n_targets) {
    this->Run(ctx, updater, "lambda", n_targets);
  }
};

TEST_F(TestLambda, Hist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", 1u);
}

TEST_F(TestLambda, MultiHist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", 3u);
}

TEST_F(TestLambda, Approx) {
  Context ctx;
  this->RunTest(&ctx, "grow_histmaker", 1u);
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestLambda, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist", 1u);
}

TEST_F(TestLambda, GpuMultiHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist", 3u);
}

TEST_F(TestLambda, GpuApprox) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_approx", 1u);
}
#endif  // defined(XGBOOST_USE_CUDA)

class TestAlpha : public TestRegularization {
 public:
  void RunTest(Context const* ctx, std::string const& updater, bst_target_t n_targets) {
    this->Run(ctx, updater, "alpha", n_targets);
  }
};

TEST_F(TestAlpha, Hist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", 1u);
}

TEST_F(TestAlpha, MultiHist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", 3u);
}

TEST_F(TestAlpha, Approx) {
  Context ctx;
  this->RunTest(&ctx, "grow_histmaker", 1u);
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestAlpha, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist", 1u);
}

TEST_F(TestAlpha, GpuMultiHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist", 3u);
}

TEST_F(TestAlpha, GpuApprox) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_approx", 1u);
}
#endif  // defined(XGBOOST_USE_CUDA)

class TestMaxDeltaStep : public ::testing::Test {
 public:
  void RunTest(Context const* ctx, std::string const& updater, bst_target_t n_targets) {
    bst_idx_t n_samples = 4096;
    bst_feature_t n_features = 32;
    auto p_fmat = RandomDataGenerator(n_samples, n_features, .0f)
                      .Seed(3)
                      .Targets(n_targets)
                      .GenerateDMatrix(true);
    auto gpairs = GenerateRandomGradients(ctx, n_samples, n_targets);

    RegTree tree_0{static_cast<bst_target_t>(gpairs.gpair.Shape(1)),
                   static_cast<bst_target_t>(p_fmat->Info().num_col_)};
    BuildTree(ctx, p_fmat.get(), &gpairs, updater, Args{{"max_delta_step", std::to_string(0.5)}}, &tree_0);
    ASSERT_EQ(tree_0.NumNodes(), 1);
  }
};

TEST_F(TestMaxDeltaStep, Hist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", 1u);
}

TEST_F(TestMaxDeltaStep, MultiHist) {
  Context ctx;
  this->RunTest(&ctx, "grow_quantile_histmaker", 3u);
}

TEST_F(TestMaxDeltaStep, Approx) {
  Context ctx;
  this->RunTest(&ctx, "grow_histmaker", 1u);
}

#if defined(XGBOOST_USE_CUDA)
TEST_F(TestMaxDeltaStep, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist", 1u);
}

TEST_F(TestMaxDeltaStep, GpuMultiHist) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_hist", 3u);
}

TEST_F(TestMaxDeltaStep, GpuApprox) {
  auto ctx = MakeCUDACtx(0);
  this->RunTest(&ctx, "grow_gpu_approx", 1u);
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
