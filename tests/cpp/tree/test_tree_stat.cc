#include <xgboost/tree_updater.h>
#include <xgboost/tree_model.h>
#include <gtest/gtest.h>

#include "../helpers.h"

namespace xgboost {
class UpdaterTreeStatTest : public ::testing::Test {
 protected:
  std::shared_ptr<DMatrix> p_dmat_;
  HostDeviceVector<GradientPair> gpairs_;
  size_t constexpr static kRows = 10;
  size_t constexpr static kCols = 10;

 protected:
  void SetUp() override {
    p_dmat_ = RandomDataGenerator(kRows, kCols, .5f).GenerateDMatrix(true);
    auto g = GenerateRandomGradients(kRows);
    gpairs_.Resize(kRows);
    gpairs_.Copy(g);
  }

  void RunTest(std::string updater) {
    auto tparam = CreateEmptyGenericParam(0);
    auto up = std::unique_ptr<TreeUpdater>{
        TreeUpdater::Create(updater, &tparam, ObjInfo{ObjInfo::kRegression})};
    up->Configure(Args{});
    RegTree tree;
    tree.param.num_feature = kCols;
    up->Update(&gpairs_, p_dmat_.get(), {&tree});

    tree.WalkTree([&tree](bst_node_t nidx) {
      if (tree[nidx].IsLeaf()) {
        // 1.0 is the default `min_child_weight`.
        CHECK_GE(tree.Stat(nidx).sum_hess, 1.0);
      }
      return true;
    });
  }
};

#if defined(XGBOOST_USE_CUDA)
TEST_F(UpdaterTreeStatTest, GpuHist) {
  this->RunTest("grow_gpu_hist");
}
#endif  // defined(XGBOOST_USE_CUDA)

TEST_F(UpdaterTreeStatTest, Hist) {
  this->RunTest("grow_quantile_histmaker");
}

TEST_F(UpdaterTreeStatTest, Exact) {
  this->RunTest("grow_colmaker");
}

TEST_F(UpdaterTreeStatTest, Approx) {
  this->RunTest("grow_histmaker");
}

class UpdaterEtaTest : public ::testing::Test {
 protected:
  std::shared_ptr<DMatrix> p_dmat_;
  HostDeviceVector<GradientPair> gpairs_;
  size_t constexpr static kRows = 10;
  size_t constexpr static kCols = 10;
  size_t constexpr static kClasses = 10;

  void SetUp() override {
    p_dmat_ = RandomDataGenerator(kRows, kCols, .5f).GenerateDMatrix(true, false, kClasses);
    auto g = GenerateRandomGradients(kRows);
    gpairs_.Resize(kRows);
    gpairs_.Copy(g);
  }

  void RunTest(std::string updater) {
    auto tparam = CreateEmptyGenericParam(0);
    float eta = 0.4;
    auto up_0 = std::unique_ptr<TreeUpdater>{
        TreeUpdater::Create(updater, &tparam, ObjInfo{ObjInfo::kClassification})};
    up_0->Configure(Args{{"eta", std::to_string(eta)}});

    auto up_1 = std::unique_ptr<TreeUpdater>{
        TreeUpdater::Create(updater, &tparam, ObjInfo{ObjInfo::kClassification})};
    up_1->Configure(Args{{"eta", "1.0"}});

    for (size_t iter = 0; iter < 4; ++iter) {
      RegTree tree_0;
      {
        tree_0.param.num_feature = kCols;
        up_0->Update(&gpairs_, p_dmat_.get(), {&tree_0});
      }

      RegTree tree_1;
      {
        tree_1.param.num_feature = kCols;
        up_1->Update(&gpairs_, p_dmat_.get(), {&tree_1});
      }
      tree_0.WalkTree([&](bst_node_t nidx) {
        if (tree_0[nidx].IsLeaf()) {
          EXPECT_NEAR(tree_1[nidx].LeafValue() * eta, tree_0[nidx].LeafValue(), kRtEps);
        }
        return true;
      });
    }
  }
};

TEST_F(UpdaterEtaTest, Hist) { this->RunTest("grow_quantile_histmaker"); }

TEST_F(UpdaterEtaTest, Exact) { this->RunTest("grow_colmaker"); }

TEST_F(UpdaterEtaTest, Approx) { this->RunTest("grow_histmaker"); }

#if defined(XGBOOST_USE_CUDA)
TEST_F(UpdaterEtaTest, GpuHist) { this->RunTest("grow_gpu_hist"); }
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost
