/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>                // for Args
#include <xgboost/context.h>             // for Context
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/json.h>                // for Json
#include <xgboost/task.h>                // for ObjInfo
#include <xgboost/tree_model.h>          // for RegTree
#include <xgboost/tree_updater.h>        // for TreeUpdater

#include <memory>  // for unique_ptr
#include <string>  // for string
#include <vector>  // for vector

#include "../../../src/common/random.h"  // for GlobalRandom
#include "../../../src/tree/param.h"     // for TrainParam
#include "../collective/test_worker.h"   // for BaseMGPUTest
#include "../helpers.h"

namespace xgboost::tree {
namespace {
void UpdateTree(Context const* ctx, linalg::Matrix<GradientPair>* gpair, DMatrix* dmat,
                RegTree* tree, HostDeviceVector<bst_float>* preds, float subsample,
                const std::string& sampling_method, bst_bin_t max_bin, bool concat_pages) {
  Args args{
      {"max_depth", "2"},
      {"max_bin", std::to_string(max_bin)},
      {"min_child_weight", "0.0"},
      {"reg_alpha", "0"},
      {"reg_lambda", "0"},
      {"subsample", std::to_string(subsample)},
      {"sampling_method", sampling_method},
  };
  TrainParam param;
  param.UpdateAllowUnknown(args);

  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> hist_maker{TreeUpdater::Create("grow_gpu_hist", ctx, &task)};
  if (subsample < 1.0) {
    hist_maker->Configure(Args{{"extmem_single_page", std::to_string(concat_pages)}});
  } else {
    hist_maker->Configure(Args{});
  }

  std::vector<HostDeviceVector<bst_node_t>> position(1);
  hist_maker->Update(&param, gpair, dmat, common::Span<HostDeviceVector<bst_node_t>>{position},
                     {tree});
  auto cache = linalg::MakeTensorView(ctx, preds->DeviceSpan(), preds->Size(), 1);
  if (subsample < 1.0 && !dmat->SingleColBlock() && concat_pages) {
    ASSERT_FALSE(hist_maker->UpdatePredictionCache(dmat, cache));
  } else {
    ASSERT_TRUE(hist_maker->UpdatePredictionCache(dmat, cache));
  }
}
}  // anonymous namespace

TEST(GpuHist, UniformSampling) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 2;
  constexpr float kSubsample = 0.9999;
  common::GlobalRandom().seed(1994);
  auto ctx = MakeCUDACtx(0);

  // Create an in-memory DMatrix.
  auto p_fmat = RandomDataGenerator{kRows, kCols, 0.0f}.GenerateDMatrix(true);
  ASSERT_TRUE(p_fmat->SingleColBlock());

  linalg::Matrix<GradientPair> gpair({kRows}, ctx.Device());
  gpair.Data()->Copy(GenerateRandomGradients(kRows));

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, ctx.Device());
  UpdateTree(&ctx, &gpair, p_fmat.get(), &tree, &preds, 1.0, "uniform", kRows, false);
  // Build another tree using sampling.
  RegTree tree_sampling;
  HostDeviceVector<bst_float> preds_sampling(kRows, 0.0, ctx.Device());
  UpdateTree(&ctx, &gpair, p_fmat.get(), &tree_sampling, &preds_sampling, kSubsample, "uniform",
             kRows, false);

  // Make sure the predictions are the same.
  auto preds_h = preds.ConstHostVector();
  auto preds_sampling_h = preds_sampling.ConstHostVector();
  for (size_t i = 0; i < kRows; i++) {
    EXPECT_NEAR(preds_h[i], preds_sampling_h[i], 1e-8);
  }
}

TEST(GpuHist, GradientBasedSampling) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 2;
  constexpr float kSubsample = 0.9999;
  common::GlobalRandom().seed(1994);
  auto ctx = MakeCUDACtx(0);

  // Create an in-memory DMatrix.
  auto p_fmat = RandomDataGenerator{kRows, kCols, 0.0f}.GenerateDMatrix(true);

  linalg::Matrix<GradientPair> gpair({kRows}, ctx.Device());
  gpair.Data()->Copy(GenerateRandomGradients(kRows));

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, ctx.Device());
  UpdateTree(&ctx, &gpair, p_fmat.get(), &tree, &preds, 1.0, "uniform", kRows, false);

  // Build another tree using sampling.
  RegTree tree_sampling;
  HostDeviceVector<bst_float> preds_sampling(kRows, 0.0, ctx.Device());
  UpdateTree(&ctx, &gpair, p_fmat.get(), &tree_sampling, &preds_sampling, kSubsample,
             "gradient_based", kRows, false);

  // Make sure the predictions are the same.
  auto preds_h = preds.ConstHostVector();
  auto preds_sampling_h = preds_sampling.ConstHostVector();
  for (size_t i = 0; i < kRows; i++) {
    EXPECT_NEAR(preds_h[i], preds_sampling_h[i], 1e-3);
  }
}

TEST(GpuHist, ExternalMemory) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 2;

  // Create a DMatrix with multiple batches.
  auto p_fmat_ext =
      RandomDataGenerator{kRows, kCols, 0.0f}.Batches(4).GenerateSparsePageDMatrix("temp", true);
  ASSERT_FALSE(p_fmat_ext->SingleColBlock());

  // Create a single batch DMatrix.
  auto p_fmat =
      RandomDataGenerator{kRows, kCols, 0.0f}.Batches(1).GenerateSparsePageDMatrix("temp", true);
  ASSERT_TRUE(p_fmat->SingleColBlock());

  auto ctx = MakeCUDACtx(0);
  linalg::Matrix<GradientPair> gpair({kRows}, ctx.Device());
  gpair.Data()->Copy(GenerateRandomGradients(kRows));

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, ctx.Device());
  UpdateTree(&ctx, &gpair, p_fmat.get(), &tree, &preds, 1.0, "uniform", kRows, true);
  // Build another tree using multiple ELLPACK pages.
  RegTree tree_ext;
  HostDeviceVector<bst_float> preds_ext(kRows, 0.0, ctx.Device());
  UpdateTree(&ctx, &gpair, p_fmat_ext.get(), &tree_ext, &preds_ext, 1.0, "uniform", kRows, true);

  // Make sure the predictions are the same.
  auto preds_h = preds.ConstHostVector();
  auto preds_ext_h = preds_ext.ConstHostVector();
  for (size_t i = 0; i < kRows; i++) {
    EXPECT_NEAR(preds_h[i], preds_ext_h[i], 1e-6);
  }
}

TEST(GpuHist, ExternalMemoryWithSampling) {
  constexpr size_t kRows = 4096, kCols = 2;
  constexpr float kSubsample = 0.5;
  const std::string kSamplingMethod = "gradient_based";
  common::GlobalRandom().seed(0);

  auto ctx = MakeCUDACtx(0);

  // Create a single batch DMatrix.
  auto p_fmat = RandomDataGenerator{kRows, kCols, 0.0f}
                    .Device(ctx.Device())
                    .Batches(1)
                    .GenerateSparsePageDMatrix("temp", true);
  ASSERT_TRUE(p_fmat->SingleColBlock());

  // Create a DMatrix with multiple batches.
  auto p_fmat_ext = RandomDataGenerator{kRows, kCols, 0.0f}
                        .Device(ctx.Device())
                        .Batches(4)
                        .GenerateSparsePageDMatrix("temp", true);
  ASSERT_FALSE(p_fmat_ext->SingleColBlock());

  linalg::Matrix<GradientPair> gpair({kRows}, ctx.Device());
  gpair.Data()->Copy(GenerateRandomGradients(kRows));

  // Build a tree using the in-memory DMatrix.
  auto rng = common::GlobalRandom();

  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, ctx.Device());
  UpdateTree(&ctx, &gpair, p_fmat.get(), &tree, &preds, kSubsample, kSamplingMethod, kRows, true);

  // Build another tree using multiple ELLPACK pages.
  common::GlobalRandom() = rng;
  RegTree tree_ext;
  HostDeviceVector<bst_float> preds_ext(kRows, 0.0, ctx.Device());
  UpdateTree(&ctx, &gpair, p_fmat_ext.get(), &tree_ext, &preds_ext, kSubsample, kSamplingMethod,
             kRows, true);

  Json jtree{Object{}};
  Json jtree_ext{Object{}};
  tree.SaveModel(&jtree);
  tree_ext.SaveModel(&jtree_ext);
  ASSERT_EQ(jtree, jtree_ext);
}

TEST(GpuHist, ConfigIO) {
  auto ctx = MakeCUDACtx(0);
  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_gpu_hist", &ctx, &task)};
  updater->Configure(Args{});

  Json j_updater{Object{}};
  updater->SaveConfig(&j_updater);
  ASSERT_TRUE(IsA<Object>(j_updater["hist_train_param"]));
  updater->LoadConfig(j_updater);

  Json j_updater_roundtrip{Object{}};
  updater->SaveConfig(&j_updater_roundtrip);
  ASSERT_TRUE(IsA<Object>(j_updater_roundtrip["hist_train_param"]));

  ASSERT_EQ(j_updater, j_updater_roundtrip);
}

TEST(GpuHist, MaxDepth) {
  auto ctx = MakeCUDACtx(0);
  size_t constexpr kRows = 16;
  size_t constexpr kCols = 4;
  auto p_mat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_mat}));
  learner->SetParam("max_depth", "32");
  learner->Configure();

  ASSERT_THROW({learner->UpdateOneIter(0, p_mat);}, dmlc::Error);
}

TEST(GpuHist, PageConcatConfig) {
  auto ctx = MakeCUDACtx(0);
  bst_idx_t n_samples = 64, n_features = 32;
  auto p_fmat = RandomDataGenerator{n_samples, n_features, 0}.Batches(2).GenerateSparsePageDMatrix(
      "temp", true);

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_fmat}));
  learner->SetParam("device", ctx.DeviceName());
  learner->SetParam("extmem_single_page", "true");
  learner->SetParam("subsample", "0.8");
  learner->Configure();

  learner->UpdateOneIter(0, p_fmat);
  learner->SetParam("extmem_single_page", "false");
  learner->Configure();
  // GPU Hist rebuilds the updater after configuration. Training continues
  learner->UpdateOneIter(1, p_fmat);

  learner->SetParam("extmem_single_page", "true");
  learner->SetParam("subsample", "1.0");
  ASSERT_THAT([&] { learner->UpdateOneIter(2, p_fmat); }, GMockThrow("extmem_single_page"));

  // Throws error on CPU.
  {
    auto learner = std::unique_ptr<Learner>(Learner::Create({p_fmat}));
    learner->SetParam("extmem_single_page", "true");
    ASSERT_THAT([&] { learner->UpdateOneIter(0, p_fmat); }, GMockThrow("extmem_single_page"));
  }
  {
    auto learner = std::unique_ptr<Learner>(Learner::Create({p_fmat}));
    learner->SetParam("extmem_single_page", "true");
    learner->SetParam("tree_method", "approx");
    ASSERT_THAT([&] { learner->UpdateOneIter(0, p_fmat); }, GMockThrow("extmem_single_page"));
  }
}

namespace {
RegTree GetHistTree(Context const* ctx, DMatrix* dmat) {
  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> hist_maker {TreeUpdater::Create("grow_gpu_hist", ctx, &task)};
  hist_maker->Configure(Args{});

  TrainParam param;
  param.UpdateAllowUnknown(Args{});

  linalg::Matrix<GradientPair> gpair({dmat->Info().num_row_}, ctx->Device());
  gpair.Data()->Copy(GenerateRandomGradients(dmat->Info().num_row_));

  std::vector<HostDeviceVector<bst_node_t>> position(1);
  RegTree tree;
  hist_maker->Update(&param, &gpair, dmat, common::Span<HostDeviceVector<bst_node_t>>{position},
                     {&tree});
  return tree;
}

void VerifyHistColumnSplit(bst_idx_t rows, bst_feature_t cols, RegTree const& expected_tree) {
  Context ctx(MakeCUDACtx(GPUIDX));

  auto Xy = RandomDataGenerator{rows, cols, 0}.GenerateDMatrix(true);
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  std::unique_ptr<DMatrix> sliced{Xy->SliceCol(world_size, rank)};

  RegTree tree = GetHistTree(&ctx, sliced.get());

  Json json{Object{}};
  tree.SaveModel(&json);
  Json expected_json{Object{}};
  expected_tree.SaveModel(&expected_json);
  ASSERT_EQ(json, expected_json);
}
}  // anonymous namespace

class MGPUHistTest : public collective::BaseMGPUTest {};

TEST_F(MGPUHistTest, HistColumnSplit) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;

  Context ctx(MakeCUDACtx(0));
  auto dmat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true);
  RegTree expected_tree = GetHistTree(&ctx, dmat.get());

  this->DoTest([&] { VerifyHistColumnSplit(kRows, kCols, expected_tree); }, true);
  this->DoTest([&] { VerifyHistColumnSplit(kRows, kCols, expected_tree); }, false);
}
}  // namespace xgboost::tree
