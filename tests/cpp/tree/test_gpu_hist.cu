/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>                // for Args
#include <xgboost/context.h>             // for Context
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/json.h>                // for Jons
#include <xgboost/task.h>                // for ObjInfo
#include <xgboost/tree_model.h>          // for RegTree
#include <xgboost/tree_updater.h>        // for TreeUpdater

#include <memory>  // for unique_ptr
#include <string>  // for string
#include <vector>  // for vector

#include "../../../src/common/random.h"      // for GlobalRandom
#include "../../../src/data/ellpack_page.h"  // for EllpackPage
#include "../../../src/tree/param.h"         // for TrainParam
#include "../collective/test_worker.h"       // for BaseMGPUTest
#include "../filesystem.h"                   // dmlc::TemporaryDirectory
#include "../helpers.h"

namespace xgboost::tree {
void UpdateTree(Context const* ctx, linalg::Matrix<GradientPair>* gpair, DMatrix* dmat,
                size_t gpu_page_size, RegTree* tree, HostDeviceVector<bst_float>* preds,
                float subsample = 1.0f, const std::string& sampling_method = "uniform",
                int max_bin = 2) {
  if (gpu_page_size > 0) {
    // Loop over the batches and count the records
    int64_t batch_count = 0;
    int64_t row_count = 0;
    for (const auto& batch : dmat->GetBatches<EllpackPage>(
             ctx, BatchParam{max_bin, TrainParam::DftSparseThreshold()})) {
      EXPECT_LT(batch.Size(), dmat->Info().num_row_);
      batch_count++;
      row_count += batch.Size();
    }
    EXPECT_GE(batch_count, 2);
    EXPECT_EQ(row_count, dmat->Info().num_row_);
  }

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
  hist_maker->Configure(Args{});

  std::vector<HostDeviceVector<bst_node_t>> position(1);
  hist_maker->Update(&param, gpair, dmat, common::Span<HostDeviceVector<bst_node_t>>{position},
                     {tree});
  auto cache = linalg::MakeTensorView(ctx, preds->DeviceSpan(), preds->Size(), 1);
  hist_maker->UpdatePredictionCache(dmat, cache);
}

TEST(GpuHist, UniformSampling) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 2;
  constexpr float kSubsample = 0.9999;
  common::GlobalRandom().seed(1994);

  // Create an in-memory DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, 0, true));

  linalg::Matrix<GradientPair> gpair({kRows}, Context{}.MakeCUDA().Device());
  gpair.Data()->Copy(GenerateRandomGradients(kRows));

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, DeviceOrd::CUDA(0));
  Context ctx(MakeCUDACtx(0));
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree, &preds, 1.0, "uniform", kRows);
  // Build another tree using sampling.
  RegTree tree_sampling;
  HostDeviceVector<bst_float> preds_sampling(kRows, 0.0, DeviceOrd::CUDA(0));
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree_sampling, &preds_sampling, kSubsample, "uniform",
             kRows);

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

  // Create an in-memory DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, 0, true));

  linalg::Matrix<GradientPair> gpair({kRows}, MakeCUDACtx(0).Device());
  gpair.Data()->Copy(GenerateRandomGradients(kRows));

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, DeviceOrd::CUDA(0));
  Context ctx(MakeCUDACtx(0));
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree, &preds, 1.0, "uniform", kRows);

  // Build another tree using sampling.
  RegTree tree_sampling;
  HostDeviceVector<bst_float> preds_sampling(kRows, 0.0, DeviceOrd::CUDA(0));
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree_sampling, &preds_sampling, kSubsample,
             "gradient_based", kRows);

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
  constexpr size_t kPageSize = 1024;

  dmlc::TemporaryDirectory tmpdir;

  // Create a DMatrix with multiple batches.
  std::unique_ptr<DMatrix> dmat_ext(
      CreateSparsePageDMatrix(kRows, kCols, kRows / kPageSize, tmpdir.path + "/cache"));

  // Create a single batch DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrix(kRows, kCols, 1, tmpdir.path + "/cache"));

  Context ctx(MakeCUDACtx(0));
  linalg::Matrix<GradientPair> gpair({kRows}, ctx.Device());
  gpair.Data()->Copy(GenerateRandomGradients(kRows));

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, DeviceOrd::CUDA(0));
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree, &preds, 1.0, "uniform", kRows);
  // Build another tree using multiple ELLPACK pages.
  RegTree tree_ext;
  HostDeviceVector<bst_float> preds_ext(kRows, 0.0, DeviceOrd::CUDA(0));
  UpdateTree(&ctx, &gpair, dmat_ext.get(), kPageSize, &tree_ext, &preds_ext, 1.0, "uniform", kRows);

  // Make sure the predictions are the same.
  auto preds_h = preds.ConstHostVector();
  auto preds_ext_h = preds_ext.ConstHostVector();
  for (size_t i = 0; i < kRows; i++) {
    EXPECT_NEAR(preds_h[i], preds_ext_h[i], 1e-6);
  }
}

TEST(GpuHist, ExternalMemoryWithSampling) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 2;
  constexpr size_t kPageSize = 1024;
  constexpr float kSubsample = 0.5;
  const std::string kSamplingMethod = "gradient_based";
  common::GlobalRandom().seed(0);

  dmlc::TemporaryDirectory tmpdir;

  // Create a single batch DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrix(kRows, kCols, 1, tmpdir.path + "/cache"));

  // Create a DMatrix with multiple batches.
  std::unique_ptr<DMatrix> dmat_ext(
      CreateSparsePageDMatrix(kRows, kCols, kRows / kPageSize, tmpdir.path + "/cache"));

  Context ctx(MakeCUDACtx(0));
  linalg::Matrix<GradientPair> gpair({kRows}, ctx.Device());
  gpair.Data()->Copy(GenerateRandomGradients(kRows));

  // Build a tree using the in-memory DMatrix.
  auto rng = common::GlobalRandom();

  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, DeviceOrd::CUDA(0));
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree, &preds, kSubsample, kSamplingMethod, kRows);

  // Build another tree using multiple ELLPACK pages.
  common::GlobalRandom() = rng;
  RegTree tree_ext;
  HostDeviceVector<bst_float> preds_ext(kRows, 0.0, DeviceOrd::CUDA(0));
  UpdateTree(&ctx, &gpair, dmat_ext.get(), kPageSize, &tree_ext, &preds_ext, kSubsample,
             kSamplingMethod, kRows);

  // Make sure the predictions are the same.
  auto preds_h = preds.ConstHostVector();
  auto preds_ext_h = preds_ext.ConstHostVector();
  for (size_t i = 0; i < kRows; i++) {
    ASSERT_NEAR(preds_h[i], preds_ext_h[i], 1e-3);
  }
}

TEST(GpuHist, ConfigIO) {
  Context ctx(MakeCUDACtx(0));
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
  Context ctx(MakeCUDACtx(0));
  size_t constexpr kRows = 16;
  size_t constexpr kCols = 4;
  auto p_mat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_mat}));
  learner->SetParam("max_depth", "32");
  learner->Configure();

  ASSERT_THROW({learner->UpdateOneIter(0, p_mat);}, dmlc::Error);
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

namespace {
RegTree GetApproxTree(Context const* ctx, DMatrix* dmat) {
  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> approx_maker{TreeUpdater::Create("grow_gpu_approx", ctx, &task)};
  approx_maker->Configure(Args{});

  TrainParam param;
  param.UpdateAllowUnknown(Args{});

  linalg::Matrix<GradientPair> gpair({dmat->Info().num_row_}, ctx->Device());
  gpair.Data()->Copy(GenerateRandomGradients(dmat->Info().num_row_));

  std::vector<HostDeviceVector<bst_node_t>> position(1);
  RegTree tree;
  approx_maker->Update(&param, &gpair, dmat, common::Span<HostDeviceVector<bst_node_t>>{position},
                       {&tree});
  return tree;
}

void VerifyApproxColumnSplit(bst_idx_t rows, bst_feature_t cols, RegTree const& expected_tree) {
  auto ctx = MakeCUDACtx(DistGpuIdx());

  auto Xy = RandomDataGenerator{rows, cols, 0}.GenerateDMatrix(true);
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  std::unique_ptr<DMatrix> sliced{Xy->SliceCol(world_size, rank)};

  RegTree tree = GetApproxTree(&ctx, sliced.get());

  Json json{Object{}};
  tree.SaveModel(&json);
  Json expected_json{Object{}};
  expected_tree.SaveModel(&expected_json);
  ASSERT_EQ(json, expected_json);
}
}  // anonymous namespace

class MGPUApproxTest : public collective::BaseMGPUTest {};

TEST_F(MGPUApproxTest, GPUApproxColumnSplit) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;

  Context ctx(MakeCUDACtx(0));
  auto dmat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true);
  RegTree expected_tree = GetApproxTree(&ctx, dmat.get());

  this->DoTest([&] { VerifyApproxColumnSplit(kRows, kCols, expected_tree); }, true);
  this->DoTest([&] { VerifyApproxColumnSplit(kRows, kCols, expected_tree); }, false);
}
}  // namespace xgboost::tree
