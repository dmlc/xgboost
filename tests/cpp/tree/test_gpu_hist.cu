/**
 * Copyright 2017-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <xgboost/base.h>

#include <random>
#include <string>
#include <vector>

#include "../../../src/common/common.h"
#include "../../../src/data/sparse_page_source.h"
#include "../../../src/tree/constraints.cuh"
#include "../../../src/tree/param.h"  // for TrainParam
#include "../../../src/tree/updater_gpu_common.cuh"
#include "../../../src/tree/updater_gpu_hist.cu"
#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"
#include "../histogram_helpers.h"
#include "xgboost/context.h"
#include "xgboost/json.h"

namespace xgboost::tree {
TEST(GpuHist, DeviceHistogram) {
  // Ensures that node allocates correctly after reaching `kStopGrowingSize`.
  dh::safe_cuda(cudaSetDevice(0));
  constexpr size_t kNBins = 128;
  constexpr int kNNodes = 4;
  constexpr size_t kStopGrowing = kNNodes * kNBins * 2u;
  DeviceHistogramStorage<kStopGrowing> histogram;
  histogram.Init(0, kNBins);
  for (int i = 0; i < kNNodes; ++i) {
    histogram.AllocateHistograms({i});
  }
  histogram.Reset();
  ASSERT_EQ(histogram.Data().size(), kStopGrowing);

  // Use allocated memory but do not erase nidx_map.
  for (int i = 0; i < kNNodes; ++i) {
    histogram.AllocateHistograms({i});
  }
  for (int i = 0; i < kNNodes; ++i) {
    ASSERT_TRUE(histogram.HistogramExists(i));
  }

  // Add two new nodes
  histogram.AllocateHistograms({kNNodes});
  histogram.AllocateHistograms({kNNodes + 1});

  // Old cached nodes should still exist
  for (int i = 0; i < kNNodes; ++i) {
    ASSERT_TRUE(histogram.HistogramExists(i));
  }

  // Should be deleted
  ASSERT_FALSE(histogram.HistogramExists(kNNodes));
  // Most recent node should exist
  ASSERT_TRUE(histogram.HistogramExists(kNNodes + 1));

  // Add same node again - should fail
  EXPECT_ANY_THROW(histogram.AllocateHistograms({kNNodes + 1}););
}

std::vector<GradientPairPrecise> GetHostHistGpair() {
  // 24 bins, 3 bins for each feature (column).
  std::vector<GradientPairPrecise> hist_gpair = {
    {0.8314f, 0.7147f}, {1.7989f, 3.7312f}, {3.3846f, 3.4598f},
    {2.9277f, 3.5886f}, {1.8429f, 2.4152f}, {1.2443f, 1.9019f},
    {1.6380f, 2.9174f}, {1.5657f, 2.5107f}, {2.8111f, 2.4776f},
    {2.1322f, 3.0651f}, {3.2927f, 3.8540f}, {0.5899f, 0.9866f},
    {1.5185f, 1.6263f}, {2.0686f, 3.1844f}, {2.4278f, 3.0950f},
    {1.5105f, 2.1403f}, {2.6922f, 4.2217f}, {1.8122f, 1.5437f},
    {0.0000f, 0.0000f}, {4.3245f, 5.7955f}, {1.6903f, 2.1103f},
    {2.4012f, 4.4754f}, {3.6136f, 3.4303f}, {0.0000f, 0.0000f}
  };
  return hist_gpair;
}

template <typename GradientSumT>
void TestBuildHist(bool use_shared_memory_histograms) {
  int const kNRows = 16, kNCols = 8;

  TrainParam param;
  Args args{
      {"max_depth", "6"},
      {"max_leaves", "0"},
  };
  param.Init(args);

  auto page = BuildEllpackPage(kNRows, kNCols);
  BatchParam batch_param{};
  Context ctx{CreateEmptyGenericParam(0)};
  GPUHistMakerDevice<GradientSumT> maker(&ctx, page.get(), {}, kNRows, param, kNCols, kNCols,
                                         batch_param);
  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(0.0f, 1.0f);
  HostDeviceVector<GradientPair> gpair(kNRows);
  for (auto &gp : gpair.HostVector()) {
    bst_float grad = dist(&gen);
    bst_float hess = dist(&gen);
    gp = GradientPair(grad, hess);
  }
  gpair.SetDevice(0);

  thrust::host_vector<common::CompressedByteT> h_gidx_buffer (page->gidx_buffer.HostVector());
  maker.row_partitioner.reset(new RowPartitioner(0, kNRows));
  maker.hist.AllocateHistograms({0});
  maker.gpair = gpair.DeviceSpan();
  maker.quantiser.reset(new GradientQuantiser(maker.gpair));

  BuildGradientHistogram(ctx.CUDACtx(), page->GetDeviceAccessor(0),
                         maker.feature_groups->DeviceAccessor(0), gpair.DeviceSpan(),
                         maker.row_partitioner->GetRows(0), maker.hist.GetNodeHistogram(0),
                         *maker.quantiser, !use_shared_memory_histograms);

  DeviceHistogramStorage<>& d_hist = maker.hist;

  auto node_histogram = d_hist.GetNodeHistogram(0);
  // d_hist.data stored in float, not gradient pair
  thrust::host_vector<GradientPairInt64> h_result (node_histogram.size());
  dh::safe_cuda(cudaMemcpy(h_result.data(), node_histogram.data(), node_histogram.size_bytes(),
                           cudaMemcpyDeviceToHost));

  std::vector<GradientPairPrecise> solution = GetHostHistGpair();
  for (size_t i = 0; i < h_result.size(); ++i) {
    auto result = maker.quantiser->ToFloatingPoint(h_result[i]);
    EXPECT_NEAR(result.GetGrad(), solution[i].GetGrad(), 0.01f);
    EXPECT_NEAR(result.GetHess(), solution[i].GetHess(), 0.01f);
  }
}

TEST(GpuHist, BuildHistGlobalMem) {
  TestBuildHist<GradientPairPrecise>(false);
}

TEST(GpuHist, BuildHistSharedMem) {
  TestBuildHist<GradientPairPrecise>(true);
}

HistogramCutsWrapper GetHostCutMatrix () {
  HistogramCutsWrapper cmat;
  cmat.SetPtrs({0, 3, 6, 9, 12, 15, 18, 21, 24});
  cmat.SetMins({0.1f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, 0.2f, 0.2f});
  // 24 cut fields, 3 cut fields for each feature (column).
  // Each row of the cut represents the cuts for a data column.
  cmat.SetValues({0.30f, 0.67f, 1.64f,
              0.32f, 0.77f, 1.95f,
              0.29f, 0.70f, 1.80f,
              0.32f, 0.75f, 1.85f,
              0.18f, 0.59f, 1.69f,
              0.25f, 0.74f, 2.00f,
              0.26f, 0.74f, 1.98f,
              0.26f, 0.71f, 1.83f});
  return cmat;
}

inline GradientQuantiser DummyRoundingFactor() {
  thrust::device_vector<GradientPair> gpair(1);
  gpair[0] = {1000.f, 1000.f};  // Tests should not exceed sum of 1000
  return GradientQuantiser(dh::ToSpan(gpair));
}

void TestHistogramIndexImpl() {
  // Test if the compressed histogram index matches when using a sparse
  // dmatrix with and without using external memory

  int constexpr kNRows = 1000, kNCols = 10;

  // Build 2 matrices and build a histogram maker with that
  Context ctx(CreateEmptyGenericParam(0));
  ObjInfo task{ObjInfo::kRegression};
  tree::GPUHistMaker hist_maker{&ctx, &task}, hist_maker_ext{&ctx, &task};
  std::unique_ptr<DMatrix> hist_maker_dmat(
    CreateSparsePageDMatrixWithRC(kNRows, kNCols, 0, true));

  dmlc::TemporaryDirectory tempdir;
  std::unique_ptr<DMatrix> hist_maker_ext_dmat(
    CreateSparsePageDMatrixWithRC(kNRows, kNCols, 128UL, true, tempdir));

  Args training_params = {{"max_depth", "10"}, {"max_leaves", "0"}};
  TrainParam param;
  param.UpdateAllowUnknown(training_params);

  hist_maker.Configure(training_params);
  hist_maker.InitDataOnce(&param, hist_maker_dmat.get());
  hist_maker_ext.Configure(training_params);
  hist_maker_ext.InitDataOnce(&param, hist_maker_ext_dmat.get());

  // Extract the device maker from the histogram makers and from that its compressed
  // histogram index
  const auto &maker = hist_maker.maker;
  auto grad = GenerateRandomGradients(kNRows);
  grad.SetDevice(0);
  maker->Reset(&grad, hist_maker_dmat.get(), kNCols);
  std::vector<common::CompressedByteT> h_gidx_buffer(maker->page->gidx_buffer.HostVector());

  const auto &maker_ext = hist_maker_ext.maker;
  maker_ext->Reset(&grad, hist_maker_ext_dmat.get(), kNCols);
  std::vector<common::CompressedByteT> h_gidx_buffer_ext(maker_ext->page->gidx_buffer.HostVector());

  ASSERT_EQ(maker->page->Cuts().TotalBins(), maker_ext->page->Cuts().TotalBins());
  ASSERT_EQ(maker->page->gidx_buffer.Size(), maker_ext->page->gidx_buffer.Size());
}

TEST(GpuHist, TestHistogramIndex) {
  TestHistogramIndexImpl();
}

void UpdateTree(Context const* ctx, HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
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
  tree::GPUHistMaker hist_maker{ctx, &task};

  std::vector<HostDeviceVector<bst_node_t>> position(1);
  hist_maker.Update(&param, gpair, dmat, common::Span<HostDeviceVector<bst_node_t>>{position},
                    {tree});
  auto cache = linalg::MakeTensorView(ctx, preds->DeviceSpan(), preds->Size(), 1);
  hist_maker.UpdatePredictionCache(dmat, cache);
}

TEST(GpuHist, UniformSampling) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 2;
  constexpr float kSubsample = 0.9999;
  common::GlobalRandom().seed(1994);

  // Create an in-memory DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, 0, true));

  auto gpair = GenerateRandomGradients(kRows);

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, 0);
  Context ctx(CreateEmptyGenericParam(0));
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree, &preds, 1.0, "uniform", kRows);
  // Build another tree using sampling.
  RegTree tree_sampling;
  HostDeviceVector<bst_float> preds_sampling(kRows, 0.0, 0);
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

  auto gpair = GenerateRandomGradients(kRows);

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, 0);
  Context ctx(CreateEmptyGenericParam(0));
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree, &preds, 1.0, "uniform", kRows);

  // Build another tree using sampling.
  RegTree tree_sampling;
  HostDeviceVector<bst_float> preds_sampling(kRows, 0.0, 0);
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

  auto gpair = GenerateRandomGradients(kRows);

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  Context ctx(CreateEmptyGenericParam(0));
  HostDeviceVector<bst_float> preds(kRows, 0.0, 0);
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree, &preds, 1.0, "uniform", kRows);
  // Build another tree using multiple ELLPACK pages.
  RegTree tree_ext;
  HostDeviceVector<bst_float> preds_ext(kRows, 0.0, 0);
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

  auto gpair = GenerateRandomGradients(kRows);

  // Build a tree using the in-memory DMatrix.
  auto rng = common::GlobalRandom();

  Context ctx(CreateEmptyGenericParam(0));
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, 0);
  UpdateTree(&ctx, &gpair, dmat.get(), 0, &tree, &preds, kSubsample, kSamplingMethod, kRows);

  // Build another tree using multiple ELLPACK pages.
  common::GlobalRandom() = rng;
  RegTree tree_ext;
  HostDeviceVector<bst_float> preds_ext(kRows, 0.0, 0);
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
  Context ctx(CreateEmptyGenericParam(0));
  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_gpu_hist", &ctx, &task)};
  updater->Configure(Args{});

  Json j_updater { Object() };
  updater->SaveConfig(&j_updater);
  ASSERT_TRUE(IsA<Object>(j_updater["gpu_hist_train_param"]));
  updater->LoadConfig(j_updater);

  Json j_updater_roundtrip { Object() };
  updater->SaveConfig(&j_updater_roundtrip);
  ASSERT_TRUE(IsA<Object>(j_updater_roundtrip["gpu_hist_train_param"]));

  ASSERT_EQ(j_updater, j_updater_roundtrip);
}

TEST(GpuHist, MaxDepth) {
  Context ctx(CreateEmptyGenericParam(0));
  size_t constexpr kRows = 16;
  size_t constexpr kCols = 4;
  auto p_mat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_mat}));
  learner->SetParam("max_depth", "32");
  learner->Configure();

  ASSERT_THROW({learner->UpdateOneIter(0, p_mat);}, dmlc::Error);
}
}  // namespace xgboost::tree
