/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <dmlc/filesystem.h>
#include <xgboost/base.h>
#include <random>
#include <string>
#include <vector>

#include "../helpers.h"
#include "../histogram_helpers.h"

#include "xgboost/json.h"
#include "../../../src/data/sparse_page_source.h"
#include "../../../src/tree/updater_gpu_hist.cu"
#include "../../../src/tree/updater_gpu_common.cuh"
#include "../../../src/common/common.h"
#include "../../../src/tree/constraints.cuh"

namespace xgboost {
namespace tree {

TEST(GpuHist, DeviceHistogram) {
  // Ensures that node allocates correctly after reaching `kStopGrowingSize`.
  dh::safe_cuda(cudaSetDevice(0));
  constexpr size_t kNBins = 128;
  constexpr size_t kNNodes = 4;
  constexpr size_t kStopGrowing = kNNodes * kNBins * 2u;
  DeviceHistogram<GradientPairPrecise, kStopGrowing> histogram;
  histogram.Init(0, kNBins);
  for (size_t i = 0; i < kNNodes; ++i) {
    histogram.AllocateHistogram(i);
  }
  histogram.Reset();
  ASSERT_EQ(histogram.Data().size(), kStopGrowing);

  // Use allocated memory but do not erase nidx_map.
  for (size_t i = 0; i < kNNodes; ++i) {
    histogram.AllocateHistogram(i);
  }
  for (size_t i = 0; i < kNNodes; ++i) {
    ASSERT_TRUE(histogram.HistogramExists(i));
  }

  // Erase existing nidx_map.
  for (size_t i = kNNodes; i < kNNodes * 2; ++i) {
    histogram.AllocateHistogram(i);
  }
  for (size_t i = 0; i < kNNodes; ++i) {
    ASSERT_FALSE(histogram.HistogramExists(i));
  }
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
  std::vector<std::pair<std::string, std::string>> args {
    {"max_depth", "6"},
    {"max_leaves", "0"},
  };
  param.Init(args);
  auto page = BuildEllpackPage(kNRows, kNCols);
  BatchParam batch_param{};
  GPUHistMakerDevice<GradientSumT> maker(0, page.get(), kNRows, param, kNCols, kNCols,
                                         true, batch_param);
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
  maker.hist.AllocateHistogram(0);
  maker.gpair = gpair.DeviceSpan();

  maker.BuildHist(0);
  DeviceHistogram<GradientSumT> d_hist = maker.hist;

  auto node_histogram = d_hist.GetNodeHistogram(0);
  // d_hist.data stored in float, not gradient pair
  thrust::host_vector<GradientSumT> h_result (d_hist.Data().size() / 2);
  size_t data_size =
      sizeof(GradientSumT) /
      (sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT));
  data_size *= d_hist.Data().size();
  dh::safe_cuda(cudaMemcpy(h_result.data(), node_histogram.data(), data_size,
                           cudaMemcpyDeviceToHost));

  std::vector<GradientPairPrecise> solution = GetHostHistGpair();
  std::cout << std::fixed;
  for (size_t i = 0; i < h_result.size(); ++i) {
    EXPECT_NEAR(h_result[i].GetGrad(), solution[i].GetGrad(), 0.01f);
    EXPECT_NEAR(h_result[i].GetHess(), solution[i].GetHess(), 0.01f);
  }
}

TEST(GpuHist, BuildHistGlobalMem) {
  TestBuildHist<GradientPairPrecise>(false);
  TestBuildHist<GradientPair>(false);
}

TEST(GpuHist, BuildHistSharedMem) {
  TestBuildHist<GradientPairPrecise>(true);
  TestBuildHist<GradientPair>(true);
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

// TODO(trivialfis): This test is over simplified.
TEST(GpuHist, EvaluateRootSplit) {
  constexpr int kNRows = 16;
  constexpr int kNCols = 8;

  TrainParam param;

  std::vector<std::pair<std::string, std::string>> args {
    {"max_depth", "1"},
  {"max_leaves", "0"},

  // Disable all other parameters.
  {"colsample_bynode", "1"},
  {"colsample_bylevel", "1"},
  {"colsample_bytree", "1"},
  {"min_child_weight", "0.01"},
  {"reg_alpha", "0"},
  {"reg_lambda", "0"},
  {"max_delta_step", "0"}
  };
  param.Init(args);
  for (size_t i = 0; i < kNCols; ++i) {
    param.monotone_constraints.emplace_back(0);
  }

  int max_bins = 4;

  // Initialize GPUHistMakerDevice
  auto page = BuildEllpackPage(kNRows, kNCols);
  BatchParam batch_param{};
  GPUHistMakerDevice<GradientPairPrecise>
    maker(0, page.get(), kNRows, param, kNCols, kNCols, true, batch_param);
  // Initialize GPUHistMakerDevice::node_sum_gradients
  maker.node_sum_gradients = {};

  // Initialize GPUHistMakerDevice::cut
  auto cmat = GetHostCutMatrix();

  // Copy cut matrix to device.
  page->Cuts() = cmat;
  maker.monotone_constraints = param.monotone_constraints;

  // Initialize GPUHistMakerDevice::hist
  maker.hist.Init(0, (max_bins - 1) * kNCols);
  maker.hist.AllocateHistogram(0);
  // Each row of hist_gpair represents gpairs for one feature.
  // Each entry represents a bin.
  std::vector<GradientPairPrecise> hist_gpair = GetHostHistGpair();
  std::vector<bst_float> hist;
  for (auto pair : hist_gpair) {
    hist.push_back(pair.GetGrad());
    hist.push_back(pair.GetHess());
  }

  ASSERT_EQ(maker.hist.Data().size(), hist.size());
  thrust::copy(hist.begin(), hist.end(),
    maker.hist.Data().begin());

  maker.column_sampler.Init(kNCols,
    param.colsample_bynode,
    param.colsample_bylevel,
    param.colsample_bytree,
    false);

  RegTree tree;
  MetaInfo info;
  info.num_row_ = kNRows;
  info.num_col_ = kNCols;

  maker.node_value_constraints.resize(1);
  maker.node_value_constraints[0].lower_bound = -1.0;
  maker.node_value_constraints[0].upper_bound = 1.0;

  DeviceSplitCandidate res = maker.EvaluateRootSplit({6.4f, 12.8f});

  ASSERT_EQ(res.findex, 7);
  ASSERT_NEAR(res.fvalue, 0.26, xgboost::kRtEps);
}

void TestHistogramIndexImpl() {
  // Test if the compressed histogram index matches when using a sparse
  // dmatrix with and without using external memory

  int constexpr kNRows = 1000, kNCols = 10;

  // Build 2 matrices and build a histogram maker with that
  tree::GPUHistMakerSpecialised<GradientPairPrecise> hist_maker, hist_maker_ext;
  std::unique_ptr<DMatrix> hist_maker_dmat(
    CreateSparsePageDMatrixWithRC(kNRows, kNCols, 0, true));

  dmlc::TemporaryDirectory tempdir;
  std::unique_ptr<DMatrix> hist_maker_ext_dmat(
    CreateSparsePageDMatrixWithRC(kNRows, kNCols, 128UL, true, tempdir));

  std::vector<std::pair<std::string, std::string>> training_params = {
    {"max_depth", "10"},
    {"max_leaves", "0"}
  };

  GenericParameter generic_param(CreateEmptyGenericParam(0));
  hist_maker.Configure(training_params, &generic_param);
  hist_maker.InitDataOnce(hist_maker_dmat.get());
  hist_maker_ext.Configure(training_params, &generic_param);
  hist_maker_ext.InitDataOnce(hist_maker_ext_dmat.get());

  // Extract the device maker from the histogram makers and from that its compressed
  // histogram index
  const auto &maker = hist_maker.maker;
  std::vector<common::CompressedByteT> h_gidx_buffer(maker->page->gidx_buffer.HostVector());

  const auto &maker_ext = hist_maker_ext.maker;
  std::vector<common::CompressedByteT> h_gidx_buffer_ext(maker_ext->page->gidx_buffer.HostVector());

  ASSERT_EQ(maker->page->Cuts().TotalBins(), maker_ext->page->Cuts().TotalBins());
  ASSERT_EQ(maker->page->gidx_buffer.Size(), maker_ext->page->gidx_buffer.Size());

}

TEST(GpuHist, TestHistogramIndex) {
  TestHistogramIndexImpl();
}

// gamma is an alias of min_split_loss
int32_t TestMinSplitLoss(DMatrix* dmat, float gamma, HostDeviceVector<GradientPair>* gpair) {
  Args args {
    {"max_depth", "1"},
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
    {"gamma", std::to_string(gamma)}
  };

  tree::GPUHistMakerSpecialised<GradientPairPrecise> hist_maker;
  GenericParameter generic_param(CreateEmptyGenericParam(0));
  hist_maker.Configure(args, &generic_param);

  RegTree tree;
  hist_maker.Update(gpair, dmat, {&tree});

  auto n_nodes = tree.NumExtraNodes();
  return n_nodes;
}

TEST(GpuHist, MinSplitLoss) {
  constexpr size_t kRows = 32;
  constexpr size_t kCols = 16;
  constexpr float kSparsity = 0.6;
  auto dmat = RandomDataGenerator(kRows, kCols, kSparsity).Seed(3).GenerateDMatrix();
  auto gpair = GenerateRandomGradients(kRows);

  {
    int32_t n_nodes = TestMinSplitLoss(dmat.get(), 0.01, &gpair);
    // This is not strictly verified, meaning the numeber `2` is whatever GPU_Hist retured
    // when writing this test, and only used for testing larger gamma (below) does prevent
    // building tree.
    ASSERT_EQ(n_nodes, 2);
  }
  {
    int32_t n_nodes = TestMinSplitLoss(dmat.get(), 100.0, &gpair);
    // No new nodes with gamma == 100.
    ASSERT_EQ(n_nodes, static_cast<decltype(n_nodes)>(0));
  }
}

void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
                size_t gpu_page_size, RegTree* tree,
                HostDeviceVector<bst_float>* preds, float subsample = 1.0f,
                const std::string& sampling_method = "uniform",
                int max_bin = 2) {

  if (gpu_page_size > 0) {
    // Loop over the batches and count the records
    int64_t batch_count = 0;
    int64_t row_count = 0;
    for (const auto& batch : dmat->GetBatches<EllpackPage>({0, max_bin, gpu_page_size})) {
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

  tree::GPUHistMakerSpecialised<GradientPairPrecise> hist_maker;
  GenericParameter generic_param(CreateEmptyGenericParam(0));
  generic_param.gpu_page_size = gpu_page_size;
  hist_maker.Configure(args, &generic_param);

  hist_maker.Update(gpair, dmat, {tree});
  hist_maker.UpdatePredictionCache(dmat, preds);
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
  UpdateTree(&gpair, dmat.get(), 0, &tree, &preds, 1.0, "uniform", kRows);
  // Build another tree using sampling.
  RegTree tree_sampling;
  HostDeviceVector<bst_float> preds_sampling(kRows, 0.0, 0);
  UpdateTree(&gpair, dmat.get(), 0, &tree_sampling, &preds_sampling, kSubsample,
             "uniform", kRows);

  // Make sure the predictions are the same.
  auto preds_h = preds.ConstHostVector();
  auto preds_sampling_h = preds_sampling.ConstHostVector();
  for (int i = 0; i < kRows; i++) {
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
  UpdateTree(&gpair, dmat.get(), 0, &tree, &preds, 1.0, "uniform", kRows);

  // Build another tree using sampling.
  RegTree tree_sampling;
  HostDeviceVector<bst_float> preds_sampling(kRows, 0.0, 0);
  UpdateTree(&gpair, dmat.get(), 0, &tree_sampling, &preds_sampling, kSubsample,
             "gradient_based", kRows);

  // Make sure the predictions are the same.
  auto preds_h = preds.ConstHostVector();
  auto preds_sampling_h = preds_sampling.ConstHostVector();
  for (int i = 0; i < kRows; i++) {
    EXPECT_NEAR(preds_h[i], preds_sampling_h[i], 1e-3);
  }
}

TEST(GpuHist, ExternalMemory) {
  constexpr size_t kRows = 4096;
  constexpr size_t kCols = 2;
  constexpr size_t kPageSize = 1024;

  // Create an in-memory DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, 0, true));

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat_ext(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));

  auto gpair = GenerateRandomGradients(kRows);

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, 0);
  UpdateTree(&gpair, dmat.get(), 0, &tree, &preds, 1.0, "uniform", kRows);
  // Build another tree using multiple ELLPACK pages.
  RegTree tree_ext;
  HostDeviceVector<bst_float> preds_ext(kRows, 0.0, 0);
  UpdateTree(&gpair, dmat_ext.get(), kPageSize, &tree_ext, &preds_ext, 1.0, "uniform", kRows);

  // Make sure the predictions are the same.
  auto preds_h = preds.ConstHostVector();
  auto preds_ext_h = preds_ext.ConstHostVector();
  for (int i = 0; i < kRows; i++) {
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

  // Create an in-memory DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, 0, true));

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat_ext(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));

  auto gpair = GenerateRandomGradients(kRows);

  // Build a tree using the in-memory DMatrix.
  RegTree tree;
  HostDeviceVector<bst_float> preds(kRows, 0.0, 0);
  UpdateTree(&gpair, dmat.get(), 0, &tree, &preds, kSubsample, kSamplingMethod,
             kRows);

  // Build another tree using multiple ELLPACK pages.
  RegTree tree_ext;
  HostDeviceVector<bst_float> preds_ext(kRows, 0.0, 0);
  UpdateTree(&gpair, dmat_ext.get(), kPageSize, &tree_ext, &preds_ext,
             kSubsample, kSamplingMethod, kRows);

  // Make sure the predictions are the same.
  auto preds_h = preds.ConstHostVector();
  auto preds_ext_h = preds_ext.ConstHostVector();
  for (int i = 0; i < kRows; i++) {
    EXPECT_NEAR(preds_h[i], preds_ext_h[i], 2e-3);
  }
}

TEST(GpuHist, ConfigIO) {
  GenericParameter generic_param(CreateEmptyGenericParam(0));
  std::unique_ptr<TreeUpdater> updater {TreeUpdater::Create("grow_gpu_hist", &generic_param) };
  updater->Configure(Args{});

  Json j_updater { Object() };
  updater->SaveConfig(&j_updater);
  ASSERT_TRUE(IsA<Object>(j_updater["gpu_hist_train_param"]));
  ASSERT_TRUE(IsA<Object>(j_updater["train_param"]));
  updater->LoadConfig(j_updater);

  Json j_updater_roundtrip { Object() };
  updater->SaveConfig(&j_updater_roundtrip);
  ASSERT_TRUE(IsA<Object>(j_updater_roundtrip["gpu_hist_train_param"]));
  ASSERT_TRUE(IsA<Object>(j_updater_roundtrip["train_param"]));

  ASSERT_EQ(j_updater, j_updater_roundtrip);
}

}  // namespace tree
}  // namespace xgboost
