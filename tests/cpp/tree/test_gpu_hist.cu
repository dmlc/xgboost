/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <dmlc/filesystem.h>
#include <xgboost/base.h>
#include <random>
#include <string>
#include <vector>

#include "../helpers.h"
#include "gtest/gtest.h"

#include "../../../src/data/sparse_page_source.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../../../src/tree/updater_gpu_hist.cu"
#include "../../../src/tree/updater_gpu_common.cuh"
#include "../../../src/common/common.h"
#include "../../../src/tree/constraints.cuh"

namespace xgboost {
namespace tree {

TEST(GpuHist, DeviceHistogram) {
  // Ensures that node allocates correctly after reaching `kStopGrowingSize`.
  dh::SaveCudaContext{
    [&]() {
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
  };
}

namespace {
class HistogramCutsWrapper : public common::HistogramCuts {
 public:
  using SuperT = common::HistogramCuts;
  void SetValues(std::vector<float> cuts) {
    SuperT::cut_values_ = cuts;
  }
  void SetPtrs(std::vector<uint32_t> ptrs) {
    SuperT::cut_ptrs_ = ptrs;
  }
  void SetMins(std::vector<float> mins) {
    SuperT::min_vals_ = mins;
  }
};
}  //  anonymous namespace

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
  GPUHistMakerDevice<GradientSumT> maker(0, page.get(), kNRows, param, kNCols, kNCols);
  maker.InitHistogram();

  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(0.0f, 1.0f);
  std::vector<GradientPair> h_gpair(kNRows);
  for (auto &gpair : h_gpair) {
    bst_float grad = dist(&gen);
    bst_float hess = dist(&gen);
    gpair = GradientPair(grad, hess);
  }

  thrust::host_vector<common::CompressedByteT> h_gidx_buffer (page->gidx_buffer.size());

  common::CompressedByteT* d_gidx_buffer_ptr = page->gidx_buffer.data();
  dh::safe_cuda(cudaMemcpy(h_gidx_buffer.data(), d_gidx_buffer_ptr,
                           sizeof(common::CompressedByteT) * page->gidx_buffer.size(),
                           cudaMemcpyDeviceToHost));

  maker.row_partitioner.reset(new RowPartitioner(0, kNRows));
  maker.hist.AllocateHistogram(0);
  dh::CopyVectorToDeviceSpan(maker.gpair, h_gpair);

  maker.use_shared_memory_histograms = use_shared_memory_histograms;
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
TEST(GpuHist, EvaluateSplits) {
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
  GPUHistMakerDevice<GradientPairPrecise> maker(0, page.get(), kNRows, param, kNCols, kNCols);
  // Initialize GPUHistMakerDevice::node_sum_gradients
  maker.node_sum_gradients = {{6.4f, 12.8f}};

  // Initialize GPUHistMakerDevice::cut
  auto cmat = GetHostCutMatrix();

  // Copy cut matrix to device.
  maker.ba.Allocate(0,
                    &(page->matrix.info.feature_segments), cmat.Ptrs().size(),
                    &(page->matrix.info.min_fvalue), cmat.MinValues().size(),
                    &(page->matrix.info.gidx_fvalue_map), 24,
                    &(maker.monotone_constraints), kNCols);
  dh::CopyVectorToDeviceSpan(page->matrix.info.feature_segments, cmat.Ptrs());
  dh::CopyVectorToDeviceSpan(page->matrix.info.gidx_fvalue_map, cmat.Values());
  dh::CopyVectorToDeviceSpan(maker.monotone_constraints, param.monotone_constraints);
  dh::CopyVectorToDeviceSpan(page->matrix.info.min_fvalue, cmat.MinValues());

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

  std::vector<DeviceSplitCandidate> res = maker.EvaluateSplits({0, 0 }, tree, kNCols);

  ASSERT_EQ(res[0].findex, 7);
  ASSERT_EQ(res[1].findex, 7);
  ASSERT_NEAR(res[0].fvalue, 0.26, xgboost::kRtEps);
  ASSERT_NEAR(res[1].fvalue, 0.26, xgboost::kRtEps);
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
  std::vector<common::CompressedByteT> h_gidx_buffer(maker->page->gidx_buffer.size());
  dh::CopyDeviceSpanToVector(&h_gidx_buffer, maker->page->gidx_buffer);

  const auto &maker_ext = hist_maker_ext.maker;
  std::vector<common::CompressedByteT> h_gidx_buffer_ext(maker_ext->page->gidx_buffer.size());
  dh::CopyDeviceSpanToVector(&h_gidx_buffer_ext, maker_ext->page->gidx_buffer);

  ASSERT_EQ(maker->page->matrix.info.n_bins, maker_ext->page->matrix.info.n_bins);
  ASSERT_EQ(maker->page->gidx_buffer.size(), maker_ext->page->gidx_buffer.size());

  ASSERT_EQ(h_gidx_buffer, h_gidx_buffer_ext);
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
  auto dmat = CreateDMatrix(kRows, kCols, kSparsity, 3);

  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(0.0f, 1.0f);
  std::vector<GradientPair> h_gpair(kRows);
  for (auto &gpair : h_gpair) {
    bst_float grad = dist(&gen);
    bst_float hess = dist(&gen);
    gpair = GradientPair(grad, hess);
  }
  HostDeviceVector<GradientPair> gpair(h_gpair);

  {
    int32_t n_nodes = TestMinSplitLoss((*dmat).get(), 0.01, &gpair);
    // This is not strictly verified, meaning the numeber `2` is whatever GPU_Hist retured
    // when writing this test, and only used for testing larger gamma (below) does prevent
    // building tree.
    ASSERT_EQ(n_nodes, 2);
  }
  {
    int32_t n_nodes = TestMinSplitLoss((*dmat).get(), 100.0, &gpair);
    // No new nodes with gamma == 100.
    ASSERT_EQ(n_nodes, static_cast<decltype(n_nodes)>(0));
  }
  delete dmat;
}

}  // namespace tree
}  // namespace xgboost
