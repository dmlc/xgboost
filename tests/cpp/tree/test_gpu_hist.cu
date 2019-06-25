/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#include <thrust/device_vector.h>
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

template <typename GradientSumT>
void BuildGidx(DeviceShard<GradientSumT>* shard, int n_rows, int n_cols,
               bst_float sparsity=0) {
  auto dmat = CreateDMatrix(n_rows, n_cols, sparsity, 3);
  const SparsePage& batch = *(*dmat)->GetRowBatches().begin();

  common::HistCutMatrix cmat;
  cmat.row_ptr = {0, 3, 6, 9, 12, 15, 18, 21, 24};
  cmat.min_val = {0.1f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, 0.2f, 0.2f};
  // 24 cut fields, 3 cut fields for each feature (column).
  cmat.cut = {0.30f, 0.67f, 1.64f,
              0.32f, 0.77f, 1.95f,
              0.29f, 0.70f, 1.80f,
              0.32f, 0.75f, 1.85f,
              0.18f, 0.59f, 1.69f,
              0.25f, 0.74f, 2.00f,
              0.26f, 0.74f, 1.98f,
              0.26f, 0.71f, 1.83f};

  auto is_dense = (*dmat)->Info().num_nonzero_ ==
                  (*dmat)->Info().num_row_ * (*dmat)->Info().num_col_;
  size_t row_stride = 0;
  const auto &offset_vec = batch.offset.ConstHostVector();
  for (size_t i = 1; i < offset_vec.size(); ++i) {
    row_stride = std::max(row_stride, offset_vec[i] - offset_vec[i-1]);
  }
  shard->InitCompressedData(cmat, row_stride, is_dense);
  shard->CreateHistIndices(
    batch, cmat, RowStateOnDevice(batch.Size(), batch.Size()), -1);

  delete dmat;
}

TEST(GpuHist, BuildGidxDense) {
  int constexpr kNRows = 16, kNCols = 8;
  tree::TrainParam param;
  std::vector<std::pair<std::string, std::string>> args {
    {"max_depth", "1"},
    {"max_leaves", "0"},
  };
  param.Init(args);
  DeviceShard<GradientPairPrecise> shard(0, 0, 0, kNRows, param, kNCols, kNCols);
  BuildGidx(&shard, kNRows, kNCols);

  std::vector<common::CompressedByteT> h_gidx_buffer(shard.gidx_buffer.size());
  dh::CopyDeviceSpanToVector(&h_gidx_buffer, shard.gidx_buffer);
  common::CompressedIterator<uint32_t> gidx(h_gidx_buffer.data(), 25);

  ASSERT_EQ(shard.ellpack_matrix.row_stride, kNCols);

  std::vector<uint32_t> solution = {
    0, 3, 8,  9, 14, 17, 20, 21,
    0, 4, 7, 10, 14, 16, 19, 22,
    1, 3, 7, 11, 14, 15, 19, 21,
    2, 3, 7,  9, 13, 16, 20, 22,
    2, 3, 6,  9, 12, 16, 20, 21,
    1, 5, 6, 10, 13, 16, 20, 21,
    2, 5, 8,  9, 13, 17, 19, 22,
    2, 4, 6, 10, 14, 17, 19, 21,
    2, 5, 7,  9, 13, 16, 19, 22,
    0, 3, 8, 10, 12, 16, 19, 22,
    1, 3, 7, 10, 13, 16, 19, 21,
    1, 3, 8, 10, 13, 17, 20, 22,
    2, 4, 6,  9, 14, 15, 19, 22,
    1, 4, 6,  9, 13, 16, 19, 21,
    2, 4, 8, 10, 14, 15, 19, 22,
    1, 4, 7, 10, 14, 16, 19, 21,
  };
  for (size_t i = 0; i < kNRows * kNCols; ++i) {
    ASSERT_EQ(solution[i], gidx[i]);
  }
}

TEST(GpuHist, BuildGidxSparse) {
  int constexpr kNRows = 16, kNCols = 8;
  TrainParam param;
  std::vector<std::pair<std::string, std::string>> args {
    {"max_depth", "1"},
    {"max_leaves", "0"},
  };
  param.Init(args);

  DeviceShard<GradientPairPrecise> shard(0, 0, 0, kNRows, param, kNCols,
                                         kNCols);
  BuildGidx(&shard, kNRows, kNCols, 0.9f);

  std::vector<common::CompressedByteT> h_gidx_buffer(shard.gidx_buffer.size());
  dh::CopyDeviceSpanToVector(&h_gidx_buffer, shard.gidx_buffer);
  common::CompressedIterator<uint32_t> gidx(h_gidx_buffer.data(), 25);

  ASSERT_LE(shard.ellpack_matrix.row_stride, 3);

  // row_stride = 3, 16 rows, 48 entries for ELLPack
  std::vector<uint32_t> solution = {
    15, 24, 24,  0, 24, 24, 24, 24, 24, 24, 24, 24, 20, 24, 24, 24,
    24, 24, 24, 24, 24,  5, 24, 24,  0, 16, 24, 15, 24, 24, 24, 24,
    24,  7, 14, 16,  4, 24, 24, 24, 24, 24,  9, 24, 24,  1, 24, 24
  };
  for (size_t i = 0; i < kNRows * shard.ellpack_matrix.row_stride; ++i) {
    ASSERT_EQ(solution[i], gidx[i]);
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
  DeviceShard<GradientSumT> shard(0, 0, 0, kNRows, param, kNCols,
                                  kNCols);
  BuildGidx(&shard, kNRows, kNCols);

  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(0.0f, 1.0f);
  std::vector<GradientPair> h_gpair(kNRows);
  for (auto &gpair : h_gpair) {
    bst_float grad = dist(&gen);
    bst_float hess = dist(&gen);
    gpair = GradientPair(grad, hess);
  }

  thrust::host_vector<common::CompressedByteT> h_gidx_buffer (
      shard.gidx_buffer.size());

  common::CompressedByteT* d_gidx_buffer_ptr = shard.gidx_buffer.data();
  dh::safe_cuda(cudaMemcpy(h_gidx_buffer.data(), d_gidx_buffer_ptr,
                           sizeof(common::CompressedByteT) * shard.gidx_buffer.size(),
                           cudaMemcpyDeviceToHost));

  shard.row_partitioner.reset(new RowPartitioner(0, kNRows));
  shard.hist.AllocateHistogram(0);
  dh::CopyVectorToDeviceSpan(shard.gpair, h_gpair);

  shard.use_shared_memory_histograms = use_shared_memory_histograms;
  shard.BuildHist(0);
  DeviceHistogram<GradientSumT> d_hist = shard.hist;

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

common::HistCutMatrix GetHostCutMatrix () {
  common::HistCutMatrix cmat;
  cmat.row_ptr = {0, 3, 6, 9, 12, 15, 18, 21, 24};
  cmat.min_val = {0.1f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, 0.2f, 0.2f};
  // 24 cut fields, 3 cut fields for each feature (column).
  // Each row of the cut represents the cuts for a data column.
  cmat.cut = {0.30f, 0.67f, 1.64f,
              0.32f, 0.77f, 1.95f,
              0.29f, 0.70f, 1.80f,
              0.32f, 0.75f, 1.85f,
              0.18f, 0.59f, 1.69f,
              0.25f, 0.74f, 2.00f,
              0.26f, 0.74f, 1.98f,
              0.26f, 0.71f, 1.83f};
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

  // Initialize DeviceShard
  std::unique_ptr<DeviceShard<GradientPairPrecise>> shard{
    new DeviceShard<GradientPairPrecise>(0, 0, 0, kNRows, param, kNCols,
                                         kNCols)};
  // Initialize DeviceShard::node_sum_gradients
  shard->node_sum_gradients = {{6.4f, 12.8f}};

  // Initialize DeviceShard::cut
  common::HistCutMatrix cmat = GetHostCutMatrix();

  // Copy cut matrix to device.
  shard->ba.Allocate(0,
                     &(shard->feature_segments), cmat.row_ptr.size(),
                     &(shard->min_fvalue), cmat.min_val.size(),
                     &(shard->gidx_fvalue_map), 24,
                     &(shard->monotone_constraints), kNCols);
  dh::CopyVectorToDeviceSpan(shard->feature_segments, cmat.row_ptr);
  dh::CopyVectorToDeviceSpan(shard->gidx_fvalue_map, cmat.cut);
  dh::CopyVectorToDeviceSpan(shard->monotone_constraints,
                             param.monotone_constraints);
  shard->ellpack_matrix.feature_segments = shard->feature_segments;
  shard->ellpack_matrix.gidx_fvalue_map = shard->gidx_fvalue_map;
  dh::CopyVectorToDeviceSpan(shard->min_fvalue, cmat.min_val);
  shard->ellpack_matrix.min_fvalue = shard->min_fvalue;

  // Initialize DeviceShard::hist
  shard->hist.Init(0, (max_bins - 1) * kNCols);
  shard->hist.AllocateHistogram(0);
  // Each row of hist_gpair represents gpairs for one feature.
  // Each entry represents a bin.
  std::vector<GradientPairPrecise> hist_gpair = GetHostHistGpair();
  std::vector<bst_float> hist;
  for (auto pair : hist_gpair) {
    hist.push_back(pair.GetGrad());
    hist.push_back(pair.GetHess());
  }

  ASSERT_EQ(shard->hist.Data().size(), hist.size());
  thrust::copy(hist.begin(), hist.end(),
               shard->hist.Data().begin());

  shard->column_sampler.Init(kNCols,
                                  param.colsample_bynode,
                                  param.colsample_bylevel,
                                  param.colsample_bytree,
                                  false);

  RegTree tree;
  MetaInfo info;
  info.num_row_ = kNRows;
  info.num_col_ = kNCols;

  shard->node_value_constraints.resize(1);
  shard->node_value_constraints[0].lower_bound = -1.0;
  shard->node_value_constraints[0].upper_bound = 1.0;

  std::vector<DeviceSplitCandidate> res =
    shard->EvaluateSplits({ 0,0 }, tree, kNCols);

  ASSERT_EQ(res[0].findex, 7);
  ASSERT_EQ(res[1].findex, 7);
  ASSERT_NEAR(res[0].fvalue, 0.26, xgboost::kRtEps);
  ASSERT_NEAR(res[1].fvalue, 0.26, xgboost::kRtEps);
}

void TestHistogramIndexImpl(int n_gpus) {
  // Test if the compressed histogram index matches when using a sparse
  // dmatrix with and without using external memory

  int constexpr kNRows = 1000, kNCols = 10;

  // Build 2 matrices and build a histogram maker with that
  tree::GPUHistMakerSpecialised<GradientPairPrecise> hist_maker, hist_maker_ext;
  std::unique_ptr<DMatrix> hist_maker_dmat(
    CreateSparsePageDMatrixWithRC(kNRows, kNCols, 0, true));
  std::unique_ptr<DMatrix> hist_maker_ext_dmat(
    CreateSparsePageDMatrixWithRC(kNRows, kNCols, 128UL, true));

  std::vector<std::pair<std::string, std::string>> training_params = {
    {"max_depth", "10"},
    {"max_leaves", "0"}
  };

  LearnerTrainParam learner_param(CreateEmptyGenericParam(0, n_gpus));
  hist_maker.Init(training_params, &learner_param);
  hist_maker.InitDataOnce(hist_maker_dmat.get());
  hist_maker_ext.Init(training_params, &learner_param);
  hist_maker_ext.InitDataOnce(hist_maker_ext_dmat.get());

  ASSERT_EQ(hist_maker.shards_.size(), hist_maker_ext.shards_.size());

  // Extract the device shards from the histogram makers and from that its compressed
  // histogram index
  for (size_t i = 0; i < hist_maker.shards_.size(); ++i) {
    const auto &dev_shard = hist_maker.shards_[i];
    std::vector<common::CompressedByteT> h_gidx_buffer(dev_shard->gidx_buffer.size());
    dh::CopyDeviceSpanToVector(&h_gidx_buffer, dev_shard->gidx_buffer);

    const auto &dev_shard_ext = hist_maker_ext.shards_[i];
    std::vector<common::CompressedByteT> h_gidx_buffer_ext(dev_shard_ext->gidx_buffer.size());
    dh::CopyDeviceSpanToVector(&h_gidx_buffer_ext, dev_shard_ext->gidx_buffer);

    ASSERT_EQ(dev_shard->n_bins, dev_shard_ext->n_bins);
    ASSERT_EQ(dev_shard->gidx_buffer.size(), dev_shard_ext->gidx_buffer.size());

    ASSERT_EQ(h_gidx_buffer, h_gidx_buffer_ext);
  }
}

TEST(GpuHist, TestHistogramIndex) {
  TestHistogramIndexImpl(1);
}

#if defined(XGBOOST_USE_NCCL)
TEST(GpuHist, MGPU_TestHistogramIndex) {
  auto devices = GPUSet::AllVisible();
  CHECK_GT(devices.Size(), 1);
  TestHistogramIndexImpl(-1);
}
#endif

}  // namespace tree
}  // namespace xgboost
