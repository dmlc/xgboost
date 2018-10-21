/*!
 * Copyright 2017-2018 XGBoost contributors
 */

#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include <random>
#include "../helpers.h"
#include "gtest/gtest.h"

#include "../../../src/data/sparse_page_source.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../../../src/tree/updater_gpu_hist.cu"
#include "../../../src/tree/updater_gpu_common.cuh"
#include "../../../src/common/common.h"

namespace xgboost {
namespace tree {

void BuildGidx(DeviceShard* shard, int n_rows, int n_cols,
               bst_float sparsity=0) {
  auto dmat = CreateDMatrix(n_rows, n_cols, sparsity, 3);
  const SparsePage& batch = *(*dmat)->GetRowBatches().begin();

  common::HistCutMatrix cmat;
  cmat.row_ptr = {0, 3, 6, 9, 12, 15, 18, 21, 24};
  cmat.min_val = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.2, 0.2};
  // 24 cut fields, 3 cut fields for each feature (column).
  cmat.cut = {0.30, 0.67, 1.64,
              0.32, 0.77, 1.95,
              0.29, 0.70, 1.80,
              0.32, 0.75, 1.85,
              0.18, 0.59, 1.69,
              0.25, 0.74, 2.00,
              0.26, 0.74, 1.98,
              0.26, 0.71, 1.83};

  shard->InitRowPtrs(batch);
  shard->InitCompressedData(cmat, batch);

  delete dmat;
}

TEST(GpuHist, BuildGidxDense) {
  int const n_rows = 16, n_cols = 8;
  TrainParam param;
  param.max_depth = 1;
  param.n_gpus = 1;
  param.max_leaves = 0;

  DeviceShard shard(0, 0, 0, n_rows, param);
  BuildGidx(&shard, n_rows, n_cols);

  std::vector<common::CompressedByteT> h_gidx_buffer;
  h_gidx_buffer = shard.gidx_buffer.AsVector();
  common::CompressedIterator<uint32_t> gidx(h_gidx_buffer.data(), 25);

  ASSERT_EQ(shard.row_stride, n_cols);

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
  for (size_t i = 0; i < n_rows * n_cols; ++i) {
    ASSERT_EQ(solution[i], gidx[i]);
  }
}

TEST(GpuHist, BuildGidxSparse) {
  int const n_rows = 16, n_cols = 8;
  TrainParam param;
  param.max_depth = 1;
  param.n_gpus = 1;
  param.max_leaves = 0;

  DeviceShard shard(0, 0, 0, n_rows, param);
  BuildGidx(&shard, n_rows, n_cols, 0.9f);

  std::vector<common::CompressedByteT> h_gidx_buffer;
  h_gidx_buffer = shard.gidx_buffer.AsVector();
  common::CompressedIterator<uint32_t> gidx(h_gidx_buffer.data(), 25);

  ASSERT_LE(shard.row_stride, 3);

  // row_stride = 3, 16 rows, 48 entries for ELLPack
  std::vector<uint32_t> solution = {
    15, 24, 24,  0, 24, 24, 24, 24, 24, 24, 24, 24, 20, 24, 24, 24,
    24, 24, 24, 24, 24,  5, 24, 24,  0, 16, 24, 15, 24, 24, 24, 24,
    24,  7, 14, 16,  4, 24, 24, 24, 24, 24,  9, 24, 24,  1, 24, 24
  };
  for (size_t i = 0; i < n_rows * shard.row_stride; ++i) {
    ASSERT_EQ(solution[i], gidx[i]);
  }
}

std::vector<GradientPairPrecise> GetHostHistGpair() {
  // 24 bins, 3 bins for each feature (column).
  std::vector<GradientPairPrecise> hist_gpair = {
    {0.8314, 0.7147}, {1.7989, 3.7312}, {3.3846, 3.4598},
    {2.9277, 3.5886}, {1.8429, 2.4152}, {1.2443, 1.9019},
    {1.6380, 2.9174}, {1.5657, 2.5107}, {2.8111, 2.4776},
    {2.1322, 3.0651}, {3.2927, 3.8540}, {0.5899, 0.9866},
    {1.5185, 1.6263}, {2.0686, 3.1844}, {2.4278, 3.0950},
    {1.5105, 2.1403}, {2.6922, 4.2217}, {1.8122, 1.5437},
    {0.0000, 0.0000}, {4.3245, 5.7955}, {1.6903, 2.1103},
    {2.4012, 4.4754}, {3.6136, 3.4303}, {0.0000, 0.0000}
  };
  return hist_gpair;
}

void TestBuildHist(GPUHistBuilderBase& builder) {
  int const n_rows = 16, n_cols = 8;

  TrainParam param;
  param.max_depth = 6;
  param.n_gpus = 1;
  param.max_leaves = 0;

  DeviceShard shard(0, 0, 0, n_rows, param);

  BuildGidx(&shard, n_rows, n_cols);

  xgboost::SimpleLCG gen;
  xgboost::SimpleRealUniformDistribution<bst_float> dist(0.0f, 1.0f);
  std::vector<GradientPair> h_gpair(n_rows);
  for (size_t i = 0; i < h_gpair.size(); ++i) {
    bst_float grad = dist(&gen);
    bst_float hess = dist(&gen);
    h_gpair[i] = GradientPair(grad, hess);
  }

  thrust::device_vector<GradientPair> gpair (n_rows);
  gpair = h_gpair;

  int num_symbols = shard.n_bins + 1;

  thrust::host_vector<common::CompressedByteT> h_gidx_buffer (
      shard.gidx_buffer.Size());

  common::CompressedByteT* d_gidx_buffer_ptr = shard.gidx_buffer.Data();
  dh::safe_cuda(cudaMemcpy(h_gidx_buffer.data(), d_gidx_buffer_ptr,
                           sizeof(common::CompressedByteT) * shard.gidx_buffer.Size(),
                           cudaMemcpyDeviceToHost));
  auto gidx = common::CompressedIterator<uint32_t>(h_gidx_buffer.data(),
                                                   num_symbols);

  shard.ridx_segments.resize(1);
  shard.ridx_segments[0] = Segment(0, n_rows);
  shard.hist.AllocateHistogram(0);
  shard.gpair.copy(gpair.begin(), gpair.end());
  thrust::sequence(shard.ridx.CurrentDVec().tbegin(),
                   shard.ridx.CurrentDVec().tend());

  builder.Build(&shard, 0);
  DeviceHistogram d_hist = shard.hist;

  GradientPairSumT* d_histptr {d_hist.GetHistPtr(0)};
  // d_hist.data stored in float, not gradient pair
  thrust::host_vector<GradientPairSumT> h_result (d_hist.data.size()/2);
  size_t data_size = sizeof(GradientPairSumT) / (
      sizeof(GradientPairSumT) / sizeof(GradientPairSumT::ValueT));
  data_size *= d_hist.data.size();
  dh::safe_cuda(cudaMemcpy(h_result.data(), d_histptr, data_size,
                           cudaMemcpyDeviceToHost));

  std::vector<GradientPairPrecise> solution = GetHostHistGpair();
  std::cout << std::fixed;
  for (size_t i = 0; i < h_result.size(); ++i) {
    EXPECT_NEAR(h_result[i].GetGrad(), solution[i].GetGrad(), 0.01f);
    EXPECT_NEAR(h_result[i].GetHess(), solution[i].GetHess(), 0.01f);
  }
}

TEST(GpuHist, BuildHistGlobalMem) {
  GlobalMemHistBuilder builder;
  TestBuildHist(builder);
}

TEST(GpuHist, BuildHistSharedMem) {
  SharedMemHistBuilder builder;
  TestBuildHist(builder);
}

common::HistCutMatrix GetHostCutMatrix () {
  common::HistCutMatrix cmat;
  cmat.row_ptr = {0, 3, 6, 9, 12, 15, 18, 21, 24};
  cmat.min_val = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.2, 0.2};
  // 24 cut fields, 3 cut fields for each feature (column).
  // Each row of the cut represents the cuts for a data column.
  cmat.cut = {0.30, 0.67, 1.64,
              0.32, 0.77, 1.95,
              0.29, 0.70, 1.80,
              0.32, 0.75, 1.85,
              0.18, 0.59, 1.69,
              0.25, 0.74, 2.00,
              0.26, 0.74, 1.98,
              0.26, 0.71, 1.83};
  return cmat;
}

// TODO(trivialfis): This test is over simplified.
TEST(GpuHist, EvaluateSplits) {
  constexpr int n_rows = 16;
  constexpr int n_cols = 8;

  TrainParam param;
  param.max_depth = 1;
  param.n_gpus = 1;
  param.colsample_bylevel = 1;
  param.colsample_bytree = 1;
  param.min_child_weight = 0.01;

  // Disable all parameters.
  param.reg_alpha = 0.0;
  param.reg_lambda = 0;
  param.max_delta_step = 0.0;

  for (size_t i = 0; i < n_cols; ++i) {
    param.monotone_constraints.emplace_back(0);
  }

  int max_bins = 4;

  // Initialize DeviceShard
  std::unique_ptr<DeviceShard> shard {new DeviceShard(0, 0, 0, n_rows, param)};
  // Initialize DeviceShard::node_sum_gradients
  shard->node_sum_gradients = {{6.4, 12.8}};

  // Initialize DeviceShard::cut
  common::HistCutMatrix cmat = GetHostCutMatrix();

  // Copy cut matrix to device.
  DeviceShard::DeviceHistCutMatrix cut;
  shard->ba.Allocate(0, true,
                     &(shard->cut_.feature_segments), cmat.row_ptr.size(),
                     &(shard->cut_.min_fvalue), cmat.min_val.size(),
                     &(shard->cut_.gidx_fvalue_map), 24,
                     &(shard->monotone_constraints), n_cols);
  shard->cut_.feature_segments.copy(cmat.row_ptr.begin(), cmat.row_ptr.end());
  shard->cut_.gidx_fvalue_map.copy(cmat.cut.begin(), cmat.cut.end());
  shard->monotone_constraints.copy(param.monotone_constraints.begin(),
                                   param.monotone_constraints.end());

  // Initialize DeviceShard::hist
  shard->hist.Init(0, (max_bins - 1) * n_cols);
  shard->hist.AllocateHistogram(0);
  // Each row of hist_gpair represents gpairs for one feature.
  // Each entry represents a bin.
  std::vector<GradientPairPrecise> hist_gpair = GetHostHistGpair();
  std::vector<bst_float> hist;
  for (auto pair : hist_gpair) {
    hist.push_back(pair.GetGrad());
    hist.push_back(pair.GetHess());
  }

  ASSERT_EQ(shard->hist.data.size(), hist.size());
  thrust::copy(hist.begin(), hist.end(),
               shard->hist.data.begin());


  // Initialize GPUHistMaker
  GPUHistMaker hist_maker = GPUHistMaker();
  hist_maker.param_ = param;
  hist_maker.shards_.push_back(std::move(shard));
  hist_maker.column_sampler_.Init(n_cols,
                                  param.colsample_bylevel,
                                  param.colsample_bytree,
                                  false);

  RegTree tree;
  tree.InitModel();

  MetaInfo info;
  info.num_row_ = n_rows;
  info.num_col_ = n_cols;

  hist_maker.info_ = &info;
  hist_maker.node_value_constraints_.resize(1);
  hist_maker.node_value_constraints_[0].lower_bound = -1.0;
  hist_maker.node_value_constraints_[0].upper_bound = 1.0;

  std::vector<DeviceSplitCandidate> res =
      hist_maker.EvaluateSplits({0}, &tree);

  ASSERT_EQ(res.size(), 1);
  ASSERT_EQ(res[0].findex, 7);
  ASSERT_NEAR(res[0].fvalue, 0.26, xgboost::kRtEps);
}

TEST(GpuHist, ApplySplit) {
  GPUHistMaker hist_maker = GPUHistMaker();
  int constexpr nid = 0;
  int constexpr n_rows = 16;
  int constexpr n_cols = 8;

  TrainParam param;
  param.silent = true;

  // Initialize shard
  for (size_t i = 0; i < n_cols; ++i) {
    param.monotone_constraints.emplace_back(0);
  }

  hist_maker.shards_.resize(1);
  hist_maker.shards_[0].reset(new DeviceShard(0, 0, 0, n_rows, param));

  auto& shard = hist_maker.shards_.at(0);
  shard->ridx_segments.resize(3);  // 3 nodes.
  shard->node_sum_gradients.resize(3);

  shard->ridx_segments[0] = Segment(0, n_rows);
  shard->ba.Allocate(0, true, &(shard->ridx), n_rows,
                     &(shard->position), n_rows);
  shard->row_stride = n_cols;
  thrust::sequence(shard->ridx.CurrentDVec().tbegin(),
                   shard->ridx.CurrentDVec().tend());
  dh::safe_cuda(cudaMallocHost(&(shard->tmp_pinned), sizeof(int64_t)));

  // Initialize GPUHistMaker
  hist_maker.param_ = param;
  RegTree tree;
  tree.InitModel();

  DeviceSplitCandidate candidate;
  candidate.Update(2, kLeftDir,
                   0.59, 4,  // fvalue has to be equal to one of the cut field
                   GradientPair(8.2, 2.8), GradientPair(6.3, 3.6),
                   GPUTrainingParam(param));
  GPUHistMaker::ExpandEntry candidate_entry {0, 0, candidate, 0};
  candidate_entry.nid = nid;

  auto const& nodes = tree.GetNodes();
  size_t n_nodes = nodes.size();

  // Used to get bin_id in update position.
  common::HistCutMatrix cmat = GetHostCutMatrix();
  hist_maker.hmat_ = cmat;

  MetaInfo info;
  info.num_row_ = n_rows;
  info.num_col_ = n_cols;
  info.num_nonzero_ = n_rows * n_cols;  // Dense

  // Initialize gidx
  int n_bins = 24;
  int row_stride = n_cols;
  int num_symbols = n_bins + 1;
  size_t compressed_size_bytes =
      common::CompressedBufferWriter::CalculateBufferSize(
          row_stride * n_rows, num_symbols);
  shard->ba.Allocate(0, param.silent,
                     &(shard->gidx_buffer), compressed_size_bytes);

  common::CompressedBufferWriter wr(num_symbols);
  std::vector<int> h_gidx (n_rows * row_stride);
  std::iota(h_gidx.begin(), h_gidx.end(), 0);
  std::vector<common::CompressedByteT> h_gidx_compressed (compressed_size_bytes);

  wr.Write(h_gidx_compressed.data(), h_gidx.begin(), h_gidx.end());
  shard->gidx_buffer.copy(h_gidx_compressed.begin(), h_gidx_compressed.end());

  shard->gidx = common::CompressedIterator<uint32_t>(
      shard->gidx_buffer.Data(), num_symbols);

  hist_maker.info_ = &info;
  hist_maker.ApplySplit(candidate_entry, &tree);

  ASSERT_FALSE(tree[nid].IsLeaf());

  int left_nidx = tree[nid].LeftChild();
  int right_nidx = tree[nid].RightChild();

  ASSERT_EQ(shard->ridx_segments[left_nidx].begin, 0);
  ASSERT_EQ(shard->ridx_segments[left_nidx].end, 6);
  ASSERT_EQ(shard->ridx_segments[right_nidx].begin, 6);
  ASSERT_EQ(shard->ridx_segments[right_nidx].end, 16);
}

TEST(GpuHist, MGPU_mock) {
  // Attempt to choose multiple GPU devices
  int ngpu;
  dh::safe_cuda(cudaGetDeviceCount(&ngpu));
  CHECK_GT(ngpu, 1);
  for (int i = 0; i < ngpu; ++i) {
    dh::safe_cuda(cudaSetDevice(i));
  }
}

}  // namespace tree
}  // namespace xgboost
