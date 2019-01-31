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

  DeviceShard<GradientPairPrecise> shard(0, 0, n_rows, param);
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

  DeviceShard<GradientPairPrecise> shard(0, 0, n_rows, param);
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
void TestBuildHist(GPUHistBuilderBase<GradientSumT>& builder) {
  int const n_rows = 16, n_cols = 8;

  TrainParam param;
  param.max_depth = 6;
  param.n_gpus = 1;
  param.max_leaves = 0;

  DeviceShard<GradientSumT> shard(0, 0, n_rows, param);

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
  DeviceHistogram<GradientSumT> d_hist = shard.hist;

  auto node_histogram = d_hist.GetNodeHistogram(0);
  // d_hist.data stored in float, not gradient pair
  thrust::host_vector<GradientSumT> h_result (d_hist.data.size()/2);
  size_t data_size =
      sizeof(GradientSumT) /
      (sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT));
  data_size *= d_hist.data.size();
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
  GlobalMemHistBuilder<GradientPairPrecise> double_builder;
  TestBuildHist(double_builder);
  GlobalMemHistBuilder<GradientPair> float_builder;
  TestBuildHist(float_builder);
}

TEST(GpuHist, BuildHistSharedMem) {
  SharedMemHistBuilder<GradientPairPrecise> double_builder;
  TestBuildHist(double_builder);
  SharedMemHistBuilder<GradientPair> float_builder;
  TestBuildHist(float_builder);
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
  constexpr int n_rows = 16;
  constexpr int n_cols = 8;

  TrainParam param;
  param.max_depth = 1;
  param.n_gpus = 1;
  param.colsample_bynode = 1;
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
  std::unique_ptr<DeviceShard<GradientPairPrecise>> shard {new DeviceShard<GradientPairPrecise>(0, 0, n_rows, param)};
  // Initialize DeviceShard::node_sum_gradients
  shard->node_sum_gradients = {{6.4f, 12.8f}};

  // Initialize DeviceShard::cut
  common::HistCutMatrix cmat = GetHostCutMatrix();

  // Copy cut matrix to device.
  DeviceShard<GradientPairPrecise>::DeviceHistCutMatrix cut;
  shard->ba.Allocate(0,
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
  GPUHistMakerSpecialised<GradientPairPrecise> hist_maker =
      GPUHistMakerSpecialised<GradientPairPrecise>();
  hist_maker.param_ = param;
  hist_maker.shards_.push_back(std::move(shard));
  hist_maker.column_sampler_.Init(n_cols,
                                  param.colsample_bynode,
                                  param.colsample_bylevel,
                                  param.colsample_bytree,
                                  false);

  RegTree tree;
  MetaInfo info;
  info.num_row_ = n_rows;
  info.num_col_ = n_cols;

  hist_maker.info_ = &info;
  hist_maker.node_value_constraints_.resize(1);
  hist_maker.node_value_constraints_[0].lower_bound = -1.0;
  hist_maker.node_value_constraints_[0].upper_bound = 1.0;

  DeviceSplitCandidate res =
      hist_maker.EvaluateSplit(0, &tree);

  ASSERT_EQ(res.findex, 7);
  ASSERT_NEAR(res.fvalue, 0.26, xgboost::kRtEps);
}

TEST(GpuHist, ApplySplit) {
  GPUHistMakerSpecialised<GradientPairPrecise> hist_maker =
      GPUHistMakerSpecialised<GradientPairPrecise>();
  int constexpr nid = 0;
  int constexpr n_rows = 16;
  int constexpr n_cols = 8;

  TrainParam param;

  // Initialize shard
  for (size_t i = 0; i < n_cols; ++i) {
    param.monotone_constraints.emplace_back(0);
  }

  hist_maker.shards_.resize(1);
  hist_maker.shards_[0].reset(new DeviceShard<GradientPairPrecise>(0, 0, n_rows, param));

  auto& shard = hist_maker.shards_.at(0);
  shard->ridx_segments.resize(3);  // 3 nodes.
  shard->node_sum_gradients.resize(3);

  shard->ridx_segments[0] = Segment(0, n_rows);
  shard->ba.Allocate(0, &(shard->ridx), n_rows,
                     &(shard->position), n_rows);
  shard->row_stride = n_cols;
  thrust::sequence(shard->ridx.CurrentDVec().tbegin(),
                   shard->ridx.CurrentDVec().tend());
  // Initialize GPUHistMaker
  hist_maker.param_ = param;
  RegTree tree;

  DeviceSplitCandidate candidate;
  candidate.Update(2, kLeftDir,
                   0.59, 4,  // fvalue has to be equal to one of the cut field
                   GradientPair(8.2, 2.8), GradientPair(6.3, 3.6),
                   GPUTrainingParam(param));
  GPUHistMakerSpecialised<GradientPairPrecise>::ExpandEntry candidate_entry {0, 0, candidate, 0};
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
  shard->ba.Allocate(0, &(shard->gidx_buffer), compressed_size_bytes);

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
  hist_maker.UpdatePosition(candidate_entry, &tree);

  ASSERT_FALSE(tree[nid].IsLeaf());

  int left_nidx = tree[nid].LeftChild();
  int right_nidx = tree[nid].RightChild();

  ASSERT_EQ(shard->ridx_segments[left_nidx].begin, 0);
  ASSERT_EQ(shard->ridx_segments[left_nidx].end, 6);
  ASSERT_EQ(shard->ridx_segments[right_nidx].begin, 6);
  ASSERT_EQ(shard->ridx_segments[right_nidx].end, 16);
}

void TestSortPosition(const std::vector<int>& position_in, int left_idx,
                      int right_idx) {
  int left_count = std::count(position_in.begin(), position_in.end(), left_idx);
  thrust::device_vector<int> position = position_in;
  thrust::device_vector<int> position_out(position.size());

  thrust::device_vector<bst_uint> ridx(position.size());
  thrust::sequence(ridx.begin(), ridx.end());
  thrust::device_vector<bst_uint> ridx_out(ridx.size());
  dh::CubMemory tmp;
  SortPosition(
      &tmp, common::Span<int>(position.data().get(), position.size()),
      common::Span<int>(position_out.data().get(), position_out.size()),
      common::Span<bst_uint>(ridx.data().get(), ridx.size()),
      common::Span<bst_uint>(ridx_out.data().get(), ridx_out.size()), left_idx,
      right_idx, left_count);
  thrust::host_vector<int> position_result = position_out;
  thrust::host_vector<int> ridx_result = ridx_out;

  // Check position is sorted
  EXPECT_TRUE(std::is_sorted(position_result.begin(), position_result.end()));
  // Check row indices are sorted inside left and right segment
  EXPECT_TRUE(
      std::is_sorted(ridx_result.begin(), ridx_result.begin() + left_count));
  EXPECT_TRUE(
      std::is_sorted(ridx_result.begin() + left_count, ridx_result.end()));

  // Check key value pairs are the same
  for (auto i = 0ull; i < ridx_result.size(); i++) {
    EXPECT_EQ(position_result[i], position_in[ridx_result[i]]);
  }
}

TEST(GpuHist, SortPosition) {
  TestSortPosition({1, 2, 1, 2, 1}, 1, 2);
  TestSortPosition({1, 1, 1, 1}, 1, 2);
  TestSortPosition({2, 2, 2, 2}, 1, 2);
  TestSortPosition({1, 2, 1, 2, 3}, 1, 2);
}
}  // namespace tree
}  // namespace xgboost
