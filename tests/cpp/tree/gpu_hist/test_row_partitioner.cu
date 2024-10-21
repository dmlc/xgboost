/**
 * Copyright 2019-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>    // for sort
#include <thrust/unique.h>  // for unique
#include <xgboost/base.h>
#include <xgboost/tree_model.h>  // for RegTree

#include <cstddef>   // for size_t
#include <cstdint>   // for uint32_t
#include <iterator>  // for distance
#include <vector>    // for vector

#include "../../../../src/data/ellpack_page.cuh"
#include "../../../../src/tree/gpu_hist/expand_entry.cuh"  // for GPUExpandEntry
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../../../../src/tree/param.h"    // for TrainParam
#include "../../collective/test_worker.h"  // for TestDistributedGlobal
#include "../../helpers.h"                 // for RandomDataGenerator

namespace xgboost::tree {
void TestUpdatePositionBatch() {
  const int kNumRows = 10;
  auto ctx = MakeCUDACtx(0);
  RowPartitioner rp;
  rp.Reset(&ctx, kNumRows, 0);
  auto rows = rp.GetRowsHost(0);
  EXPECT_EQ(rows.size(), kNumRows);
  for (auto i = 0ull; i < kNumRows; i++) {
    EXPECT_EQ(rows[i], i);
  }
  std::vector<int> extra_data = {0};
  // Send the first five training instances to the right node
  // and the second 5 to the left node
  rp.UpdatePositionBatch(
      &ctx, {0}, {1}, {2}, extra_data,
      [=] __device__(RowPartitioner::RowIndexT ridx, int, int) { return ridx > 4; });
  rows = rp.GetRowsHost(1);
  for (auto r : rows) {
    EXPECT_GT(r, 4);
  }
  rows = rp.GetRowsHost(2);
  for (auto r : rows) {
    EXPECT_LT(r, 5);
  }

  // Split the left node again
  rp.UpdatePositionBatch(
      &ctx, {1}, {3}, {4}, extra_data,
      [=] __device__(RowPartitioner::RowIndexT ridx, int, int) { return ridx < 7; });
  EXPECT_EQ(rp.GetRows(3).size(), 2);
  EXPECT_EQ(rp.GetRows(4).size(), 3);
}

TEST(RowPartitioner, Batch) { TestUpdatePositionBatch(); }

void TestSortPositionBatch(const std::vector<int>& ridx_in, const std::vector<Segment>& segments) {
  auto ctx = MakeCUDACtx(0);
  thrust::device_vector<cuda_impl::RowIndexT> ridx = ridx_in;
  thrust::device_vector<cuda_impl::RowIndexT> ridx_tmp(ridx_in.size());
  thrust::device_vector<cuda_impl::RowIndexT> counts(segments.size());

  auto op = [=] __device__(auto ridx, int split_index, int data) {
    return ridx % 2 == 0;
  };
  std::vector<int> op_data(segments.size());
  std::vector<PerNodeData<int>> h_batch_info(segments.size());
  dh::TemporaryArray<PerNodeData<int>> d_batch_info(segments.size());

  std::size_t total_rows = 0;
  for (size_t i = 0; i < segments.size(); i++) {
    h_batch_info[i] = {segments.at(i), 0};
    total_rows += segments.at(i).Size();
  }
  dh::safe_cuda(cudaMemcpyAsync(d_batch_info.data().get(), h_batch_info.data(),
                                h_batch_info.size() * sizeof(PerNodeData<int>), cudaMemcpyDefault,
                                nullptr));
  dh::DeviceUVector<std::int8_t> tmp;
  SortPositionBatch<decltype(op), int>(&ctx, dh::ToSpan(d_batch_info), dh::ToSpan(ridx),
                                       dh::ToSpan(ridx_tmp), dh::ToSpan(counts), total_rows, op,
                                       &tmp);

  auto op_without_data = [=] __device__(auto ridx) {
    return ridx % 2 == 0;
  };
  for (size_t i = 0; i < segments.size(); i++) {
    auto begin = ridx.begin() + segments[i].begin;
    auto end = ridx.begin() + segments[i].end;
    bst_uint count = counts[i];
    auto left_partition_count =
        thrust::count_if(thrust::device, begin, begin + count, op_without_data);
    EXPECT_EQ(left_partition_count, count);
    auto right_partition_count =
        thrust::count_if(thrust::device, begin + count, end, op_without_data);
    EXPECT_EQ(right_partition_count, 0);
  }
}

TEST(RowPartitioner, SortPositionBatch) {
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{0, 3}, {3, 6}});
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{0, 1}, {3, 6}});
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{0, 6}});
  TestSortPositionBatch({0, 1, 2, 3, 4, 5}, {{3, 6}, {0, 2}});
}

namespace {
void GetSplit(RegTree* tree, float split_value, std::vector<GPUExpandEntry>* candidates) {
  CHECK(!tree->IsMultiTarget());
  tree->ExpandNode(
      /*nid=*/RegTree::kRoot, /*split_index=*/0, /*split_value=*/split_value,
      /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      /*left_sum=*/0.0f,
      /*right_sum=*/0.0f);
  candidates->front().nid = 0;
  candidates->front().depth = 0;
  candidates->front().split.fvalue = split_value;
  candidates->front().split.findex = 0;
}

void TestExternalMemory() {
  auto ctx = MakeCUDACtx(0);

  bst_bin_t max_bin = 32;
  auto p_fmat =
      RandomDataGenerator{256, 16, 0.0f}.Batches(4).GenerateSparsePageDMatrix("temp", true);

  std::vector<std::unique_ptr<RowPartitioner>> partitioners;
  RegTree tree;
  std::vector<GPUExpandEntry> candidates(1);

  auto param = BatchParam{max_bin, TrainParam::DftSparseThreshold()};
  float split_value{0.0f};
  bst_feature_t const split_ind = 0;
  dh::device_vector<bst_node_t> position(p_fmat->Info().num_row_, 0);

  auto encode_op = [=] __device__(bst_idx_t, bst_node_t nidx) {
    return nidx;
  };  // NOLINT

  for (auto const& page : p_fmat->GetBatches<EllpackPage>(&ctx, param)) {
    if (partitioners.empty()) {
      auto ptr = page.Impl()->Cuts().Ptrs()[split_ind + 1];
      split_value = page.Impl()->Cuts().Values().at(ptr / 2);
      GetSplit(&tree, split_value, &candidates);
    }

    partitioners.emplace_back(std::make_unique<RowPartitioner>());
    partitioners.back()->Reset(&ctx, page.Size(), page.BaseRowId());
    std::vector<RegTree::Node> splits{tree[0]};
    auto acc = page.Impl()->GetDeviceAccessor(&ctx);
    partitioners.back()->UpdatePositionBatch(
        &ctx, {0}, {1}, {2}, splits,
        [=] __device__(bst_idx_t ridx, std::int32_t nidx_in_batch, RegTree::Node const& node) {
          auto fvalue = acc.GetFvalue(ridx, node.SplitIndex());
          return fvalue <= node.SplitCond();
        });
    partitioners.back()->FinalisePosition(
        &ctx, dh::ToSpan(position).subspan(page.BaseRowId(), page.Size()), page.BaseRowId(),
        encode_op);
  }

  bst_idx_t n_left{0};
  for (auto const& page : p_fmat->GetBatches<SparsePage>()) {
    auto batch = page.GetView();
    for (size_t i = 0; i < batch.Size(); ++i) {
      if (batch[i][split_ind].fvalue < split_value) {
        n_left++;
      }
    }
  }

  RegTree::Node node = tree[RegTree::kRoot];
  auto n_left_pos =
      thrust::count_if(position.cbegin(), position.cend(),
                       [=] XGBOOST_DEVICE(bst_node_t v) { return v == node.LeftChild(); });
  ASSERT_EQ(n_left, n_left_pos);
  thrust::sort(position.begin(), position.end());
  auto end_it = thrust::unique(position.begin(), position.end());
  ASSERT_EQ(std::distance(position.begin(), end_it), 2);
}
}  // anonymous namespace

TEST(RowPartitioner, LeafPartitionExternalMemory) { TestExternalMemory(); }

namespace {
void TestEmptyNode(std::int32_t n_workers) {
  collective::TestDistributedGlobal(n_workers, [] {
    auto ctx = MakeCUDACtx(DistGpuIdx());
    RowPartitioner partitioner;
    bst_idx_t n_samples = (collective::GetRank() == 0) ? 0 : 1024;
    bst_idx_t base_rowid = 0;
    partitioner.Reset(&ctx, n_samples, base_rowid);
    std::vector<RegTree::Node> splits(1);
    partitioner.UpdatePositionBatch(
        &ctx, {0}, {1}, {2}, splits,
        [] XGBOOST_DEVICE(bst_idx_t ridx, std::int32_t /*nidx_in_batch*/, RegTree::Node) {
          return ridx < 3;
        });
    ASSERT_EQ(partitioner.GetNumNodes(), 3);
    if (collective::GetRank() == 0) {
      for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_TRUE(partitioner.GetRows(i).empty());
      }
    }
    ctx.CUDACtx()->Stream().Sync();
  });
}
}  // anonymous namespace

TEST(RowPartitioner, MGPUEmpty) {
  std::int32_t n_workers = curt::AllVisibleGPUs();
  TestEmptyNode(n_workers);
}
}  // namespace xgboost::tree
