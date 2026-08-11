/**
 * Copyright 2019-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>  // for sort
#include <thrust/transform.h>
#include <thrust/unique.h>  // for unique
#include <xgboost/base.h>
#include <xgboost/tree_model.h>  // for RegTree

#include <algorithm>  // for sort
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t
#include <iterator>   // for distance
#include <vector>     // for vector

#include "../../../../src/data/ellpack_page.cuh"
#include "../../../../src/tree/gpu_hist/expand_entry.cuh"  // for GPUExpandEntry
#include "../../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../../../../src/tree/param.h"  // for TrainParam
#include "../../../../src/tree/sample_position.h"
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
  dh::DeviceUVector<cuda_impl::RowIndexT> ridx_tmp(kNumRows);
  // Send the first five training instances to the right node
  // and the second 5 to the left node
  rp.UpdatePositionBatch(&ctx, {0}, {1}, {2}, extra_data, dh::ToSpan(ridx_tmp),
                         [=] __device__(RowPartitioner::RowIndexT ridx, int) { return ridx > 4; });
  rows = rp.GetRowsHost(1);
  for (auto r : rows) {
    EXPECT_GT(r, 4);
  }
  rows = rp.GetRowsHost(2);
  for (auto r : rows) {
    EXPECT_LT(r, 5);
  }

  // Split the left node again
  rp.UpdatePositionBatch(&ctx, {1}, {3}, {4}, extra_data, dh::ToSpan(ridx_tmp),
                         [=] __device__(RowPartitioner::RowIndexT ridx, int) { return ridx < 7; });
  EXPECT_EQ(rp.GetRows(3).size(), 2);
  EXPECT_EQ(rp.GetRows(4).size(), 3);
}

TEST(RowPartitioner, Batch) { TestUpdatePositionBatch(); }

namespace {
// The rows of a node are not kept in any particular order, the right child comes out of the
// scatter reversed. Sort so that the assertions are about membership only.
[[nodiscard]] std::vector<RowPartitioner::RowIndexT> SortedRows(RowPartitioner* rp,
                                                                bst_node_t nidx) {
  auto rows = rp->GetRowsHost(nidx);
  std::sort(rows.begin(), rows.end());
  return rows;
}

// Seeding from an explicit subset, as cross-validation does for a fold.
void TestResetSubset() {
  auto ctx = MakeCUDACtx(0);
  bst_idx_t constexpr kNumRows = 16;
  // A non-contiguous subset that does not start at zero, to catch a stray base row index.
  std::vector<bst_idx_t> const h_ridx{3, 4, 7, 8, 11, 15};
  dh::device_vector<bst_idx_t> d_ridx{h_ridx};

  RowPartitioner rp;
  rp.Reset(&ctx, kNumRows, dh::ToSpan(d_ridx));
  ASSERT_EQ(rp.Size(), h_ridx.size());
  auto rows = rp.GetRowsHost(RegTree::kRoot);
  ASSERT_EQ(rows.size(), h_ridx.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    // The stored indices are the input, unshifted.
    ASSERT_EQ(rows[i], h_ridx[i]);
  }

  // The children must partition the subset. Compared against the full expected lists, a
  // duplicated or dropped row would satisfy a range check.
  std::vector<int> extra_data = {0};
  dh::DeviceUVector<cuda_impl::RowIndexT> ridx_tmp(rp.Size());
  rp.UpdatePositionBatch(&ctx, {RegTree::kRoot}, {1}, {2}, extra_data, dh::ToSpan(ridx_tmp),
                         [=] __device__(RowPartitioner::RowIndexT ridx, int) { return ridx < 8; });
  ASSERT_EQ(SortedRows(&rp, 1), (std::vector<RowPartitioner::RowIndexT>{3, 4, 7}));
  ASSERT_EQ(SortedRows(&rp, 2), (std::vector<RowPartitioner::RowIndexT>{8, 11, 15}));
}

// The batched wrapper, which also sizes the shared sort scratch from the largest subset.
void TestResetSubsetBatches() {
  auto ctx = MakeCUDACtx(0);
  bst_idx_t constexpr kNumRows = 16;
  std::vector<bst_idx_t> const h_batch_0{0, 2, 5};
  std::vector<bst_idx_t> const h_batch_1{8, 9, 12, 13, 15};
  dh::device_vector<bst_idx_t> d_batch_0{h_batch_0}, d_batch_1{h_batch_1};

  RowPartitionerBatches rps;
  rps.Reset(&ctx, kNumRows, {dh::ToSpan(d_batch_0), dh::ToSpan(d_batch_1)});
  ASSERT_EQ(rps.Size(), 2);
  ASSERT_EQ(rps.At(0)->Size(), h_batch_0.size());
  ASSERT_EQ(rps.At(1)->Size(), h_batch_1.size());

  // The partitioners must survive a re-seed, which is what a boosting round does.
  std::vector<RowPartitioner*> const reused{rps.At(0).get(), rps.At(1).get()};
  rps.Reset(&ctx, kNumRows, {dh::ToSpan(d_batch_0), dh::ToSpan(d_batch_1)});
  ASSERT_EQ(rps.At(0).get(), reused[0]);
  ASSERT_EQ(rps.At(1).get(), reused[1]);

  using RowIndexT = RowPartitioner::RowIndexT;
  std::vector<int> extra_data = {0};
  for (std::int32_t batch_idx = 0; batch_idx < 2; ++batch_idx) {
    // The sort scratch is shared by the batches. Sized from the first subset instead of the
    // largest, the second batch aborts in `subspan` inside the wrapper.
    rps.UpdatePositionBatch(&ctx, batch_idx, {RegTree::kRoot}, {1}, {2}, extra_data,
                            [=] __device__(RowIndexT ridx, int) { return ridx % 2 == 0; });
  }
  ASSERT_EQ(SortedRows(rps.At(0).get(), 1), (std::vector<RowIndexT>{0, 2}));
  ASSERT_EQ(SortedRows(rps.At(0).get(), 2), (std::vector<RowIndexT>{5}));
  ASSERT_EQ(SortedRows(rps.At(1).get(), 1), (std::vector<RowIndexT>{8, 12}));
  ASSERT_EQ(SortedRows(rps.At(1).get(), 2), (std::vector<RowIndexT>{9, 13, 15}));
}
}  // anonymous namespace

TEST(RowPartitioner, ResetSubset) { TestResetSubset(); }

TEST(RowPartitioner, ResetSubsetBatches) { TestResetSubsetBatches(); }

void TestSortPositionBatch(const std::vector<int>& ridx_in, const std::vector<Segment>& segments) {
  auto ctx = MakeCUDACtx(0);
  thrust::device_vector<cuda_impl::RowIndexT> ridx = ridx_in;
  thrust::device_vector<cuda_impl::RowIndexT> ridx_tmp(ridx_in.size());
  thrust::device_vector<cuda_impl::RowIndexT> counts(segments.size());

  auto op = [=] __device__(auto ridx, int data) {
    return ridx % 2 == 0;
  };  // NOLINT
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
  };  // NOLINT
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
  candidates->front().nidx = 0;
  candidates->front().depth = 0;
  candidates->front().split.fvalue = split_value;
  candidates->front().split.findex = 0;
}

namespace {
template <typename Accessor>
struct LessThanOp {
  Accessor acc;
  explicit LessThanOp(Accessor acc) : acc{acc} {}
  __device__ bool operator()(bst_idx_t ridx, RegTree::Node const& node) const {
    auto fvalue = acc.GetFvalue(ridx, node.SplitIndex());
    return fvalue <= node.SplitCond();
  }
};
}  // namespace

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

  auto n_rows = p_fmat->Info().num_row_;
  auto encode_op = [=] __device__(bst_idx_t ridx, bst_node_t nidx) {
    return SamplePosition::Encode(nidx, ridx < n_rows / 2);
  };  // NOLINT

  for (auto const& page : p_fmat->GetBatches<EllpackPage>(&ctx, param)) {
    if (partitioners.empty()) {
      auto ptr = page.Impl()->Cuts().Ptrs()[split_ind + 1];
      split_value = page.Impl()->Cuts().Values().at(ptr / 2);
      GetSplit(&tree, split_value, &candidates);
    }

    partitioners.emplace_back(std::make_unique<RowPartitioner>());
    partitioners.back()->Reset(&ctx, page.Size(), page.BaseRowId());
    dh::DeviceUVector<cuda_impl::RowIndexT> ridx_tmp(page.Size());
    std::vector<RegTree::Node> splits{tree[0]};
    page.Impl()->Visit(&ctx, {}, [&](auto&& acc) {
      partitioners.back()->UpdatePositionBatch(&ctx, {0}, {1}, {2}, splits, dh::ToSpan(ridx_tmp),
                                               LessThanOp{acc});
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
  auto n_left_pos = thrust::count_if(
      position.cbegin(), position.cend(),
      [=] XGBOOST_DEVICE(bst_node_t v) { return SamplePosition::Decode(v) == node.LeftChild(); });
  ASSERT_EQ(n_left, n_left_pos);

  std::vector<bst_node_t> h_position(position.size());
  dh::CopyDeviceSpanToVector(&h_position, dh::ToSpan(position));
  for (bst_idx_t ridx = 0; ridx < n_rows; ++ridx) {
    EXPECT_EQ(SamplePosition::IsValid(h_position[ridx]), ridx < n_rows / 2);
  }

  thrust::transform(thrust::device, position.cbegin(), position.cend(), position.begin(),
                    [] XGBOOST_DEVICE(bst_node_t nidx) { return SamplePosition::Decode(nidx); });
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
    dh::DeviceUVector<cuda_impl::RowIndexT> ridx_tmp(n_samples);
    partitioner.UpdatePositionBatch(
        &ctx, {0}, {1}, {2}, splits, dh::ToSpan(ridx_tmp),
        [] XGBOOST_DEVICE(bst_idx_t ridx, RegTree::Node) { return ridx < 3; });
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
