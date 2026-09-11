/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <thrust/reverse.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "../../../src/cross_validate/kfolds.cuh"
#include "../../../src/tree/gpu_hist/row_partitioner.cuh"
#include "../helpers.h"

namespace xgboost::cv {
TEST(FoldAssignmentGPU, RangesAndPartitioner) {
  auto ctx = MakeCUDACtx(0);
  std::vector<std::int64_t> expected{0, 0, 2, 1, 2, 0, 1, 2, 1, 0};
  dh::DeviceUVector<std::int64_t> source(expected.size());
  dh::safe_cuda(cudaMemcpy(source.data(), expected.data(), expected.size() * sizeof(std::int64_t),
                           cudaMemcpyHostToDevice));
  FoldAssignment assignment{&ctx, 3, dh::ToSpan(source)};
  thrust::fill(ctx.CUDACtx()->CTP(), source.begin(), source.end(), 0);  // Import owns its copy.
  EXPECT_EQ(assignment.Counts(), (std::vector<bst_idx_t>{4, 3, 3}));
  auto first = assignment.ReadRows({1, 7});
  auto second = assignment.ReadRows({4, 9});
  EXPECT_EQ(first.ids.data(), assignment.Ids().data() + 1);
  EXPECT_EQ(second.ids.data(), assignment.Ids().data() + 4);
  EXPECT_THROW(static_cast<void>(assignment.ReadRows({9, 11})), dmlc::Error);
  EXPECT_THROW(static_cast<void>(assignment.ReadRows({2, 1})), dmlc::Error);
  tree::RowPartitionerBatches partitions;
  for (auto const& boundaries :
       {std::vector<bst_idx_t>{0, 2, 2, 3, 7, 10}, std::vector<bst_idx_t>{0, 3, 8, 10}}) {
    for (std::size_t k = 0; k < assignment.KFolds(); ++k) {
      std::vector<bst_idx_t> counts;
      for (std::size_t b = 1; b < boundaries.size(); ++b) {
        auto begin = boundaries[b - 1], end = boundaries[b];
        auto valid = assignment.CountValidation(&ctx, {begin, end})[k];
        EXPECT_EQ(valid, std::count(expected.begin() + begin, expected.begin() + end, k));
        counts.push_back(end - begin - valid);
      }
      for (std::size_t round = 0; round < 2; ++round) {
        partitions.Reset(boundaries, counts, [&](bst_idx_t begin, bst_idx_t end, auto out) {
          assignment.ReadRows({begin, end}).Select(&ctx, k, false, out);
        });
        for (std::size_t b = 0; b + 1 < boundaries.size(); ++b) {
          std::vector<tree::RowPartitioner::RowIndexT> train;
          for (auto r = boundaries[b]; r < boundaries[b + 1]; ++r) {
            if (expected[r] != static_cast<std::int64_t>(k)) {
              train.push_back(r);
            }
          }
          EXPECT_EQ(partitions[b]->GetRowsHost(0), train);
          // Another tree must reset the mutated roots, including empty subsets/pages.
          auto rows = partitions[b]->GetRows();
          auto ptr = const_cast<tree::RowPartitioner::RowIndexT*>(rows.data());
          thrust::reverse(ctx.CUDACtx()->CTP(), ptr, ptr + rows.size());
        }
      }
    }
  }
}

TEST(FoldAssignmentGPU, RowIndexLimit) {
  auto ctx = MakeCUDACtx(0);
  tree::RowPartitioner partition;
  auto limit = std::numeric_limits<tree::RowPartitioner::RowIndexT>::max();
  EXPECT_THROW(partition.Reset(&ctx, 1, static_cast<bst_idx_t>(limit) + 1), dmlc::Error);
  EXPECT_THROW(partition.Reset(&ctx, static_cast<bst_idx_t>(limit) + 1, 0), dmlc::Error);
}
}  // namespace xgboost::cv
