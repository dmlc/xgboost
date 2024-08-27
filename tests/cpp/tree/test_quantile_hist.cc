/**
 * Copyright 2018-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_updater.h>

#include <algorithm>
#include <cstddef>  // for size_t
#include <string>
#include <vector>

#include "../../../src/tree/common_row_partitioner.h"
#include "../../../src/tree/hist/expand_entry.h"  // for MultiExpandEntry, CPUExpandEntry
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../helpers.h"
#include "test_column_split.h"  // for TestColumnSplit
#include "test_partitioner.h"
#include "xgboost/data.h"

namespace xgboost::tree {
namespace {
template <typename ExpandEntry>
void TestPartitioner(bst_target_t n_targets) {
  std::size_t n_samples = 1024, base_rowid = 0;
  bst_feature_t n_features = 1;

  Context ctx;
  ctx.InitAllowUnknown(Args{});

  CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, false};
  ASSERT_EQ(partitioner.base_rowid, base_rowid);
  ASSERT_EQ(partitioner.Size(), 1);
  ASSERT_EQ(partitioner.Partitions()[0].Size(), n_samples);

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  std::vector<ExpandEntry> candidates{{0, 0}};
  candidates.front().split.loss_chg = 0.4;

  auto cuts = common::SketchOnDMatrix(&ctx, Xy.get(), 64);

  for (auto const& page : Xy->GetBatches<SparsePage>()) {
    GHistIndexMatrix gmat(page, {}, cuts, 64, true, 0.5, ctx.Threads());
    bst_feature_t const split_ind = 0;
    common::ColumnMatrix column_indices;
    column_indices.InitFromSparse(page, gmat, 0.5, ctx.Threads());
    {
      auto min_value = gmat.cut.MinValues()[split_ind];
      RegTree tree{n_targets, n_features};
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, false};
      if constexpr (std::is_same_v<ExpandEntry, CPUExpandEntry>) {
        GetSplit(&tree, min_value, &candidates);
      } else {
        GetMultiSplitForTest(&tree, min_value, &candidates);
      }
      partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates, &tree);
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, false};
      auto ptr = gmat.cut.Ptrs()[split_ind + 1];
      float split_value = gmat.cut.Values().at(ptr / 2);
      RegTree tree{n_targets, n_features};
      if constexpr (std::is_same<ExpandEntry, CPUExpandEntry>::value) {
        GetSplit(&tree, split_value, &candidates);
      } else {
        GetMultiSplitForTest(&tree, split_value, &candidates);
      }
      auto left_nidx = tree.LeftChild(RegTree::kRoot);
      partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates, &tree);

      auto elem = partitioner[left_nidx];
      ASSERT_LT(elem.Size(), n_samples);
      ASSERT_GT(elem.Size(), 1);
      for (auto it = elem.begin; it != elem.end; ++it) {
        auto value = gmat.cut.Values().at(gmat.index[*it]);
        ASSERT_LE(value, split_value);
      }
      auto right_nidx = tree.RightChild(RegTree::kRoot);
      elem = partitioner[right_nidx];
      for (auto it = elem.begin; it != elem.end; ++it) {
        auto value = gmat.cut.Values().at(gmat.index[*it]);
        ASSERT_GT(value, split_value);
      }
    }
  }
}
}  // anonymous namespace

TEST(QuantileHist, Partitioner) { TestPartitioner<CPUExpandEntry>(1); }

TEST(QuantileHist, MultiPartitioner) { TestPartitioner<MultiExpandEntry>(3); }

namespace {

template <typename ExpandEntry>
void VerifyColumnSplitPartitioner(bst_target_t n_targets, size_t n_samples,
                                  bst_feature_t n_features, size_t base_rowid,
                                  std::shared_ptr<DMatrix> Xy, float min_value, float mid_value,
                                  CommonRowPartitioner const& expected_mid_partitioner) {
  auto dmat =
      std::unique_ptr<DMatrix>{Xy->SliceCol(collective::GetWorldSize(), collective::GetRank())};

  Context ctx;
  ctx.InitAllowUnknown(Args{});

  std::vector<ExpandEntry> candidates{{0, 0}};
  candidates.front().split.loss_chg = 0.4;
  auto cuts = common::SketchOnDMatrix(&ctx, dmat.get(), 64);

  for (auto const& page : Xy->GetBatches<SparsePage>()) {
    GHistIndexMatrix gmat(page, {}, cuts, 64, true, 0.5, ctx.Threads());
    common::ColumnMatrix column_indices;
    column_indices.InitFromSparse(page, gmat, 0.5, ctx.Threads());
    {
      RegTree tree{n_targets, n_features};
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, true};
      if constexpr (std::is_same<ExpandEntry, CPUExpandEntry>::value) {
        GetSplit(&tree, min_value, &candidates);
      } else {
        GetMultiSplitForTest(&tree, min_value, &candidates);
      }
      partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates, &tree);
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      RegTree tree{n_targets, n_features};
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, true};
      if constexpr (std::is_same<ExpandEntry, CPUExpandEntry>::value) {
        GetSplit(&tree, mid_value, &candidates);
      } else {
        GetMultiSplitForTest(&tree, mid_value, &candidates);
      }
      auto left_nidx = tree.LeftChild(RegTree::kRoot);
      partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates, &tree);

      auto elem = partitioner[left_nidx];
      ASSERT_LT(elem.Size(), n_samples);
      ASSERT_GT(elem.Size(), 1);
      auto expected_elem = expected_mid_partitioner[left_nidx];
      ASSERT_EQ(elem.Size(), expected_elem.Size());
      for (auto it = elem.begin, eit = expected_elem.begin; it != elem.end; ++it, ++eit) {
        ASSERT_EQ(*it, *eit);
      }

      auto right_nidx = tree.RightChild(RegTree::kRoot);
      elem = partitioner[right_nidx];
      expected_elem = expected_mid_partitioner[right_nidx];
      ASSERT_EQ(elem.Size(), expected_elem.Size());
      for (auto it = elem.begin, eit = expected_elem.begin; it != elem.end; ++it, ++eit) {
        ASSERT_EQ(*it, *eit);
      }
    }
  }
}

template <typename ExpandEntry>
void TestColumnSplitPartitioner(bst_target_t n_targets) {
  std::size_t n_samples = 1024, base_rowid = 0;
  bst_feature_t n_features = 16;
  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  std::vector<ExpandEntry> candidates{{0, 0}};
  candidates.front().split.loss_chg = 0.4;

  Context ctx;
  ctx.InitAllowUnknown(Args{});
  auto cuts = common::SketchOnDMatrix(&ctx, Xy.get(), 64);

  float min_value, mid_value;
  CommonRowPartitioner mid_partitioner{&ctx, n_samples, base_rowid, false};
  for (auto const& page : Xy->GetBatches<SparsePage>()) {
    GHistIndexMatrix gmat(page, {}, cuts, 64, true, 0.5, ctx.Threads());
    bst_feature_t const split_ind = 0;
    common::ColumnMatrix column_indices;
    column_indices.InitFromSparse(page, gmat, 0.5, ctx.Threads());
    min_value = gmat.cut.MinValues()[split_ind];

    auto ptr = gmat.cut.Ptrs()[split_ind + 1];
    mid_value = gmat.cut.Values().at(ptr / 2);
    RegTree tree{n_targets, n_features};
    if constexpr (std::is_same<ExpandEntry, CPUExpandEntry>::value) {
      GetSplit(&tree, mid_value, &candidates);
    } else {
      GetMultiSplitForTest(&tree, mid_value, &candidates);
    }
    mid_partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates, &tree);
  }

  auto constexpr kWorkers = 4;
  collective::TestDistributedGlobal(kWorkers, [&] {
    VerifyColumnSplitPartitioner<ExpandEntry>(n_targets, n_samples, n_features, base_rowid, Xy,
                                              min_value, mid_value, mid_partitioner);
  });
}
}  // anonymous namespace

TEST(QuantileHist, PartitionerColumnSplit) { TestColumnSplitPartitioner<CPUExpandEntry>(1); }

TEST(QuantileHist, MultiPartitionerColumnSplit) { TestColumnSplitPartitioner<MultiExpandEntry>(3); }

namespace {
class TestHistColumnSplit : public ::testing::TestWithParam<std::tuple<bst_target_t, bool, float>> {
 public:
  void Run() {
    auto [n_targets, categorical, sparsity] = GetParam();
    TestColumnSplit(n_targets, categorical, "grow_quantile_histmaker", sparsity);
  }
};
}  // anonymous namespace

TEST_P(TestHistColumnSplit, Basic) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(ColumnSplit, TestHistColumnSplit, ::testing::ValuesIn([]() {
                           std::vector<std::tuple<bst_target_t, bool, float>> params;
                           for (auto categorical : {true, false}) {
                             for (auto sparsity : {0.0f, 0.6f}) {
                               for (bst_target_t n_targets : {1u, 3u}) {
                                 params.emplace_back(n_targets, categorical, sparsity);
                               }
                             }
                           }
                           return params;
                         }()));
}  // namespace xgboost::tree
