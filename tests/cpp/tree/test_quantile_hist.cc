/**
 * Copyright 2018-2023 by XGBoost Contributors
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
#include "../../../src/tree/param.h"
#include "../../../src/tree/split_evaluator.h"
#include "../helpers.h"
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
  RunWithInMemoryCommunicator(kWorkers, VerifyColumnSplitPartitioner<ExpandEntry>, n_targets,
                              n_samples, n_features, base_rowid, Xy, min_value, mid_value,
                              mid_partitioner);
}
}  // anonymous namespace

TEST(QuantileHist, PartitionerColSplit) { TestColumnSplitPartitioner<CPUExpandEntry>(1); }

TEST(QuantileHist, MultiPartitionerColSplit) { TestColumnSplitPartitioner<MultiExpandEntry>(3); }

namespace {
void VerifyColumnSplit(bst_row_t rows, bst_feature_t cols, bst_target_t n_targets,
                       RegTree const& expected_tree) {
  auto Xy = RandomDataGenerator{rows, cols, 0}.GenerateDMatrix(true);
  auto p_gradients = GenerateGradients(rows, n_targets);
  Context ctx;
  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_quantile_histmaker", &ctx, &task)};
  std::vector<HostDeviceVector<bst_node_t>> position(1);

  std::unique_ptr<DMatrix> sliced{Xy->SliceCol(collective::GetWorldSize(), collective::GetRank())};

  RegTree tree{n_targets, cols};
  TrainParam param;
  param.Init(Args{});
  updater->Update(&param, p_gradients.get(), sliced.get(), position, {&tree});

  Json json{Object{}};
  tree.SaveModel(&json);
  Json expected_json{Object{}};
  expected_tree.SaveModel(&expected_json);
  ASSERT_EQ(json, expected_json);
}

void TestColumnSplit(bst_target_t n_targets) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;

  RegTree expected_tree{n_targets, kCols};
  ObjInfo task{ObjInfo::kRegression};
  {
    auto Xy = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true);
    auto p_gradients = GenerateGradients(kRows, n_targets);
    Context ctx;
    std::unique_ptr<TreeUpdater> updater{
        TreeUpdater::Create("grow_quantile_histmaker", &ctx, &task)};
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    TrainParam param;
    param.Init(Args{});
    updater->Update(&param, p_gradients.get(), Xy.get(), position, {&expected_tree});
  }

  auto constexpr kWorldSize = 2;
  RunWithInMemoryCommunicator(kWorldSize, VerifyColumnSplit, kRows, kCols, n_targets,
                              std::cref(expected_tree));
}
}  // anonymous namespace

TEST(QuantileHist, ColumnSplit) { TestColumnSplit(1); }

TEST(QuantileHist, ColumnSplitMultiTarget) { TestColumnSplit(3); }

}  // namespace xgboost::tree
