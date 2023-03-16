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

  auto cuts = common::SketchOnDMatrix(Xy.get(), 64, ctx.Threads());

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

TEST(QuantileHist, Partitioner) { TestPartitioner<CPUExpandEntry>(1); }

TEST(QuantileHist, MultiPartitioner) { TestPartitioner<MultiExpandEntry>(3); }
}  // namespace xgboost::tree
