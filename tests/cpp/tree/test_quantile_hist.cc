/*!
 * Copyright 2018-2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_updater.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../../../src/tree/param.h"
#include "../../../src/tree/split_evaluator.h"
#include "../../../src/tree/updater_quantile_hist.h"
#include "../helpers.h"
#include "test_partitioner.h"
#include "xgboost/data.h"

namespace xgboost {
namespace tree {
TEST(QuantileHist, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;
  GenericParameter ctx;
  ctx.InitAllowUnknown(Args{});

  HistRowPartitioner partitioner{n_samples, base_rowid, ctx.Threads()};
  ASSERT_EQ(partitioner.base_rowid, base_rowid);
  ASSERT_EQ(partitioner.Size(), 1);
  ASSERT_EQ(partitioner.Partitions()[0].Size(), n_samples);

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  std::vector<CPUExpandEntry> candidates{{0, 0, 0.4}};

  auto cuts = common::SketchOnDMatrix(Xy.get(), 64, ctx.Threads());

  for (auto const& page : Xy->GetBatches<SparsePage>()) {
    GHistIndexMatrix gmat;
    gmat.Init(page, {}, cuts, 64, false, 0.5, ctx.Threads());
    bst_feature_t const split_ind = 0;
    common::ColumnMatrix column_indices;
    column_indices.Init(page, gmat, 0.5, ctx.Threads());
    {
      auto min_value = gmat.cut.MinValues()[split_ind];
      RegTree tree;
      HistRowPartitioner partitioner{n_samples, base_rowid, ctx.Threads()};
      GetSplit(&tree, min_value, &candidates);
      partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates, &tree);
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      HistRowPartitioner partitioner{n_samples, base_rowid, ctx.Threads()};
      auto ptr = gmat.cut.Ptrs()[split_ind + 1];
      float split_value = gmat.cut.Values().at(ptr / 2);
      RegTree tree;
      GetSplit(&tree, split_value, &candidates);
      auto left_nidx = tree[RegTree::kRoot].LeftChild();
      partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates, &tree);

      auto elem = partitioner[left_nidx];
      ASSERT_LT(elem.Size(), n_samples);
      ASSERT_GT(elem.Size(), 1);
      for (auto it = elem.begin; it != elem.end; ++it) {
        auto value = gmat.cut.Values().at(gmat.index[*it]);
        ASSERT_LE(value, split_value);
      }
      auto right_nidx = tree[RegTree::kRoot].RightChild();
      elem = partitioner[right_nidx];
      for (auto it = elem.begin; it != elem.end; ++it) {
        auto value = gmat.cut.Values().at(gmat.index[*it]);
        ASSERT_GT(value, split_value) << *it;
      }
    }
  }
}
}  // namespace tree
}  // namespace xgboost
