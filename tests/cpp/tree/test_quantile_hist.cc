/**
 * Copyright 2018-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/gradient.h>  // for GradientContainer
#include <xgboost/host_device_vector.h>
#include <xgboost/linalg.h>
#include <xgboost/tree_updater.h>

#include <cmath>
#include <cstddef>  // for size_t
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../../../src/tree/common_row_partitioner.h"
#include "../../../src/tree/hist/expand_entry.h"  // for MultiExpandEntry, CPUExpandEntry
#include "../collective/test_worker.h"            // for TestDistributedGlobal
#include "../helpers.h"
#include "test_partitioner.h"
#include "xgboost/data.h"
#include "xgboost/task.h"

namespace xgboost::tree {
namespace {
template <typename ExpandEntry>
void TestPartitioner(bst_target_t n_targets) {
  std::size_t n_samples = 1024, base_rowid = 0;
  bst_feature_t n_features = 1;

  Context ctx;
  ctx.InitAllowUnknown(Args{});

  CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid};
  ASSERT_EQ(partitioner.base_rowid, base_rowid);
  ASSERT_EQ(partitioner.Size(), 1);
  ASSERT_EQ(partitioner.Partitions()[0].Size(), n_samples);

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  std::vector<ExpandEntry> candidates{{0, 0}};
  candidates.front().split.loss_chg = 0.4;

  auto cuts = common::SketchOnDMatrix(&ctx, Xy.get(), 64);

  for (auto const& page : Xy->GetBatches<SparsePage>()) {
    GHistIndexMatrix gmat{&ctx, page, {}, cuts, 64, true, 0.5};
    bst_feature_t const split_ind = 0;
    common::ColumnMatrix column_indices;
    column_indices.InitFromSparse(page, gmat, 0.5, ctx.Threads());
    {
      auto min_value = -std::numeric_limits<float>::infinity();
      RegTree tree{n_targets, n_features};
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid};
      if constexpr (std::is_same_v<ExpandEntry, CPUExpandEntry>) {
        GetSplit(&tree, min_value, &candidates);
        partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates,
                                                tree.HostScView());
      } else {
        GetMultiSplitForTest(&tree, min_value, &candidates);
        partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates,
                                                tree.HostMtView());
      }
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid};
      auto ptr = gmat.cut.Ptrs()[split_ind + 1];
      float split_value = gmat.cut.Values().at(ptr / 2);
      RegTree tree{n_targets, n_features};
      if constexpr (std::is_same_v<ExpandEntry, CPUExpandEntry>) {
        GetSplit(&tree, split_value, &candidates);
        partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates,
                                                tree.HostScView());
      } else {
        GetMultiSplitForTest(&tree, split_value, &candidates);
        partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates,
                                                tree.HostMtView());
      }

      {
        auto left_nidx = tree.LeftChild(RegTree::kRoot);
        auto const& elem = partitioner[left_nidx];
        ASSERT_LT(elem.Size(), n_samples);
        ASSERT_GT(elem.Size(), 1);
        for (auto& it : elem) {
          auto value = gmat.cut.Values().at(gmat.index[it]);
          ASSERT_LE(value, split_value);
        }
      }
      {
        auto right_nidx = tree.RightChild(RegTree::kRoot);
        auto const& elem = partitioner[right_nidx];
        for (auto& it : elem) {
          auto value = gmat.cut.Values().at(gmat.index[it]);
          ASSERT_GT(value, split_value);
        }
      }
    }
  }
}
}  // anonymous namespace

TEST(QuantileHist, Partitioner) { TestPartitioner<CPUExpandEntry>(1); }

TEST(QuantileHist, MultiPartitioner) { TestPartitioner<MultiExpandEntry>(3); }

namespace {
// Verify that partitioners from a previous multi-batch DMatrix are not reused.
void TestPartitionerOverrun(bst_target_t n_targets) {
  // Update with a multi-batch DMatrix, then reuse the updater on a single-batch DMatrix
  // with the same number of rows. Partitioners left over from the first DMatrix would
  // overwrite leaf positions of the second one, so the positions must match the ones from
  // a fresh updater. Both DMatrix objects have the same number of rows to keep any stale
  // write inside the position vector instead of its reserved storage.
  constexpr bst_idx_t kRows = 1 << 16;
  constexpr int kCols = 3;

  Context ctx;
  ctx.InitAllowUnknown(Args{{"nthread", "1"}});

  ObjInfo task{ObjInfo::kRegression, true};
  auto make_updater = [&] {
    auto updater =
        std::unique_ptr<TreeUpdater>{TreeUpdater::Create("grow_quantile_histmaker", &ctx, &task)};
    updater->Configure(Args{});
    return updater;
  };

  TrainParam param;
  param.InitAllowUnknown(Args{{"max_depth", "1"},
                              {"max_bin", "32"},
                              {"lambda", "0"},
                              {"gamma", "0"},
                              {"min_child_weight", "0"}});

  auto update = [&](TreeUpdater* updater, DMatrix* dmat) {
    // Random gradients so that the tree has more than one leaf.
    auto gpair = GenerateRandomGradients(&ctx, dmat->Info().num_row_, n_targets);

    RegTree tree{n_targets, static_cast<bst_feature_t>(kCols)};
    std::vector<RegTree*> trees{&tree};
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    updater->Update(&param, &gpair, dmat, common::Span{position.data(), 1}, trees);
    return position.front().ConstHostVector();
  };

  auto dmat_multi = RandomDataGenerator{kRows, kCols, 0.0f}.Batches(8).GenerateSparsePageDMatrix(
      "part_resize_multi_first", true);
  // In-memory DMatrix with a single page. Unlike the external memory one above, it uses
  // the seed, so the two DMatrix objects produce different trees.
  auto dmat_single = RandomDataGenerator{kRows, kCols, 0.0f}.Seed(1).GenerateDMatrix(false);

  auto reused = make_updater();
  update(reused.get(), dmat_multi.get());
  auto position = update(reused.get(), dmat_single.get());

  auto fresh = make_updater();
  auto expected = update(fresh.get(), dmat_single.get());

  ASSERT_EQ(position.size(), kRows);
  EXPECT_EQ(position, expected) << "Leaf positions were written by a stale partitioner left "
                                   "over from the previous multi-batch DMatrix.";
}
}  // anonymous namespace

TEST(QuantileHist, HistUpdaterPartitionerOverrun) { TestPartitionerOverrun(1); }

TEST(QuantileHist, MultiTargetHistBuilderPartitionerOverrun) { TestPartitionerOverrun(3); }
}  // namespace xgboost::tree
