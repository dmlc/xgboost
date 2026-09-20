/**
 * Copyright 2021-2025, XGBoost contributors.
 */
#include <gtest/gtest.h>
#include <xgboost/gradient.h>      // for GradientContainer
#include <xgboost/tree_model.h>    // for RegTree
#include <xgboost/tree_updater.h>  // for TreeUpdater

#include <algorithm>  // for transform
#include <limits>
#include <memory>  // for unique_ptr
#include <vector>  // for vector

#include "../../../src/tree/common_row_partitioner.h"
#include "../../../src/tree/param.h"    // for TrainParam
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../helpers.h"
#include "test_partitioner.h"

namespace xgboost::tree {
namespace {
std::vector<float> GenerateHess(size_t n_samples) {
  auto grad = GenerateRandomGradients(n_samples);
  std::vector<float> hess(grad.Size());
  std::transform(grad.HostVector().cbegin(), grad.HostVector().cend(), hess.begin(),
                 [](auto gpair) { return gpair.GetHess(); });
  return hess;
}
}  // anonymous namespace

TEST(Approx, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;
  Context ctx;
  ctx.InitAllowUnknown(Args{});
  CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid};
  ASSERT_EQ(partitioner.base_rowid, base_rowid);
  ASSERT_EQ(partitioner.Size(), 1);
  ASSERT_EQ(partitioner.Partitions()[0].Size(), n_samples);

  auto const Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  auto hess = GenerateHess(n_samples);
  std::vector<CPUExpandEntry> candidates{{0, 0}};
  candidates.front().split.loss_chg = 0.4;

  for (auto const& page : Xy->GetBatches<GHistIndexMatrix>(&ctx, {64, hess, true})) {
    bst_feature_t const split_ind = 0;
    {
      auto min_value = -std::numeric_limits<float>::infinity();
      RegTree tree;
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid};
      GetSplit(&tree, min_value, &candidates);
      partitioner.UpdatePosition(&ctx, page, candidates, tree.HostScView());
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid};
      auto ptr = page.cut.Ptrs()[split_ind + 1];
      float split_value = page.cut.Values().at(ptr / 2);
      RegTree tree;
      GetSplit(&tree, split_value, &candidates);
      partitioner.UpdatePosition(&ctx, page, candidates, tree.HostScView());

      {
        auto left_nidx = tree[RegTree::kRoot].LeftChild();
        auto const& elem = partitioner[left_nidx];
        ASSERT_LT(elem.Size(), n_samples);
        ASSERT_GT(elem.Size(), 1);
        for (auto& it : elem) {
          auto value = page.cut.Values().at(page.index[it]);
          ASSERT_LE(value, split_value);
        }
      }
      {
        auto right_nidx = tree[RegTree::kRoot].RightChild();
        auto const& elem = partitioner[right_nidx];
        for (auto& it : elem) {
          auto value = page.cut.Values().at(page.index[it]);
          ASSERT_GT(value, split_value) << it;
        }
      }
    }
  }
}

TEST(Approx, InteractionConstraint) {
  auto constexpr kRows = 32;
  auto constexpr kCols = 16;
  auto p_dmat = RandomDataGenerator{kRows, kCols, 0.6f}.GenerateDMatrix();
  Context ctx;

  GradientContainer gpair = GenerateRandomGradients(&ctx, kRows, 1);

  ObjInfo task{ObjInfo::kRegression};
  {
    // With constraints
    RegTree tree{1, kCols};

    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_histmaker", &ctx, &task)};
    TrainParam param;
    param.UpdateAllowUnknown(
        Args{{"interaction_constraints", "[[0, 1]]"}, {"num_feature", std::to_string(kCols)}});
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    updater->Configure(Args{});
    updater->Update(&param, &gpair, p_dmat.get(), position, {&tree});

    ASSERT_EQ(tree.NumExtraNodes(), 4);
    ASSERT_EQ(tree[0].SplitIndex(), 1);

    ASSERT_EQ(tree[tree[0].LeftChild()].SplitIndex(), 0);
    ASSERT_EQ(tree[tree[0].RightChild()].SplitIndex(), 0);
  }
  {
    // Without constraints
    RegTree tree{1u, kCols};

    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_histmaker", &ctx, &task)};
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    TrainParam param;
    param.Init(Args{});
    updater->Configure(Args{});
    updater->Update(&param, &gpair, p_dmat.get(), position, {&tree});

    ASSERT_EQ(tree.NumExtraNodes(), 10);
    ASSERT_EQ(tree[0].SplitIndex(), 1);

    ASSERT_NE(tree[tree[0].LeftChild()].SplitIndex(), 0);
    ASSERT_NE(tree[tree[0].RightChild()].SplitIndex(), 0);
  }
}
}  // namespace xgboost::tree
