/**
 * Copyright 2021-2023 by XGBoost contributors.
 */
#include <gtest/gtest.h>

#include "../../../src/common/numeric.h"
#include "../../../src/tree/common_row_partitioner.h"
#include "../helpers.h"
#include "test_partitioner.h"

namespace xgboost {
namespace tree {
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
  CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, false};
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
      auto min_value = page.cut.MinValues()[split_ind];
      RegTree tree;
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, false};
      GetSplit(&tree, min_value, &candidates);
      partitioner.UpdatePosition(&ctx, page, candidates, &tree);
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, false};
      auto ptr = page.cut.Ptrs()[split_ind + 1];
      float split_value = page.cut.Values().at(ptr / 2);
      RegTree tree;
      GetSplit(&tree, split_value, &candidates);
      partitioner.UpdatePosition(&ctx, page, candidates, &tree);

      auto left_nidx = tree[RegTree::kRoot].LeftChild();
      auto elem = partitioner[left_nidx];
      ASSERT_LT(elem.Size(), n_samples);
      ASSERT_GT(elem.Size(), 1);
      for (auto it = elem.begin; it != elem.end; ++it) {
        auto value = page.cut.Values().at(page.index[*it]);
        ASSERT_LE(value, split_value);
      }

      auto right_nidx = tree[RegTree::kRoot].RightChild();
      elem = partitioner[right_nidx];
      for (auto it = elem.begin; it != elem.end; ++it) {
        auto value = page.cut.Values().at(page.index[*it]);
        ASSERT_GT(value, split_value) << *it;
      }
    }
  }
}

namespace {
void TestColumnSplitPartitioner(size_t n_samples, size_t base_rowid, std::shared_ptr<DMatrix> Xy,
                                std::vector<float>* hess, float min_value, float mid_value,
                                CommonRowPartitioner const& expected_mid_partitioner) {
  auto dmat =
      std::unique_ptr<DMatrix>{Xy->SliceCol(collective::GetWorldSize(), collective::GetRank())};
  std::vector<CPUExpandEntry> candidates{{0, 0}};
  candidates.front().split.loss_chg = 0.4;

  Context ctx;
  ctx.InitAllowUnknown(Args{});
  for (auto const& page : dmat->GetBatches<GHistIndexMatrix>(&ctx, {64, *hess, true})) {
    {
      RegTree tree;
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, true};
      GetSplit(&tree, min_value, &candidates);
      partitioner.UpdatePosition(&ctx, page, candidates, &tree);
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, true};
      RegTree tree;
      GetSplit(&tree, mid_value, &candidates);
      partitioner.UpdatePosition(&ctx, page, candidates, &tree);

      auto left_nidx = tree[RegTree::kRoot].LeftChild();
      auto elem = partitioner[left_nidx];
      ASSERT_LT(elem.Size(), n_samples);
      ASSERT_GT(elem.Size(), 1);
      auto expected_elem = expected_mid_partitioner[left_nidx];
      ASSERT_EQ(elem.Size(), expected_elem.Size());
      for (auto it = elem.begin, eit = expected_elem.begin; it != elem.end; ++it, ++eit) {
        ASSERT_EQ(*it, *eit);
      }

      auto right_nidx = tree[RegTree::kRoot].RightChild();
      elem = partitioner[right_nidx];
      expected_elem = expected_mid_partitioner[right_nidx];
      ASSERT_EQ(elem.Size(), expected_elem.Size());
      for (auto it = elem.begin, eit = expected_elem.begin; it != elem.end; ++it, ++eit) {
        ASSERT_EQ(*it, *eit);
      }
    }
  }
}
}  // anonymous namespace

TEST(Approx, PartitionerColSplit) {
  size_t n_samples = 1024, n_features = 16, base_rowid = 0;
  auto const Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  auto hess = GenerateHess(n_samples);
  std::vector<CPUExpandEntry> candidates{{0, 0}};
  candidates.front().split.loss_chg = 0.4;

  float min_value, mid_value;
  Context ctx;
  ctx.InitAllowUnknown(Args{});
  CommonRowPartitioner mid_partitioner{&ctx, n_samples, base_rowid, false};
  for (auto const& page : Xy->GetBatches<GHistIndexMatrix>(&ctx, {64, hess, true})) {
    bst_feature_t const split_ind = 0;
    min_value = page.cut.MinValues()[split_ind];

    auto ptr = page.cut.Ptrs()[split_ind + 1];
    mid_value = page.cut.Values().at(ptr / 2);
    RegTree tree;
    GetSplit(&tree, mid_value, &candidates);
    mid_partitioner.UpdatePosition(&ctx, page, candidates, &tree);
  }

  auto constexpr kWorkers = 4;
  RunWithInMemoryCommunicator(kWorkers, TestColumnSplitPartitioner, n_samples, base_rowid, Xy,
                              &hess, min_value, mid_value, mid_partitioner);
}
}  // namespace tree
}  // namespace xgboost
