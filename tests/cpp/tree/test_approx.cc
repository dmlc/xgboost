/*!
 * Copyright 2021-2022, XGBoost contributors.
 */
#include <gtest/gtest.h>

#include "../../../src/tree/updater_approx.h"
#include "../helpers.h"
#include "test_partitioner.h"

namespace xgboost {
namespace tree {
TEST(Approx, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;
  ApproxRowPartitioner partitioner{n_samples, base_rowid};
  ASSERT_EQ(partitioner.base_rowid, base_rowid);
  ASSERT_EQ(partitioner.Size(), 1);
  ASSERT_EQ(partitioner.Partitions()[0].Size(), n_samples);

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  GenericParameter ctx;
  ctx.InitAllowUnknown(Args{});
  std::vector<CPUExpandEntry> candidates{{0, 0, 0.4}};

  auto grad = GenerateRandomGradients(n_samples);
  std::vector<float> hess(grad.Size());
  std::transform(grad.HostVector().cbegin(), grad.HostVector().cend(), hess.begin(),
                 [](auto gpair) { return gpair.GetHess(); });

  for (auto const &page : Xy->GetBatches<GHistIndexMatrix>({64, hess, true})) {
    bst_feature_t const split_ind = 0;
    {
      auto min_value = page.cut.MinValues()[split_ind];
      RegTree tree;
      ApproxRowPartitioner partitioner{n_samples, base_rowid};
      GetSplit(&tree, min_value, &candidates);
      partitioner.UpdatePosition(&ctx, page, candidates, &tree);
      ASSERT_EQ(partitioner.Size(), 3);
      ASSERT_EQ(partitioner[1].Size(), 0);
      ASSERT_EQ(partitioner[2].Size(), n_samples);
    }
    {
      ApproxRowPartitioner partitioner{n_samples, base_rowid};
      auto ptr = page.cut.Ptrs()[split_ind + 1];
      float split_value = page.cut.Values().at(ptr / 2);
      RegTree tree;
      GetSplit(&tree, split_value, &candidates);
      auto left_nidx = tree[RegTree::kRoot].LeftChild();
      partitioner.UpdatePosition(&ctx, page, candidates, &tree);

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
}  // namespace tree
}  // namespace xgboost
