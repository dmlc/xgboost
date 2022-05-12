/*!
 * Copyright 2021-2022, XGBoost contributors.
 */
#include <gtest/gtest.h>

#include "../../../src/common/numeric.h"
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

  for (auto const& page : Xy->GetBatches<GHistIndexMatrix>({64, hess, true})) {
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
void TestLeafPartition(size_t n_samples) {
  size_t const n_features = 2, base_rowid = 0;
  common::RowSetCollection row_set;
  ApproxRowPartitioner partitioner{n_samples, base_rowid};

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  GenericParameter ctx;
  std::vector<CPUExpandEntry> candidates{{0, 0, 0.4}};
  RegTree tree;
  std::vector<float> hess(n_samples, 0);
  // emulate sampling
  auto not_sampled = [](size_t i) {
    size_t const kSampleFactor{3};
    return i % kSampleFactor != 0;
  };
  size_t n{0};
  for (size_t i = 0; i < hess.size(); ++i) {
    if (not_sampled(i)) {
      hess[i] = 1.0f;
      ++n;
    }
  }

  std::vector<size_t> h_nptr;
  float split_value{0};
  for (auto const& page : Xy->GetBatches<GHistIndexMatrix>({Context::kCpuId, 64})) {
    bst_feature_t const split_ind = 0;
    auto ptr = page.cut.Ptrs()[split_ind + 1];
    split_value = page.cut.Values().at(ptr / 2);
    GetSplit(&tree, split_value, &candidates);
    partitioner.UpdatePosition(&ctx, page, candidates, &tree);
    std::vector<bst_node_t> position;
    partitioner.LeafPartition(&ctx, tree, hess, &position);
    std::sort(position.begin(), position.end());
    size_t beg = std::distance(
        position.begin(),
        std::find_if(position.begin(), position.end(), [&](bst_node_t nidx) { return nidx >= 0; }));
    std::vector<size_t> nptr;
    common::RunLengthEncode(position.cbegin() + beg, position.cend(), &nptr);
    std::transform(nptr.begin(), nptr.end(), nptr.begin(), [&](size_t x) { return x + beg; });
    auto n_uniques = std::unique(position.begin() + beg, position.end()) - (position.begin() + beg);
    ASSERT_EQ(nptr.size(), n_uniques + 1);
    ASSERT_EQ(nptr[0], beg);
    ASSERT_EQ(nptr.back(), n_samples);

    h_nptr = nptr;
  }

  if (h_nptr.front() == n_samples) {
    return;
  }

  ASSERT_GE(h_nptr.size(), 2);

  for (auto const& page : Xy->GetBatches<SparsePage>()) {
    auto batch = page.GetView();
    size_t left{0};
    for (size_t i = 0; i < batch.Size(); ++i) {
      if (not_sampled(i) && batch[i].front().fvalue < split_value) {
        left++;
      }
    }
    ASSERT_EQ(left, h_nptr[1] - h_nptr[0]);  // equal to number of sampled assigned to left
  }
}
}  // anonymous namespace

TEST(Approx, LeafPartition) {
  for (auto n_samples : {0ul, 1ul, 128ul, 256ul}) {
    TestLeafPartition(n_samples);
  }
}
}  // namespace tree
}  // namespace xgboost
