/**
 * Copyright 2022-2023 by XGBoost contributors.
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>                         // for bst_node_t
#include <xgboost/context.h>                      // for Context

#include <algorithm>                              // for transform
#include <iterator>                               // for distance
#include <vector>                                 // for vector

#include "../../../src/common/numeric.h"          // for ==RunLengthEncode
#include "../../../src/common/row_set.h"          // for RowSetCollection
#include "../../../src/data/gradient_index.h"     // for GHistIndexMatrix
#include "../../../src/tree/common_row_partitioner.h"
#include "../../../src/tree/hist/expand_entry.h"  // for CPUExpandEntry
#include "../helpers.h"                           // for RandomDataGenerator
#include "test_partitioner.h"                     // for GetSplit

namespace xgboost::tree {
namespace {
void TestLeafPartition(size_t n_samples) {
  size_t const n_features = 2, base_rowid = 0;
  Context ctx;
  common::RowSetCollection row_set;
  CommonRowPartitioner partitioner{&ctx, n_samples, base_rowid, false};

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  std::vector<CPUExpandEntry> candidates{{0, 0}};
  candidates.front().split.loss_chg = 0.4;
  RegTree tree;
  std::vector<float> hess(n_samples, 0);
  // emulate sampling
  auto not_sampled = [](size_t i) {
    size_t const kSampleFactor{3};
    return i % kSampleFactor != 0;
  };
  for (size_t i = 0; i < hess.size(); ++i) {
    if (not_sampled(i)) {
      hess[i] = 1.0f;
    }
  }

  std::vector<size_t> h_nptr;
  float split_value{0};
  for (auto const& page : Xy->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{64, 0.2})) {
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

TEST(CommonRowPartitioner, LeafPartition) {
  for (auto n_samples : {0ul, 1ul, 128ul, 256ul}) {
    TestLeafPartition(n_samples);
  }
}
}  // namespace xgboost::tree
