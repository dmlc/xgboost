/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include <cstddef>  // for size_t
#include <set>      // for set
#include <vector>   // for vector

#include "../../../src/cross_validate/kfolds.h"
#include "../helpers.h"  // for AssertVecEq, MakeCUDACtx

namespace xgboost::cv {
namespace {
using Indices = std::vector<bst_idx_t>;

struct ExpectedFold {
  Indices train;
  Indices valid;
};

void CheckKFold(std::size_t n_rows, std::vector<ExpectedFold> const& expected,
                bst_idx_t begin = 0) {
  auto ctx = MakeCUDACtx(0);
  auto k_folds = expected.size();

  FoldInfo out;
  for (std::size_t k = 0; k < k_folds; ++k) {
    KFold(&ctx, k_folds, begin, begin + n_rows, k, &out);
  }

  std::set<bst_idx_t> full_ridxs;
  for (std::size_t i = 0; i < n_rows; ++i) {
    full_ridxs.emplace(begin + static_cast<bst_idx_t>(i));
  }

  ASSERT_EQ(out.KFolds(), k_folds);
  ASSERT_EQ(out.valid_ridxs.size(), k_folds);
  for (std::size_t k = 0; k < k_folds; ++k) {
    ASSERT_EQ(out.ridxs[k].Device(), ctx.Device());
    ASSERT_EQ(out.valid_ridxs[k].Device(), ctx.Device());
    auto train_ridxs = out.ridxs[k].HostVector();
    auto valid_ridxs = out.valid_ridxs[k].HostVector();
    AssertVecEq(train_ridxs, expected[k].train);
    AssertVecEq(valid_ridxs, expected[k].valid);

    std::set<bst_idx_t> fold_ridxs{train_ridxs.cbegin(), train_ridxs.cend()};
    fold_ridxs.insert(valid_ridxs.cbegin(), valid_ridxs.cend());
    ASSERT_EQ(fold_ridxs, full_ridxs);
  }
}
}  // namespace

TEST(KFold, Indices) {
  CheckKFold(10, {{{4, 5, 6, 7, 8, 9}, {0, 1, 2, 3}},
                  {{0, 1, 2, 3, 7, 8, 9}, {4, 5, 6}},
                  {{0, 1, 2, 3, 4, 5, 6}, {7, 8, 9}}});
}

TEST(KFold, GlobalIndices) {
  CheckKFold(9,
             {{{13, 14, 15, 16, 17, 18}, {10, 11, 12}},
              {{10, 11, 12, 15, 16, 17, 18}, {13, 14}},
              {{10, 11, 12, 13, 14, 17, 18}, {15, 16}},
              {{10, 11, 12, 13, 14, 15, 16}, {17, 18}}},
             10);
}

TEST(KFold, EmptyOutput) {
  CheckKFold(0, {{{}, {}}, {{}, {}}, {{}, {}}});
  CheckKFold(4, {{{}, {0, 1, 2, 3}}});
}
}  // namespace xgboost::cv
