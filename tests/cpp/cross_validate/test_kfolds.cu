/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <vector>   // for vector

#include "../../../src/cross_validate/kfolds.h"
#include "../helpers.h"  // for AssertVecEq, MakeCUDACtx

namespace xgboost::cv {
namespace {
void CheckKFold(std::size_t n_rows, std::size_t k_folds, std::int32_t k,
                std::vector<bst_idx_t> expected, bst_idx_t begin = 0) {
  auto ctx = MakeCUDACtx(0);

  MetaInfo info;
  info.num_row_ = n_rows;

  HostDeviceVector<bst_idx_t> out;
  KFold(&ctx, k_folds, begin, begin + info.num_row_, k, &out);

  ASSERT_EQ(out.Device(), ctx.Device());
  AssertVecEq(out.HostVector(), expected);
}
}  // namespace

TEST(KFold, TrainingIndices) {
  CheckKFold(10, 3, 0, {4, 5, 6, 7, 8, 9});
  CheckKFold(10, 3, 1, {0, 1, 2, 3, 7, 8, 9});
  CheckKFold(10, 3, 2, {0, 1, 2, 3, 4, 5, 6});
}

TEST(KFold, BatchLocalIndices) {
  CheckKFold(9, 4, 2, {0, 1, 2, 3, 4, 7, 8}, 10);
}

TEST(KFold, EmptyOutput) {
  CheckKFold(0, 3, 0, {});
  CheckKFold(4, 1, 0, {});
}
}  // namespace xgboost::cv
