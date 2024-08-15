/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>        // for Json
#include <xgboost/tree_model.h>  // for RegTree

#include "../../../../src/common/categorical.h"  // for CatBitField
#include "../../../../src/tree/hist/expand_entry.h"

namespace xgboost::tree {
TEST(ExpandEntry, IO) {
  CPUExpandEntry entry{RegTree::kRoot, 0};
  entry.split.Update(1.0, 1, /*new_split_value=*/0.3, true, true, GradStats{1.0, 1.0},
                     GradStats{2.0, 2.0});
  bst_bin_t n_bins_feature = 256;
  auto n = common::CatBitField::ComputeStorageSize(n_bins_feature);
  entry.split.cat_bits = decltype(entry.split.cat_bits)(n, 0);
  common::CatBitField cat_bits{entry.split.cat_bits};
  cat_bits.Set(n_bins_feature / 2);

  Json je{Object{}};
  entry.Save(&je);

  CPUExpandEntry loaded;
  loaded.Load(je);

  ASSERT_EQ(loaded.split.is_cat, entry.split.is_cat);
  ASSERT_EQ(loaded.split.cat_bits, entry.split.cat_bits);
  ASSERT_EQ(loaded.split.left_sum.GetGrad(), entry.split.left_sum.GetGrad());
  ASSERT_EQ(loaded.split.right_sum.GetHess(), entry.split.right_sum.GetHess());
}

TEST(ExpandEntry, IOMulti) {
  MultiExpandEntry entry{RegTree::kRoot, 0};
  auto left_sum = std::vector<GradientPairPrecise>{{1.0, 1.0}, {1.0, 1.0}};
  auto right_sum = std::vector<GradientPairPrecise>{{2.0, 2.0}, {2.0, 2.0}};
  entry.split.Update(1.0, 1, /*new_split_value=*/0.3, true, true,
                     linalg::MakeVec(left_sum.data(), left_sum.size()),
                     linalg::MakeVec(right_sum.data(), right_sum.size()));
  bst_bin_t n_bins_feature = 256;
  auto n = common::CatBitField::ComputeStorageSize(n_bins_feature);
  entry.split.cat_bits = decltype(entry.split.cat_bits)(n, 0);
  common::CatBitField cat_bits{entry.split.cat_bits};
  cat_bits.Set(n_bins_feature / 2);

  Json je{Object{}};
  entry.Save(&je);

  MultiExpandEntry loaded;
  loaded.Load(je);

  ASSERT_EQ(loaded.split.is_cat, entry.split.is_cat);
  ASSERT_EQ(loaded.split.cat_bits, entry.split.cat_bits);
  ASSERT_EQ(loaded.split.left_sum, entry.split.left_sum);
  ASSERT_EQ(loaded.split.right_sum, entry.split.right_sum);
}
}  // namespace xgboost::tree
