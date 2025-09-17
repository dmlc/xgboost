/**
 * Copyright 2025, XGBoost contributors
 */

#include "test_cat_container.h"

#include <gtest/gtest.h>

#include "../encoder/df_mock.h"

namespace xgboost {
using DfTest = enc::cpu_impl::DfTest;

auto eq_check = [](common::Span<bst_cat_t const> sorted_idx, std::vector<bst_cat_t> const& sol) {
  ASSERT_EQ(sorted_idx, common::Span{sol});
};

TEST(CatContainer, Str) {
  Context ctx;
  TestCatContainerStr<DfTest>(&ctx, eq_check);
}

TEST(CatContainer, Mixed) {
  Context ctx;
  TestCatContainerMixed<DfTest>(&ctx, eq_check);
}
}  // namespace xgboost
