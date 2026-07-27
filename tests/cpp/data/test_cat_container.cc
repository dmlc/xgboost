/**
 * Copyright 2025, XGBoost contributors
 */

#include "test_cat_container.h"

#include <gtest/gtest.h>

#include <utility>  // for move
#include <vector>   // for vector

#include "../encoder/df_mock.h"
#include "../helpers.h"  // for GMockThrow

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

TEST(CatContainer, RejectFloat) {
  Json column{Object{}};
  column["type"] = static_cast<std::int64_t>(Value::ValueKind::kF32Array);
  column["values"] = F32Array{1};

  Json in{Object{}};
  in["enc"] = Array(std::vector<Json>{std::move(column)});

  CatContainer cats;
  EXPECT_THAT([&] { cats.Load(in); }, GMockThrow("floating point dtype"));
}
}  // namespace xgboost
