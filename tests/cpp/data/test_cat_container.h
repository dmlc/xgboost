/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once

#include <gtest/gtest.h>

#include "../../../src/data/cat_container.h"

namespace xgboost {
namespace test_cat_detail {
inline void HostCheck(CatContainer const& cats) {
  ASSERT_TRUE(cats.HasCategorical());
  ASSERT_FALSE(cats.Empty());
  ASSERT_TRUE(cats.HostCanRead());
  ASSERT_FALSE(cats.DeviceCanRead());
}

inline void DeviceCheck(CatContainer const& cats) {
  ASSERT_TRUE(cats.HasCategorical());
  ASSERT_FALSE(cats.Empty());
  ASSERT_TRUE(cats.HostCanRead());
  ASSERT_FALSE(cats.DeviceCanRead());
}

[[nodiscard]] inline CatContainer FromDf(Context const*, enc::HostColumnsView df) {
  return CatContainer{df};
}

#if defined(XGBOOST_USE_CUDA)
[[nodiscard]] inline CatContainer FromDf(Context const* ctx, enc::DeviceColumnsView df) {
  return CatContainer{ctx->Device(), df};
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace test_cat_detail

template <typename DfTest, typename EqCheck>
auto TestCatContainerStr(Context const* ctx, EqCheck&& is_eq) {
  auto df = DfTest::Make(DfTest::MakeStrs("abc", "bcd", "cde", "ab"));
  auto h_df = df.View();
  auto cats = test_cat_detail::FromDf(ctx, h_df);
  if (ctx->IsCPU()) {
    test_cat_detail::HostCheck(cats);
  } else {
    test_cat_detail::DeviceCheck(cats);
  }

  [&] {
    ASSERT_EQ(df.View().columns.size(), cats.NumFeatures());
  }();

  cats.Sort(ctx);

  auto sol = std::vector<bst_cat_t>{3, 0, 1, 2};
  auto sorted_idx = cats.RefSortedIndex(ctx);
  is_eq(sorted_idx, sol);
  [&] {
    auto view = cats.HostView();
    ASSERT_EQ(view.n_total_cats, sol.size());
    ASSERT_EQ(view.feature_segments.size(), 2ul);
    ASSERT_EQ(view.feature_segments[0], 0);
    ASSERT_EQ(view.feature_segments[1], static_cast<bst_cat_t>(sol.size()));
  }();

  return df;
}

template <typename DfTest, typename EqCheck>
auto TestCatContainerMixed(Context const* ctx, EqCheck&& is_eq) {
  auto df =
      DfTest::Make(DfTest::MakeStrs("abc", "bcd", "cde", "ab"), DfTest::MakeInts(2, 2, 3, 0, 4));
  auto h_df = df.View();
  auto cats = test_cat_detail::FromDf(ctx, h_df);
  if (ctx->IsCPU()) {
    test_cat_detail::HostCheck(cats);
  } else {
    test_cat_detail::DeviceCheck(cats);
  }

  cats.Sort(ctx);
  auto sorted_idx = cats.RefSortedIndex(ctx);
  auto sol = std::vector<bst_cat_t>{3, 0, 1, 2, 3, 0, 1, 2, 4};
  is_eq(sorted_idx, sol);
  auto view = cats.HostView();
  [&] {
    ASSERT_EQ(view.n_total_cats, sol.size());
    ASSERT_EQ(view.feature_segments.size(), 3ul);
    ASSERT_EQ(view.feature_segments[0], 0);
    ASSERT_EQ(view.feature_segments[1], 4);
    ASSERT_EQ(view.feature_segments[2], static_cast<bst_cat_t>(sol.size()));
  }();

  return df;
}
}  // namespace xgboost
