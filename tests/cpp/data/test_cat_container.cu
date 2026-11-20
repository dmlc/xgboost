/**
 * Copyright 2025, XGBoost contributors
 */

#include <gtest/gtest.h>
#include <xgboost/base.h>  // for bst_cat_t
#include <xgboost/span.h>  // for Span

#include <cstdint>  // for int32_t
#include <vector>   // for vector

#include "../../../src/common/common.h"           // for safe_cuda
#include "../../../src/common/threading_utils.h"  // for ParallelFor
#include "../encoder/df_mock.h"
#include "../helpers.h"  // for MakeCUDACtx
#include "test_cat_container.h"

namespace xgboost {
// Doesn't support GPU input yet since cuDF doesn't have cuda arrow export.
using DfTest = enc::cpu_impl::DfTest;
namespace {
auto eq_check = [](common::Span<bst_cat_t const> sorted_idx, std::vector<bst_cat_t> const& sol) {
  std::vector<bst_cat_t> h_sorted(sorted_idx.size());
  dh::safe_cuda(cudaMemcpyAsync(h_sorted.data(), sorted_idx.data(), sorted_idx.size_bytes(),
                                cudaMemcpyDefault));
  ASSERT_EQ(h_sorted, sol);
};
}  // namespace

TEST(CatContainer, StrGpu) {
  auto ctx = MakeCUDACtx(0);
  auto df = TestCatContainerStr<DfTest>(&ctx, eq_check);
}

TEST(CatContainer, MixedGpu) {
  auto ctx = MakeCUDACtx(0);
  auto df = TestCatContainerMixed<DfTest>(&ctx, eq_check);
}

TEST(CatContainer, EmptyGpu) {
  auto ctx = MakeCUDACtx(0);
  dh::device_vector<enc::DeviceCatIndexView> columns(2);
  dh::device_vector<std::int32_t> segments{0, 0, 0};
  enc::DeviceColumnsView view{dh::ToSpan(columns), dh::ToSpan(segments), 0};
  CatContainer cats{&ctx, view, false};
  ASSERT_TRUE(cats.Empty());
  ASSERT_FALSE(cats.HasCategorical());
  ASSERT_EQ(cats.NumFeatures(), 0);
  ASSERT_TRUE(cats.HostView().Empty());
}

TEST(CatContainer, ThreadSafety) {
  auto ctx = MakeCUDACtx(0);
  auto df = DfTest::Make(DfTest::MakeStrs("abc", "bcd", "cde", "ab"), DfTest::MakeInts(2, 2, 3, 0));
  auto h_df = df.View();
  auto cats = test_cat_detail::FromDf(&ctx, h_df);
  cats.Sort(&ctx);  // not thread safe

  common::ParallelFor(ctx.Threads(), 64, [&](auto i) {
    auto sorted_idx = cats.RefSortedIndex(&ctx);
    if (i % 2 == 0) {
      auto h_cats = cats.HostView();
      ASSERT_EQ(h_cats.n_total_cats, 8);
    } else {
      auto d_cats = cats.DeviceView(&ctx);
      ASSERT_EQ(d_cats.n_total_cats, 8);
    }
    auto sol = std::vector<bst_cat_t>{3, 0, 1, 2, 3, 0, 1, 2};
    eq_check(sorted_idx, sol);
  });
}
}  // namespace xgboost
