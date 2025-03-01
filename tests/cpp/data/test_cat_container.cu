/**
 * Copyright 2025, XGBoost contributors
 */

#include <gtest/gtest.h>

#include "../../../src/common/common.h"
#include "../encoder/df_mock.h"
#include "../helpers.h"
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
}  // namespace xgboost
