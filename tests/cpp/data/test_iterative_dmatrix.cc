/*!
 * Copyright 2022 XGBoost contributors
 */
#include "test_iterative_dmatrix.h"

#include <gtest/gtest.h>

#include "../../../src/data/gradient_index.h"
#include "../../../src/data/iterative_dmatrix.h"
#include "../helpers.h"

namespace xgboost {
namespace data {
TEST(IterativeDMatrix, Ref) {
  TestRefDMatrix<GHistIndexMatrix, NumpyArrayIterForTest>(
      [&](GHistIndexMatrix const& page) { return page.cut; });
}

TEST(IterativeDMatrix, IsDense) {
  int n_bins = 16;
  auto test = [n_bins](float sparsity) {
    NumpyArrayIterForTest iter(sparsity);
    IterativeDMatrix m(&iter, iter.Proxy(), nullptr, Reset, Next,
                       std::numeric_limits<float>::quiet_NaN(), 0, n_bins);
    if (sparsity == 0.0) {
      ASSERT_TRUE(m.IsDense());
    } else {
      ASSERT_FALSE(m.IsDense());
    }
  };
  test(0.0);
  test(0.1);
  test(1.0);
}
}  // namespace data
}  // namespace xgboost
