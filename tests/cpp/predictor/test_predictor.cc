/*!
 * Copyright 2020 by Contributors
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <xgboost/predictor.h>
#include <xgboost/data.h>

#include "../helpers.h"
#include "xgboost/generic_parameters.h"

namespace xgboost {
TEST(Predictor, PredictionCache) {
  size_t constexpr kRows = 16, kCols = 4;

  PredictionContainer container;
  DMatrix* m;
  // Add a cache that is immediately expired.
  auto add_cache = [&]() {
    auto *pp_dmat = CreateDMatrix(kRows, kCols, 0);
    auto p_dmat = *pp_dmat;
    container.Cache(p_dmat, GenericParameter::kCpuId);
    m = p_dmat.get();
    delete pp_dmat;
  };

  add_cache();
  ASSERT_EQ(container.Container().size(), 0);
  add_cache();
  EXPECT_ANY_THROW(container.Entry(m));
}
}  // namespace xgboost
