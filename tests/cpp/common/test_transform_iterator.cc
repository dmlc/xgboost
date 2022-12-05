/**
 * Copyright 2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstddef>  // std::size_t

#include "../../../src/common/transform_iterator.h"

namespace xgboost {
namespace common {
TEST(IndexTransformIter, Basic) {
  auto sqr = [](std::size_t i) { return i * i; };
  auto iter = MakeIndexTransformIter(sqr);
  for (std::size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(iter[i], sqr(i));
  }
}
}  // namespace common
}  // namespace xgboost
