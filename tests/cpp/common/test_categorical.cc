/*!
 * Copyright 2021 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <limits>

#include "../../../src/common/categorical.h"

namespace xgboost {
namespace common {
TEST(Categorical, Decision) {
  // inf
  float a = std::numeric_limits<float>::infinity();

  ASSERT_TRUE(common::InvalidCat(a));
  std::vector<uint32_t> cats(256, 0);
  ASSERT_TRUE(Decision(cats, a));

  // larger than size
  a = 256;
  ASSERT_TRUE(Decision(cats, a));

  // negative
  a = -1;
  ASSERT_TRUE(Decision(cats, a));

  CatBitField bits{cats};
  bits.Set(0);
  a = -0.5;
  ASSERT_TRUE(Decision(cats, a));

  // round toward 0
  a = 0.5;
  ASSERT_FALSE(Decision(cats, a));

  // valid
  a = 13;
  bits.Set(a);
  ASSERT_FALSE(Decision(bits.Bits(), a));
}
}  // namespace common
}  // namespace xgboost
