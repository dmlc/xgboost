/**
 * Copyright 2024-2025, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/common/common.h"

namespace xgboost::common {
TEST(Common, HumanMemUnit) {
  auto name = HumanMemUnit(1024 * 1024 * 1024ul);
  ASSERT_EQ(name, "1GB");
  name = HumanMemUnit(1024 * 1024ul);
  ASSERT_EQ(name, "1MB");
  name = HumanMemUnit(1024);
  ASSERT_EQ(name, "1KB");
  name = HumanMemUnit(1);
  ASSERT_EQ(name, "1B");
}

TEST(Common, TrimLast) {
  {
    std::string in{"foobar "};
    auto out = TrimLast(in);
    ASSERT_EQ(out, "foobar");
  }
  {
    std::string in{R"(foobar
)"};
    auto out = TrimLast(in);
    ASSERT_EQ(out, "foobar");
  }
}
}  // namespace xgboost::common
