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

TEST(Common, Split) {
  auto check = [](char const* chars, std::size_t n) {
    std::string str{chars};
    auto res_str = Split(str, ',');
    std::string_view view{chars};
    auto res_view = Split(str, ',');
    ASSERT_EQ(res_view.size(), res_str.size());
    ASSERT_EQ(res_view.size(), n);
    for (std::size_t i = 0; i < res_str.size(); ++i) {
      ASSERT_EQ(res_str[i].size(), res_view[i].size());
      auto eq = std::equal(res_str[i].cbegin(), res_str[i].cend(), res_view[i].cbegin());
      ASSERT_TRUE(eq);
    }
  };
  check("foo,bar", 2);
  check("foo,bar,", 2);
  check(",foo,bar", 3);
  check(",foo,bar,", 3);  // last is ignored
  check(",,,,foo,bar", 6);
  check(",foo,,,,bar", 6);
}

TEST(Common, Trim) {
  {
    auto res = TrimFirst(" foo ");
    ASSERT_EQ(res, std::string_view{"foo "});
  }
  {
    auto res = TrimLast(" foo ");
    ASSERT_EQ(res, std::string_view{" foo"});
  }
}
}  // namespace xgboost::common
