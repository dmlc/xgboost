/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/string_view.h>

#include <algorithm>  // std::equal
#include <sstream>    // std::stringstream
#include <string>     // std::string

namespace xgboost {
TEST(StringView, Basic) {
  StringView str{"This is a string."};
  std::stringstream ss;
  ss << str;

  std::string res = ss.str();
  ASSERT_EQ(str.size(), res.size());
  ASSERT_TRUE(std::equal(res.cbegin(), res.cend(), str.cbegin()));

  auto substr = str.substr(5, 2);
  ASSERT_EQ(substr.size(), 2);

  ASSERT_EQ(StringView{"is"}.size(), 2);
  ASSERT_TRUE(substr == "is");
  ASSERT_FALSE(substr != "is");
  ASSERT_FALSE(substr == "foobar");
  ASSERT_FALSE(substr == "i");

  ASSERT_TRUE(std::equal(substr.crbegin(), substr.crend(), StringView{"si"}.cbegin()));

  {
    StringView empty{nullptr};
    ASSERT_TRUE(empty.empty());
  }
  {
    StringView empty{""};
    ASSERT_TRUE(empty.empty());
    StringView empty2{nullptr};
    ASSERT_EQ(empty, empty2);
  }
}
}  // namespace xgboost
