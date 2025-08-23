/**
 * Copyright 2025, XGBoost contributors
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <xgboost/base.h>         // for kRtEps
#include <xgboost/json.h>         // for Json
#include <xgboost/string_view.h>  // for StringView

#include <sstream>  // for istringstream, ostringstream
#include <string>   // for string

#include "../../../src/common/param_array.h"
#include "../helpers.h"

namespace xgboost::common {
TEST(ParamArray, Float) {
  ParamArray<float, false> values{"values"};
  {
    std::istringstream sin{"1.1"};
    sin >> values;
    ASSERT_EQ(values.size(), 1);
    ASSERT_NEAR(values[0], 1.1, kRtEps);
    std::ostringstream sout;
    sout << values;
    auto jarr = Json::Load(StringView{sout.str()});
    for (std::size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(get<Number const>(jarr[i]), values[i]);
    }
  }
  {
    std::string str = "[1.1, 1.3]";
    std::istringstream sin{str};
    sin >> values;
    ASSERT_EQ(values.size(), 2);
    ASSERT_NEAR(values[0], 1.1, kRtEps);
    ASSERT_NEAR(values[1], 1.3, kRtEps);
    std::ostringstream sout;
    sout << values;
    auto jarr = Json::Load(StringView{sout.str()});
    for (std::size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(get<Number const>(jarr[i]), values[i]);
    }
  }
  {
    ParamArray<float, true> values{"values"};
    std::istringstream sin{"1.1"};
    sin >> values;
    ASSERT_EQ(values.size(), 1);
    ASSERT_NEAR(values[0], 1.1, kRtEps);
    std::ostringstream sout;
    sout << values;
    auto jarr = Json::Load(StringView{sout.str()});
    ASSERT_TRUE(IsA<Number>(jarr));
    ASSERT_NEAR(get<Number>(jarr), 1.1, kRtEps);
  }
  {
    ParamArray<float, true> values{"values"};
    std::istringstream sin{"[\"foo\"]"};
    ASSERT_THAT([&] { sin >> values; }, GMockThrow(R"(`Number`, `Integer`)"));
  }
}
}  // namespace xgboost::common
