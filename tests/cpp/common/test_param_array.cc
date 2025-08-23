/**
 * Copyright 2025, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>  // for kRtEps
#include <xgboost/json.h>
#include <xgboost/string_view.h>

#include <sstream>  // for istringstream, ostringstream
#include <string>   // for string

#include "../../../src/common/param_array.h"

namespace xgboost::common {
TEST(ParamArray, Float) {
  ParamArray<float> values;
  {
    std::istringstream sin{"1.1"};
    sin >> values;
    ASSERT_EQ(values.size(), 1);
    ASSERT_NEAR(values[0], 1.1, kRtEps);
    std::ostringstream sout;
    sout << values;
    ASSERT_EQ(sout.str(), sin.str());
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
}
}  // namespace xgboost::common
