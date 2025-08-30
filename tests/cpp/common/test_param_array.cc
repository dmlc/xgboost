/**
 * Copyright 2025, XGBoost contributors
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <xgboost/base.h>         // for kRtEps
#include <xgboost/json.h>         // for Json
#include <xgboost/parameter.h>    // for XGBoostParameter
#include <xgboost/string_view.h>  // for StringView

#include <sstream>  // for istringstream, ostringstream
#include <string>   // for string

#include "../../../src/common/param_array.h"
#include "../helpers.h"

namespace xgboost::common {
TEST(ParamArray, Float) {
  ParamArray<float> values{"values"};
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
    ParamArray<float> values{"values"};
    std::istringstream sin{"[\"foo\"]"};
    ASSERT_THAT(
        [&] { sin >> values; },
        GMockThrow(
            R"(Invalid type for: `values`, expecting one of the: {`Number`, `Integer`}, got: `String`)"));
  }
}

namespace {
struct TestParamArray : public XGBoostParameter<TestParamArray> {
  ParamArray<float> test_key{"test_key", 0.2f};
  DMLC_DECLARE_PARAMETER(TestParamArray) {
    DMLC_DECLARE_FIELD(test_key).describe("test").set_default(ParamArray<float>{"test_key", 0.2f});
  }
};

DMLC_REGISTER_PARAMETER(TestParamArray);
}  // namespace

TEST(ParamArray, Update) {
  TestParamArray param;
  param.UpdateAllowUnknown(Args{{}});
  ASSERT_EQ(param.test_key.size(), 1);
  ASSERT_EQ(param.test_key.Name(), "test_key");
}
}  // namespace xgboost::common
