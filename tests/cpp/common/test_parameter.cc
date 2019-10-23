/*!
 * Copyright (c) by Contributors 2019
 */
#include <gtest/gtest.h>

#include <xgboost/base.h>
#include <xgboost/parameter.h>

enum class Foo : int {
  kBar = 0, kFrog = 1, kCat = 2, kDog = 3
};

DECLARE_FIELD_ENUM_CLASS(Foo);

struct MyEnumParam : xgboost::XGBoostParameter<MyEnumParam> {
  Foo foo;
  int bar;
  DMLC_DECLARE_PARAMETER(MyEnumParam) {
    DMLC_DECLARE_FIELD(foo)
      .set_default(Foo::kBar)
      .add_enum("bar", Foo::kBar)
      .add_enum("frog", Foo::kFrog)
      .add_enum("cat", Foo::kCat)
      .add_enum("dog", Foo::kDog);
    DMLC_DECLARE_FIELD(bar)
      .set_default(-1);
  }
};

DMLC_REGISTER_PARAMETER(MyEnumParam);

TEST(EnumClassParam, Basic) {
  MyEnumParam param;
  std::map<std::string, std::string> kwargs{
    {"foo", "frog"}, {"bar", "10"}
  };
  // try initializing
  param.Init(kwargs); // NOLINT(clang-analyzer-core.UndefinedBinaryOperatorResult)
  ASSERT_EQ(param.foo, Foo::kFrog);
  ASSERT_EQ(param.bar, 10);

  // try all possible enum values
  kwargs["foo"] = "bar";
  param.Init(kwargs);
  ASSERT_EQ(param.foo, Foo::kBar);
  kwargs["foo"] = "frog";
  param.Init(kwargs);
  ASSERT_EQ(param.foo, Foo::kFrog);
  kwargs["foo"] = "cat";
  param.Init(kwargs);
  ASSERT_EQ(param.foo, Foo::kCat);
  kwargs["foo"] = "dog";
  param.Init(kwargs);
  ASSERT_EQ(param.foo, Foo::kDog);

  // try setting non-existent enum value
  kwargs["foo"] = "human";
  ASSERT_THROW(param.Init(kwargs), dmlc::ParamError);
}

struct UpdatableParam : xgboost::XGBoostParameter<UpdatableParam> {
  float f { 0.0f };
  double d { 0.0 };

  DMLC_DECLARE_PARAMETER(UpdatableParam) {
    DMLC_DECLARE_FIELD(f)
        .set_default(11.0f);
    DMLC_DECLARE_FIELD(d)
        .set_default(2.71828f);
  }
};

DMLC_REGISTER_PARAMETER(UpdatableParam);

TEST(XGBoostParameter, Update) {
  {
    UpdatableParam p;
    auto constexpr kRtEps = xgboost::kRtEps;

    p.UpdateAllowUnknown(xgboost::Args{});
    // When it's not initialized, perform set_default.
    ASSERT_NEAR(p.f, 11.0f, kRtEps);
    ASSERT_NEAR(p.d, 2.71828f, kRtEps);

    p.d = 3.14149;

    p.UpdateAllowUnknown(xgboost::Args{{"f", "2.71828"}});
    ASSERT_NEAR(p.f, 2.71828f, kRtEps);

    // p.d is un-effected by the update.
    ASSERT_NEAR(p.d, 3.14149, kRtEps);
  }
  {
    UpdatableParam p;
    auto constexpr kRtEps = xgboost::kRtEps;
    p.UpdateAllowUnknown(xgboost::Args{{"f", "2.71828"}});
    ASSERT_NEAR(p.f, 2.71828f, kRtEps);
    ASSERT_NEAR(p.d, 2.71828, kRtEps);  // default
  }
}
