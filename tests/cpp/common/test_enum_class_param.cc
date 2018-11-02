#include "../../../src/common/enum_class_param.h"
#include <dmlc/parameter.h>
#include <gtest/gtest.h>

enum class Foo : int {
  kBar = 0, kFrog = 1, kCat = 2, kDog = 3
};

DECLARE_FIELD_ENUM_CLASS(Foo);

struct MyParam : dmlc::Parameter<MyParam> {
  Foo foo;
  int bar;
  DMLC_DECLARE_PARAMETER(MyParam) {
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

DMLC_REGISTER_PARAMETER(MyParam);

TEST(EnumClassParam, Basic) {
  MyParam param;
  std::map<std::string, std::string> kwargs{
    {"foo", "frog"}, {"bar", "10"}
  };
  // try initializing
  param.Init(kwargs);
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
