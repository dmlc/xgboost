// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(Objective, UnknownFunction) {
  xgboost::ObjFunction* obj = nullptr;
  EXPECT_ANY_THROW(obj = xgboost::ObjFunction::Create("unknown_name"));
  EXPECT_NO_THROW(obj = xgboost::ObjFunction::Create("reg:squarederror"));
  if (obj) {
    delete obj;
  }
}
