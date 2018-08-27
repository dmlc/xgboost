// Copyright by Contributors
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(Objective, UnknownFunction) {
  xgboost::ObjFunction* obj;
  EXPECT_ANY_THROW(obj = xgboost::ObjFunction::Create("unknown_name"));
  EXPECT_NO_THROW(obj = xgboost::ObjFunction::Create("reg:linear"));
  delete obj;
}
