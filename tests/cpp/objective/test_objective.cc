// Copyright by Contributors
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(Objective, UnknownFunction) {
  EXPECT_ANY_THROW(xgboost::ObjFunction::Create("unknown_name"));
  EXPECT_NO_THROW(xgboost::ObjFunction::Create("reg:linear"));
}
