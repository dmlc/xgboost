// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/objective.h>
#include <xgboost/generic_parameters.h>

#include "../helpers.h"

TEST(Objective, UnknownFunction) {
  xgboost::ObjFunction* obj = nullptr;
  xgboost::LearnerTrainParam tparam;
  std::vector<std::pair<std::string, std::string>> args;
  tparam.InitAllowUnknown(args);

  EXPECT_ANY_THROW(obj = xgboost::ObjFunction::Create(&tparam, "unknown_name"));
  EXPECT_NO_THROW(obj = xgboost::ObjFunction::Create(&tparam, "reg:squarederror"));
  if (obj) {
    delete obj;
  }
}
