// Copyright by Contributors
#include <xgboost/objective.h>
#include <limits>

#include "../helpers.h"

TEST(Objective, HingeObj) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("binary:hinge");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  xgboost::bst_float eps = std::numeric_limits<xgboost::bst_float>::min();
  CheckObjFunction(obj,
                   {-1.0f, -0.5f, 0.5f, 1.0f, -1.0f, -0.5f,  0.5f, 1.0f},
                   { 0.0f,  0.0f, 0.0f, 0.0f,  1.0f,  1.0f,  1.0f, 1.0f},
                   { 1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,  1.0f, 1.0f},
                   { 0.0f,  1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 0.0f},
                   {  eps,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,  1.0f, eps });

  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}
