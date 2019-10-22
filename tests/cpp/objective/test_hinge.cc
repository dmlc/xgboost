// Copyright by Contributors
#include <xgboost/objective.h>
#include <xgboost/generic_parameters.h>
#include <limits>

#include "../helpers.h"

TEST(Objective, DeclareUnifiedTest(HingeObj)) {
  xgboost::GenericParameter tparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<xgboost::ObjFunction> obj {
    xgboost::ObjFunction::Create("binary:hinge", &tparam)
  };

  xgboost::bst_float eps = std::numeric_limits<xgboost::bst_float>::min();
  CheckObjFunction(obj,
                   {-1.0f, -0.5f, 0.5f, 1.0f, -1.0f, -0.5f,  0.5f, 1.0f},
                   { 0.0f,  0.0f, 0.0f, 0.0f,  1.0f,  1.0f,  1.0f, 1.0f},
                   { 1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,  1.0f, 1.0f},
                   { 0.0f,  1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 0.0f},
                   {  eps,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,  1.0f, eps });
  CheckObjFunction(obj,
                   {-1.0f, -0.5f, 0.5f, 1.0f, -1.0f, -0.5f,  0.5f, 1.0f},
                   { 0.0f,  0.0f, 0.0f, 0.0f,  1.0f,  1.0f,  1.0f, 1.0f},
                   {},  // Empty weight.
                   { 0.0f,  1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 0.0f},
                   {  eps,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,  1.0f, eps });

  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}
