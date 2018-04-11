/*!
 * Copyright 2017 XGBoost contributors
 */
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(Objective, GPULinearRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("gpu:reg:linear");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {0, 0.1f, 0.9f,   1,    0,  0.1f, 0.9f,  1},
                   {0,   0,   0,   0,    1,    1,    1, 1},
                   {1,   1,   1,   1,    1,    1,    1, 1},
                   {0, 0.1f, 0.9f, 1.0f, -1.0f, -0.9f, -0.1f, 0},
                   {1,   1,   1,   1,    1,    1,    1, 1});

  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}

TEST(Objective, GPULogisticRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("gpu:reg:logistic");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,  0.9f,      1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f},
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f});
}

TEST(Objective, GPULogisticRegressionBasic) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("gpu:reg:logistic");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {10}, {1}, {0}, {0}))
    << "Expected error when label not in range [0,1f] for LogisticRegression";

  // test ProbToMargin
  EXPECT_NEAR(obj->ProbToMargin(0.1f), -2.197f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.5f), 0, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.9f), 2.197f, 0.01f);
  EXPECT_ANY_THROW(obj->ProbToMargin(10))
    << "Expected error when base_score not in range [0,1f] for LogisticRegression";

  // test PredTransform
  xgboost::HostDeviceVector<xgboost::bst_float> io_preds = {0, 0.1f, 0.5f, 0.9f, 1};
  std::vector<xgboost::bst_float> out_preds = {0.5f, 0.524f, 0.622f, 0.710f, 0.731f};
  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();
  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
}

TEST(Objective, GPULogisticRawGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("gpu:binary:logitraw");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,   0.9f,     1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f},
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f});
}
