// Copyright by Contributors
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(Objective, LinearRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:linear");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {0, 0.1, 0.9,   1,    0,  0.1, 0.9,  1},
                   {0,   0,   0,   0,    1,    1,    1, 1},
                   {1,   1,   1,   1,    1,    1,    1, 1},
                   {0, 0.1, 0.9, 1.0, -1.0, -0.9, -0.1, 0},
                   {1,   1,   1,   1,    1,    1,    1, 1});

  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}

TEST(Objective, LogisticRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:logistic");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1,  0.9,    1,    0,   0.1,  0.9,      1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5, 0.52, 0.71, 0.73, -0.5, -0.47, -0.28, -0.26},
                   {0.25, 0.24, 0.20, 0.19, 0.25,  0.24,  0.20,  0.19});
}

TEST(Objective, LogisticRegressionBasic) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:logistic");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {10}, {1}, {0}, {0}))
    << "Expected error when label not in range [0,1] for LogisticRegression";

  // test ProbToMargin
  EXPECT_NEAR(obj->ProbToMargin(0.1), -2.197, 0.01);
  EXPECT_NEAR(obj->ProbToMargin(0.5), 0, 0.01);
  EXPECT_NEAR(obj->ProbToMargin(0.9), 2.197, 0.01);
  EXPECT_ANY_THROW(obj->ProbToMargin(10))
    << "Expected error when base_score not in range [0,1] for LogisticRegression";

  // test PredTransform
  std::vector<xgboost::bst_float> preds = {0, 0.1, 0.5, 0.9, 1};
  std::vector<xgboost::bst_float> out_preds = {0.5, 0.524, 0.622, 0.710, 0.731};
  obj->PredTransform(&preds);
  for (int i = 0; i < preds.size(); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01);
  }
}

TEST(Objective, LogisticRawGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("binary:logitraw");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1,  0.9,    1,    0,   0.1,   0.9,     1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5, 0.52, 0.71, 0.73, -0.5, -0.47, -0.28, -0.26},
                   {0.25, 0.24, 0.20, 0.19, 0.25,  0.24,  0.20,  0.19});
}

TEST(Objective, PoissonRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("count:poisson");
  std::vector<std::pair<std::string, std::string> > args;
  args.push_back(std::make_pair("max_delta_step", "0.1"));
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1,  0.9,    1,    0,  0.1,  0.9,    1},
                   {   0,    0,    0,    0,    1,    1,    1,    1},
                   {   1,    1,    1,    1,    1,    1,    1,    1},
                   {   1, 1.10, 2.45, 2.71,    0, 0.10, 1.45, 1.71},
                   {1.10, 1.22, 2.71, 3.00, 1.10, 1.22, 2.71, 3.00});
}

TEST(Objective, PoissonRegressionBasic) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("count:poisson");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {-1}, {1}, {0}, {0}))
    << "Expected error when label < 0 for PoissonRegression";

  // test ProbToMargin
  EXPECT_NEAR(obj->ProbToMargin(0.1), -2.30, 0.01);
  EXPECT_NEAR(obj->ProbToMargin(0.5), -0.69, 0.01);
  EXPECT_NEAR(obj->ProbToMargin(0.9), -0.10, 0.01);

  // test PredTransform
  std::vector<xgboost::bst_float> preds = {0, 0.1, 0.5, 0.9, 1};
  std::vector<xgboost::bst_float> out_preds = {1, 1.10, 1.64, 2.45, 2.71};
  obj->PredTransform(&preds);
  for (int i = 0; i < preds.size(); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01);
  }
}
