// Copyright by Contributors
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(Objective, LinearRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:linear");
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

TEST(Objective, LogisticRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:logistic");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,  0.9f,      1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f},
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f});
}

TEST(Objective, LogisticRegressionBasic) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:logistic");
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

TEST(Objective, LogisticRawGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("binary:logitraw");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,   0.9f,     1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f},
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f});
}

TEST(Objective, PoissonRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("count:poisson");
  std::vector<std::pair<std::string, std::string> > args;
  args.push_back(std::make_pair("max_delta_step", "0.1f"));
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,  0.1f,  0.9f,    1},
                   {   0,    0,    0,    0,    1,    1,    1,    1},
                   {   1,    1,    1,    1,    1,    1,    1,    1},
                   {   1, 1.10f, 2.45f, 2.71f,    0, 0.10f, 1.45f, 1.71f},
                   {1.10f, 1.22f, 2.71f, 3.00f, 1.10f, 1.22f, 2.71f, 3.00f});
}

TEST(Objective, PoissonRegressionBasic) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("count:poisson");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {-1}, {1}, {0}, {0}))
    << "Expected error when label < 0 for PoissonRegression";

  // test ProbToMargin
  EXPECT_NEAR(obj->ProbToMargin(0.1f), -2.30f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.5f), -0.69f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.9f), -0.10f, 0.01f);

  // test PredTransform
  xgboost::HostDeviceVector<xgboost::bst_float> io_preds = {0, 0.1f, 0.5f, 0.9f, 1};
  std::vector<xgboost::bst_float> out_preds = {1, 1.10f, 1.64f, 2.45f, 2.71f};
  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();
  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
}

TEST(Objective, GammaRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:gamma");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   {0, 0.1f, 0.9f, 1, 0,  0.1f,  0.9f,    1},
                   {0,   0,   0, 0, 1,    1,    1,    1},
                   {1,   1,   1, 1, 1,    1,    1,    1},
                   {1,   1,   1, 1, 0, 0.09f, 0.59f, 0.63f},
                   {0,   0,   0, 0, 1, 0.90f, 0.40f, 0.36f});
}

TEST(Objective, GammaRegressionBasic) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:gamma");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {-1}, {1}, {0}, {0}))
    << "Expected error when label < 0 for GammaRegression";

  // test ProbToMargin
  EXPECT_NEAR(obj->ProbToMargin(0.1f), -2.30f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.5f), -0.69f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.9f), -0.10f, 0.01f);

  // test PredTransform
  xgboost::HostDeviceVector<xgboost::bst_float> io_preds = {0, 0.1f, 0.5f, 0.9f, 1};
  std::vector<xgboost::bst_float> out_preds = {1, 1.10f, 1.64f, 2.45f, 2.71f};
  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();
  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
}

TEST(Objective, TweedieRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:tweedie");
  std::vector<std::pair<std::string, std::string> > args;
  args.push_back(std::make_pair("tweedie_variance_power", "1.1f"));
  obj->Configure(args);
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1, 0,  0.1f,  0.9f,    1},
                   {   0,    0,    0,    0, 1,    1,    1,    1},
                   {   1,    1,    1,    1, 1,    1,    1,    1},
                   {   1, 1.09f, 2.24f, 2.45f, 0, 0.10f, 1.33f, 1.55f},
                   {0.89f, 0.98f, 2.02f, 2.21f, 1, 1.08f, 2.11f, 2.30f});
}

TEST(Objective, TweedieRegressionBasic) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("reg:tweedie");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {-1}, {1}, {0}, {0}))
    << "Expected error when label < 0 for TweedieRegression";

  // test ProbToMargin
  EXPECT_NEAR(obj->ProbToMargin(0.1f), -2.30f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.5f), -0.69f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.9f), -0.10f, 0.01f);

  // test PredTransform
  xgboost::HostDeviceVector<xgboost::bst_float> io_preds = {0, 0.1f, 0.5f, 0.9f, 1};
  std::vector<xgboost::bst_float> out_preds = {1, 1.10f, 1.64f, 2.45f, 2.71f};
  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();
  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
}

TEST(Objective, CoxRegressionGPair) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("survival:cox");
  std::vector<std::pair<std::string, std::string> > args;
  obj->Configure(args);
  CheckObjFunction(obj,
                   { 0, 0.1f, 0.9f,       1,       0,    0.1f,   0.9f,       1},
                   { 0,   -2,   -2,       2,       3,       5,    -10,     100},
                   { 1,    1,    1,       1,       1,       1,      1,       1},
                   { 0,    0,    0, -0.799f, -0.788f, -0.590f, 0.910f,  1.006f},
                   { 0,    0,    0,  0.160f,  0.186f,  0.348f, 0.610f,  0.639f});
}
