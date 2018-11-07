/*!
 * Copyright 2018 XGBoost contributors
 */
#include <xgboost/objective.h>

#include "../helpers.h"

TEST(Objective, DeclareUnifiedTest(SoftmaxMultiClassObjGPair)) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("multi:softmax");
  std::vector<std::pair<std::string, std::string>> args {{"num_class", "3"}};
  obj->Configure(args);
  CheckObjFunction(obj,
		   {1.0f, 0.0f, 2.0f, 2.0f, 0.0f, 1.0f}, // preds
		   {1.0f, 0.0f},	       // labels
		   {1.0f, 1.0f},	       // weights
		   {0.24f, -0.91f, 0.66f, -0.33f, 0.09f, 0.24f}, // grad
		   {0.36f, 0.16f, 0.44f, 0.45f, 0.16f, 0.37f});	 // hess

  ASSERT_NO_THROW(obj->DefaultEvalMetric());

  delete obj;
}

TEST(Objective, DeclareUnifiedTest(SoftmaxMultiClassBasic)) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("multi:softmax");
  std::vector<std::pair<std::string, std::string>> args
    {std::pair<std::string, std::string>("num_class", "3")};
  obj->Configure(args);

  xgboost::HostDeviceVector<xgboost::bst_float>  io_preds = {2.0f, 0.0f, 1.0f,
							     1.0f, 0.0f, 2.0f};
  std::vector<xgboost::bst_float> out_preds = {0.0f, 2.0f};
  obj->PredTransform(&io_preds);

  auto& preds = io_preds.HostVector();

  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }

  delete obj;
}

TEST(Objective, DeclareUnifiedTest(SoftprobMultiClassBasic)) {
  xgboost::ObjFunction * obj = xgboost::ObjFunction::Create("multi:softprob");
  std::vector<std::pair<std::string, std::string>> args
    {std::pair<std::string, std::string>("num_class", "3")};
  obj->Configure(args);

  xgboost::HostDeviceVector<xgboost::bst_float>  io_preds = {2.0f, 0.0f, 1.0f};
  std::vector<xgboost::bst_float> out_preds = {0.66524096f, 0.09003057f, 0.24472847f};

  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();

  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
  delete obj;
}
