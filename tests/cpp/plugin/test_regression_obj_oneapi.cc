/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/objective.h>
#include <xgboost/context.h>
#include <xgboost/json.h>
#include "../helpers.h"
namespace xgboost {

TEST(Plugin, LinearRegressionGPairOneAPI) {
  Context tparam = CreateEmptyGenericParam(0);
  std::vector<std::pair<std::string, std::string>> args;

  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("reg:squarederror_oneapi", &tparam)
  };

  obj->Configure(args);
  CheckObjFunction(obj,
                   {0, 0.1f, 0.9f,   1,    0,  0.1f, 0.9f,  1},
                   {0,   0,   0,   0,    1,    1,    1, 1},
                   {1,   1,   1,   1,    1,    1,    1, 1},
                   {0, 0.1f, 0.9f, 1.0f, -1.0f, -0.9f, -0.1f, 0},
                   {1,   1,   1,   1,    1,    1,    1, 1});
  CheckObjFunction(obj,
                   {0, 0.1f, 0.9f,   1,    0,  0.1f, 0.9f,  1},
                   {0,   0,   0,   0,    1,    1,    1, 1},
                   {},  // empty weight
                   {0, 0.1f, 0.9f, 1.0f, -1.0f, -0.9f, -0.1f, 0},
                   {1,   1,   1,   1,    1,    1,    1, 1});
  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}

TEST(Plugin, SquaredLogOneAPI) {
  Context tparam = CreateEmptyGenericParam(0);
  std::vector<std::pair<std::string, std::string>> args;

  std::unique_ptr<ObjFunction> obj { ObjFunction::Create("reg:squaredlogerror_oneapi", &tparam) };
  obj->Configure(args);
  CheckConfigReload(obj, "reg:squaredlogerror_oneapi");

  CheckObjFunction(obj,
                   {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},  // pred
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},  // labels
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},  // weights
                   {-0.5435f, -0.4257f, -0.25475f, -0.05855f, 0.1009f},
                   { 1.3205f,  1.0492f,  0.69215f,  0.34115f, 0.1091f});
  CheckObjFunction(obj,
                   {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},  // pred
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},  // labels
                   {},                              // empty weights
                   {-0.5435f, -0.4257f, -0.25475f, -0.05855f, 0.1009f},
                   { 1.3205f,  1.0492f,  0.69215f,  0.34115f, 0.1091f});
  ASSERT_EQ(obj->DefaultEvalMetric(), std::string{"rmsle"});
}

TEST(Plugin, LogisticRegressionGPairOneAPI) {
  Context tparam = CreateEmptyGenericParam(0);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj { ObjFunction::Create("reg:logistic_oneapi", &tparam) };

  obj->Configure(args);
  CheckConfigReload(obj, "reg:logistic_oneapi");

  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,  0.9f,      1}, // preds
                   {   0,    0,    0,    0,    1,     1,     1,     1}, // labels
                   {   1,    1,    1,    1,    1,     1,     1,     1}, // weights
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f}, // out_grad
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f}); // out_hess
}

TEST(Plugin, LogisticRegressionBasicOneAPI) {
  Context lparam = CreateEmptyGenericParam(0);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("reg:logistic_oneapi", &lparam)
  };

  obj->Configure(args);
  CheckConfigReload(obj, "reg:logistic_oneapi");

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
  HostDeviceVector<bst_float> io_preds = {0, 0.1f, 0.5f, 0.9f, 1};
  std::vector<bst_float> out_preds = {0.5f, 0.524f, 0.622f, 0.710f, 0.731f};
  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();
  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
}

TEST(Plugin, LogisticRawGPairOneAPI) {
  Context lparam = CreateEmptyGenericParam(0);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction>  obj {
    ObjFunction::Create("binary:logitraw_oneapi", &lparam)
  };

  obj->Configure(args);

  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,   0.9f,     1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f},
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f});
}

TEST(Plugin, CPUvsOneAPI) {
  Context ctx = CreateEmptyGenericParam(0);

  ObjFunction * obj_cpu =
      ObjFunction::Create("reg:squarederror", &ctx);
  ObjFunction * obj_oneapi =
      ObjFunction::Create("reg:squarederror_oneapi", &ctx);
  HostDeviceVector<GradientPair> cpu_out_preds;
  HostDeviceVector<GradientPair> oneapi_out_preds;

  constexpr size_t kRows = 400;
  constexpr size_t kCols = 100;
  auto pdmat = RandomDataGenerator(kRows, kCols, 0).Seed(0).GenerateDMatrix();
  HostDeviceVector<float> preds;
  preds.Resize(kRows);
  auto& h_preds = preds.HostVector();
  for (size_t i = 0; i < h_preds.size(); ++i) {
    h_preds[i] = static_cast<float>(i);
  }
  auto& info = pdmat->Info();

  info.labels_.Resize(kRows);
  auto& h_labels = info.labels_.HostVector();
  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = 1 / static_cast<float>(i+1);
  }

  {
    // CPU
    ctx.gpu_id = -1;
    obj_cpu->GetGradient(preds, info, 0, &cpu_out_preds);
  }
  {
    // oneapi
    ctx.gpu_id = 0;
    obj_oneapi->GetGradient(preds, info, 0, &oneapi_out_preds);
  }

  auto& h_cpu_out = cpu_out_preds.HostVector();
  auto& h_oneapi_out = oneapi_out_preds.HostVector();

  float sgrad = 0;
  float shess = 0;
  for (size_t i = 0; i < kRows; ++i) {
    sgrad += std::pow(h_cpu_out[i].GetGrad() - h_oneapi_out[i].GetGrad(), 2);
    shess += std::pow(h_cpu_out[i].GetHess() - h_oneapi_out[i].GetHess(), 2);
  }
  ASSERT_NEAR(sgrad, 0.0f, kRtEps);
  ASSERT_NEAR(shess, 0.0f, kRtEps);

  delete obj_cpu;
  delete obj_oneapi;
}

}  // namespace xgboost
