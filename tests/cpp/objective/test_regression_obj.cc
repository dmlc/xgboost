/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include "test_regression_obj.h"

#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/json.h>
#include <xgboost/objective.h>
#include <xgboost/tree_model.h>  // for RegTree

#include <cmath>    // for hypot, sqrt
#include <memory>   // for unique_ptr
#include <utility>  // for pair

#include "../../../src/common/linalg_op.h"  // for begin, end
#include "../../../src/common/math.h"       // for SoftPlus
#include "../../../src/tree/tree_view.h"    // for MultiTargetTreeView
#include "../helpers.h"
#include "../tree/test_multi_target_tree_model.h"  // for MakeMtTreeForTest
#include "test_objective_helpers.h"  // for MakePositionsForTest, MakeIotaLabelsForTest
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/linalg.h"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost {
namespace {
void CheckProbaToMargin(std::unique_ptr<ObjFunction> const& obj, float in, float expect,
                        float abs_error = 1e-2f) {
  linalg::Vector<float> t{{in}, {1}, obj->Ctx()->Device()};
  obj->ProbToMargin(&t);
  ASSERT_NEAR(t(0), expect, abs_error);
}
}  // namespace

void TestLinearRegressionGPair(const Context* ctx) {
  std::string obj_name = "reg:squarederror";

  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(obj_name, ctx)};

  obj->Configure(args);
  // clang-format off
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
  // clang-format on
  ASSERT_NO_THROW({ [[maybe_unused]] auto _ = obj->DefaultEvalMetric(); });
}

void TestSquaredLog(const Context* ctx) {
  std::string obj_name = "reg:squaredlogerror";
  std::vector<std::pair<std::string, std::string>> args;

  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(obj_name, ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, obj_name);
  // clang-format off
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
  // clang-format on
  ASSERT_EQ(obj->DefaultEvalMetric(), std::string{"rmsle"});
}

void TestLogisticRegressionGPair(const Context* ctx) {
  std::string obj_name = "reg:logistic";
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(obj_name, ctx)};

  obj->Configure(args);
  CheckConfigReload(obj, obj_name);
  // clang-format off
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,  0.9f,      1},  // preds
                   {   0,    0,    0,    0,    1,     1,     1,     1},  // labels
                   {   1,    1,    1,    1,    1,     1,     1,     1},  // weights
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f},  // out_grad
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f});  // out_hess
  // clang-format on
}

void TestLogisticRegressionBasic(const Context* ctx) {
  std::string obj_name = "reg:logistic";
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(obj_name, ctx)};

  obj->Configure(args);
  CheckConfigReload(obj, obj_name);

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {10}, {1}, {0}, {0}))
      << "Expected error when label not in range [0,1f] for LogisticRegression";

  // test ProbToMargin
  CheckProbaToMargin(obj, 0.1f, -2.197f);
  CheckProbaToMargin(obj, 0.5f, 0);
  CheckProbaToMargin(obj, 0.9f, 2.197f);
  ASSERT_THAT([&] { CheckProbaToMargin(obj, 10, 0); }, GMockThrow("base_score must be in (0,1)"));

  // test PredTransform
  HostDeviceVector<bst_float> io_preds = {0, 0.1f, 0.5f, 0.9f, 1};
  std::vector<bst_float> out_preds = {0.5f, 0.524f, 0.622f, 0.710f, 0.731f};
  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();
  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
}

void TestsLogisticRawGPair(const Context* ctx) {
  std::string obj_name = "binary:logitraw";
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(obj_name, ctx)};
  obj->Configure(args);
  // clang-format off
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,   0.9f,     1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f},
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f});
  // clang-format on
}

void TestPoissonRegressionGPair(const Context* ctx) {
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("count:poisson", ctx)};

  obj->Configure(args);
  // clang-format off
  CheckObjFunction(obj,
                   {  -2,    -1,     0,    1,   -2,    -1,     0,    1},
                   {   0,     0,     0,    0,    1,     2,     3,    4},
                   {   1,     1,     1,    1,    1,     1,     1,    1},
                   { .14f,  .37f,     1, 2.71f, -.86f, -1.63f,    -2, -1.28f},
                   {.068f, .184f,   .5f, 1.359f, .568f, 1.184f,     2, 3.359f});
  CheckObjFunction(obj,
                   {  -2,    -1,     0,    1,   -2,    -1,     0,    1},
                   {   0,     0,     0,    0,    1,     2,     3,    4},
                   {},  // Empty weight
                   { .14f,  .37f,     1, 2.71f, -.86f, -1.63f,    -2, -1.28f},
                   {.068f, .184f,   .5f, 1.359f, .568f, 1.184f,     2, 3.359f});
  // clang-format on
}

void TestPoissonRegressionBasic(const Context* ctx) {
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("count:poisson", ctx)};

  Json legacy_config{Object{}};
  legacy_config["name"] = String{"count:poisson"};
  legacy_config["poisson_regression_param"] = Object{};
  legacy_config["poisson_regression_param"]["max_delta_step"] = String{"7E-1"};
  ASSERT_NO_THROW(obj->LoadConfig(legacy_config));

  obj->Configure(args);
  auto config = CheckConfigReload(obj, "count:poisson");
  auto const& config_obj = get<Object const>(config);
  ASSERT_EQ(config_obj.size(), 1);

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {-1}, {1}, {0}, {0}))
      << "Expected error when label < 0 for PoissonRegression";

  // test ProbToMargin
  CheckProbaToMargin(obj, 0.1f, -2.30f);
  CheckProbaToMargin(obj, 0.5f, -0.69f);
  CheckProbaToMargin(obj, 0.9f, -0.10f);

  // test PredTransform
  HostDeviceVector<bst_float> io_preds = {0, 0.1f, 0.5f, 0.9f, 1};
  std::vector<bst_float> out_preds = {1, 1.10f, 1.64f, 2.45f, 2.71f};
  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();
  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
}

void TestGammaRegressionGPair(const Context* ctx) {
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:gamma", ctx)};

  obj->Configure(args);
  // clang-format off
  CheckObjFunction(obj,
                   {0, 0.1f, 0.9f, 1, 0,  0.1f,  0.9f,    1},
                   {2,   2,   2,   2, 1,    1,    1,    1},
                   {1,   1,   1,   1, 1,    1,    1,    1},
                   {-1,  -0.809, 0.187, 0.264, 0, 0.09f, 0.59f, 0.63f},
                   {2,   1.809,  0.813, 0.735, 1, 0.90f, 0.40f, 0.36f});
  CheckObjFunction(obj,
                   {0, 0.1f, 0.9f, 1, 0,  0.1f,  0.9f,    1},
                   {2,   2,   2,   2, 1,    1,    1,    1},
                   {},  // Empty weight
                   {-1,  -0.809, 0.187, 0.264, 0, 0.09f, 0.59f, 0.63f},
                   {2,   1.809,  0.813, 0.735, 1, 0.90f, 0.40f, 0.36f});
  // clang-format on
}

void TestGammaRegressionBasic(const Context* ctx) {
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:gamma", ctx)};

  obj->Configure(args);
  CheckConfigReload(obj, "reg:gamma");

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {0}, {1}, {0}, {0}))
      << "Expected error when label = 0 for GammaRegression";
  EXPECT_ANY_THROW(CheckObjFunction(obj, {-1}, {-1}, {1}, {-1}, {-3}))
      << "Expected error when label < 0 for GammaRegression";

  // test ProbToMargin
  CheckProbaToMargin(obj, 0.1f, -2.30f);
  CheckProbaToMargin(obj, 0.5f, -0.69f);
  CheckProbaToMargin(obj, 0.9f, -0.10f);

  // test PredTransform
  HostDeviceVector<bst_float> io_preds = {0, 0.1f, 0.5f, 0.9f, 1};
  std::vector<bst_float> out_preds = {1, 1.10f, 1.64f, 2.45f, 2.71f};
  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();
  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
}

void TestTweedieRegressionGPair(const Context* ctx) {
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:tweedie", ctx)};

  args.emplace_back("tweedie_variance_power", "1.1f");
  obj->Configure(args);
  // clang-format off
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1, 0,  0.1f,  0.9f,    1},
                   {   0,    0,    0,    0, 1,    1,    1,    1},
                   {   1,    1,    1,    1, 1,    1,    1,    1},
                   {   1, 1.09f, 2.24f, 2.45f, 0, 0.10f, 1.33f, 1.55f},
                   {0.89f, 0.98f, 2.02f, 2.21f, 1, 1.08f, 2.11f, 2.30f});
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1, 0,  0.1f,  0.9f,    1},
                   {   0,    0,    0,    0, 1,    1,    1,    1},
                   {},  // Empty weight.
                   {   1, 1.09f, 2.24f, 2.45f, 0, 0.10f, 1.33f, 1.55f},
                   {0.89f, 0.98f, 2.02f, 2.21f, 1, 1.08f, 2.11f, 2.30f});
  // clang-format on
  ASSERT_EQ(obj->DefaultEvalMetric(), std::string{"tweedie-nloglik@1.1"});
}

void TestTweedieRegressionBasic(const Context* ctx) {
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:tweedie", ctx)};

  obj->Configure(args);
  CheckConfigReload(obj, "reg:tweedie");

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {-1}, {1}, {0}, {0}))
      << "Expected error when label < 0 for TweedieRegression";

  // test ProbToMargin
  CheckProbaToMargin(obj, 0.1f, -2.30f);
  CheckProbaToMargin(obj, 0.5f, -0.69f);
  CheckProbaToMargin(obj, 0.9f, -0.10f);

  // test PredTransform
  HostDeviceVector<bst_float> io_preds = {0, 0.1f, 0.5f, 0.9f, 1};
  std::vector<bst_float> out_preds = {1, 1.10f, 1.64f, 2.45f, 2.71f};
  obj->PredTransform(&io_preds);
  auto& preds = io_preds.HostVector();
  for (int i = 0; i < static_cast<int>(io_preds.Size()); ++i) {
    EXPECT_NEAR(preds[i], out_preds[i], 0.01f);
  }
}

void TestCoxRegressionGPair(const Context* ctx) {
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("survival:cox", ctx)};

  obj->Configure(args);
  // clang-format off
  CheckObjFunction(obj,
                   { 0, 0.1f, 0.9f,       1,       0,    0.1f,   0.9f,       1},
                   { 0,   -2,   -2,       2,       3,       5,    -10,     100},
                   { 1,    1,    1,       1,       1,       1,      1,       1},
                   { 0,    0,    0, -0.799f, -0.788f, -0.590f, 0.910f,  1.006f},
                   { 0,    0,    0,  0.160f,  0.186f,  0.348f, 0.610f,  0.639f});
  // clang-format on
}

void TestAbsoluteError(const Context* ctx) {
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:absoluteerror", ctx)};
  obj->Configure({});
  CheckConfigReload(obj, "reg:absoluteerror");
  ASSERT_FALSE(obj->Task().const_hess);

  auto check = [&](std::vector<float> const& predts, std::vector<float> const& labels,
                   std::vector<float> const& weights) {
    double sum_weight{0.0};
    double sum_root_residual{0.0};
    for (std::size_t i{0}; i < labels.size(); ++i) {
      auto const w = weights.empty() ? 1.0f : weights[i];
      sum_weight += w;
      sum_root_residual += w * std::sqrt(std::abs(predts[i] - labels[i]));
    }
    auto const root_mean = sum_weight == 0.0 ? 0.0 : sum_root_residual / sum_weight;
    auto const delta = static_cast<float>(root_mean * root_mean);
    std::vector<float> grad(labels.size());
    std::vector<float> hess(labels.size());
    for (std::size_t i{0}; i < labels.size(); ++i) {
      auto const residual = predts[i] - labels[i];
      auto const norm = std::hypot(delta, residual);
      auto const curvature = norm > 0.0f ? delta / norm : 1.0f;
      auto const w = weights.empty() ? 1.0f : weights[i];
      grad[i] = w * residual * curvature;
      hess[i] = w * curvature;
    }
    CheckObjFunction(obj, predts, labels, weights, grad, hess);
  };

  check({0.0f, 2.0f, 5.0f}, {1.0f, 0.0f, 1.0f}, {1.0f, 2.0f, 0.5f});
  check({0.0f, 2.0f, 5.0f}, {1.0f, 0.0f, 1.0f}, {});
  check({1.0f, 2.0f}, {1.0f, 2.0f}, {});

  MetaInfo info;
  info.num_row_ = 2;
  info.labels.Reshape(2, 2);
  info.labels.Data()->HostVector() = {0.0f, 0.0f, 0.0f, 0.0f};
  HostDeviceVector<float> predts{{1.0f, 100.0f, 4.0f, 400.0f}};
  linalg::Matrix<GradientPair> gpair;
  obj->GetGradient(predts, info, 0, &gpair);
  auto h_gpair = gpair.HostView();
  for (std::size_t row{0}; row < 2; ++row) {
    auto const residual = row == 0 ? 1.0f : 4.0f;
    auto const delta = 2.25f;
    auto const curvature = delta / std::hypot(delta, residual);
    ASSERT_NEAR(h_gpair(row, 0).GetGrad(), residual * curvature, kRtEps);
    ASSERT_NEAR(h_gpair(row, 1).GetGrad(), 100.0f * residual * curvature, 1.0e-4f);
    ASSERT_NEAR(h_gpair(row, 0).GetHess(), curvature, kRtEps);
    ASSERT_NEAR(h_gpair(row, 1).GetHess(), curvature, kRtEps);
  }

  auto expected_intercept = [](std::vector<float> const& labels,
                               std::vector<float> const& weights) {
    double sum_weight{0.0};
    double mean{0.0};
    for (std::size_t i{0}; i < labels.size(); ++i) {
      auto const w = weights.empty() ? 1.0f : weights[i];
      sum_weight += w;
      mean += w * labels[i];
    }
    mean /= sum_weight;

    double root_residual{0.0};
    for (std::size_t i{0}; i < labels.size(); ++i) {
      auto const w = weights.empty() ? 1.0f : weights[i];
      root_residual += w * std::sqrt(std::abs(mean - labels[i]));
    }
    auto const delta = std::pow(root_residual / sum_weight, 2.0);

    double sum_grad{0.0};
    double sum_hess{0.0};
    for (std::size_t i{0}; i < labels.size(); ++i) {
      auto const w = weights.empty() ? 1.0f : weights[i];
      auto const residual = mean - labels[i];
      auto const norm = std::hypot(delta, residual);
      auto const curvature = norm > 0.0 ? delta / norm : 1.0;
      sum_grad += w * residual * curvature;
      sum_hess += w * curvature;
    }
    return static_cast<float>(mean - sum_grad / sum_hess);
  };

  auto init = [&](std::vector<float> labels, std::vector<float> const& weights) {
    MetaInfo init_info;
    init_info.num_row_ = labels.size();
    init_info.labels.Reshape(labels.size(), 1);
    init_info.labels.Data()->HostVector() = std::move(labels);
    init_info.weights_.HostVector() = weights;
    linalg::Vector<float> base_score;
    obj->InitEstimation(init_info, &base_score);
    return base_score.HostView()(0);
  };

  for (auto const& weights : std::vector<std::vector<float>>{{}, {1.0f, 2.0f, 3.0f, 4.0f}}) {
    std::vector<float> labels{0.0f, 0.0f, 0.0f, 1000.0f};
    auto const expected = expected_intercept(labels, weights);
    ASSERT_NEAR(init(labels, weights), expected, 1.0e-4f);
    std::transform(labels.cbegin(), labels.cend(), labels.begin(),
                   [](float label) { return label + 1000.0f; });
    ASSERT_NEAR(init(labels, weights), expected + 1000.0f, 1.0e-4f);
  }
  ASSERT_EQ(obj->DefaultEvalMetric(), std::string{"mae"});
}

void TestExpectileRegressionGPair(const Context* ctx) {
  Args args{{"expectile_alpha", "0.8"}};

  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:expectileerror", ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, "reg:expectileerror");

  std::vector<float> predts{1.0f, 2.0f, 3.0f};
  std::vector<float> labels{3.0f, 2.0f, 1.0f};
  std::vector<float> weights{1.0f, 1.0f, 1.0f};
  std::vector<float> grad{-1.6f, 0.0f, 0.4f};
  std::vector<float> hess{0.8f, 0.2f, 0.2f};
  CheckObjFunction(obj, predts, labels, weights, grad, hess);
  CheckObjFunction(obj, predts, labels, {}, grad, hess);

  ASSERT_EQ(obj->DefaultEvalMetric(), std::string{"expectile"});
}

void TestExpectileRegressionMultiAlpha(const Context* ctx) {
  Args args{{"expectile_alpha", "[0.2, 0.8]"}};

  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:expectileerror", ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, "reg:expectileerror");

  std::vector<float> predts{0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> labels{1.0f, 2.0f};
  auto gap = kRtEps + common::SoftPlus(0.0f);
  std::vector<float> grad{-0.2f + 0.8f * (gap - 1.0f), 0.5f * 0.8f * (gap - 1.0f),
                          -0.4f + 0.8f * (gap - 2.0f), 0.5f * 0.8f * (gap - 2.0f)};
  std::vector<float> hess{1.0f, 0.2f, 1.0f, 0.2f};
  CheckObjFunction(obj, predts, labels, {}, grad, hess);
}

void TestExpectileRegressionInitEstimation(const Context* ctx) {
  Args args{{"expectile_alpha", "[0.2, 0.8]"}};
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:expectileerror", ctx)};
  obj->Configure(args);

  MetaInfo info;
  info.num_row_ = 10;
  info.labels.ModifyInplace([&](HostDeviceVector<float>* data, common::Span<std::size_t> shape) {
    data->SetDevice(ctx->Device());
    data->Resize(info.num_row_);
    shape[0] = info.num_row_;
    shape[1] = 1;

    auto& h_labels = data->HostVector();
    for (std::size_t i = 0; i < info.num_row_; ++i) {
      h_labels[i] = static_cast<float>(i);
    }
  });

  linalg::Vector<float> base_scores;
  obj->InitEstimation(info, &base_scores);
  ASSERT_EQ(base_scores.Size(), 2);
  auto one_step = [&](float alpha) {
    double sum_w = 0.0;
    double sum_wy = 0.0;
    double mean = 4.5;
    for (std::size_t i = 0; i < info.num_row_; ++i) {
      double label = static_cast<double>(i);
      double diff = mean - label;
      double w = diff >= 0.0 ? (1.0 - alpha) : alpha;
      sum_w += w;
      sum_wy += w * label;
    }
    return static_cast<float>(sum_wy / sum_w);
  };
  ASSERT_NEAR(base_scores(0), one_step(0.2f), kRtEps);
  ASSERT_NEAR(base_scores(1), one_step(0.8f), kRtEps);
}

void TestPseudoHuber(const Context* ctx) {
  Args args;

  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:pseudohubererror", ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, "reg:pseudohubererror");

  CheckObjFunction(obj, {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},                          // pred
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},                               // labels
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},                               // weights
                   {-0.668965f, -0.624695f, -0.514496f, -0.196116f, 0.514496f},  // out_grad
                   {0.410660f, 0.476140f, 0.630510f, 0.9428660f, 0.630510f});    // out_hess
  CheckObjFunction(obj, {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},                          // pred
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},                               // labels
                   {},                                                           // empty weights
                   {-0.668965f, -0.624695f, -0.514496f, -0.196116f, 0.514496f},  // out_grad
                   {0.410660f, 0.476140f, 0.630510f, 0.9428660f, 0.630510f});    // out_hess
  ASSERT_EQ(obj->DefaultEvalMetric(), std::string{"mphe"});

  obj->Configure({{"huber_slope", "0.1"}});
  CheckConfigReload(obj, "reg:pseudohubererror");
  CheckObjFunction(obj, {0.1f, 0.2f, 0.4f, 0.8f, 1.6f},                          // pred
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},                               // labels
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},                               // weights
                   {-0.099388f, -0.099228f, -0.098639f, -0.089443f, 0.098639f},  // out_grad
                   {0.0013467f, 0.001908f, 0.004443f, 0.089443f, 0.004443f});    // out_hess
}

}  // namespace xgboost
