/**
 * Copyright 2017-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/json.h>
#include <xgboost/objective.h>

#include <numeric>  // for iota

#include "../../../src/common/linalg_op.h"  // for begin, end
#include "../../../src/objective/adaptive.h"
#include "../../../src/tree/param.h"        // for TrainParam
#include "../helpers.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/linalg.h"

#include "test_regression_obj.h"

namespace xgboost {

void TestLinearRegressionGPair(const Context* ctx) {
  std::string obj_name = "reg:squarederror";

  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(obj_name, ctx)};

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

void TestSquaredLog(const Context* ctx) {
  std::string obj_name = "reg:squaredlogerror";
  std::vector<std::pair<std::string, std::string>> args;

  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(obj_name, ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, obj_name);

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

void TestLogisticRegressionGPair(const Context* ctx) {
  std::string obj_name = "reg:logistic";
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(obj_name, ctx)};

  obj->Configure(args);
  CheckConfigReload(obj, obj_name);

  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,  0.9f,      1}, // preds
                   {   0,    0,    0,    0,    1,     1,     1,     1}, // labels
                   {   1,    1,    1,    1,    1,     1,     1,     1}, // weights
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f}, // out_grad
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f}); // out_hess
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
  EXPECT_NEAR(obj->ProbToMargin(0.1f), -2.197f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.5f), 0, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.9f), 2.197f, 0.01f);
  EXPECT_ANY_THROW((void)obj->ProbToMargin(10))
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

void TestsLogisticRawGPair(const Context* ctx) {
  std::string obj_name = "binary:logitraw";
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction>  obj {ObjFunction::Create(obj_name, ctx)};
  obj->Configure(args);

  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,   0.9f,     1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f},
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f});
}

void TestPoissonRegressionGPair(const Context* ctx) {
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("count:poisson", ctx)
  };

  args.emplace_back("max_delta_step", "0.1f");
  obj->Configure(args);

  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,  0.1f,  0.9f,    1},
                   {   0,    0,    0,    0,    1,    1,    1,    1},
                   {   1,    1,    1,    1,    1,    1,    1,    1},
                   {   1, 1.10f, 2.45f, 2.71f,    0, 0.10f, 1.45f, 1.71f},
                   {1.10f, 1.22f, 2.71f, 3.00f, 1.10f, 1.22f, 2.71f, 3.00f});
  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,  0.1f,  0.9f,    1},
                   {   0,    0,    0,    0,    1,    1,    1,    1},
                   {},  // Empty weight
                   {   1, 1.10f, 2.45f, 2.71f,    0, 0.10f, 1.45f, 1.71f},
                   {1.10f, 1.22f, 2.71f, 3.00f, 1.10f, 1.22f, 2.71f, 3.00f});
}

void TestPoissonRegressionBasic(const Context* ctx) {
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("count:poisson", ctx)
  };

  obj->Configure(args);
  CheckConfigReload(obj, "count:poisson");

  // test label validation
  EXPECT_ANY_THROW(CheckObjFunction(obj, {0}, {-1}, {1}, {0}, {0}))
    << "Expected error when label < 0 for PoissonRegression";

  // test ProbToMargin
  EXPECT_NEAR(obj->ProbToMargin(0.1f), -2.30f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.5f), -0.69f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.9f), -0.10f, 0.01f);

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
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("reg:gamma", ctx)
  };

  obj->Configure(args);
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
  EXPECT_NEAR(obj->ProbToMargin(0.1f), -2.30f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.5f), -0.69f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.9f), -0.10f, 0.01f);

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
  EXPECT_NEAR(obj->ProbToMargin(0.1f), -2.30f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.5f), -0.69f, 0.01f);
  EXPECT_NEAR(obj->ProbToMargin(0.9f), -0.10f, 0.01f);

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
  CheckObjFunction(obj,
                   { 0, 0.1f, 0.9f,       1,       0,    0.1f,   0.9f,       1},
                   { 0,   -2,   -2,       2,       3,       5,    -10,     100},
                   { 1,    1,    1,       1,       1,       1,      1,       1},
                   { 0,    0,    0, -0.799f, -0.788f, -0.590f, 0.910f,  1.006f},
                   { 0,    0,    0,  0.160f,  0.186f,  0.348f, 0.610f,  0.639f});
}

void TestAbsoluteError(const Context* ctx) {
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:absoluteerror", ctx)};
  obj->Configure({});
  CheckConfigReload(obj, "reg:absoluteerror");

  MetaInfo info;
  std::vector<float> labels{0.f, 3.f, 2.f, 5.f, 4.f, 7.f};
  info.labels.Reshape(6, 1);
  info.labels.Data()->HostVector() = labels;
  info.num_row_ = labels.size();
  HostDeviceVector<float> predt{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  info.weights_.HostVector() = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

  CheckObjFunction(obj, predt.HostVector(), labels, info.weights_.HostVector(),
                   {1.f, -1.f, 1.f, -1.f, 1.f, -1.f}, info.weights_.HostVector());

  RegTree tree;
  tree.ExpandNode(0, /*split_index=*/1, 2, true, 0.0f, 2.f, 3.f, 4.f, 2.f, 1.f, 1.f);

  HostDeviceVector<bst_node_t> position(labels.size(), 0);
  auto& h_position = position.HostVector();
  for (size_t i = 0; i < labels.size(); ++i) {
    if (i < labels.size() / 2) {
      h_position[i] = 1;  // left
    } else {
      h_position[i] = 2;  // right
    }
  }

  auto& h_predt = predt.HostVector();
  for (size_t i = 0; i < h_predt.size(); ++i) {
    h_predt[i] = labels[i] + i;
  }

  tree::TrainParam param;
  param.Init(Args{});
  auto lr = param.learning_rate;

  obj->UpdateTreeLeaf(position, info, param.learning_rate, predt, 0, &tree);
  ASSERT_EQ(tree[1].LeafValue(), -1.0f * lr);
  ASSERT_EQ(tree[2].LeafValue(), -4.0f * lr);
}

void TestAbsoluteErrorLeaf(const Context* ctx) {
  bst_target_t constexpr kTargets = 3, kRows = 16;
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:absoluteerror", ctx)};
  obj->Configure({});

  MetaInfo info;
  info.num_row_ = kRows;
  info.labels.Reshape(16, kTargets);
  HostDeviceVector<float> predt(info.labels.Size());

  for (bst_target_t t{0}; t < kTargets; ++t) {
    auto h_labels = info.labels.HostView().Slice(linalg::All(), t);
    std::iota(linalg::begin(h_labels), linalg::end(h_labels), 0);

    auto h_predt =
        linalg::MakeTensorView(ctx, predt.HostSpan(), kRows, kTargets).Slice(linalg::All(), t);
    for (size_t i = 0; i < h_predt.Size(); ++i) {
      h_predt(i) = h_labels(i) + i;
    }

    HostDeviceVector<bst_node_t> position(h_labels.Size(), 0);
    auto& h_position = position.HostVector();
    for (int32_t i = 0; i < 3; ++i) {
      h_position[i] = ~i;  // negation for sampled nodes.
    }
    for (size_t i = 3; i < 8; ++i) {
      h_position[i] = 3;
    }
    // empty leaf for node 4
    for (size_t i = 8; i < 13; ++i) {
      h_position[i] = 5;
    }
    for (size_t i = 13; i < h_labels.Size(); ++i) {
      h_position[i] = 6;
    }

    RegTree tree;
    tree.ExpandNode(0, /*split_index=*/1, 2, true, 0.0f, 2.f, 3.f, 4.f, 2.f, 1.f, 1.f);
    tree.ExpandNode(1, /*split_index=*/1, 2, true, 0.0f, 2.f, 3.f, 4.f, 2.f, 1.f, 1.f);
    tree.ExpandNode(2, /*split_index=*/1, 2, true, 0.0f, 2.f, 3.f, 4.f, 2.f, 1.f, 1.f);
    ASSERT_EQ(tree.GetNumLeaves(), 4);

    auto empty_leaf = tree[4].LeafValue();

    tree::TrainParam param;
    param.Init(Args{});
    auto lr = param.learning_rate;

    obj->UpdateTreeLeaf(position, info, lr, predt, t, &tree);
    ASSERT_EQ(tree[3].LeafValue(), -5.0f * lr);
    ASSERT_EQ(tree[4].LeafValue(), empty_leaf * lr);
    ASSERT_EQ(tree[5].LeafValue(), -10.0f * lr);
    ASSERT_EQ(tree[6].LeafValue(), -14.0f * lr);
  }
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
