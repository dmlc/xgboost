/*!
 * Copyright 2017-2022 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/objective.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/json.h>
#include "../helpers.h"
namespace xgboost {

TEST(Objective, DeclareUnifiedTest(LinearRegressionGPair)) {
  GenericParameter tparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;

  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("reg:squarederror", &tparam)
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

TEST(Objective, DeclareUnifiedTest(SquaredLog)) {
  GenericParameter tparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;

  std::unique_ptr<ObjFunction> obj { ObjFunction::Create("reg:squaredlogerror", &tparam) };
  obj->Configure(args);
  CheckConfigReload(obj, "reg:squaredlogerror");

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

TEST(Objective, DeclareUnifiedTest(PseudoHuber)) {
  GenericParameter tparam = CreateEmptyGenericParam(GPUIDX);
  Args args;

  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:pseudohubererror", &tparam)};
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

TEST(Objective, DeclareUnifiedTest(LogisticRegressionGPair)) {
  GenericParameter tparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj { ObjFunction::Create("reg:logistic", &tparam) };

  obj->Configure(args);
  CheckConfigReload(obj, "reg:logistic");

  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,  0.9f,      1}, // preds
                   {   0,    0,    0,    0,    1,     1,     1,     1}, // labels
                   {   1,    1,    1,    1,    1,     1,     1,     1}, // weights
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f}, // out_grad
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f}); // out_hess
}

TEST(Objective, DeclareUnifiedTest(LogisticRegressionBasic)) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("reg:logistic", &lparam)
  };

  obj->Configure(args);
  CheckConfigReload(obj, "reg:logistic");

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

TEST(Objective, DeclareUnifiedTest(LogisticRawGPair)) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction>  obj {
    ObjFunction::Create("binary:logitraw", &lparam)
  };
  obj->Configure(args);

  CheckObjFunction(obj,
                   {   0,  0.1f,  0.9f,    1,    0,   0.1f,   0.9f,     1},
                   {   0,    0,    0,    0,    1,     1,     1,     1},
                   {   1,    1,    1,    1,    1,     1,     1,     1},
                   { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f},
                   {0.25f, 0.24f, 0.20f, 0.19f, 0.25f,  0.24f,  0.20f,  0.19f});
}

TEST(Objective, DeclareUnifiedTest(PoissonRegressionGPair)) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("count:poisson", &lparam)
  };

  args.emplace_back(std::make_pair("max_delta_step", "0.1f"));
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

TEST(Objective, DeclareUnifiedTest(PoissonRegressionBasic)) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("count:poisson", &lparam)
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

TEST(Objective, DeclareUnifiedTest(GammaRegressionGPair)) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("reg:gamma", &lparam)
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

TEST(Objective, DeclareUnifiedTest(GammaRegressionBasic)) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("reg:gamma", &lparam)
  };

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

TEST(Objective, DeclareUnifiedTest(TweedieRegressionGPair)) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("reg:tweedie", &lparam)
  };

  args.emplace_back(std::make_pair("tweedie_variance_power", "1.1f"));
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

#if defined(__CUDACC__)
TEST(Objective, CPU_vs_CUDA) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);

  ObjFunction * obj =
      ObjFunction::Create("reg:squarederror", &lparam);
  HostDeviceVector<GradientPair> cpu_out_preds;
  HostDeviceVector<GradientPair> cuda_out_preds;

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

  info.labels.Reshape(kRows);
  auto& h_labels = info.labels.Data()->HostVector();
  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = 1 / (float)(i+1);
  }

  {
    // CPU
    lparam.gpu_id = -1;
    obj->GetGradient(preds, info, 0, &cpu_out_preds);
  }
  {
    // CUDA
    lparam.gpu_id = 0;
    obj->GetGradient(preds, info, 0, &cuda_out_preds);
  }

  auto& h_cpu_out = cpu_out_preds.HostVector();
  auto& h_cuda_out = cuda_out_preds.HostVector();

  float sgrad = 0;
  float shess = 0;
  for (size_t i = 0; i < kRows; ++i) {
    sgrad += std::pow(h_cpu_out[i].GetGrad() - h_cuda_out[i].GetGrad(), 2);
    shess += std::pow(h_cpu_out[i].GetHess() - h_cuda_out[i].GetHess(), 2);
  }
  ASSERT_NEAR(sgrad, 0.0f, kRtEps);
  ASSERT_NEAR(shess, 0.0f, kRtEps);

  delete obj;
}
#endif

TEST(Objective, DeclareUnifiedTest(TweedieRegressionBasic)) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("reg:tweedie", &lparam)
  };

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

// CoxRegression not implemented in GPU code, no need for testing.
#if !defined(__CUDACC__)
TEST(Objective, CoxRegressionGPair) {
  GenericParameter lparam = CreateEmptyGenericParam(GPUIDX);
  std::vector<std::pair<std::string, std::string>> args;
  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("survival:cox", &lparam)
  };

  obj->Configure(args);
  CheckObjFunction(obj,
                   { 0, 0.1f, 0.9f,       1,       0,    0.1f,   0.9f,       1},
                   { 0,   -2,   -2,       2,       3,       5,    -10,     100},
                   { 1,    1,    1,       1,       1,       1,      1,       1},
                   { 0,    0,    0, -0.799f, -0.788f, -0.590f, 0.910f,  1.006f},
                   { 0,    0,    0,  0.160f,  0.186f,  0.348f, 0.610f,  0.639f});
}
#endif

TEST(Objective, DeclareUnifiedTest(AbsoluteError)) {
  Context ctx = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:absoluteerror", &ctx)};
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
  std::vector<RowIndexCache> row_idx;
  row_idx.emplace_back(&ctx, info.labels.Shape(0));

  row_idx.back().node_idx.HostVector().push_back(1);  // left
  row_idx.back().node_idx.HostVector().push_back(2);  // right
  auto& ptr = row_idx.back().node_ptr.HostVector();
  ptr.push_back(0);
  ptr.push_back(3);
  ptr.push_back(info.labels.Size());
  auto& h_row_idx = row_idx.back().row_index.HostVector();
  for (size_t i = info.labels.Size() - 1;; --i) {
    h_row_idx[i] = i;
    if (i == 0) {
      break;
    }
  }

  auto& h_predt = predt.HostVector();
  for (size_t i = 0; i < h_predt.size(); ++i) {
    h_predt[i] = labels[i] + i;
  }
  // obj->UpdateTreeLeaf(common::Span<RowIndexCache const>{row_idx}, info, predt, &tree);
  ASSERT_EQ(tree[1].LeafValue(), -1);
  ASSERT_EQ(tree[2].LeafValue(), -4);
}

TEST(Objective, DeclareUnifiedTest(AbsoluteErrorLeaf)) {
  Context ctx = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:absoluteerror", &ctx)};
  obj->Configure({});

  MetaInfo info;
  info.labels.Reshape(16, 1);
  info.num_row_ = info.labels.Size();
  CHECK_EQ(info.num_row_, 16);
  auto h_labels = info.labels.HostView().Values();
  std::iota(h_labels.begin(), h_labels.end(), 0);
  HostDeviceVector<float> predt(h_labels.size());
  auto& h_predt = predt.HostVector();
  for (size_t i = 0; i < h_predt.size(); ++i) {
    h_predt[i] = h_labels[i] + i;
  }

  std::vector<RowIndexCache> row_idx_v;
  row_idx_v.emplace_back(&ctx, info.labels.Shape(0));

  auto& part = row_idx_v.back();
  part.node_idx = {3, 4, 5, 6};
  // starting from 3 to emulate subsampling, empty leaaft for node 4.
  part.node_ptr = {3, 8, 8, 13, 16};
  auto& h_row_idx = part.row_index.HostVector();
  std::iota(h_row_idx.begin(), h_row_idx.end(), 0);

  RegTree tree;
  tree.ExpandNode(0, /*split_index=*/1, 2, true, 0.0f, 2.f, 3.f, 4.f, 2.f, 1.f, 1.f);
  tree.ExpandNode(1, /*split_index=*/1, 2, true, 0.0f, 2.f, 3.f, 4.f, 2.f, 1.f, 1.f);
  tree.ExpandNode(2, /*split_index=*/1, 2, true, 0.0f, 2.f, 3.f, 4.f, 2.f, 1.f, 1.f);
  ASSERT_EQ(tree.GetNumLeaves(), 4);

  auto empty_leaf = tree[4].LeafValue();
  // obj->UpdateTreeLeaf(row_idx_v, info, predt, &tree);
  ASSERT_EQ(tree[3].LeafValue(), -5);
  ASSERT_EQ(tree[4].LeafValue(), empty_leaf);
  ASSERT_EQ(tree[5].LeafValue(), -10);
  ASSERT_EQ(tree[6].LeafValue(), -14);
}
}  // namespace xgboost
