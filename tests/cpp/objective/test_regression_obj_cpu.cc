/**
 * Copyright 2018-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/objective.h>

#include "../../../src/objective/adaptive.h"
#include "../../../src/tree/param.h"  // for TrainParam
#include "../helpers.h"
#include "test_regression_obj.h"

namespace xgboost {
TEST(Objective, DeclareUnifiedTest(LinearRegressionGPair)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestLinearRegressionGPair(&ctx);
}

TEST(Objective, DeclareUnifiedTest(SquaredLog)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestSquaredLog(&ctx);
}

TEST(Objective, DeclareUnifiedTest(PseudoHuber)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestPseudoHuber(&ctx);
}

TEST(Objective, DeclareUnifiedTest(LogisticRegressionGPair)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestLogisticRegressionGPair(&ctx);
}

TEST(Objective, DeclareUnifiedTest(LogisticRegressionBasic)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestLogisticRegressionBasic(&ctx);
}

TEST(Objective, DeclareUnifiedTest(LogisticRawGPair)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestsLogisticRawGPair(&ctx);
}

TEST(Objective, DeclareUnifiedTest(PoissonRegressionGPair)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestPoissonRegressionGPair(&ctx);
}

TEST(Objective, DeclareUnifiedTest(PoissonRegressionBasic)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestPoissonRegressionBasic(&ctx);
}

TEST(Objective, DeclareUnifiedTest(GammaRegressionGPair)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestGammaRegressionGPair(&ctx);
}

TEST(Objective, DeclareUnifiedTest(GammaRegressionBasic)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestGammaRegressionBasic(&ctx);
}

TEST(Objective, DeclareUnifiedTest(TweedieRegressionGPair)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestTweedieRegressionGPair(&ctx);
}

#if defined(__CUDACC__)
TEST(Objective, CPU_vs_CUDA) {
  Context ctx = MakeCUDACtx(GPUIDX);

  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:squarederror", &ctx)};
  linalg::Matrix<GradientPair> cpu_out_preds;
  linalg::Matrix<GradientPair> cuda_out_preds;

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
    h_labels[i] = 1 / static_cast<float>(i+1);
  }

  {
    // CPU
    ctx = ctx.MakeCPU();
    obj->GetGradient(preds, info, 0, &cpu_out_preds);
  }
  {
    // CUDA
    ctx = ctx.MakeCUDA(0);
    obj->GetGradient(preds, info, 0, &cuda_out_preds);
  }

  auto h_cpu_out = cpu_out_preds.HostView();
  auto h_cuda_out = cuda_out_preds.HostView();

  float sgrad = 0;
  float shess = 0;
  for (size_t i = 0; i < kRows; ++i) {
    sgrad += std::pow(h_cpu_out(i).GetGrad() - h_cuda_out(i).GetGrad(), 2);
    shess += std::pow(h_cpu_out(i).GetHess() - h_cuda_out(i).GetHess(), 2);
  }
  ASSERT_NEAR(sgrad, 0.0f, kRtEps);
  ASSERT_NEAR(shess, 0.0f, kRtEps);
}
#endif

TEST(Objective, DeclareUnifiedTest(TweedieRegressionBasic)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestTweedieRegressionBasic(&ctx);
}

// CoxRegression not implemented in GPU code, no need for testing.
#if !defined(__CUDACC__)
TEST(Objective, CoxRegressionGPair) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestCoxRegressionGPair(&ctx);
}
#endif

TEST(Objective, DeclareUnifiedTest(AbsoluteError)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestAbsoluteError(&ctx);
}

TEST(Objective, DeclareUnifiedTest(AbsoluteErrorLeaf)) {
  Context ctx = MakeCUDACtx(GPUIDX);
  TestAbsoluteErrorLeaf(&ctx);
}

TEST(Adaptive, DeclareUnifiedTest(MissingLeaf)) {
  std::vector<bst_node_t> missing{1, 3};

  std::vector<bst_node_t> h_nidx = {2, 4, 5};
  std::vector<size_t> h_nptr = {0, 4, 8, 16};

  obj::detail::FillMissingLeaf(missing, &h_nidx, &h_nptr);

  ASSERT_EQ(h_nidx[0], missing[0]);
  ASSERT_EQ(h_nidx[2], missing[1]);
  ASSERT_EQ(h_nidx[1], 2);
  ASSERT_EQ(h_nidx[3], 4);
  ASSERT_EQ(h_nidx[4], 5);

  ASSERT_EQ(h_nptr[0], 0);
  ASSERT_EQ(h_nptr[1], 0);  // empty
  ASSERT_EQ(h_nptr[2], 4);
  ASSERT_EQ(h_nptr[3], 4);  // empty
  ASSERT_EQ(h_nptr[4], 8);
  ASSERT_EQ(h_nptr[5], 16);
}
}  // namespace xgboost
