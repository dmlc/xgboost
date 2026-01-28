/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>
#include <xgboost/learner.h>

#include <memory>

#include "../helpers.h"
#include "test_predictor.h"
#include "test_shap.h"

namespace xgboost {

TEST(GPUPredictor, CompareCPUShap) {
  auto ctx = MakeCUDACtx(0);
  Context cpu_ctx;
  bst_feature_t constexpr kCols{10};
  bst_idx_t constexpr kRows{1000};
  std::size_t constexpr kIters{10};

  HostDeviceVector<float> predictions;
  HostDeviceVector<float> cpu_predictions;
  HostDeviceVector<float> interactions;
  HostDeviceVector<float> cpu_interactions;

  auto dmat = RandomDataGenerator(kRows, kCols, 0.0).Device(ctx.Device()).GenerateDMatrix();
  dmat->Info().labels.Reshape(kRows, 1);
  auto& h_labels = dmat->Info().labels.Data()->HostVector();
  for (size_t i = 0; i < kRows; ++i) {
    h_labels[i] = i % 2;
  }

  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->SetParams(Args{{"objective", "binary:logistic"},
                          {"max_depth", "12"},
                          {"min_split_loss", "0"},
                          {"min_child_weight", "0"},
                          {"reg_lambda", "0"},
                          {"reg_alpha", "0"},
                          {"subsample", "1"},
                          {"colsample_bytree", "1"},
                          {"device", ctx.DeviceName()}});
  learner->Configure();
  for (std::size_t i = 0; i < kIters; ++i) {
    learner->UpdateOneIter(i, dmat);
  }

  Json model{Object{}};
  learner->SaveModel(&model);

  std::unique_ptr<Learner> learner_gpu{Learner::Create({})};
  learner_gpu->LoadModel(model);
  learner_gpu->SetParam("device", ctx.DeviceName());
  learner_gpu->Configure();

  std::unique_ptr<Learner> learner_cpu{Learner::Create({})};
  learner_cpu->LoadModel(model);
  learner_cpu->SetParam("device", cpu_ctx.DeviceName());
  learner_cpu->Configure();

  learner_gpu->Predict(dmat, false, &predictions, 0, 0, false, false, true, false, false);
  learner_cpu->Predict(dmat, false, &cpu_predictions, 0, 0, false, false, true, false, false);
  learner_gpu->Predict(dmat, false, &interactions, 0, 0, false, false, false, false, true);
  learner_cpu->Predict(dmat, false, &cpu_interactions, 0, 0, false, false, false, false, true);
  auto& phis = predictions.HostVector();
  auto& cpu_phis = cpu_predictions.HostVector();
  for (auto i = 0ull; i < phis.size(); i++) {
    EXPECT_NEAR(cpu_phis[i], phis[i], 1e-4);
  }

  auto& inter = interactions.HostVector();
  auto& cpu_inter = cpu_interactions.HostVector();
  for (auto i = 0ull; i < inter.size(); i++) {
    EXPECT_NEAR(cpu_inter[i], inter[i], 1e-3);
  }
}

TEST(GPUPredictor, ShapOutputCasesGPU) {
  auto ctx = MakeCUDACtx(0);
  auto cases = BuildShapTestCases(&ctx);
  for (auto const& [dmat, args] : cases) {
    CheckShapOutput(&ctx, dmat.get(), args);
  }
}
}  // namespace xgboost
