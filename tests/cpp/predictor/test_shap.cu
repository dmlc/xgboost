/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/predictor.h>

#include <memory>

#include "../helpers.h"
#include "test_predictor.h"
#include "test_shap.h"

namespace xgboost {
// Very basic test of empty model
TEST(GPUPredictor, ShapStump) {
  cudaSetDevice(0);

  auto ctx = MakeCUDACtx(0);
  LearnerModelParam mparam{MakeMP(1, .5, 1, ctx.Device())};
  gbm::GBTreeModel model(&mparam, &ctx);

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::make_unique<RegTree>());
  model.CommitModelGroup(std::move(trees), 0);

  auto gpu_lparam = MakeCUDACtx(0);
  std::unique_ptr<Predictor> gpu_predictor = std::unique_ptr<Predictor>(
      Predictor::Create("gpu_predictor", &gpu_lparam));
  gpu_predictor->Configure({});
  HostDeviceVector<float> predictions;
  auto dmat = RandomDataGenerator(3, 1, 0).GenerateDMatrix();
  gpu_predictor->PredictContribution(dmat.get(), &predictions, model);
  auto& phis = predictions.HostVector();
  auto base_score = mparam.BaseScore(DeviceOrd::CPU())(0);
  EXPECT_EQ(phis[0], 0.0);
  EXPECT_EQ(phis[1], base_score);
  EXPECT_EQ(phis[2], 0.0);
  EXPECT_EQ(phis[3], base_score);
  EXPECT_EQ(phis[4], 0.0);
  EXPECT_EQ(phis[5], base_score);
}

TEST(GPUPredictor, Shap) {
  auto ctx = MakeCUDACtx(0);
  LearnerModelParam mparam{MakeMP(1, .5, 1, ctx.Device())};
  gbm::GBTreeModel model(&mparam, &ctx);

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::make_unique<RegTree>());
  trees[0]->ExpandNode(0, 0, 0.5, true, 1.0, -1.0, 1.0, 0.0, 5.0, 2.0, 3.0);
  model.CommitModelGroup(std::move(trees), 0);

  auto cpu_lparam = MakeCUDACtx(-1);
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &ctx));
  std::unique_ptr<Predictor> cpu_predictor = std::unique_ptr<Predictor>(
      Predictor::Create("cpu_predictor", &cpu_lparam));
  gpu_predictor->Configure({});
  cpu_predictor->Configure({});
  HostDeviceVector<float> predictions;
  HostDeviceVector<float> cpu_predictions;
  auto dmat = RandomDataGenerator(3, 1, 0).GenerateDMatrix();
  gpu_predictor->PredictContribution(dmat.get(), &predictions, model);
  cpu_predictor->PredictContribution(dmat.get(), &cpu_predictions, model);
  auto& phis = predictions.HostVector();
  auto& cpu_phis = cpu_predictions.HostVector();
  for (auto i = 0ull; i < phis.size(); i++) {
    EXPECT_NEAR(cpu_phis[i], phis[i], 1e-3);
  }
}
}  // namespace xgboost
