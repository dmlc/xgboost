/**
 * Copyright 2017-2025, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/c_api.h>
#include <xgboost/learner.h>
#include <xgboost/logging.h>
#include <xgboost/predictor.h>

#include <string>

#include "../../../src/data/device_adapter.cuh"
#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../collective/test_worker.h"  // for TestDistributedGlobal, BaseMGPUTest
#include "../helpers.h"
#include "test_predictor.h"
#include "test_shap.h"

namespace xgboost::predictor {
TEST(GPUPredictor, Basic) {
  auto cpu_lparam = MakeCUDACtx(-1);
  auto gpu_lparam = MakeCUDACtx(0);

  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &gpu_lparam));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &cpu_lparam));

  gpu_predictor->Configure({});
  cpu_predictor->Configure({});

  for (size_t i = 1; i < 33; i *= 2) {
    int n_row = i, n_col = i;
    auto dmat = RandomDataGenerator(n_row, n_col, 0).GenerateDMatrix();

    auto ctx = MakeCUDACtx(0);
    LearnerModelParam mparam{MakeMP(n_col, .5, 1, ctx.Device())};
    std::unique_ptr<gbm::GBTreeModel> p_model = CreateTestModel(&mparam, &ctx);
    auto const& model = *p_model;

    // Test predict batch
    PredictionCacheEntry gpu_out_predictions;
    PredictionCacheEntry cpu_out_predictions;

    gpu_predictor->InitOutPredictions(dmat->Info(), &gpu_out_predictions.predictions, model);
    gpu_predictor->PredictBatch(dmat.get(), &gpu_out_predictions, model, 0);
    cpu_predictor->InitOutPredictions(dmat->Info(), &cpu_out_predictions.predictions, model);
    cpu_predictor->PredictBatch(dmat.get(), &cpu_out_predictions, model, 0);

    std::vector<float>& gpu_out_predictions_h = gpu_out_predictions.predictions.HostVector();
    std::vector<float>& cpu_out_predictions_h = cpu_out_predictions.predictions.HostVector();
    float abs_tolerance = 0.001;
    for (size_t j = 0; j < gpu_out_predictions.predictions.Size(); j++) {
      ASSERT_NEAR(gpu_out_predictions_h[j], cpu_out_predictions_h[j], abs_tolerance);
    }
  }
}

TEST(GPUPredictor, BatchPredictionWithWeights) {
  auto ctx = MakeCUDACtx(0);
  TestBatchPredictionWithWeights(&ctx);
}

TEST(GPUPredictor, InplacePredictionWithWeights) {
  auto ctx = MakeCUDACtx(0);
  TestInplacePredictionWithWeights(&ctx);
}

namespace {

}  // namespace xgboost::predictor
