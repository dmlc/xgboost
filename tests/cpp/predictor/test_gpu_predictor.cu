/**
 * Copyright 2017-2023, XGBoost contributors
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
#include "../helpers.h"
#include "test_predictor.h"

namespace xgboost {
namespace predictor {

TEST(GPUPredictor, Basic) {
  auto cpu_lparam = CreateEmptyGenericParam(-1);
  auto gpu_lparam = CreateEmptyGenericParam(0);

  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &gpu_lparam));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &cpu_lparam));

  gpu_predictor->Configure({});
  cpu_predictor->Configure({});

  for (size_t i = 1; i < 33; i *= 2) {
    int n_row = i, n_col = i;
    auto dmat = RandomDataGenerator(n_row, n_col, 0).GenerateDMatrix();

    Context ctx;
    ctx.gpu_id = 0;
    LearnerModelParam mparam{MakeMP(n_col, .5, 1, ctx.gpu_id)};
    gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

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

TEST(GPUPredictor, EllpackBasic) {
  size_t constexpr kCols {8};
  for (size_t bins = 2; bins < 258; bins += 16) {
    size_t rows = bins * 16;
    auto p_m = RandomDataGenerator{rows, kCols, 0.0}.Bins(bins).Device(0).GenerateDeviceDMatrix();
    ASSERT_FALSE(p_m->PageExists<SparsePage>());
    TestPredictionFromGradientIndex<EllpackPage>("gpu_predictor", rows, kCols, p_m);
    TestPredictionFromGradientIndex<EllpackPage>("gpu_predictor", bins, kCols, p_m);
  }
}

TEST(GPUPredictor, EllpackTraining) {
  size_t constexpr kRows { 128 }, kCols { 16 }, kBins { 64 };
  auto p_ellpack =
      RandomDataGenerator{kRows, kCols, 0.0}.Bins(kBins).Device(0).GenerateDeviceDMatrix();
  HostDeviceVector<float> storage(kRows * kCols);
  auto columnar = RandomDataGenerator{kRows, kCols, 0.0}
       .Device(0)
       .GenerateArrayInterface(&storage);
  auto adapter = data::CupyAdapter(columnar);
  std::shared_ptr<DMatrix> p_full {
    DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)
  };
  TestTrainingPrediction(kRows, kBins, "gpu_hist", p_full, p_ellpack);
}

TEST(GPUPredictor, ExternalMemoryTest) {
  auto lparam = CreateEmptyGenericParam(0);
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &lparam));
  gpu_predictor->Configure({});

  const int n_classes = 3;
  Context ctx;
  ctx.gpu_id = 0;
  LearnerModelParam mparam{MakeMP(5, .5, n_classes, ctx.gpu_id)};

  gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx, n_classes);
  std::vector<std::unique_ptr<DMatrix>> dmats;

  dmats.push_back(CreateSparsePageDMatrix(400));
  dmats.push_back(CreateSparsePageDMatrix(800));
  dmats.push_back(CreateSparsePageDMatrix(8000));

  for (const auto& dmat: dmats) {
    dmat->Info().base_margin_ = decltype(dmat->Info().base_margin_){
        {dmat->Info().num_row_, static_cast<size_t>(n_classes)}, 0};
    dmat->Info().base_margin_.Data()->Fill(0.5);
    PredictionCacheEntry out_predictions;
    gpu_predictor->InitOutPredictions(dmat->Info(), &out_predictions.predictions, model);
    gpu_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);
    EXPECT_EQ(out_predictions.predictions.Size(), dmat->Info().num_row_ * n_classes);
    const std::vector<float> &host_vector = out_predictions.predictions.ConstHostVector();
    for (size_t i = 0; i < host_vector.size() / n_classes; i++) {
      ASSERT_EQ(host_vector[i * n_classes], 2.0);
      ASSERT_EQ(host_vector[i * n_classes + 1], 0.5);
      ASSERT_EQ(host_vector[i * n_classes + 2], 0.5);
    }
  }
}

TEST(GPUPredictor, InplacePredictCupy) {
  size_t constexpr kRows{128}, kCols{64};
  RandomDataGenerator gen(kRows, kCols, 0.5);
  gen.Device(0);
  HostDeviceVector<float> data;
  std::string interface_str = gen.GenerateArrayInterface(&data);
  std::shared_ptr<DMatrix> p_fmat{new data::DMatrixProxy};
  dynamic_cast<data::DMatrixProxy*>(p_fmat.get())->SetCUDAArray(interface_str.c_str());
  TestInplacePrediction(p_fmat, "gpu_predictor", kRows, kCols, 0);
}

TEST(GPUPredictor, InplacePredictCuDF) {
  size_t constexpr kRows{128}, kCols{64};
  RandomDataGenerator gen(kRows, kCols, 0.5);
  gen.Device(0);
  std::vector<HostDeviceVector<float>> storage(kCols);
  auto interface_str = gen.GenerateColumnarArrayInterface(&storage);
  std::shared_ptr<DMatrix> p_fmat{new data::DMatrixProxy};
  dynamic_cast<data::DMatrixProxy*>(p_fmat.get())->SetCUDAArray(interface_str.c_str());
  TestInplacePrediction(p_fmat, "gpu_predictor", kRows, kCols, 0);
}

TEST(GpuPredictor, LesserFeatures) {
  TestPredictionWithLesserFeatures("gpu_predictor");
}

// Very basic test of empty model
TEST(GPUPredictor, ShapStump) {
  cudaSetDevice(0);

  Context ctx;
  ctx.gpu_id = 0;
  LearnerModelParam mparam{MakeMP(1, .5, 1, ctx.gpu_id)};
  gbm::GBTreeModel model(&mparam, &ctx);

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::unique_ptr<RegTree>(new RegTree));
  model.CommitModelGroup(std::move(trees), 0);

  auto gpu_lparam = CreateEmptyGenericParam(0);
  std::unique_ptr<Predictor> gpu_predictor = std::unique_ptr<Predictor>(
      Predictor::Create("gpu_predictor", &gpu_lparam));
  gpu_predictor->Configure({});
  HostDeviceVector<float> predictions;
  auto dmat = RandomDataGenerator(3, 1, 0).GenerateDMatrix();
  gpu_predictor->PredictContribution(dmat.get(), &predictions, model);
  auto& phis = predictions.HostVector();
  auto base_score = mparam.BaseScore(Context::kCpuId)(0);
  EXPECT_EQ(phis[0], 0.0);
  EXPECT_EQ(phis[1], base_score);
  EXPECT_EQ(phis[2], 0.0);
  EXPECT_EQ(phis[3], base_score);
  EXPECT_EQ(phis[4], 0.0);
  EXPECT_EQ(phis[5], base_score);
}

TEST(GPUPredictor, Shap) {
  Context ctx;
  ctx.gpu_id = 0;
  LearnerModelParam mparam{MakeMP(1, .5, 1, ctx.gpu_id)};
  gbm::GBTreeModel model(&mparam, &ctx);

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::unique_ptr<RegTree>(new RegTree));
  trees[0]->ExpandNode(0, 0, 0.5, true, 1.0, -1.0, 1.0, 0.0, 5.0, 2.0, 3.0);
  model.CommitModelGroup(std::move(trees), 0);

  auto gpu_lparam = CreateEmptyGenericParam(0);
  auto cpu_lparam = CreateEmptyGenericParam(-1);
  std::unique_ptr<Predictor> gpu_predictor = std::unique_ptr<Predictor>(
      Predictor::Create("gpu_predictor", &gpu_lparam));
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

TEST(GPUPredictor, IterationRange) {
  TestIterationRange("gpu_predictor");
}


TEST(GPUPredictor, CategoricalPrediction) {
  TestCategoricalPrediction("gpu_predictor");
}

TEST(GPUPredictor, CategoricalPredictLeaf) {
  TestCategoricalPredictLeaf(StringView{"gpu_predictor"});
}

TEST(GPUPredictor, PredictLeafBasic) {
  size_t constexpr kRows = 5, kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).Device(0).GenerateDMatrix();
  auto lparam = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &lparam));
  gpu_predictor->Configure({});

  LearnerModelParam mparam{MakeMP(kCols, .0, 1)};
  Context ctx;
  gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

  HostDeviceVector<float> leaf_out_predictions;
  gpu_predictor->PredictLeaf(dmat.get(), &leaf_out_predictions, model);
  auto const& h_leaf_out_predictions = leaf_out_predictions.ConstHostVector();
  for (auto v : h_leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }
}

TEST(GPUPredictor, Sparse) {
  TestSparsePrediction(0.2, "gpu_predictor");
  TestSparsePrediction(0.8, "gpu_predictor");
}
}  // namespace predictor
}  // namespace xgboost
