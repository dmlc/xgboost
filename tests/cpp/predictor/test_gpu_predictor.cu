/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <xgboost/c_api.h>
#include <xgboost/predictor.h>
#include <xgboost/logging.h>
#include <xgboost/learner.h>
#include <string>

#include "../helpers.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../../../src/data/device_adapter.cuh"
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

    LearnerModelParam param;
    param.num_feature = n_col;
    param.num_output_group = 1;
    param.base_score = 0.5;

    gbm::GBTreeModel model = CreateTestModel(&param);

    // Test predict batch
    PredictionCacheEntry gpu_out_predictions;
    PredictionCacheEntry cpu_out_predictions;

    gpu_predictor->PredictBatch(dmat.get(), &gpu_out_predictions, model, 0);
    ASSERT_EQ(model.trees.size(), gpu_out_predictions.version);
    cpu_predictor->PredictBatch(dmat.get(), &cpu_out_predictions, model, 0);

    std::vector<float>& gpu_out_predictions_h = gpu_out_predictions.predictions.HostVector();
    std::vector<float>& cpu_out_predictions_h = cpu_out_predictions.predictions.HostVector();
    float abs_tolerance = 0.001;
    for (int j = 0; j < gpu_out_predictions.predictions.Size(); j++) {
      ASSERT_NEAR(gpu_out_predictions_h[j], cpu_out_predictions_h[j], abs_tolerance);
    }
  }
}

TEST(GPUPredictor, EllpackBasic) {
  size_t constexpr kCols {8};
  for (size_t bins = 2; bins < 258; bins += 16) {
    size_t rows = bins * 16;
    auto p_m = RandomDataGenerator{rows, kCols, 0.0}
         .Bins(bins)
         .Device(0)
         .GenerateDeviceDMatrix(true);
    TestPredictionFromGradientIndex<EllpackPage>("gpu_predictor", rows, kCols, p_m);
    TestPredictionFromGradientIndex<EllpackPage>("gpu_predictor", bins, kCols, p_m);
  }
}

TEST(GPUPredictor, EllpackTraining) {
  size_t constexpr kRows { 128 }, kCols { 16 }, kBins { 64 };
  auto p_ellpack = RandomDataGenerator{kRows, kCols, 0.0}
       .Bins(kBins)
       .Device(0)
       .GenerateDeviceDMatrix(true);
  std::vector<HostDeviceVector<float>> storage(kCols);
  auto columnar = RandomDataGenerator{kRows, kCols, 0.0}
       .Device(0)
       .GenerateColumnarArrayInterface(&storage);
  auto adapter = data::CudfAdapter(columnar);
  std::shared_ptr<DMatrix> p_full {
    DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)
  };
  TestTrainingPrediction(kRows, "gpu_hist", p_full, p_ellpack);
}

TEST(GPUPredictor, ExternalMemoryTest) {
  auto lparam = CreateEmptyGenericParam(0);
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &lparam));
  gpu_predictor->Configure({});

  LearnerModelParam param;
  param.num_feature = 2;
  const int n_classes = 3;
  param.num_output_group = n_classes;
  param.base_score = 0.5;

  gbm::GBTreeModel model = CreateTestModel(&param, n_classes);
  std::vector<std::unique_ptr<DMatrix>> dmats;
  dmlc::TemporaryDirectory tmpdir;
  std::string file0 = tmpdir.path + "/big_0.libsvm";
  std::string file1 = tmpdir.path + "/big_1.libsvm";
  std::string file2 = tmpdir.path + "/big_2.libsvm";
  dmats.push_back(CreateSparsePageDMatrix(9, 64UL, file0));
  dmats.push_back(CreateSparsePageDMatrix(128, 128UL, file1));
  dmats.push_back(CreateSparsePageDMatrix(1024, 1024UL, file2));

  for (const auto& dmat: dmats) {
    dmat->Info().base_margin_.Resize(dmat->Info().num_row_ * n_classes, 0.5);
    PredictionCacheEntry out_predictions;
    gpu_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);
    EXPECT_EQ(out_predictions.predictions.Size(), dmat->Info().num_row_ * n_classes);
    const std::vector<float> &host_vector = out_predictions.predictions.ConstHostVector();
    for (int i = 0; i < host_vector.size() / n_classes; i++) {
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
  auto x = std::make_shared<data::CupyAdapter>(interface_str);
  TestInplacePrediction(x, "gpu_predictor", kRows, kCols, 0);
}

TEST(GPUPredictor, InplacePredictCuDF) {
  size_t constexpr kRows{128}, kCols{64};
  RandomDataGenerator gen(kRows, kCols, 0.5);
  gen.Device(0);
  std::vector<HostDeviceVector<float>> storage(kCols);
  auto interface_str = gen.GenerateColumnarArrayInterface(&storage);
  auto x = std::make_shared<data::CudfAdapter>(interface_str);
  TestInplacePrediction(x, "gpu_predictor", kRows, kCols, 0);
}

TEST(GPUPredictor, MGPU_InplacePredict) {  // NOLINT
  int32_t n_gpus = xgboost::common::AllVisibleGPUs();
  if (n_gpus <= 1) {
    LOG(WARNING) << "GPUPredictor.MGPU_InplacePredict is skipped.";
    return;
  }
  size_t constexpr kRows{128}, kCols{64};
  RandomDataGenerator gen(kRows, kCols, 0.5);
  gen.Device(1);
  HostDeviceVector<float> data;
  std::string interface_str = gen.GenerateArrayInterface(&data);
  auto x = std::make_shared<data::CupyAdapter>(interface_str);
  TestInplacePrediction(x, "gpu_predictor", kRows, kCols, 1);
  EXPECT_THROW(TestInplacePrediction(x, "gpu_predictor", kRows, kCols, 0),
               dmlc::Error);
}

}  // namespace predictor
}  // namespace xgboost
