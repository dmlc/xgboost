/*!
 * Copyright 2017-2023 XGBoost contributors
 */
#include <gtest/gtest.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/predictor.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../../../src/data/adapter.h"
#include "../../../src/gbm/gbtree.h"
#pragma GCC diagnostic pop

#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"
#include "../predictor/test_predictor.h"

namespace xgboost {

TEST(SyclPredictor, Basic) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
  TestBasic(dmat.get(), &ctx);
}

TEST(SyclPredictor, ExternalMemory) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  bst_idx_t constexpr kRows{64};
  bst_feature_t constexpr kCols{12};
  auto dmat =
      RandomDataGenerator{kRows, kCols, 0.5f}.Batches(3).GenerateSparsePageDMatrix("temp", true);
  TestBasic(dmat.get(), &ctx);
}

TEST(SyclPredictor, InplacePredict) {
  bst_idx_t constexpr kRows{128};
  bst_feature_t constexpr kCols{64};
  Context ctx;
  auto gen = RandomDataGenerator{kRows, kCols, 0.5}.Device(ctx.Device());
  {
    HostDeviceVector<float> data;
    gen.GenerateDense(&data);
    ASSERT_EQ(data.Size(), kRows * kCols);
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy{}};
    auto array_interface = GetArrayInterface(&data, kRows, kCols);
    std::string arr_str;
    Json::Dump(array_interface, &arr_str);
    x->SetArrayData(arr_str.data());
    TestInplacePrediction(&ctx, x, kRows, kCols);
  }
}

TEST(SyclPredictor, IterationRange) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestIterationRange(&ctx);
}

TEST(SyclPredictor, GHistIndexTraining) {
  size_t constexpr kRows{128}, kCols{16}, kBins{64};
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  auto p_hist = RandomDataGenerator{kRows, kCols, 0.0}.Bins(kBins).GenerateDMatrix(false);
  HostDeviceVector<float> storage(kRows * kCols);
  auto columnar = RandomDataGenerator{kRows, kCols, 0.0}.GenerateArrayInterface(&storage);
  auto adapter = data::ArrayAdapter(columnar.c_str());
  std::shared_ptr<DMatrix> p_full{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)};
  TestTrainingPrediction(&ctx, kRows, kBins, p_full, p_hist);
}

TEST(SyclPredictor, CategoricalPredictLeaf) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestCategoricalPredictLeaf(&ctx, false);
}

TEST(SyclPredictor, LesserFeatures) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestPredictionWithLesserFeatures(&ctx);
}

TEST(SyclPredictor, Sparse) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestSparsePrediction(&ctx, 0.2);
  TestSparsePrediction(&ctx, 0.8);
}

TEST(SyclPredictor, Multi) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestVectorLeafPrediction(&ctx);
}

}  // namespace xgboost
