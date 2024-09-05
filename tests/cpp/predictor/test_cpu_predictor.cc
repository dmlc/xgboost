/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/predictor.h>

#include "../../../src/collective/communicator-inl.h"
#include "../../../src/data/adapter.h"
#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/gbm/gbtree.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "../helpers.h"
#include "test_predictor.h"

namespace xgboost {
TEST(CpuPredictor, Basic) {
  Context ctx;
  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
  TestBasic(dmat.get(), &ctx);
}

namespace {
void TestColumnSplit() {
  Context ctx;
  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  dmat = std::unique_ptr<DMatrix>{dmat->SliceCol(world_size, rank)};

  TestBasic(dmat.get(), &ctx);
}
}  // anonymous namespace

TEST(CpuPredictor, BasicColumnSplit) {
  auto constexpr kWorldSize = 2;
  collective::TestDistributedGlobal(kWorldSize, TestColumnSplit);
}

TEST(CpuPredictor, IterationRange) {
  Context ctx;
  TestIterationRange(&ctx);
}

TEST(CpuPredictor, IterationRangeColmnSplit) {
  auto constexpr kWorldSize = 2;
  TestIterationRangeColumnSplit(kWorldSize, false);
}

TEST(CpuPredictor, ExternalMemory) {
  Context ctx;
  bst_idx_t constexpr kRows{64};
  bst_feature_t constexpr kCols{12};
  auto dmat =
      RandomDataGenerator{kRows, kCols, 0.5f}.Batches(3).GenerateSparsePageDMatrix("temp", true);
  TestBasic(dmat.get(), &ctx);
}

TEST_P(ShapExternalMemoryTest, CPUPredictor) {
  Context ctx;
  auto [is_qdm, is_interaction] = this->GetParam();
  this->Run(&ctx, is_qdm, is_interaction);
}

TEST(CpuPredictor, InplacePredict) {
  bst_idx_t constexpr kRows{128};
  bst_feature_t constexpr kCols{64};
  Context ctx;
  auto gen = RandomDataGenerator{kRows, kCols, 0.5}.Device(ctx.Device());
  {
    HostDeviceVector<float> data;
    gen.GenerateDense(&data);
    ASSERT_EQ(data.Size(), kRows * kCols);
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy{}};
    auto array_interface = GetArrayInterface(&data, kRows, kCols);
    std::string arr_str;
    Json::Dump(array_interface, &arr_str);
    x->SetArrayData(arr_str.data());
    TestInplacePrediction(&ctx, x, kRows, kCols);
  }

  {
    HostDeviceVector<float> data;
    HostDeviceVector<std::size_t> rptrs;
    HostDeviceVector<bst_feature_t> columns;
    gen.GenerateCSR(&data, &rptrs, &columns);
    auto data_interface = GetArrayInterface(&data, kRows * kCols, 1);
    auto rptr_interface = GetArrayInterface(&rptrs, kRows + 1, 1);
    auto col_interface = GetArrayInterface(&columns, kRows * kCols, 1);
    std::string data_str, rptr_str, col_str;
    Json::Dump(data_interface, &data_str);
    Json::Dump(rptr_interface, &rptr_str);
    Json::Dump(col_interface, &col_str);
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy};
    x->SetCSRData(rptr_str.data(), col_str.data(), data_str.data(), kCols, true);
    TestInplacePrediction(&ctx, x, kRows, kCols);
  }
}

namespace {
void TestUpdatePredictionCache(bool use_subsampling) {
  std::size_t constexpr kRows = 64, kCols = 16, kClasses = 4;
  LearnerModelParam mparam{MakeMP(kCols, .0, kClasses)};
  Context ctx;

  std::unique_ptr<gbm::GBTree> gbm;
  gbm.reset(static_cast<gbm::GBTree*>(GradientBooster::Create("gbtree", &ctx, &mparam)));
  Args args{{"tree_method", "hist"}};
  if (use_subsampling) {
    args.emplace_back("subsample", "0.5");
  }
  gbm->Configure(args);

  auto dmat = RandomDataGenerator(kRows, kCols, 0).Classes(kClasses).GenerateDMatrix(true);

  linalg::Matrix<GradientPair> gpair({kRows, kClasses}, ctx.Device());
  auto h_gpair = gpair.HostView();
  for (size_t i = 0; i < kRows * kClasses; ++i) {
    std::apply(h_gpair, linalg::UnravelIndex(i, kRows, kClasses)) = {static_cast<float>(i), 1};
  }

  PredictionCacheEntry predtion_cache;
  predtion_cache.predictions.Resize(kRows * kClasses, 0);
  // after one training iteration predtion_cache is filled with cached in QuantileHistMaker
  // prediction values
  gbm->DoBoost(dmat.get(), &gpair, &predtion_cache, nullptr);

  PredictionCacheEntry out_predictions;
  // perform prediction from scratch on the same input data, should be equal to cached result
  gbm->PredictBatch(dmat.get(), &out_predictions, false, 0, 0);

  std::vector<float>& out_predictions_h = out_predictions.predictions.HostVector();
  std::vector<float>& predtion_cache_from_train = predtion_cache.predictions.HostVector();
  for (size_t i = 0; i < out_predictions_h.size(); ++i) {
    ASSERT_NEAR(out_predictions_h[i], predtion_cache_from_train[i], kRtEps);
  }
}
}  // namespace

TEST(CPUPredictor, GHistIndexTraining) {
  size_t constexpr kRows{128}, kCols{16}, kBins{64};
  Context ctx;
  auto p_hist = RandomDataGenerator{kRows, kCols, 0.0}.Bins(kBins).GenerateQuantileDMatrix(false);
  HostDeviceVector<float> storage(kRows * kCols);
  auto columnar = RandomDataGenerator{kRows, kCols, 0.0}.GenerateArrayInterface(&storage);
  auto adapter = data::ArrayAdapter(columnar.c_str());
  std::shared_ptr<DMatrix> p_full{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)};
  TestTrainingPrediction(&ctx, kRows, kBins, p_full, p_hist);
}

TEST(CPUPredictor, CategoricalPrediction) {
  TestCategoricalPrediction(false, false);
}

TEST(CPUPredictor, CategoricalPredictionColumnSplit) {
  auto constexpr kWorldSize = 2;
  collective::TestDistributedGlobal(kWorldSize, [] { TestCategoricalPrediction(false, true); });
}

TEST(CPUPredictor, CategoricalPredictLeaf) {
  Context ctx;
  TestCategoricalPredictLeaf(&ctx, false);
}

TEST(CPUPredictor, CategoricalPredictLeafColumnSplit) {
  auto constexpr kWorldSize = 2;
  Context ctx;
  collective::TestDistributedGlobal(kWorldSize, [&] { TestCategoricalPredictLeaf(&ctx, true); });
}

TEST(CpuPredictor, UpdatePredictionCache) {
  TestUpdatePredictionCache(false);
  TestUpdatePredictionCache(true);
}

TEST(CpuPredictor, LesserFeatures) {
  Context ctx;
  TestPredictionWithLesserFeatures(&ctx);
}

TEST(CpuPredictor, LesserFeaturesColumnSplit) {
  auto constexpr kWorldSize = 2;
  collective::TestDistributedGlobal(kWorldSize,
                                    [] { TestPredictionWithLesserFeaturesColumnSplit(false); });
}

TEST(CpuPredictor, Sparse) {
  Context ctx;
  TestSparsePrediction(&ctx, 0.2);
  TestSparsePrediction(&ctx, 0.8);
}

TEST(CpuPredictor, SparseColumnSplit) {
  auto constexpr kWorldSize = 2;
  TestSparsePredictionColumnSplit(kWorldSize, false, 0.2);
  TestSparsePredictionColumnSplit(kWorldSize, false, 0.8);
}

TEST(CpuPredictor, Multi) {
  Context ctx;
  ctx.nthread = 1;
  TestVectorLeafPrediction(&ctx);
}

TEST(CpuPredictor, Access) { TestPredictionDeviceAccess(); }
}  // namespace xgboost
