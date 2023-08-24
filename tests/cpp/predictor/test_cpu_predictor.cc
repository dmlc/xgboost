/**
 * Copyright 2017-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/predictor.h>

#include <cstdint>
#include <thread>

#include "../../../src/collective/communicator-inl.h"
#include "../../../src/data/adapter.h"
#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/gbm/gbtree.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"
#include "test_predictor.h"

namespace xgboost {

namespace {
void TestBasic(DMatrix* dmat) {
  Context ctx;
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &ctx));

  size_t const kRows = dmat->Info().num_row_;
  size_t const kCols = dmat->Info().num_col_;

  LearnerModelParam mparam{MakeMP(kCols, .0, 1)};

  ctx.UpdateAllowUnknown(Args{});
  gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

  // Test predict batch
  PredictionCacheEntry out_predictions;
  cpu_predictor->InitOutPredictions(dmat->Info(), &out_predictions.predictions, model);
  cpu_predictor->PredictBatch(dmat, &out_predictions, model, 0);

  std::vector<float>& out_predictions_h = out_predictions.predictions.HostVector();
  for (size_t i = 0; i < out_predictions.predictions.Size(); i++) {
    ASSERT_EQ(out_predictions_h[i], 1.5);
  }

  // Test predict instance
  auto const& batch = *dmat->GetBatches<xgboost::SparsePage>().begin();
  auto page = batch.GetView();
  for (size_t i = 0; i < batch.Size(); i++) {
    std::vector<float> instance_out_predictions;
    cpu_predictor->PredictInstance(page[i], &instance_out_predictions, model, 0,
                                   dmat->Info().IsColumnSplit());
    ASSERT_EQ(instance_out_predictions[0], 1.5);
  }

  // Test predict leaf
  HostDeviceVector<float> leaf_out_predictions;
  cpu_predictor->PredictLeaf(dmat, &leaf_out_predictions, model);
  auto const& h_leaf_out_predictions = leaf_out_predictions.ConstHostVector();
  for (auto v : h_leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  if (dmat->Info().IsColumnSplit()) {
    // Predict contribution is not supported for column split.
    return;
  }

  // Test predict contribution
  HostDeviceVector<float> out_contribution_hdv;
  auto& out_contribution = out_contribution_hdv.HostVector();
  cpu_predictor->PredictContribution(dmat, &out_contribution_hdv, model);
  ASSERT_EQ(out_contribution.size(), kRows * (kCols + 1));
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is
    // filled with LeafValue().
    if ((i + 1) % (kCols + 1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }
  // Test predict contribution (approximate method)
  cpu_predictor->PredictContribution(dmat, &out_contribution_hdv, model, 0, nullptr, true);
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is
    // filled with LeafValue().
    if ((i + 1) % (kCols + 1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }
}
}  // anonymous namespace

TEST(CpuPredictor, Basic) {
  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
  TestBasic(dmat.get());
}

namespace {
void TestColumnSplit() {
  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  dmat = std::unique_ptr<DMatrix>{dmat->SliceCol(world_size, rank)};

  TestBasic(dmat.get());
}
}  // anonymous namespace

TEST(CpuPredictor, BasicColumnSplit) {
  auto constexpr kWorldSize = 2;
  RunWithInMemoryCommunicator(kWorldSize, TestColumnSplit);
}

TEST(CpuPredictor, IterationRange) {
  Context ctx;
  TestIterationRange(&ctx);
}

TEST(CpuPredictor, IterationRangeColmnSplit) {
  Context ctx;
  TestIterationRangeColumnSplit(&ctx);
}

TEST(CpuPredictor, ExternalMemory) {
  size_t constexpr kPageSize = 64, kEntriesPerCol = 3;
  size_t constexpr kEntries = kPageSize * kEntriesPerCol * 2;
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(kEntries);
  TestBasic(dmat.get());
}

TEST(CpuPredictor, InplacePredict) {
  bst_row_t constexpr kRows{128};
  bst_feature_t constexpr kCols{64};
  Context ctx;
  auto gen = RandomDataGenerator{kRows, kCols, 0.5}.Device(ctx.gpu_id);
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
    HostDeviceVector<bst_row_t> rptrs;
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

  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix(true, true, kClasses);

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
  Context ctx;
  TestCategoricalPrediction(&ctx, false);
}

TEST(CPUPredictor, CategoricalPredictionColumnSplit) {
  Context ctx;
  TestCategoricalPredictionColumnSplit(&ctx);
}

TEST(CPUPredictor, CategoricalPredictLeaf) {
  Context ctx;
  TestCategoricalPredictLeaf(&ctx, false);
}

TEST(CPUPredictor, CategoricalPredictLeafColumnSplit) {
  Context ctx;
  TestCategoricalPredictLeafColumnSplit(&ctx);
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
  Context ctx;
  TestPredictionWithLesserFeaturesColumnSplit(&ctx);
}

TEST(CpuPredictor, Sparse) {
  Context ctx;
  TestSparsePrediction(&ctx, 0.2);
  TestSparsePrediction(&ctx, 0.8);
}

TEST(CpuPredictor, SparseColumnSplit) {
  Context ctx;
  TestSparsePredictionColumnSplit(&ctx, 0.2);
  TestSparsePredictionColumnSplit(&ctx, 0.8);
}

TEST(CpuPredictor, Multi) {
  Context ctx;
  ctx.nthread = 1;
  TestVectorLeafPrediction(&ctx);
}

TEST(CpuPredictor, Access) { TestPredictionDeviceAccess(); }
}  // namespace xgboost
