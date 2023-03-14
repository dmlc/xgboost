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
TEST(CpuPredictor, Basic) {
  auto lparam = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &lparam));

  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;

  LearnerModelParam mparam{MakeMP(kCols, .0, 1)};

  Context ctx;
  ctx.UpdateAllowUnknown(Args{});
  gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  // Test predict batch
  PredictionCacheEntry out_predictions;
  cpu_predictor->InitOutPredictions(dmat->Info(), &out_predictions.predictions, model);
  cpu_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);

  std::vector<float>& out_predictions_h = out_predictions.predictions.HostVector();
  for (size_t i = 0; i < out_predictions.predictions.Size(); i++) {
    ASSERT_EQ(out_predictions_h[i], 1.5);
  }

  // Test predict instance
  auto const &batch = *dmat->GetBatches<xgboost::SparsePage>().begin();
  auto page = batch.GetView();
  for (size_t i = 0; i < batch.Size(); i++) {
    std::vector<float> instance_out_predictions;
    cpu_predictor->PredictInstance(page[i], &instance_out_predictions, model);
    ASSERT_EQ(instance_out_predictions[0], 1.5);
  }

  // Test predict leaf
  HostDeviceVector<float> leaf_out_predictions;
  cpu_predictor->PredictLeaf(dmat.get(), &leaf_out_predictions, model);
  auto const& h_leaf_out_predictions = leaf_out_predictions.ConstHostVector();
  for (auto v : h_leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  // Test predict contribution
  HostDeviceVector<float> out_contribution_hdv;
  auto& out_contribution = out_contribution_hdv.HostVector();
  cpu_predictor->PredictContribution(dmat.get(), &out_contribution_hdv, model);
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
  cpu_predictor->PredictContribution(dmat.get(), &out_contribution_hdv, model,
                                     0, nullptr, true);
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

namespace {
void TestColumnSplitPredictBatch() {
  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();

  auto lparam = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &lparam));

  LearnerModelParam mparam{MakeMP(kCols, .0, 1)};

  Context ctx;
  ctx.UpdateAllowUnknown(Args{});
  gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

  // Test predict batch
  PredictionCacheEntry out_predictions;
  cpu_predictor->InitOutPredictions(dmat->Info(), &out_predictions.predictions, model);
  auto sliced = std::unique_ptr<DMatrix>{dmat->SliceCol(world_size, rank)};
  cpu_predictor->PredictBatch(sliced.get(), &out_predictions, model, 0);

  std::vector<float>& out_predictions_h = out_predictions.predictions.HostVector();
  for (size_t i = 0; i < out_predictions.predictions.Size(); i++) {
    ASSERT_EQ(out_predictions_h[i], 1.5);
  }
}
}  // anonymous namespace

TEST(CpuPredictor, ColumnSplit) {
  auto constexpr kWorldSize = 2;
  RunWithInMemoryCommunicator(kWorldSize, TestColumnSplitPredictBatch);
}

TEST(CpuPredictor, IterationRange) {
  TestIterationRange("cpu_predictor");
}

TEST(CpuPredictor, ExternalMemory) {
  size_t constexpr kPageSize = 64, kEntriesPerCol = 3;
  size_t constexpr kEntries = kPageSize * kEntriesPerCol * 2;

  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(kEntries);
  auto lparam = CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &lparam));

  LearnerModelParam mparam{MakeMP(dmat->Info().num_col_, .0, 1)};

  Context ctx;
  ctx.UpdateAllowUnknown(Args{});
  gbm::GBTreeModel model = CreateTestModel(&mparam, &ctx);

  // Test predict batch
  PredictionCacheEntry out_predictions;
  cpu_predictor->InitOutPredictions(dmat->Info(), &out_predictions.predictions, model);
  cpu_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);
  std::vector<float> &out_predictions_h = out_predictions.predictions.HostVector();
  ASSERT_EQ(out_predictions.predictions.Size(), dmat->Info().num_row_);
  for (const auto& v : out_predictions_h) {
    ASSERT_EQ(v, 1.5);
  }

  // Test predict leaf
  HostDeviceVector<float> leaf_out_predictions;
  cpu_predictor->PredictLeaf(dmat.get(), &leaf_out_predictions, model);
  auto const& h_leaf_out_predictions = leaf_out_predictions.ConstHostVector();
  ASSERT_EQ(h_leaf_out_predictions.size(), dmat->Info().num_row_);
  for (const auto& v : h_leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  // Test predict contribution
  HostDeviceVector<float> out_contribution_hdv;
  auto& out_contribution = out_contribution_hdv.HostVector();
  cpu_predictor->PredictContribution(dmat.get(), &out_contribution_hdv, model);
  ASSERT_EQ(out_contribution.size(), dmat->Info().num_row_ * (dmat->Info().num_col_ + 1));
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is filled with LeafValue().
    if ((i + 1) % (dmat->Info().num_col_ + 1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }

  // Test predict contribution (approximate method)
  HostDeviceVector<float> out_contribution_approximate_hdv;
  auto& out_contribution_approximate = out_contribution_approximate_hdv.HostVector();
  cpu_predictor->PredictContribution(
      dmat.get(), &out_contribution_approximate_hdv, model, 0, nullptr, true);
  ASSERT_EQ(out_contribution_approximate.size(),
            dmat->Info().num_row_ * (dmat->Info().num_col_ + 1));
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is filled with LeafValue().
    if ((i + 1) % (dmat->Info().num_col_ + 1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }
}

TEST(CpuPredictor, InplacePredict) {
  bst_row_t constexpr kRows{128};
  bst_feature_t constexpr kCols{64};
  auto gen = RandomDataGenerator{kRows, kCols, 0.5}.Device(-1);
  {
    HostDeviceVector<float> data;
    gen.GenerateDense(&data);
    ASSERT_EQ(data.Size(), kRows * kCols);
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy{}};
    auto array_interface = GetArrayInterface(&data, kRows, kCols);
    std::string arr_str;
    Json::Dump(array_interface, &arr_str);
    x->SetArrayData(arr_str.data());
    TestInplacePrediction(x, "cpu_predictor", kRows, kCols, Context::kCpuId);
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
    TestInplacePrediction(x, "cpu_predictor", kRows, kCols, Context::kCpuId);
  }
}

void TestUpdatePredictionCache(bool use_subsampling) {
  size_t constexpr kRows = 64, kCols = 16, kClasses = 4;
  LearnerModelParam mparam{MakeMP(kCols, .0, kClasses)};
  Context ctx;

  std::unique_ptr<gbm::GBTree> gbm;
  gbm.reset(static_cast<gbm::GBTree*>(GradientBooster::Create("gbtree", &ctx, &mparam)));
  std::map<std::string, std::string> cfg;
  cfg["tree_method"] = "hist";
  cfg["predictor"]   = "cpu_predictor";
  if (use_subsampling) {
    cfg["subsample"] = "0.5";
  }
  Args args = {cfg.cbegin(), cfg.cend()};
  gbm->Configure(args);

  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix(true, true, kClasses);

  HostDeviceVector<GradientPair> gpair;
  auto& h_gpair = gpair.HostVector();
  h_gpair.resize(kRows*kClasses);
  for (size_t i = 0; i < kRows*kClasses; ++i) {
    h_gpair[i] = {static_cast<float>(i), 1};
  }

  PredictionCacheEntry predtion_cache;
  predtion_cache.predictions.Resize(kRows*kClasses, 0);
  // after one training iteration predtion_cache is filled with cached in QuantileHistMaker::Builder prediction values
  gbm->DoBoost(dmat.get(), &gpair, &predtion_cache, nullptr);

  PredictionCacheEntry out_predictions;
  // perform fair prediction on the same input data, should be equal to cached result
  gbm->PredictBatch(dmat.get(), &out_predictions, false, 0, 0);

  std::vector<float> &out_predictions_h = out_predictions.predictions.HostVector();
  std::vector<float> &predtion_cache_from_train = predtion_cache.predictions.HostVector();
  for (size_t i = 0; i < out_predictions_h.size(); ++i) {
    ASSERT_NEAR(out_predictions_h[i], predtion_cache_from_train[i], kRtEps);
  }
}

TEST(CPUPredictor, GHistIndex) {
  size_t constexpr kRows{128}, kCols{16}, kBins{64};
  auto p_hist = RandomDataGenerator{kRows, kCols, 0.0}.Bins(kBins).GenerateQuantileDMatrix();
  HostDeviceVector<float> storage(kRows * kCols);
  auto columnar = RandomDataGenerator{kRows, kCols, 0.0}.GenerateArrayInterface(&storage);
  auto adapter = data::ArrayAdapter(columnar.c_str());
  std::shared_ptr<DMatrix> p_full{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)};
  TestTrainingPrediction(kRows, kBins, "hist", p_full, p_hist);
}

TEST(CPUPredictor, CategoricalPrediction) {
  TestCategoricalPrediction("cpu_predictor");
}

TEST(CPUPredictor, CategoricalPredictLeaf) {
  TestCategoricalPredictLeaf(StringView{"cpu_predictor"});
}

TEST(CpuPredictor, UpdatePredictionCache) {
  TestUpdatePredictionCache(false);
  TestUpdatePredictionCache(true);
}

TEST(CpuPredictor, LesserFeatures) {
  TestPredictionWithLesserFeatures("cpu_predictor");
}

TEST(CpuPredictor, Sparse) {
  TestSparsePrediction(0.2, "cpu_predictor");
  TestSparsePrediction(0.8, "cpu_predictor");
}

TEST(CpuPredictor, Multi) {
  Context ctx;
  ctx.nthread = 1;
  TestVectorLeafPrediction(&ctx);
}
}  // namespace xgboost
