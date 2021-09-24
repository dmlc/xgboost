/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <dmlc/filesystem.h>
#include <gtest/gtest.h>
#include <xgboost/predictor.h>

#include "../helpers.h"
#include "test_predictor.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../../../src/gbm/gbtree.h"
#include "../../../src/data/adapter.h"

namespace xgboost {
TEST(CpuPredictor, Basic) {
  auto lparam = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &lparam));

  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;

  LearnerModelParam param;
  param.num_feature = kCols;
  param.base_score = 0.0;
  param.num_output_group = 1;

  gbm::GBTreeModel model = CreateTestModel(&param);

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

  LearnerModelParam param;
  param.base_score = 0;
  param.num_feature = dmat->Info().num_col_;
  param.num_output_group = 1;

  gbm::GBTreeModel model = CreateTestModel(&param);

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
    std::shared_ptr<data::DenseAdapter> x{
      new data::DenseAdapter(data.HostPointer(), kRows, kCols)};
    TestInplacePrediction(x, "cpu_predictor", kRows, kCols, -1);
  }

  {
    HostDeviceVector<float> data;
    HostDeviceVector<bst_row_t> rptrs;
    HostDeviceVector<bst_feature_t> columns;
    gen.GenerateCSR(&data, &rptrs, &columns);
    std::shared_ptr<data::CSRAdapter> x{new data::CSRAdapter(
        rptrs.HostPointer(), columns.HostPointer(), data.HostPointer(), kRows,
        data.Size(), kCols)};
    TestInplacePrediction(x, "cpu_predictor", kRows, kCols, -1);
  }
}

void TestUpdatePredictionCache(bool use_subsampling) {
  size_t constexpr kRows = 64, kCols = 16, kClasses = 4;
  LearnerModelParam mparam;
  mparam.num_feature = kCols;
  mparam.num_output_group = kClasses;
  mparam.base_score = 0;

  GenericParameter gparam;
  gparam.Init(Args{});

  std::unique_ptr<gbm::GBTree> gbm;
  gbm.reset(static_cast<gbm::GBTree*>(GradientBooster::Create("gbtree", &gparam, &mparam)));
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
  gbm->DoBoost(dmat.get(), &gpair, &predtion_cache);

  PredictionCacheEntry out_predictions;
  // perform fair prediction on the same input data, should be equal to cached result
  gbm->PredictBatch(dmat.get(), &out_predictions, false, 0, 0);

  std::vector<float> &out_predictions_h = out_predictions.predictions.HostVector();
  std::vector<float> &predtion_cache_from_train = predtion_cache.predictions.HostVector();
  for (size_t i = 0; i < out_predictions_h.size(); ++i) {
    ASSERT_NEAR(out_predictions_h[i], predtion_cache_from_train[i], kRtEps);
  }
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
}  // namespace xgboost
