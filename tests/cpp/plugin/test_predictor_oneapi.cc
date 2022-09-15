/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/predictor.h>

#include "../../../src/data/adapter.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"
#include "../predictor/test_predictor.h"

namespace xgboost {
TEST(Plugin, OneAPIPredictorBasic) {
  auto lparam = CreateEmptyGenericParam(0);
  std::unique_ptr<Predictor> oneapi_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("oneapi_predictor", &lparam));

  int kRows = 5;
  int kCols = 5;

  LearnerModelParam param;
  param.num_feature = kCols;
  param.base_score = 0.0;
  param.num_output_group = 1;

  gbm::GBTreeModel model = CreateTestModel(&param);

  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  // Test predict batch
  PredictionCacheEntry out_predictions;
  oneapi_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);
  ASSERT_EQ(model.trees.size(), out_predictions.version);
  std::vector<float>& out_predictions_h = out_predictions.predictions.HostVector();
  for (size_t i = 0; i < out_predictions.predictions.Size(); i++) {
    ASSERT_EQ(out_predictions_h[i], 1.5);
  }

  // Test predict instance
  auto const &batch = *dmat->GetBatches<xgboost::SparsePage>().begin();
  for (size_t i = 0; i < batch.Size(); i++) {
    std::vector<float> instance_out_predictions;
    oneapi_predictor->PredictInstance(batch[i], &instance_out_predictions, model);
    ASSERT_EQ(instance_out_predictions[0], 1.5);
  }

  // Test predict leaf
  std::vector<float> leaf_out_predictions;
  oneapi_predictor->PredictLeaf(dmat.get(), &leaf_out_predictions, model);
  for (auto v : leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  // Test predict contribution
  std::vector<float> out_contribution;
  oneapi_predictor->PredictContribution(dmat.get(), &out_contribution, model);
  ASSERT_EQ(out_contribution.size(), kRows * (kCols + 1));
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is filled with LeafValue().
    if ((i+1) % (kCols+1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }
  // Test predict contribution (approximate method)
  oneapi_predictor->PredictContribution(dmat.get(), &out_contribution, model, 0, nullptr, true);
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is filled with LeafValue().
    if ((i+1) % (kCols+1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }
}

TEST(Plugin, OneAPIPredictorExternalMemory) {
  dmlc::TemporaryDirectory tmpdir;
  std::string filename = tmpdir.path + "/big.libsvm";
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(12, 64, filename);
  auto lparam = CreateEmptyGenericParam(0);

  std::unique_ptr<Predictor> oneapi_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("oneapi_predictor", &lparam));

  LearnerModelParam param;
  param.base_score = 0;
  param.num_feature = dmat->Info().num_col_;
  param.num_output_group = 1;

  gbm::GBTreeModel model = CreateTestModel(&param);

  // Test predict batch
  PredictionCacheEntry out_predictions;
  oneapi_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);
  std::vector<float> &out_predictions_h = out_predictions.predictions.HostVector();
  ASSERT_EQ(out_predictions.predictions.Size(), dmat->Info().num_row_);
  for (const auto& v : out_predictions_h) {
    ASSERT_EQ(v, 1.5);
  }

  // Test predict leaf
  std::vector<float> leaf_out_predictions;
  oneapi_predictor->PredictLeaf(dmat.get(), &leaf_out_predictions, model);
  ASSERT_EQ(leaf_out_predictions.size(), dmat->Info().num_row_);
  for (const auto& v : leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  // Test predict contribution
  std::vector<float> out_contribution;
  oneapi_predictor->PredictContribution(dmat.get(), &out_contribution, model);
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
  std::vector<float> out_contribution_approximate;
  oneapi_predictor->PredictContribution(dmat.get(), &out_contribution_approximate, model, 0, nullptr, true);
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

TEST(Plugin, OneAPIPredictorInplacePredict) {
  bst_row_t constexpr kRows{128};
  bst_feature_t constexpr kCols{64};
  auto gen = RandomDataGenerator{kRows, kCols, 0.5}.Device(-1);
  {
    HostDeviceVector<float> data;
    gen.GenerateDense(&data);
    ASSERT_EQ(data.Size(), kRows * kCols);
    std::shared_ptr<data::DenseAdapter> x{
      new data::DenseAdapter(data.HostPointer(), kRows, kCols)};
    TestInplacePrediction(x, "oneapi_predictor", kRows, kCols, -1);
  }

  {
    HostDeviceVector<float> data;
    HostDeviceVector<bst_row_t> rptrs;
    HostDeviceVector<bst_feature_t> columns;
    gen.GenerateCSR(&data, &rptrs, &columns);
    std::shared_ptr<data::CSRAdapter> x{new data::CSRAdapter(
        rptrs.HostPointer(), columns.HostPointer(), data.HostPointer(), kRows,
        data.Size(), kCols)};
    TestInplacePrediction(x, "oneapi_predictor", kRows, kCols, -1);
  }
}
}  // namespace xgboost
