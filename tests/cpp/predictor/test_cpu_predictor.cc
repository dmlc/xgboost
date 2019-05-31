// Copyright by Contributors
#include <dmlc/filesystem.h>
#include <gtest/gtest.h>
#include <xgboost/predictor.h>
#include "../helpers.h"

namespace xgboost {
TEST(cpu_predictor, Test) {
  auto lparam = CreateEmptyGenericParam(0, 0);
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &lparam));

  gbm::GBTreeModel model = CreateTestModel();

  int n_row = 5;
  int n_col = 5;

  auto dmat = CreateDMatrix(n_row, n_col, 0);

  // Test predict batch
  HostDeviceVector<float> out_predictions;
  cpu_predictor->PredictBatch((*dmat).get(), &out_predictions, model, 0);
  std::vector<float>& out_predictions_h = out_predictions.HostVector();
  for (int i = 0; i < out_predictions.Size(); i++) {
    ASSERT_EQ(out_predictions_h[i], 1.5);
  }

  // Test predict instance
  auto &batch = *(*dmat)->GetRowBatches().begin();
  for (int i = 0; i < batch.Size(); i++) {
    std::vector<float> instance_out_predictions;
    cpu_predictor->PredictInstance(batch[i], &instance_out_predictions, model);
    ASSERT_EQ(instance_out_predictions[0], 1.5);
  }

  // Test predict leaf
  std::vector<float> leaf_out_predictions;
  cpu_predictor->PredictLeaf((*dmat).get(), &leaf_out_predictions, model);
  for (auto v : leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  // Test predict contribution
  std::vector<float> out_contribution;
  cpu_predictor->PredictContribution((*dmat).get(), &out_contribution, model);
  for (auto const& contri : out_contribution) {
    ASSERT_EQ(contri, 1.5);
  }
  // Test predict contribution (approximate method)
  cpu_predictor->PredictContribution((*dmat).get(), &out_contribution, model, true);
  for (auto const& contri : out_contribution) {
    ASSERT_EQ(contri, 1.5);
  }

  delete dmat;
}

TEST(cpu_predictor, ExternalMemoryTest) {
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(12, 64);
  auto lparam = CreateEmptyGenericParam(0, 0);
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &lparam));

  gbm::GBTreeModel model = CreateTestModel();

  // Test predict batch
  HostDeviceVector<float> out_predictions;
  cpu_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);
  std::vector<float> &out_predictions_h = out_predictions.HostVector();
  EXPECT_EQ(out_predictions.Size(), dmat->Info().num_row_);
  for (const auto& v : out_predictions_h) {
    ASSERT_EQ(v, 1.5);
  }

  // Test predict leaf
  std::vector<float> leaf_out_predictions;
  cpu_predictor->PredictLeaf(dmat.get(), &leaf_out_predictions, model);
  EXPECT_EQ(leaf_out_predictions.size(), dmat->Info().num_row_);
  for (const auto& v : leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  // Test predict contribution
  std::vector<float> out_contribution;
  cpu_predictor->PredictContribution(dmat.get(), &out_contribution, model);
  EXPECT_EQ(out_contribution.size(), dmat->Info().num_row_);
  for (const auto& v : out_contribution) {
    ASSERT_EQ(v, 1.5);
  }

  // Test predict contribution (approximate method)
  std::vector<float> out_contribution_approximate;
  cpu_predictor->PredictContribution(dmat.get(), &out_contribution_approximate, model, true);
  EXPECT_EQ(out_contribution_approximate.size(), dmat->Info().num_row_);
  for (const auto& v : out_contribution_approximate) {
    ASSERT_EQ(v, 1.5);
  }
}
}  // namespace xgboost
