
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <xgboost/c_api.h>
#include <xgboost/predictor.h>
#include "../helpers.h"
#include "gtest/gtest.h"

namespace xgboost {
namespace predictor {

gbm::GBTreeModel CreateModel(int num_trees, int num_group,
                             std::vector<float> leaf_weights,
                             float base_margin) {
  gbm::GBTreeModel model(base_margin);
  model.param.num_output_group = num_group;
  for (int i = 0; i < num_trees; i++) {
    std::vector<std::unique_ptr<RegTree>> trees;
    trees.push_back(std::unique_ptr<RegTree>(new RegTree()));
    trees.back()->InitModel();
    (*trees.back())[0].set_leaf(leaf_weights[i % num_group]);
    (*trees.back()).stat(0).sum_hess = 1.0f;
    model.CommitModel(std::move(trees), i % num_group);
  }

  return model;
}

TEST(gpu_predictor, test_partial_prediction) {
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor"));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor"));
  gpu_predictor->Init({}, {});
  cpu_predictor->Init({}, {});

  auto model = CreateModel(5, 1, { 1.5 }, 0.5);
  auto dmat = CreateDMatrix(100, 20, 0);

  std::vector<float> gpu_out_predictions;
  std::vector<float> cpu_out_predictions;
  gpu_predictor->PredictBatch(dmat.get(), &gpu_out_predictions, model, 4, 5,
                              false);
  cpu_predictor->PredictBatch(dmat.get(), &cpu_out_predictions, model, 4, 5,
                              false);
  for (int i = 0; i < gpu_out_predictions.size(); i++) {
    ASSERT_EQ(gpu_out_predictions[i], 1.5f);
    ASSERT_EQ(cpu_out_predictions[i], 1.5f);
  }
}

TEST(gpu_predictor, test_partial_prediction_multi) {
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor"));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor"));
  gpu_predictor->Init({}, {});
  cpu_predictor->Init({}, {});

  auto model = CreateModel(9, 3, { 0.5f,1.0f,1.5f }, 0.5);
  auto dmat = CreateDMatrix(100, 20, 0);

  std::vector<float> gpu_out_predictions;
  std::vector<float> cpu_out_predictions;
  gpu_predictor->PredictBatch(dmat.get(), &gpu_out_predictions, model, 3, 6,
                              false);
  cpu_predictor->PredictBatch(dmat.get(), &cpu_out_predictions, model, 3, 6,
                              false);
  for (int i = 0; i < gpu_out_predictions.size(); i++) {
    if (i % 3 == 0) {
      ASSERT_EQ(gpu_out_predictions[i], 0.5f);
      ASSERT_EQ(cpu_out_predictions[i], 0.5f);
    }
    else if (i % 3 == 1) {
      ASSERT_EQ(gpu_out_predictions[i], 1.0f);
      ASSERT_EQ(cpu_out_predictions[i], 1.0f);
    }
    else if (i % 3 == 2) {
      ASSERT_EQ(gpu_out_predictions[i], 1.5f);
      ASSERT_EQ(cpu_out_predictions[i], 1.5f);
    }
  }
}

TEST(gpu_predictor, Test) {
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor"));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor"));
  gpu_predictor->Init({}, {});
  cpu_predictor->Init({}, {});

  auto model = CreateModel(1, 1, { 1.5f }, 0.5);
  int n_row = 5;
  int n_col = 5;

  auto dmat = CreateDMatrix(n_row, n_col, 0);

  // Test predict batch
  HostDeviceVector<float> gpu_out_predictions;
  HostDeviceVector<float> cpu_out_predictions;
  gpu_predictor->PredictBatch(dmat.get(), &gpu_out_predictions, model, 0);
  cpu_predictor->PredictBatch(dmat.get(), &cpu_out_predictions, model, 0);
  std::vector<float>& gpu_out_predictions_h = gpu_out_predictions.HostVector();
  std::vector<float>& cpu_out_predictions_h = cpu_out_predictions.HostVector();
  float abs_tolerance = 0.001;
  for (int i = 0; i < gpu_out_predictions.Size(); i++) {
    ASSERT_LT(std::abs(gpu_out_predictions_h[i] - cpu_out_predictions_h[i]),
              abs_tolerance);
  }
  // Test predict instance
  auto batch = dmat->RowIterator()->Value();
  for (int i = 0; i < batch.size; i++) {
    std::vector<float> gpu_instance_out_predictions;
    std::vector<float> cpu_instance_out_predictions;
    cpu_predictor->PredictInstance(batch[i], &cpu_instance_out_predictions,
                                   model);
    gpu_predictor->PredictInstance(batch[i], &gpu_instance_out_predictions,
                                   model);
    ASSERT_EQ(gpu_instance_out_predictions[0], cpu_instance_out_predictions[0]);
  }

  // Test predict leaf
  std::vector<float> gpu_leaf_out_predictions;
  std::vector<float> cpu_leaf_out_predictions;
  cpu_predictor->PredictLeaf(dmat.get(), &cpu_leaf_out_predictions, model);
  gpu_predictor->PredictLeaf(dmat.get(), &gpu_leaf_out_predictions, model);
  for (int i = 0; i < gpu_leaf_out_predictions.size(); i++) {
    ASSERT_EQ(gpu_leaf_out_predictions[i], cpu_leaf_out_predictions[i]);
  }

  // Test predict contribution
  std::vector<float> gpu_out_contribution;
  std::vector<float> cpu_out_contribution;
  cpu_predictor->PredictContribution(dmat.get(), &cpu_out_contribution, model);
  gpu_predictor->PredictContribution(dmat.get(), &gpu_out_contribution, model);
  for (int i = 0; i < gpu_out_contribution.size(); i++) {
    ASSERT_EQ(gpu_out_contribution[i], cpu_out_contribution[i]);
  }
}
}  // namespace predictor
}  // namespace xgboost
