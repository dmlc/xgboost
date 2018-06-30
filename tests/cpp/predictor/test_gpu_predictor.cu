
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <xgboost/c_api.h>
#include <xgboost/predictor.h>
#include "gtest/gtest.h"
#include "../helpers.h"

namespace xgboost {
namespace predictor {
TEST(gpu_predictor, Test) {
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor"));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor"));

  gpu_predictor->Init({}, {});
  cpu_predictor->Init({}, {});

  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::unique_ptr<RegTree>(new RegTree()));
  trees.back()->InitModel();
  (*trees.back())[0].SetLeaf(1.5f);
  (*trees.back()).Stat(0).sum_hess = 1.0f;
  gbm::GBTreeModel model(0.5);
  model.CommitModel(std::move(trees), 0);
  model.param.num_output_group = 1;

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
  for (int i = 0; i < batch.Size(); i++) {
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
