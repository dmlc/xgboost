
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <dmlc/logging.h>
#include <dmlc/filesystem.h>
#include <xgboost/c_api.h>
#include <xgboost/predictor.h>
#include <string>
#include "gtest/gtest.h"
#include "../helpers.h"

namespace {

inline void CheckCAPICall(int ret) {
  ASSERT_EQ(ret, 0) << XGBGetLastError();
}

}  // namespace anonymous

extern const std::map<std::string, std::string>&
QueryBoosterConfigurationArguments(BoosterHandle handle);

namespace xgboost {
namespace predictor {

TEST(gpu_predictor, Test) {
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor"));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor"));

  gpu_predictor->Init({}, {});
  cpu_predictor->Init({}, {});

  gbm::GBTreeModel model = CreateTestModel();

  int n_row = 5;
  int n_col = 5;

  auto dmat = CreateDMatrix(n_row, n_col, 0);

  // Test predict batch
  HostDeviceVector<float> gpu_out_predictions;
  HostDeviceVector<float> cpu_out_predictions;
  gpu_predictor->PredictBatch((*dmat).get(), &gpu_out_predictions, model, 0);
  cpu_predictor->PredictBatch((*dmat).get(), &cpu_out_predictions, model, 0);
  std::vector<float>& gpu_out_predictions_h = gpu_out_predictions.HostVector();
  std::vector<float>& cpu_out_predictions_h = cpu_out_predictions.HostVector();
  float abs_tolerance = 0.001;
  for (int i = 0; i < gpu_out_predictions.Size(); i++) {
    ASSERT_NEAR(gpu_out_predictions_h[i], cpu_out_predictions_h[i], abs_tolerance);
  }
  // Test predict instance
  const auto &batch = *(*dmat)->GetRowBatches().begin();
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
  cpu_predictor->PredictLeaf((*dmat).get(), &cpu_leaf_out_predictions, model);
  gpu_predictor->PredictLeaf((*dmat).get(), &gpu_leaf_out_predictions, model);
  for (int i = 0; i < gpu_leaf_out_predictions.size(); i++) {
    ASSERT_EQ(gpu_leaf_out_predictions[i], cpu_leaf_out_predictions[i]);
  }

  // Test predict contribution
  std::vector<float> gpu_out_contribution;
  std::vector<float> cpu_out_contribution;
  cpu_predictor->PredictContribution((*dmat).get(), &cpu_out_contribution, model);
  gpu_predictor->PredictContribution((*dmat).get(), &gpu_out_contribution, model);
  for (int i = 0; i < gpu_out_contribution.size(); i++) {
    ASSERT_EQ(gpu_out_contribution[i], cpu_out_contribution[i]);
  }

  delete dmat;
}

TEST(gpu_predictor, ExternalMemoryTest) {
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor"));
  gpu_predictor->Init({}, {});
  gbm::GBTreeModel model = CreateTestModel();
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(32, 64);

  // Test predict batch
  HostDeviceVector<float> out_predictions;
  gpu_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);
  EXPECT_EQ(out_predictions.Size(), dmat->Info().num_row_);
  for (const auto& v : out_predictions.HostVector()) {
    ASSERT_EQ(v, 1.5);
  }

  // Test predict leaf
  std::vector<float> leaf_out_predictions;
  gpu_predictor->PredictLeaf(dmat.get(), &leaf_out_predictions, model);
  EXPECT_EQ(leaf_out_predictions.size(), dmat->Info().num_row_);
  for (const auto& v : leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  // Test predict contribution
  std::vector<float> out_contribution;
  gpu_predictor->PredictContribution(dmat.get(), &out_contribution, model);
  EXPECT_EQ(out_contribution.size(), dmat->Info().num_row_);
  for (const auto& v : out_contribution) {
    ASSERT_EQ(v, 1.5);
  }

  // Test predict contribution (approximate method)
  std::vector<float> out_contribution_approximate;
  gpu_predictor->PredictContribution(dmat.get(), &out_contribution_approximate, model, true);
  EXPECT_EQ(out_contribution_approximate.size(), dmat->Info().num_row_);
  for (const auto& v : out_contribution_approximate) {
    ASSERT_EQ(v, 1.5);
  }
}

#if defined(XGBOOST_USE_NCCL)
// Test whether pickling preserves predictor parameters
TEST(gpu_predictor, MGPU_PicklingTest) {
  int ngpu;
  dh::safe_cuda(cudaGetDeviceCount(&ngpu));

  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateBigTestData(tmp_file, 600);

  DMatrixHandle dmat[1];
  BoosterHandle bst, bst2;
  std::vector<bst_float> label;
  for (int i = 0; i < 200; ++i) {
    label.push_back((i % 2 ? 1 : 0));
  }

  // Load data matrix
  CheckCAPICall(XGDMatrixCreateFromFile(tmp_file.c_str(), 0, &dmat[0]));
  CheckCAPICall(XGDMatrixSetFloatInfo(dmat[0], "label", label.data(), 200));
  // Create booster
  CheckCAPICall(XGBoosterCreate(dmat, 1, &bst));
  // Set parameters
  CheckCAPICall(XGBoosterSetParam(bst, "seed", "0"));
  CheckCAPICall(XGBoosterSetParam(bst, "base_score", "0.5"));
  CheckCAPICall(XGBoosterSetParam(bst, "booster", "gbtree"));
  CheckCAPICall(XGBoosterSetParam(bst, "learning_rate", "0.01"));
  CheckCAPICall(XGBoosterSetParam(bst, "max_depth", "8"));
  CheckCAPICall(XGBoosterSetParam(bst, "objective", "binary:logistic"));
  CheckCAPICall(XGBoosterSetParam(bst, "seed", "123"));
  CheckCAPICall(XGBoosterSetParam(bst, "tree_method", "gpu_hist"));
  CheckCAPICall(XGBoosterSetParam(bst, "n_gpus", std::to_string(ngpu).c_str()));
  CheckCAPICall(XGBoosterSetParam(bst, "predictor", "gpu_predictor"));

  // Run boosting iterations
  for (int i = 0; i < 10; ++i) {
    CheckCAPICall(XGBoosterUpdateOneIter(bst, i, dmat[0]));
  }

  // Delete matrix
  CheckCAPICall(XGDMatrixFree(dmat[0]));

  // Pickle
  const char* dptr;
  bst_ulong len;
  std::string buf;
  CheckCAPICall(XGBoosterGetModelRaw(bst, &len, &dptr));
  buf = std::string(dptr, len);
  CheckCAPICall(XGBoosterFree(bst));

  // Unpickle
  CheckCAPICall(XGBoosterCreate(nullptr, 0, &bst2));
  CheckCAPICall(XGBoosterLoadModelFromBuffer(bst2, buf.c_str(), len));

  {  // Query predictor
    const auto& kwargs = QueryBoosterConfigurationArguments(bst2);
    ASSERT_EQ(kwargs.at("predictor"), "gpu_predictor");
    ASSERT_EQ(kwargs.at("n_gpus"), std::to_string(ngpu).c_str());
  }

  {  // Change n_gpus and query again
    CheckCAPICall(XGBoosterSetParam(bst2, "n_gpus", "1"));
    const auto& kwargs = QueryBoosterConfigurationArguments(bst2);
    ASSERT_EQ(kwargs.at("n_gpus"), "1");
  }

  {  // Change predictor and query again
    CheckCAPICall(XGBoosterSetParam(bst2, "predictor", "cpu_predictor"));
    const auto& kwargs = QueryBoosterConfigurationArguments(bst2);
    ASSERT_EQ(kwargs.at("predictor"), "cpu_predictor");
  }

  CheckCAPICall(XGBoosterFree(bst2));
}

// multi-GPU predictor test
TEST(gpu_predictor, MGPU_Test) {
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor"));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor"));

  gpu_predictor->Init({std::pair<std::string, std::string>("n_gpus", "-1")}, {});
  cpu_predictor->Init({}, {});

  for (size_t i = 1; i < 33; i *= 2) {
    int n_row = i, n_col = i;
    auto dmat = CreateDMatrix(n_row, n_col, 0);

    gbm::GBTreeModel model = CreateTestModel();

    // Test predict batch
    HostDeviceVector<float> gpu_out_predictions;
    HostDeviceVector<float> cpu_out_predictions;

    gpu_predictor->PredictBatch((*dmat).get(), &gpu_out_predictions, model, 0);
    cpu_predictor->PredictBatch((*dmat).get(), &cpu_out_predictions, model, 0);

    std::vector<float>& gpu_out_predictions_h = gpu_out_predictions.HostVector();
    std::vector<float>& cpu_out_predictions_h = cpu_out_predictions.HostVector();
    float abs_tolerance = 0.001;
    for (int j = 0; j < gpu_out_predictions.Size(); j++) {
      ASSERT_NEAR(gpu_out_predictions_h[j], cpu_out_predictions_h[j], abs_tolerance);
    }
    delete dmat;
  }
}

// multi-GPU predictor external memory test
TEST(gpu_predictor, MGPU_ExternalMemoryTest) {
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor"));
  gpu_predictor->Init({std::pair<std::string, std::string>("n_gpus", "-1")}, {});

  gbm::GBTreeModel model = CreateTestModel();
  std::vector<std::unique_ptr<DMatrix>> dmats;
  dmats.push_back(CreateSparsePageDMatrix(9, 64UL));
  dmats.push_back(CreateSparsePageDMatrix(128, 128UL));

  for (const auto& dmat: dmats) {
    // Test predict batch
    HostDeviceVector<float> out_predictions;
    gpu_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);
    EXPECT_EQ(out_predictions.Size(), dmat->Info().num_row_);
    for (const auto& v : out_predictions.HostVector()) {
      ASSERT_EQ(v, 1.5);
    }
  }
}
#endif  // defined(XGBOOST_USE_NCCL)
}  // namespace predictor
}  // namespace xgboost
