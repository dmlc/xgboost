
/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#include <dmlc/filesystem.h>
#include <xgboost/c_api.h>
#include <xgboost/predictor.h>
#include <xgboost/logging.h>
#include <xgboost/learner.h>

#include <string>
#include "gtest/gtest.h"
#include "../helpers.h"

namespace {

inline void CheckCAPICall(int ret) {
  ASSERT_EQ(ret, 0) << XGBGetLastError();
}

}  // namespace anonymous

const std::map<std::string, std::string>&
QueryBoosterConfigurationArguments(BoosterHandle handle) {
  CHECK_NE(handle, static_cast<void*>(nullptr));
  auto* bst = static_cast<xgboost::Learner*>(handle);
  bst->Configure();
  return bst->GetConfigurationArguments();
}


namespace xgboost {
namespace predictor {

TEST(gpu_predictor, Test) {
  auto cpu_lparam = CreateEmptyGenericParam(0, 0);
  auto gpu_lparam = CreateEmptyGenericParam(0, 1);

  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &gpu_lparam));
  std::unique_ptr<Predictor> cpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("cpu_predictor", &cpu_lparam));

  gpu_predictor->Configure({}, {});
  cpu_predictor->Configure({}, {});

  for (size_t i = 1; i < 33; i *= 2) {
    int n_row = i, n_col = i;
    auto dmat = CreateDMatrix(n_row, n_col, 0);

    gbm::GBTreeModel model = CreateTestModel();
    model.param.num_feature = n_col;

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

TEST(gpu_predictor, ExternalMemoryTest) {
  auto lparam = CreateEmptyGenericParam(0, 1);
  std::unique_ptr<Predictor> gpu_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("gpu_predictor", &lparam));
  gpu_predictor->Configure({}, {});
  gbm::GBTreeModel model = CreateTestModel();
  model.param.num_feature = 3;
  const int n_classes = 3;
  model.param.num_output_group = n_classes;
  std::vector<std::unique_ptr<DMatrix>> dmats;
  dmlc::TemporaryDirectory tmpdir;
  std::string file0 = tmpdir.path + "/big_0.libsvm";
  std::string file1 = tmpdir.path + "/big_1.libsvm";
  std::string file2 = tmpdir.path + "/big_2.libsvm";
  dmats.push_back(CreateSparsePageDMatrix(9, 64UL, file0));
  dmats.push_back(CreateSparsePageDMatrix(128, 128UL, file1));
  dmats.push_back(CreateSparsePageDMatrix(1024, 1024UL, file2));

  for (const auto& dmat: dmats) {
    // Test predict batch
    HostDeviceVector<float> out_predictions;
    gpu_predictor->PredictBatch(dmat.get(), &out_predictions, model, 0);
    EXPECT_EQ(out_predictions.Size(), dmat->Info().num_row_ * n_classes);
    const std::vector<float> &host_vector = out_predictions.ConstHostVector();
    for (int i = 0; i < host_vector.size() / n_classes; i++) {
      ASSERT_EQ(host_vector[i * n_classes], 1.5);
      ASSERT_EQ(host_vector[i * n_classes + 1], 0.);
      ASSERT_EQ(host_vector[i * n_classes + 2], 0.);
    }
  }
}

// Test whether pickling preserves predictor parameters
TEST(gpu_predictor, PicklingTest) {
  int const ngpu = 1;

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
  ASSERT_EQ(XGDMatrixCreateFromFile(
      tmp_file.c_str(), 0, &dmat[0]), 0) << XGBGetLastError();
  ASSERT_EQ(XGDMatrixSetFloatInfo(
      dmat[0], "label", label.data(), 200), 0) << XGBGetLastError();
  // Create booster
  ASSERT_EQ(XGBoosterCreate(dmat, 1, &bst), 0) << XGBGetLastError();
  // Set parameters
  ASSERT_EQ(XGBoosterSetParam(bst, "seed", "0"), 0) << XGBGetLastError();
  ASSERT_EQ(XGBoosterSetParam(bst, "base_score", "0.5"), 0) << XGBGetLastError();
  ASSERT_EQ(XGBoosterSetParam(bst, "booster", "gbtree"), 0) << XGBGetLastError();
  ASSERT_EQ(XGBoosterSetParam(bst, "learning_rate", "0.01"), 0) << XGBGetLastError();
  ASSERT_EQ(XGBoosterSetParam(bst, "max_depth", "8"), 0) << XGBGetLastError();
  ASSERT_EQ(XGBoosterSetParam(
      bst, "objective", "binary:logistic"), 0) << XGBGetLastError();
  ASSERT_EQ(XGBoosterSetParam(bst, "seed", "123"), 0) << XGBGetLastError();
  ASSERT_EQ(XGBoosterSetParam(
      bst, "tree_method", "gpu_hist"), 0) << XGBGetLastError();
  ASSERT_EQ(XGBoosterSetParam(
      bst, "n_gpus", std::to_string(ngpu).c_str()), 0) << XGBGetLastError();
  ASSERT_EQ(XGBoosterSetParam(bst, "predictor", "gpu_predictor"), 0) << XGBGetLastError();

  // Run boosting iterations
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(XGBoosterUpdateOneIter(bst, i, dmat[0]), 0) << XGBGetLastError();
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

  {  // Change predictor and query again
    CheckCAPICall(XGBoosterSetParam(bst2, "predictor", "cpu_predictor"));
    const auto& kwargs = QueryBoosterConfigurationArguments(bst2);
    ASSERT_EQ(kwargs.at("predictor"), "cpu_predictor");
  }

  CheckCAPICall(XGBoosterFree(bst2));
}
}  // namespace predictor
}  // namespace xgboost
