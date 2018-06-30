#include "./helpers.h"
#include "xgboost/c_api.h"
#include <random>

std::string TempFileName() {
  std::string tmp = std::tmpnam(nullptr);
  std::replace(tmp.begin(), tmp.end(), '\\',
               '/');  // Remove windows backslashes
  // Remove drive prefix for windows
  if (tmp.find("C:") != std::string::npos)
    tmp.erase(tmp.find("C:"), 2);
  return tmp;
}

bool FileExists(const std::string name) {
  struct stat st;
  return stat(name.c_str(), &st) == 0; 
}

long GetFileSize(const std::string filename) {
  struct stat st;
  stat(filename.c_str(), &st);
  return st.st_size;
}

std::string CreateSimpleTestData() {
  return CreateBigTestData(6);
}

std::string CreateBigTestData(size_t n_entries) {
  std::string tmp_file = TempFileName();
  std::ofstream fo;
  fo.open(tmp_file);
  const size_t entries_per_row = 3;
  size_t n_rows = (n_entries + entries_per_row - 1) / entries_per_row;
  for (size_t i = 0; i < n_rows; ++i) {
    const char* row = i % 2 == 0 ? " 0:0 1:10 2:20\n" : " 0:0 3:30 4:40\n";
    fo << i << row;
  }
  fo.close();
  return tmp_file;
}

void _CheckObjFunction(xgboost::ObjFunction * obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      xgboost::MetaInfo info,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess) {
  xgboost::HostDeviceVector<xgboost::bst_float> in_preds(preds);

  xgboost::HostDeviceVector<xgboost::GradientPair> out_gpair;
  obj->GetGradient(&in_preds, info, 1, &out_gpair);
  std::vector<xgboost::GradientPair>& gpair = out_gpair.HostVector();

  ASSERT_EQ(gpair.size(), in_preds.Size());
  for (int i = 0; i < static_cast<int>(gpair.size()); ++i) {
    EXPECT_NEAR(gpair[i].GetGrad(), out_grad[i], 0.01)
      << "Unexpected grad for pred=" << preds[i] << " label=" << labels[i]
      << " weight=" << weights[i];
    EXPECT_NEAR(gpair[i].GetHess(), out_hess[i], 0.01)
      << "Unexpected hess for pred=" << preds[i] << " label=" << labels[i]
      << " weight=" << weights[i];
  }
}

void CheckObjFunction(xgboost::ObjFunction * obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess) {
  xgboost::MetaInfo info;
  info.num_row_ = labels.size();
  info.labels_ = labels;
  info.weights_ = weights;

  _CheckObjFunction(obj, preds, labels, weights, info, out_grad, out_hess);
}

void CheckRankingObjFunction(xgboost::ObjFunction * obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      std::vector<xgboost::bst_uint> groups,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess) {
  xgboost::MetaInfo info;
  info.num_row_ = labels.size();
  info.labels_ = labels;
  info.weights_ = weights;
  info.group_ptr_ = groups;

  _CheckObjFunction(obj, preds, labels, weights, info, out_grad, out_hess);
}


xgboost::bst_float GetMetricEval(xgboost::Metric * metric,
                                 std::vector<xgboost::bst_float> preds,
                                 std::vector<xgboost::bst_float> labels,
                                 std::vector<xgboost::bst_float> weights) {
  xgboost::MetaInfo info;
  info.num_row_ = labels.size();
  info.labels_ = labels;
  info.weights_ = weights;
  return metric->Eval(preds, info, false);
}

std::shared_ptr<xgboost::DMatrix> CreateDMatrix(int rows, int columns,
                                                float sparsity, int seed) {
  const float missing_value = -1;
  std::vector<float> test_data(rows * columns);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (auto &e : test_data) {
    if (dis(gen) < sparsity) {
      e = missing_value;
    } else {
      e = dis(gen);
    }
  }

  DMatrixHandle handle;
  XGDMatrixCreateFromMat(test_data.data(), rows, columns, missing_value,
                         &handle);
  return *static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
}
