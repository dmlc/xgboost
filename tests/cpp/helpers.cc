/*!
 * Copyright 2016-2018 XGBoost contributors
 */
#include "./helpers.h"
#include "xgboost/c_api.h"
#include <random>

bool FileExists(const std::string name) {
  struct stat st;
  return stat(name.c_str(), &st) == 0;
}

long GetFileSize(const std::string filename) {
  struct stat st;
  stat(filename.c_str(), &st);
  return st.st_size;
}

void CreateSimpleTestData(const std::string& filename) {
  CreateBigTestData(filename, 6);
}

void CreateBigTestData(const std::string& filename, size_t n_entries) {
  std::ofstream fo(filename.c_str());
  const size_t entries_per_row = 3;
  size_t n_rows = (n_entries + entries_per_row - 1) / entries_per_row;
  for (size_t i = 0; i < n_rows; ++i) {
    const char* row = i % 2 == 0 ? " 0:0 1:10 2:20\n" : " 0:0 3:30 4:40\n";
    fo << i << row;
  }
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
  obj->GetGradient(in_preds, info, 1, &out_gpair);
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
  info.labels_.HostVector() = labels;
  info.weights_.HostVector() = weights;

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
  info.labels_.HostVector() = labels;
  info.weights_.HostVector() = weights;
  info.group_ptr_ = groups;

  _CheckObjFunction(obj, preds, labels, weights, info, out_grad, out_hess);
}


xgboost::bst_float GetMetricEval(xgboost::Metric * metric,
                                 std::vector<xgboost::bst_float> preds,
                                 std::vector<xgboost::bst_float> labels,
                                 std::vector<xgboost::bst_float> weights) {
  xgboost::MetaInfo info;
  info.num_row_ = labels.size();
  info.labels_.HostVector() = labels;
  info.weights_.HostVector() = weights;
  return metric->Eval(preds, info, false);
}

namespace xgboost {
bool IsNear(std::vector<xgboost::bst_float>::const_iterator _beg1,
            std::vector<xgboost::bst_float>::const_iterator _end1,
            std::vector<xgboost::bst_float>::const_iterator _beg2) {
  for (auto iter1 = _beg1, iter2 = _beg2; iter1 != _end1; ++iter1, ++iter2) {
    if (std::abs(*iter1 - *iter2) > xgboost::kRtEps){
      return false;
    }
  }
  return true;
}

SimpleLCG::StateType SimpleLCG::operator()() {
  state_ = (alpha_ * state_) % mod_;
  return state_;
}
SimpleLCG::StateType SimpleLCG::Min() const {
  return seed_ * alpha_;
}
SimpleLCG::StateType SimpleLCG::Max() const {
  return max_value_;
}

std::shared_ptr<xgboost::DMatrix>* CreateDMatrix(int rows, int columns,
                                                 float sparsity, int seed) {
  const float missing_value = -1;
  std::vector<float> test_data(rows * columns);

  xgboost::SimpleLCG gen(seed);
  SimpleRealUniformDistribution<float> dis(0.0f, 1.0f);

  for (auto &e : test_data) {
    if (dis(&gen) < sparsity) {
      e = missing_value;
    } else {
      e = dis(&gen);
    }
  }

  DMatrixHandle handle;
  XGDMatrixCreateFromMat(test_data.data(), rows, columns, missing_value,
                         &handle);
  return static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
}

}  // namespace xgboost
