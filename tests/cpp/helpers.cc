/*!
 * Copyright 2016-2019 XGBoost contributors
 */
#include <dmlc/filesystem.h>
#include <xgboost/logging.h>
#include <xgboost/gbm.h>
#include <xgboost/json.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <cinttypes>

#include "helpers.h"
#include "xgboost/c_api.h"

#include "../../src/data/simple_csr_source.h"
#include "../../src/gbm/gbtree_model.h"

bool FileExists(const std::string& filename) {
  struct stat st;
  return stat(filename.c_str(), &st) == 0;
}

int64_t GetFileSize(const std::string& filename) {
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

void CheckObjFunctionImpl(std::unique_ptr<xgboost::ObjFunction> const& obj,
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

void CheckObjFunction(std::unique_ptr<xgboost::ObjFunction> const& obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess) {
  xgboost::MetaInfo info;
  info.num_row_ = labels.size();
  info.labels_.HostVector() = labels;
  info.weights_.HostVector() = weights;

  CheckObjFunctionImpl(obj, preds, labels, weights, info, out_grad, out_hess);
}

xgboost::Json CheckConfigReloadImpl(xgboost::Configurable* const configurable,
                                    std::string name) {
  xgboost::Json config_0 { xgboost::Object() };
  configurable->SaveConfig(&config_0);
  configurable->LoadConfig(config_0);

  xgboost::Json config_1 { xgboost::Object() };
  configurable->SaveConfig(&config_1);

  std::string str_0, str_1;
  xgboost::Json::Dump(config_0, &str_0);
  xgboost::Json::Dump(config_1, &str_1);
  EXPECT_EQ(str_0, str_1);

  if (name != "") {
    EXPECT_EQ(xgboost::get<xgboost::String>(config_1["name"]), name);
  }
  return config_1;
}

void CheckRankingObjFunction(std::unique_ptr<xgboost::ObjFunction> const& obj,
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

  CheckObjFunctionImpl(obj, preds, labels, weights, info, out_grad, out_hess);
}

xgboost::bst_float GetMetricEval(xgboost::Metric * metric,
                                 xgboost::HostDeviceVector<xgboost::bst_float> preds,
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

std::unique_ptr<DMatrix> CreateSparsePageDMatrix(
    size_t n_entries, size_t page_size, std::string tmp_file) {
  // Create sufficiently large data to make two row pages
  CreateBigTestData(tmp_file, n_entries);
  std::unique_ptr<DMatrix> dmat { DMatrix::Load(
      tmp_file + "#" + tmp_file + ".cache", true, false, "auto", page_size)};
  EXPECT_TRUE(FileExists(tmp_file + ".cache.row.page"));

  // Loop over the batches and count the records
  int64_t batch_count = 0;
  int64_t row_count = 0;
  for (const auto &batch : dmat->GetBatches<xgboost::SparsePage>()) {
    batch_count++;
    row_count += batch.Size();
  }
  EXPECT_GE(batch_count, 2);
  EXPECT_EQ(row_count, dmat->Info().num_row_);

  return dmat;
}


std::unique_ptr<DMatrix> CreateSparsePageDMatrixWithRC(
    size_t n_rows, size_t n_cols, size_t page_size, bool deterministic,
    const dmlc::TemporaryDirectory& tempdir) {
  if (!n_rows || !n_cols) {
    return nullptr;
  }

  // Create the svm file in a temp dir
  const std::string tmp_file = tempdir.path + "/big.libsvm";

  std::ofstream fo(tmp_file.c_str());
  size_t cols_per_row = ((std::max(n_rows, n_cols) - 1) / std::min(n_rows, n_cols)) + 1;
  int64_t rem_cols = n_cols;
  size_t col_idx = 0;

  // Random feature id generator
  std::random_device rdev;
  std::unique_ptr<std::mt19937> gen;
  if (deterministic) {
     // Seed it with a constant value for this configuration - without getting too fancy
     // like ordered pairing functions and its likes to make it truely unique
     gen.reset(new std::mt19937(n_rows * n_cols));
  } else {
     gen.reset(new std::mt19937(rdev()));
  }
  std::uniform_int_distribution<size_t> label(0, 1);
  std::uniform_int_distribution<size_t> dis(1, n_cols);

  for (size_t i = 0; i < n_rows; ++i) {
    // Make sure that all cols are slotted in the first few rows; randomly distribute the
    // rest
    std::stringstream row_data;
    size_t j = 0;
    if (rem_cols > 0) {
       for (; j < std::min(static_cast<size_t>(rem_cols), cols_per_row); ++j) {
         row_data << label(*gen) << " " << (col_idx+j) << ":" << (col_idx+j+1)*10*i;
       }
       rem_cols -= cols_per_row;
    } else {
       // Take some random number of colums in [1, n_cols] and slot them here
       size_t ncols = dis(*gen);
       for (; j < ncols; ++j) {
         size_t fid = (col_idx+j) % n_cols;
         row_data << label(*gen) << " " << fid << ":" << (fid+1)*10*i;
       }
    }
    col_idx += j;

    fo << row_data.str() << "\n";
  }
  fo.close();

  std::unique_ptr<DMatrix> dmat(DMatrix::Load(
    tmp_file + "#" + tmp_file + ".cache", true, false, "auto", page_size));
  EXPECT_TRUE(FileExists(tmp_file + ".cache.row.page"));

  if (!page_size) {
    std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource);
    source->CopyFrom(dmat.get());
    return std::unique_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  } else {
    return dmat;
  }
}

gbm::GBTreeModel CreateTestModel(LearnerModelParam const* param) {
  std::vector<std::unique_ptr<RegTree>> trees;
  trees.push_back(std::unique_ptr<RegTree>(new RegTree));
  (*trees.back())[0].SetLeaf(1.5f);
  (*trees.back()).Stat(0).sum_hess = 1.0f;
  gbm::GBTreeModel model(param);
  model.CommitModel(std::move(trees), 0);
  return model;
}

std::unique_ptr<GradientBooster> CreateTrainedGBM(
    std::string name, Args kwargs, size_t kRows, size_t kCols,
    LearnerModelParam const* learner_model_param,
    GenericParameter const* generic_param) {
  std::unique_ptr<GradientBooster> gbm {
    GradientBooster::Create(name, generic_param, learner_model_param, {})};
  gbm->Configure(kwargs);
  auto pp_dmat = CreateDMatrix(kRows, kCols, 0);
  auto p_dmat = *pp_dmat;

  std::vector<float> labels(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    labels[i] = i;
  }
  p_dmat->Info().labels_.HostVector() = labels;
  HostDeviceVector<GradientPair> gpair;
  auto& h_gpair = gpair.HostVector();
  h_gpair.resize(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    h_gpair[i] = {static_cast<float>(i), 1};
  }

  gbm->DoBoost(p_dmat.get(), &gpair, nullptr);

  delete pp_dmat;
  return gbm;
}

}  // namespace xgboost
