/*!
 * Copyright 2016-2020 XGBoost contributors
 */
#include <dmlc/filesystem.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <xgboost/metric.h>
#include <xgboost/learner.h>
#include <xgboost/gbm.h>
#include <xgboost/json.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <cinttypes>

#include "helpers.h"
#include "xgboost/c_api.h"
#include "../../src/data/adapter.h"
#include "../../src/gbm/gbtree_model.h"
#include "xgboost/predictor.h"

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
                          xgboost::MetaInfo const& info,
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
                                 std::vector<xgboost::bst_float> weights,
                                 std::vector<xgboost::bst_uint> groups) {
  xgboost::MetaInfo info;
  info.num_row_ = labels.size();
  info.labels_.HostVector() = labels;
  info.weights_.HostVector() = weights;
  info.group_ptr_ = groups;

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

void RandomDataGenerator::GenerateDense(HostDeviceVector<float> *out) const {
  xgboost::SimpleRealUniformDistribution<bst_float> dist(lower_, upper_);
  CHECK(out);

  SimpleLCG lcg{lcg_};
  out->Resize(rows_ * cols_, 0);
  auto &h_data = out->HostVector();
  float sparsity = sparsity_ * (upper_ - lower_) + lower_;
  for (auto &v : h_data) {
    auto g = dist(&lcg);
    if (g < sparsity) {
      v = std::numeric_limits<float>::quiet_NaN();
    } else {
      v = dist(&lcg);
    }
  }
  if (device_ >= 0) {
    out->SetDevice(device_);
    out->DeviceSpan();
  }
}

Json RandomDataGenerator::ArrayInterfaceImpl(HostDeviceVector<float> *storage,
                                             size_t rows, size_t cols) const {
  this->GenerateDense(storage);
  Json array_interface {Object()};
  array_interface["data"] = std::vector<Json>(2);
  array_interface["data"][0] = Integer(reinterpret_cast<int64_t>(storage->DevicePointer()));
  array_interface["data"][1] = Boolean(false);

  array_interface["shape"] = std::vector<Json>(2);
  array_interface["shape"][0] = rows;
  array_interface["shape"][1] = cols;

  array_interface["typestr"] = String("<f4");
  array_interface["version"] = 1;
  return array_interface;
}

std::string RandomDataGenerator::GenerateArrayInterface(
    HostDeviceVector<float> *storage) const {
  auto array_interface = this->ArrayInterfaceImpl(storage, rows_, cols_);
  std::string out;
  Json::Dump(array_interface, &out);
  return out;
}

std::pair<std::vector<std::string>, std::string>
RandomDataGenerator::GenerateArrayInterfaceBatch(
    HostDeviceVector<float> *storage, size_t batches) const {
  this->GenerateDense(storage);
  std::vector<std::string> result(batches);
  std::vector<Json> objects;

  size_t const rows_per_batch = rows_ / batches;

  auto make_interface = [storage, this](size_t offset, size_t rows) {
    Json array_interface{Object()};
    array_interface["data"] = std::vector<Json>(2);
    if (device_ >= 0) {
      array_interface["data"][0] =
          Integer(reinterpret_cast<int64_t>(storage->DevicePointer() + offset));
    } else {
      array_interface["data"][0] =
          Integer(reinterpret_cast<int64_t>(storage->HostPointer() + offset));
    }

    array_interface["data"][1] = Boolean(false);

    array_interface["shape"] = std::vector<Json>(2);
    array_interface["shape"][0] = rows;
    array_interface["shape"][1] = cols_;

    array_interface["typestr"] = String("<f4");
    array_interface["version"] = 1;
    return array_interface;
  };

  auto j_interface = make_interface(0, rows_);
  size_t offset = 0;
  for (size_t i = 0; i < batches - 1; ++i) {
    objects.emplace_back(make_interface(offset, rows_per_batch));
    offset += rows_per_batch * cols_;
  }

  size_t const remaining = rows_ - offset / cols_;
  CHECK_LE(offset, rows_ * cols_);
  objects.emplace_back(make_interface(offset, remaining));

  for (size_t i = 0; i < batches; ++i) {
    Json::Dump(objects[i], &result[i]);
  }

  std::string interface_str;
  Json::Dump(j_interface, &interface_str);
  return {result, interface_str};
}

std::string RandomDataGenerator::GenerateColumnarArrayInterface(
    std::vector<HostDeviceVector<float>> *data) const {
  CHECK(data);
  CHECK_EQ(data->size(), cols_);
  auto& storage = *data;
  Json arr { Array() };
  for (size_t i = 0; i < cols_; ++i) {
    auto column = this->ArrayInterfaceImpl(&storage[i], rows_, 1);
    get<Array>(arr).emplace_back(column);
  }
  std::string out;
  Json::Dump(arr, &out);
  return out;
}

void RandomDataGenerator::GenerateCSR(
    HostDeviceVector<float>* value, HostDeviceVector<bst_row_t>* row_ptr,
    HostDeviceVector<bst_feature_t>* columns) const {
  auto& h_value = value->HostVector();
  auto& h_rptr = row_ptr->HostVector();
  auto& h_cols = columns->HostVector();
  SimpleLCG lcg{lcg_};

  xgboost::SimpleRealUniformDistribution<bst_float> dist(lower_, upper_);
  float sparsity = sparsity_ * (upper_ - lower_) + lower_;

  h_rptr.emplace_back(0);
  for (size_t i = 0; i < rows_; ++i) {
    size_t rptr = h_rptr.back();
    for (size_t j = 0; j < cols_; ++j) {
      auto g = dist(&lcg);
      if (g >= sparsity) {
        g = dist(&lcg);
        h_value.emplace_back(g);
        rptr++;
        h_cols.emplace_back(j);
      }
    }
    h_rptr.emplace_back(rptr);
  }

  if (device_ >= 0) {
    value->SetDevice(device_);
    value->DeviceSpan();
    row_ptr->SetDevice(device_);
    row_ptr->DeviceSpan();
    columns->SetDevice(device_);
    columns->DeviceSpan();
  }

  CHECK_LE(h_value.size(), rows_ * cols_);
  CHECK_EQ(value->Size(), h_rptr.back());
  CHECK_EQ(columns->Size(), value->Size());
}

std::shared_ptr<DMatrix>
RandomDataGenerator::GenerateDMatrix(bool with_label, bool float_label,
                                     size_t classes) const {
  HostDeviceVector<float> data;
  HostDeviceVector<bst_row_t> rptrs;
  HostDeviceVector<bst_feature_t> columns;
  this->GenerateCSR(&data, &rptrs, &columns);
  data::CSRAdapter adapter(rptrs.HostPointer(), columns.HostPointer(),
                           data.HostPointer(), rows_, data.Size(), cols_);
  std::shared_ptr<DMatrix> out{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)};

  if (with_label) {
    RandomDataGenerator gen(rows_, 1, 0);
    if (!float_label) {
      gen.Lower(0).Upper(classes).GenerateDense(&out->Info().labels_);
      auto& h_labels = out->Info().labels_.HostVector();
      for (auto& v : h_labels) {
        v = static_cast<float>(static_cast<uint32_t>(v));
      }
    } else {
      gen.GenerateDense(&out->Info().labels_);
    }
  }
  return out;
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
#if defined(_OPENMP)
  EXPECT_GE(batch_count, 2);
  EXPECT_EQ(row_count, dmat->Info().num_row_);
#else
#warning "External memory doesn't work with Non-OpenMP build "
#endif  // defined(_OPENMP)
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
        row_data << label(*gen) << " " << (col_idx + j) << ":"
                 << (col_idx + j + 1) * 10 * i;
      }
      rem_cols -= cols_per_row;
    } else {
      // Take some random number of colums in [1, n_cols] and slot them here
      std::vector<size_t> random_columns;
      size_t ncols = dis(*gen);
      for (; j < ncols; ++j) {
        size_t fid = (col_idx + j) % n_cols;
        random_columns.push_back(fid);
      }
      std::sort(random_columns.begin(), random_columns.end());
      for (auto fid : random_columns) {
        row_data << label(*gen) << " " << fid << ":" << (fid + 1) * 10 * i;
      }
    }
    col_idx += j;

    fo << row_data.str() << "\n";
  }
  fo.close();

  std::string uri = tmp_file;
  if (page_size > 0) {
    uri += "#" + tmp_file + ".cache";
  }
  std::unique_ptr<DMatrix> dmat(
      DMatrix::Load(uri, true, false, "auto", page_size));
  return dmat;
}

gbm::GBTreeModel CreateTestModel(LearnerModelParam const* param, size_t n_classes) {
  gbm::GBTreeModel model(param);

  for (size_t i = 0; i < n_classes; ++i) {
    std::vector<std::unique_ptr<RegTree>> trees;
    trees.push_back(std::unique_ptr<RegTree>(new RegTree));
    if (i == 0) {
      (*trees.back())[0].SetLeaf(1.5f);
      (*trees.back()).Stat(0).sum_hess = 1.0f;
    }
    model.CommitModel(std::move(trees), i);
  }

  return model;
}

std::unique_ptr<GradientBooster> CreateTrainedGBM(
    std::string name, Args kwargs, size_t kRows, size_t kCols,
    LearnerModelParam const* learner_model_param,
    GenericParameter const* generic_param) {
  auto caches = std::make_shared< PredictionContainer >();;
  std::unique_ptr<GradientBooster> gbm {
    GradientBooster::Create(name, generic_param, learner_model_param)};
  gbm->Configure(kwargs);
  auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

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

  PredictionCacheEntry predts;

  gbm->DoBoost(p_dmat.get(), &gpair, &predts);

  return gbm;
}

}  // namespace xgboost
