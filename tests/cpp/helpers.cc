/**
 * Copyright 2016-2023 by XGBoost contributors
 */
#include "helpers.h"

#include <gtest/gtest.h>
#include <xgboost/gbm.h>
#include <xgboost/json.h>
#include <xgboost/learner.h>
#include <xgboost/logging.h>
#include <xgboost/metric.h>
#include <xgboost/objective.h>

#include <algorithm>
#include <cinttypes>
#include <random>

#include "../../src/data/adapter.h"
#include "../../src/data/iterative_dmatrix.h"
#include "../../src/data/simple_dmatrix.h"
#include "../../src/data/sparse_page_dmatrix.h"
#include "../../src/gbm/gbtree_model.h"
#include "filesystem.h"  // dmlc::TemporaryDirectory
#include "xgboost/c_api.h"
#include "xgboost/predictor.h"

#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
#include <memory>
#include <numeric>
#include <vector>
#include "rmm/mr/device/per_device_resource.hpp"
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

bool FileExists(const std::string& filename) {
  struct stat st;
  return stat(filename.c_str(), &st) == 0;
}

void CreateSimpleTestData(const std::string& filename) {
  CreateBigTestData(filename, 6);
}

void CreateBigTestData(const std::string& filename, size_t n_entries, bool zero_based) {
  std::ofstream fo(filename.c_str());
  const size_t entries_per_row = 3;
  std::string odd_row;
  if (zero_based) {
    odd_row = " 0:0 3:30 4:40\n";
  } else {
    odd_row = " 1:0 4:30 5:40\n";
  }
  std::string even_row;
  if (zero_based) {
    even_row = " 0:0 1:10 2:20\n";
  } else {
    even_row = " 1:0 2:10 3:20\n";
  }

  size_t n_rows = (n_entries + entries_per_row - 1) / entries_per_row;
  for (size_t i = 0; i < n_rows; ++i) {
    auto row = i % 2 == 0 ? even_row : odd_row;
    fo << i << row;
  }
}

void CreateTestCSV(std::string const& path, size_t rows, size_t cols) {
  std::vector<float> data(rows * cols);

  for (size_t i = 0; i < rows * cols; ++i) {
    data[i] = i;
  }

  std::ofstream fout(path);
  size_t i = 0;
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      fout << data[i];
      i++;
      if (c != cols - 1) {
        fout << ",";
      }
    }
    fout << "\n";
  }
  fout.flush();
  fout.close();
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
  info.labels =
      xgboost::linalg::Tensor<float, 2>{labels.cbegin(), labels.cend(), {labels.size()}, -1};
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
  info.labels = xgboost::linalg::Tensor<float, 2>{
      labels.cbegin(), labels.cend(), {labels.size(), static_cast<size_t>(1)}, -1};
  info.weights_.HostVector() = weights;
  info.group_ptr_ = groups;

  CheckObjFunctionImpl(obj, preds, labels, weights, info, out_grad, out_hess);
}

xgboost::bst_float GetMetricEval(xgboost::Metric* metric,
                                 xgboost::HostDeviceVector<xgboost::bst_float> const& preds,
                                 std::vector<xgboost::bst_float> labels,
                                 std::vector<xgboost::bst_float> weights,
                                 std::vector<xgboost::bst_uint> groups,
                                 xgboost::DataSplitMode data_split_mode) {
  return GetMultiMetricEval(
      metric, preds,
      xgboost::linalg::Tensor<float, 2>{labels.begin(), labels.end(), {labels.size()}, -1}, weights,
      groups, data_split_mode);
}

double GetMultiMetricEval(xgboost::Metric* metric,
                          xgboost::HostDeviceVector<xgboost::bst_float> const& preds,
                          xgboost::linalg::Tensor<float, 2> const& labels,
                          std::vector<xgboost::bst_float> weights,
                          std::vector<xgboost::bst_uint> groups,
                          xgboost::DataSplitMode data_split_mode) {
  std::shared_ptr<xgboost::DMatrix> p_fmat{xgboost::RandomDataGenerator{0, 0, 0}.GenerateDMatrix()};
  auto& info = p_fmat->Info();
  info.num_row_ = labels.Shape(0);
  info.labels.Reshape(labels.Shape()[0], labels.Shape()[1]);
  info.labels.Data()->Copy(*labels.Data());
  info.weights_.HostVector() = weights;
  info.group_ptr_ = groups;
  info.data_split_mode = data_split_mode;
  if (info.IsVerticalFederated() && xgboost::collective::GetRank() != 0) {
    info.labels.Reshape(0);
  }
  return metric->Evaluate(preds, p_fmat);
}

namespace xgboost {

float GetBaseScore(Json const &config) {
  return std::stof(get<String const>(config["learner"]["learner_model_param"]["base_score"]));
}

SimpleLCG::StateType SimpleLCG::operator()() {
  state_ = (alpha_ * state_ + (state_ == 0 ? kDefaultInit : 0)) % mod_;
  return state_;
}
SimpleLCG::StateType SimpleLCG::Min() const { return min(); }
SimpleLCG::StateType SimpleLCG::Max() const { return max(); }
// Make sure it's compile time constant.
static_assert(SimpleLCG::max() - SimpleLCG::min());

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
  return GetArrayInterface(storage, rows, cols);
}

std::string RandomDataGenerator::GenerateArrayInterface(
    HostDeviceVector<float> *storage) const {
  auto array_interface = this->ArrayInterfaceImpl(storage, rows_, cols_);
  std::string out;
  Json::Dump(array_interface, &out);
  return out;
}

std::pair<std::vector<std::string>, std::string> MakeArrayInterfaceBatch(
    HostDeviceVector<float> const* storage, std::size_t n_samples, bst_feature_t n_features,
    std::size_t batches, std::int32_t device) {
  std::vector<std::string> result(batches);
  std::vector<Json> objects;

  size_t const rows_per_batch = n_samples / batches;

  auto make_interface = [storage, device, n_features](std::size_t offset, std::size_t rows) {
    Json array_interface{Object()};
    array_interface["data"] = std::vector<Json>(2);
    if (device >= 0) {
      array_interface["data"][0] =
          Integer(reinterpret_cast<int64_t>(storage->DevicePointer() + offset));
      array_interface["stream"] = Null{};
    } else {
      array_interface["data"][0] =
          Integer(reinterpret_cast<int64_t>(storage->HostPointer() + offset));
    }

    array_interface["data"][1] = Boolean(false);

    array_interface["shape"] = std::vector<Json>(2);
    array_interface["shape"][0] = rows;
    array_interface["shape"][1] = n_features;

    array_interface["typestr"] = String("<f4");
    array_interface["version"] = 3;
    return array_interface;
  };

  auto j_interface = make_interface(0, n_samples);
  size_t offset = 0;
  for (size_t i = 0; i < batches - 1; ++i) {
    objects.emplace_back(make_interface(offset, rows_per_batch));
    offset += rows_per_batch * n_features;
  }

  size_t const remaining = n_samples - offset / n_features;
  CHECK_LE(offset, n_samples * n_features);
  objects.emplace_back(make_interface(offset, remaining));

  for (size_t i = 0; i < batches; ++i) {
    Json::Dump(objects[i], &result[i]);
  }

  std::string interface_str;
  Json::Dump(j_interface, &interface_str);
  return {result, interface_str};
}

std::pair<std::vector<std::string>, std::string> RandomDataGenerator::GenerateArrayInterfaceBatch(
    HostDeviceVector<float>* storage, size_t batches) const {
  this->GenerateDense(storage);
  return MakeArrayInterfaceBatch(storage, rows_, cols_, batches, device_);
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
  SimpleRealUniformDistribution<bst_float> cat(0.0, max_cat_);

  h_rptr.emplace_back(0);
  for (size_t i = 0; i < rows_; ++i) {
    size_t rptr = h_rptr.back();
    for (size_t j = 0; j < cols_; ++j) {
      auto g = dist(&lcg);
      if (g >= sparsity) {
        if (common::IsCat(ft_, j)) {
          g = common::AsCat(cat(&lcg));
        } else {
          g = dist(&lcg);
        }
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

std::shared_ptr<DMatrix> RandomDataGenerator::GenerateDMatrix(bool with_label, bool float_label,
                                                              size_t classes) const {
  HostDeviceVector<float> data;
  HostDeviceVector<bst_row_t> rptrs;
  HostDeviceVector<bst_feature_t> columns;
  this->GenerateCSR(&data, &rptrs, &columns);
  data::CSRAdapter adapter(rptrs.HostPointer(), columns.HostPointer(), data.HostPointer(), rows_,
                           data.Size(), cols_);
  std::shared_ptr<DMatrix> out{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)};

  if (with_label) {
    RandomDataGenerator gen{rows_, n_targets_, 0.0f};
    if (!float_label) {
      gen.Lower(0).Upper(classes).GenerateDense(out->Info().labels.Data());
      out->Info().labels.Reshape(this->rows_, this->n_targets_);
      auto& h_labels = out->Info().labels.Data()->HostVector();
      for (auto& v : h_labels) {
        v = static_cast<float>(static_cast<uint32_t>(v));
      }
    } else {
      gen.GenerateDense(out->Info().labels.Data());
      CHECK_EQ(out->Info().labels.Size(), this->rows_ * this->n_targets_);
      out->Info().labels.Reshape(this->rows_, this->n_targets_);
    }
  }
  if (device_ >= 0) {
    out->Info().labels.SetDevice(device_);
    out->Info().feature_types.SetDevice(device_);
    for (auto const& page : out->GetBatches<SparsePage>()) {
      page.data.SetDevice(device_);
      page.offset.SetDevice(device_);
    }
  }
  if (!ft_.empty()) {
    out->Info().feature_types.HostVector() = ft_;
  }
  return out;
}

std::shared_ptr<DMatrix> RandomDataGenerator::GenerateQuantileDMatrix() {
  NumpyArrayIterForTest iter{this->sparsity_, this->rows_, this->cols_, 1};
  auto m = std::make_shared<data::IterativeDMatrix>(
      &iter, iter.Proxy(), nullptr, Reset, Next, std::numeric_limits<float>::quiet_NaN(), 0, bins_);
  return m;
}

NumpyArrayIterForTest::NumpyArrayIterForTest(float sparsity, size_t rows, size_t cols,
                                             size_t batches)
    : ArrayIterForTest{sparsity, rows, cols, batches} {
  rng_->Device(Context::kCpuId);
  std::tie(batches_, interface_) = rng_->GenerateArrayInterfaceBatch(&data_, n_batches_);
  this->Reset();
}

int NumpyArrayIterForTest::Next() {
  if (iter_ == n_batches_) {
    return 0;
  }
  XGProxyDMatrixSetDataDense(proxy_, batches_[iter_].c_str());
  iter_++;
  return 1;
}

std::shared_ptr<DMatrix> GetDMatrixFromData(const std::vector<float>& x, std::size_t num_rows,
                                            bst_feature_t num_columns) {
  data::DenseAdapter adapter(x.data(), num_rows, num_columns);
  auto p_fmat = std::shared_ptr<DMatrix>(
      new data::SimpleDMatrix(&adapter, std::numeric_limits<float>::quiet_NaN(), 1));
  CHECK_EQ(p_fmat->Info().num_row_, num_rows);
  CHECK_EQ(p_fmat->Info().num_col_, num_columns);
  return p_fmat;
}

std::unique_ptr<DMatrix> CreateSparsePageDMatrix(bst_row_t n_samples, bst_feature_t n_features,
                                                 size_t n_batches, std::string prefix) {
  CHECK_GE(n_samples, n_batches);
  NumpyArrayIterForTest iter(0, n_samples, n_features, n_batches);

  std::unique_ptr<DMatrix> dmat{
      DMatrix::Create(static_cast<DataIterHandle>(&iter), iter.Proxy(), Reset, Next,
                      std::numeric_limits<float>::quiet_NaN(), omp_get_max_threads(), prefix)};

  auto row_page_path =
      data::MakeId(prefix, dynamic_cast<data::SparsePageDMatrix*>(dmat.get())) + ".row.page";
  EXPECT_TRUE(FileExists(row_page_path)) << row_page_path;

  // Loop over the batches and count the number of pages
  int64_t batch_count = 0;
  int64_t row_count = 0;
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    batch_count++;
    row_count += batch.Size();
  }

  EXPECT_GE(batch_count, n_batches);
  EXPECT_EQ(row_count, dmat->Info().num_row_);
  return dmat;
}

std::unique_ptr<DMatrix> CreateSparsePageDMatrix(size_t n_entries,
                                                 std::string prefix) {
  size_t n_columns = 3;
  size_t n_rows = n_entries / n_columns;
  NumpyArrayIterForTest iter(0, n_rows, n_columns, 2);

  std::unique_ptr<DMatrix> dmat{
      DMatrix::Create(static_cast<DataIterHandle>(&iter), iter.Proxy(), Reset, Next,
                      std::numeric_limits<float>::quiet_NaN(), 0, prefix)};
  auto row_page_path =
      data::MakeId(prefix,
                   dynamic_cast<data::SparsePageDMatrix *>(dmat.get())) +
      ".row.page";
  EXPECT_TRUE(FileExists(row_page_path)) << row_page_path;

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

  std::string uri = tmp_file + "?format=libsvm";
  if (page_size > 0) {
    uri += "#" + tmp_file + ".cache";
  }
  std::unique_ptr<DMatrix> dmat(DMatrix::Load(uri));
  return dmat;
}

std::unique_ptr<GradientBooster> CreateTrainedGBM(std::string name, Args kwargs, size_t kRows,
                                                  size_t kCols,
                                                  LearnerModelParam const* learner_model_param,
                                                  Context const* ctx) {
  auto caches = std::make_shared<PredictionContainer>();
  std::unique_ptr<GradientBooster> gbm{GradientBooster::Create(name, ctx, learner_model_param)};
  gbm->Configure(kwargs);
  auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  std::vector<float> labels(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    labels[i] = i;
  }
  p_dmat->Info().labels =
      linalg::Tensor<float, 2>{labels.cbegin(), labels.cend(), {labels.size()}, -1};
  HostDeviceVector<GradientPair> gpair;
  auto& h_gpair = gpair.HostVector();
  h_gpair.resize(kRows);
  for (size_t i = 0; i < kRows; ++i) {
    h_gpair[i] = GradientPair{static_cast<float>(i), 1};
  }

  PredictionCacheEntry predts;

  gbm->DoBoost(p_dmat.get(), &gpair, &predts, nullptr);

  return gbm;
}

ArrayIterForTest::ArrayIterForTest(float sparsity, size_t rows, size_t cols, size_t batches)
    : rows_{rows}, cols_{cols}, n_batches_{batches} {
  XGProxyDMatrixCreate(&proxy_);
  rng_.reset(new RandomDataGenerator{rows_, cols_, sparsity});
  std::tie(batches_, interface_) = rng_->GenerateArrayInterfaceBatch(&data_, n_batches_);
}

ArrayIterForTest::ArrayIterForTest(Context const* ctx, HostDeviceVector<float> const& data,
                                   std::size_t n_samples, bst_feature_t n_features,
                                   std::size_t n_batches)
    : rows_{n_samples}, cols_{n_features}, n_batches_{n_batches} {
  XGProxyDMatrixCreate(&proxy_);
  this->data_.Resize(data.Size());
  CHECK_EQ(this->data_.Size(), rows_ * cols_ * n_batches);
  this->data_.Copy(data);
  std::tie(batches_, interface_) =
      MakeArrayInterfaceBatch(&data_, rows_, cols_, n_batches_, ctx->gpu_id);
}

ArrayIterForTest::~ArrayIterForTest() { XGDMatrixFree(proxy_); }

void DMatrixToCSR(DMatrix *dmat, std::vector<float> *p_data,
                  std::vector<size_t> *p_row_ptr,
                  std::vector<bst_feature_t> *p_cids) {
  auto &data = *p_data;
  auto &row_ptr = *p_row_ptr;
  auto &cids = *p_cids;

  data.resize(dmat->Info().num_nonzero_);
  cids.resize(data.size());
  row_ptr.resize(dmat->Info().num_row_ + 1);
  SparsePage page;
  for (const auto &batch : dmat->GetBatches<SparsePage>()) {
    page.Push(batch);
  }

  auto const& in_offset = page.offset.HostVector();
  auto const& in_data = page.data.HostVector();

  CHECK_EQ(in_offset.size(), row_ptr.size());
  std::copy(in_offset.cbegin(), in_offset.cend(), row_ptr.begin());
  ASSERT_EQ(in_data.size(), data.size());
  std::transform(in_data.cbegin(), in_data.cend(), data.begin(), [](Entry const& e) {
    return e.fvalue;
  });
  ASSERT_EQ(in_data.size(), cids.size());
  std::transform(in_data.cbegin(), in_data.cend(), cids.begin(), [](Entry const& e) {
    return e.index;
  });
}

#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

using CUDAMemoryResource = rmm::mr::cuda_memory_resource;
using PoolMemoryResource = rmm::mr::pool_memory_resource<CUDAMemoryResource>;
class RMMAllocator {
 public:
  std::vector<std::unique_ptr<CUDAMemoryResource>> cuda_mr;
  std::vector<std::unique_ptr<PoolMemoryResource>> pool_mr;
  int n_gpu;
  RMMAllocator() : n_gpu(common::AllVisibleGPUs()) {
    int current_device;
    CHECK_EQ(cudaGetDevice(&current_device), cudaSuccess);
    for (int i = 0; i < n_gpu; ++i) {
      CHECK_EQ(cudaSetDevice(i), cudaSuccess);
      cuda_mr.push_back(std::make_unique<CUDAMemoryResource>());
      pool_mr.push_back(std::make_unique<PoolMemoryResource>(cuda_mr[i].get()));
    }
    CHECK_EQ(cudaSetDevice(current_device), cudaSuccess);
  }
  ~RMMAllocator() = default;
};

void DeleteRMMResource(RMMAllocator* r) {
  delete r;
}

RMMAllocatorPtr SetUpRMMResourceForCppTests(int argc, char** argv) {
  bool use_rmm_pool = false;
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == std::string("--use-rmm-pool")) {
      use_rmm_pool = true;
    }
  }
  if (!use_rmm_pool) {
    return RMMAllocatorPtr(nullptr, DeleteRMMResource);
  }
  LOG(INFO) << "Using RMM memory pool";
  auto ptr = RMMAllocatorPtr(new RMMAllocator(), DeleteRMMResource);
  for (int i = 0; i < ptr->n_gpu; ++i) {
    rmm::mr::set_per_device_resource(rmm::cuda_device_id(i), ptr->pool_mr[i].get());
  }
  return ptr;
}
#else  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
class RMMAllocator {};

void DeleteRMMResource(RMMAllocator*) {}

RMMAllocatorPtr SetUpRMMResourceForCppTests(int, char**) { return {nullptr, DeleteRMMResource}; }
#endif  // !defined(XGBOOST_USE_RMM) || XGBOOST_USE_RMM != 1
} // namespace xgboost
