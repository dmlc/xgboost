#pragma once
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include "../../../src/common/hist_util.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../../../src/data/adapter.h"

#ifdef __CUDACC__
#include <xgboost/json.h>
#include "../../../src/data/device_adapter.cuh"
#endif  // __CUDACC__

// Some helper functions used to test both GPU and CPU algorithms
//
namespace xgboost {
namespace common {

  // Generate columns with different ranges
inline std::vector<float> GenerateRandom(int num_rows, int num_columns) {
  std::vector<float> x(num_rows*num_columns);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::generate(x.begin(), x.end(), [&]() { return dist(rng); });
  for (auto i = 0; i < num_columns; i++) {
    for (auto j = 0; j < num_rows; j++) {
      x[j * num_columns + i] += i;
    }
  }
  return x;
}

inline std::vector<float> GenerateRandomWeights(int num_rows) {
  std::vector<float> w(num_rows);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::generate(w.begin(), w.end(), [&]() { return dist(rng); });
  return w;
}

#ifdef __CUDACC__
inline data::CupyAdapter AdapterFromData(const thrust::device_vector<float> &x,
  int num_rows, int num_columns) {
  Json array_interface{Object()};
  std::vector<Json> shape = {Json(static_cast<Integer::Int>(num_rows)),
    Json(static_cast<Integer::Int>(num_columns))};
  array_interface["shape"] = Array(shape);
  std::vector<Json> j_data{
    Json(Integer(reinterpret_cast<Integer::Int>(x.data().get()))),
    Json(Boolean(false))};
  array_interface["data"] = j_data;
  array_interface["version"] = Integer(static_cast<Integer::Int>(1));
  array_interface["typestr"] = String("<f4");
  std::stringstream ss;
  Json::Dump(array_interface, &ss);
  std::string str = ss.str();
  return data::CupyAdapter(str);
}
#endif

inline std::vector<float> GenerateRandomCategoricalSingleColumn(int n,
                                                                int num_categories) {
  std::vector<float> x(n);
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> dist(0, num_categories - 1);
  std::generate(x.begin(), x.end(), [&]() { return dist(rng); });
  // Make sure each category is present
  for(auto i = 0; i < num_categories; i++) {
    x[i] = i;
  }
  return x;
}

inline std::shared_ptr<data::SimpleDMatrix>
GetDMatrixFromData(const std::vector<float> &x, int num_rows, int num_columns) {
  data::DenseAdapter adapter(x.data(), num_rows, num_columns);
  return std::shared_ptr<data::SimpleDMatrix>(new data::SimpleDMatrix(
      &adapter, std::numeric_limits<float>::quiet_NaN(), 1));
}

inline std::shared_ptr<DMatrix> GetExternalMemoryDMatrixFromData(
    const std::vector<float>& x, int num_rows, int num_columns,
    size_t page_size, const dmlc::TemporaryDirectory& tempdir) {
  // Create the svm file in a temp dir
  const std::string tmp_file = tempdir.path + "/temp.libsvm";
  std::ofstream fo(tmp_file.c_str());
  for (auto i = 0; i < num_rows; i++) {
    std::stringstream row_data;
    for (auto j = 0; j < num_columns; j++) {
      row_data << 1 << " " << j << ":" << std::setprecision(15)
               << x[i * num_columns + j];
    }
    fo << row_data.str() << "\n";
  }
  fo.close();
  return std::shared_ptr<DMatrix>(DMatrix::Load(
      tmp_file + "#" + tmp_file + ".cache", true, false, "auto", page_size));
}

// Test that elements are approximately equally distributed among bins
inline void TestBinDistribution(const HistogramCuts &cuts, int column_idx,
                                const std::vector<float> &sorted_column,
                                const std::vector<float> &sorted_weights,
                                int num_bins) {
  std::map<int, int> bin_weights;
  for (auto i = 0ull; i < sorted_column.size(); i++) {
    bin_weights[cuts.SearchBin(sorted_column[i], column_idx)] += sorted_weights[i];
  }
  int local_num_bins = cuts.Ptrs()[column_idx + 1] - cuts.Ptrs()[column_idx];
  auto total_weight = std::accumulate(sorted_weights.begin(), sorted_weights.end(),0);
  int expected_bin_weight = total_weight / local_num_bins;
  // Allow up to 30% deviation. This test is not very strict, it only ensures
  // roughly equal distribution
  int allowable_error = std::max(2, int(expected_bin_weight * 0.3));

  // First and last bin can have smaller
  for (auto& kv : bin_weights) {
    EXPECT_LE(std::abs(bin_weights[kv.first] - expected_bin_weight),
              allowable_error);
  }
}

// Test sketch quantiles against the real quantiles Not a very strict
// test
inline void TestRank(const std::vector<float> &column_cuts,
                     const std::vector<float> &sorted_x,
                     const std::vector<float> &sorted_weights) {
  double eps = 0.05;
  auto total_weight =
      std::accumulate(sorted_weights.begin(), sorted_weights.end(), 0.0);
  // Ignore the last cut, its special
  double sum_weight = 0.0;
  size_t j = 0;
  for (size_t i = 0; i < column_cuts.size() - 1; i++) {
    while (column_cuts[i] > sorted_x[j]) {
      sum_weight += sorted_weights[j];
      j++;
    }
    double expected_rank = ((i + 1) * total_weight) / column_cuts.size();
    double acceptable_error = std::max(2.9, total_weight * eps);
    EXPECT_LE(std::abs(expected_rank - sum_weight), acceptable_error);
  }
}

inline void ValidateColumn(const HistogramCuts& cuts, int column_idx,
                           const std::vector<float>& sorted_column,
                           const std::vector<float>& sorted_weights,
                           size_t num_bins) {

  // Check the endpoints are correct
  CHECK_GT(sorted_column.size(), 0);
  EXPECT_LT(cuts.MinValues().at(column_idx), sorted_column.front());
  EXPECT_GT(cuts.Values()[cuts.Ptrs()[column_idx]], sorted_column.front());
  EXPECT_GE(cuts.Values()[cuts.Ptrs()[column_idx+1]-1], sorted_column.back());

  // Check the cuts are sorted
  auto cuts_begin = cuts.Values().begin() + cuts.Ptrs()[column_idx];
  auto cuts_end = cuts.Values().begin() + cuts.Ptrs()[column_idx + 1];
  EXPECT_TRUE(std::is_sorted(cuts_begin, cuts_end));

  // Check all cut points are unique
  EXPECT_EQ(std::set<float>(cuts_begin, cuts_end).size(),
            cuts_end - cuts_begin);

  auto unique = std::set<float>(sorted_column.begin(), sorted_column.end());
  if (unique.size() <= num_bins) {
    // Less unique values than number of bins
    // Each value should get its own bin
    int i = 0;
    for (auto v : unique) {
      ASSERT_EQ(cuts.SearchBin(v, column_idx), cuts.Ptrs()[column_idx] + i);
      i++;
    }
  } else {
    int num_cuts_column = cuts.Ptrs()[column_idx + 1] - cuts.Ptrs()[column_idx];
    std::vector<float> column_cuts(num_cuts_column);
    std::copy(cuts.Values().begin() + cuts.Ptrs()[column_idx],
      cuts.Values().begin() + cuts.Ptrs()[column_idx + 1],
      column_cuts.begin());
    TestBinDistribution(cuts, column_idx, sorted_column, sorted_weights, num_bins);
    TestRank(column_cuts, sorted_column, sorted_weights);
  }
}

inline void ValidateCuts(const HistogramCuts& cuts, DMatrix* dmat,
                         int num_bins) {
  // Collect data into columns
  std::vector<std::vector<float>> columns(dmat->Info().num_col_);
  for (auto& batch : dmat->GetBatches<SparsePage>()) {
    CHECK_GT(batch.Size(), 0);
    for (auto i = 0ull; i < batch.Size(); i++) {
      for (auto e : batch[i]) {
        columns[e.index].push_back(e.fvalue);
      }
    }
  }
  // Sort
  for (auto i = 0ull; i < columns.size(); i++) {
    auto& col = columns.at(i);
    const auto& w = dmat->Info().weights_.HostVector();
    std::vector<size_t > index(col.size());
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(),
              [=](size_t a, size_t b) { return col[a] < col[b]; });

    std::vector<float> sorted_column(col.size());
    std::vector<float> sorted_weights(col.size(), 1.0);
    for (auto j = 0ull; j < col.size(); j++) {
      sorted_column[j] = col[index[j]];
      if (w.size() == col.size()) {
        sorted_weights[j] = w[index[j]];
      }
    }

    ValidateColumn(cuts, i, sorted_column, sorted_weights, num_bins);
  }
}

}  // namespace common
}  // namespace xgboost
