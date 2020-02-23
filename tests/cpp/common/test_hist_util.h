#pragma once
#include <cstddef>
#include <random>
#include <fstream>
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>

#include <random>
#include <vector>
#include <string>
#include <fstream>
#include "../../../src/common/hist_util.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../../../src/data/adapter.h"

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

inline std::shared_ptr<data::SimpleDMatrix> GetDMatrixFromData(const std::vector<float>& x, int num_rows, int num_columns) {
  data::DenseAdapter adapter(x.data(), num_rows, num_columns);
  return std::shared_ptr<data::SimpleDMatrix>(new data::SimpleDMatrix(
      &adapter, std::numeric_limits<float>::quiet_NaN(),
                             1));
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
inline void TestBinDistribution(const HistogramCuts& cuts, int column_idx,
                                const std::vector<float>& column,
                                int num_bins) {
  std::map<int, int> counts;
  for (auto& v : column) {
    counts[cuts.SearchBin(v, column_idx)]++;
  }
  int local_num_bins = cuts.Ptrs()[column_idx + 1] - cuts.Ptrs()[column_idx];
  int expected_num_elements = column.size() / local_num_bins;
  // Allow about 30% deviation. This test is not very strict, it only ensures
  // roughly equal distribution
  int allowable_error = std::max(2, int(expected_num_elements * 0.3));

  // First and last bin can have smaller
  for (auto& kv : counts) {
    EXPECT_LE(std::abs(counts[kv.first] - expected_num_elements),
              allowable_error);
  }
}

  // Test sketch quantiles against the real quantiles
  // Not a very strict test
inline void TestRank(const std::vector<float>& cuts,
              const std::vector<float>& sorted_x) {
  float eps = 0.05;
  // Ignore the last cut, its special
  size_t j = 0;
  for (size_t i = 0; i < cuts.size() - 1; i++) {
    int expected_rank = ((i+1) * sorted_x.size()) / cuts.size();
    while (cuts[i] > sorted_x[j]) {
      j++;
    }
    int actual_rank = j;
    int acceptable_error = std::max(2, int(sorted_x.size() * eps));
    ASSERT_LE(std::abs(expected_rank - actual_rank), acceptable_error);
  }
}

inline void ValidateColumn(const HistogramCuts& cuts, int column_idx,
                           const std::vector<float>& column,
                           size_t num_bins) {
  std::vector<float> sorted_column(column);
  std::sort(sorted_column.begin(), sorted_column.end());

  // Check the endpoints are correct
  EXPECT_LT(cuts.MinValues()[column_idx], sorted_column.front());
  EXPECT_GT(cuts.Values()[cuts.Ptrs()[column_idx]], sorted_column.front());
  EXPECT_GE(cuts.Values()[cuts.Ptrs()[column_idx+1]-1], sorted_column.back());

  // Check the cuts are sorted
  auto cuts_begin = cuts.Values().begin() + cuts.Ptrs()[column_idx];
  auto cuts_end = cuts.Values().begin() + cuts.Ptrs()[column_idx + 1];
  EXPECT_TRUE(std::is_sorted(cuts_begin, cuts_end));

  // Check all cut points are unique
  EXPECT_EQ(std::set<float>(cuts_begin, cuts_end).size(),
            cuts_end - cuts_begin);

  if (sorted_column.size() <= num_bins) {
    // Less unique values than number of bins
    // Each value should get its own bin

    // First check the inputs are unique
    int num_unique =
        std::set<float>(sorted_column.begin(), sorted_column.end()).size();
    EXPECT_EQ(num_unique, sorted_column.size());
    for (auto i = 0ull; i < sorted_column.size(); i++) {
      ASSERT_EQ(cuts.SearchBin(sorted_column[i], column_idx),
                cuts.Ptrs()[column_idx] + i);
    }
  }
  int num_cuts_column = cuts.Ptrs()[column_idx + 1] - cuts.Ptrs()[column_idx];
  std::vector<float> column_cuts(num_cuts_column);
  std::copy(cuts.Values().begin() + cuts.Ptrs()[column_idx],
            cuts.Values().begin() + cuts.Ptrs()[column_idx + 1],
            column_cuts.begin());
  TestBinDistribution(cuts, column_idx, sorted_column, num_bins);
  TestRank(column_cuts, sorted_column);
}

// x is dense and row major
inline void ValidateCuts(const HistogramCuts& cuts, std::vector<float>& x,
                         int num_rows, int num_columns,
                         int num_bins) {
   for (auto i = 0; i < num_columns; i++) {
     // Extract the column
     std::vector<float> column(num_rows);
     for (auto j = 0; j < num_rows; j++) {
       column[j] = x[j*num_columns + i];
     }
     ValidateColumn(cuts,i, column, num_bins);
   }
}

}  // namespace common
}  // namespace xgboost
