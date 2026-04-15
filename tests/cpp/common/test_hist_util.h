/**
 * Copyright 2019-2025, XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>  // for shared_ptr
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../../../src/common/hist_util.h"
#include "../../../src/data/adapter.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../helpers.h"
#include "test_quantile.h"

#ifdef __CUDACC__
#include <xgboost/json.h>

#include "../../../src/data/device_adapter.cuh"
#endif  // __CUDACC__

// Some helper functions used to test both GPU and CPU algorithms
//
namespace xgboost::common {
// Generate columns with different ranges
inline std::vector<float> GenerateRandom(int num_rows, int num_columns) {
  std::vector<float> x(num_rows * num_columns);
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
inline data::CupyAdapter AdapterFromData(const thrust::device_vector<float>& x, int num_rows,
                                         int num_columns) {
  Json array_interface{Object()};
  std::vector<Json> shape = {Json(static_cast<Integer::Int>(num_rows)),
                             Json(static_cast<Integer::Int>(num_columns))};
  array_interface["shape"] = Array(shape);
  std::vector<Json> j_data{Json(Integer(reinterpret_cast<Integer::Int>(x.data().get()))),
                           Json(Boolean(false))};
  array_interface["data"] = j_data;
  array_interface["version"] = 3;
  array_interface["typestr"] = String("<f4");
  std::string str;
  Json::Dump(array_interface, &str);
  return data::CupyAdapter(str);
}
#endif

inline std::shared_ptr<data::SimpleDMatrix> GetDMatrixFromData(const std::vector<float>& x,
                                                               int num_rows, int num_columns) {
  data::DenseAdapter adapter(x.data(), num_rows, num_columns);
  return std::shared_ptr<data::SimpleDMatrix>(
      new data::SimpleDMatrix(&adapter, std::numeric_limits<float>::quiet_NaN(), 1));
}

inline void ValidateCuts(const HistogramCuts& cuts, DMatrix* dmat, int num_bins) {
  auto columns = quantile_test::CollectWeightedColumns(dmat);
  auto ft = dmat->Info().feature_types.ConstHostSpan();
  auto max_normalized_rank_error = dmat->Info().weights_.Empty() && dmat->Info().group_ptr_.empty()
                                       ? quantile_test::kMaxNormalizedRankError
                                       : quantile_test::kMaxWeightedNormalizedRankError;

  for (auto i = 0ull; i < columns.size(); i++) {
    if (columns[i].empty()) {
      continue;
    }
    if (!ft.empty() && IsCat(ft, i)) {
      quantile_test::ValidateCategoricalCuts(cuts, i, columns[i]);
    } else {
      auto beg = cuts.Values().cbegin() + cuts.Ptrs().at(i);
      auto end = cuts.Values().cbegin() + cuts.Ptrs().at(i + 1);
      EXPECT_TRUE(std::is_sorted(beg, end));
      EXPECT_EQ(std::adjacent_find(beg, end), end);
      quantile_test::ValidateNumericalCuts(cuts, i, columns.at(i), num_bins,
                                           max_normalized_rank_error);
    }
  }
}

}  // namespace xgboost::common
