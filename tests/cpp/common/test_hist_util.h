/**
 * Copyright 2019-2025, XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>  // for shared_ptr
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "../../../src/common/hist_util.h"
#include "../../../src/data/adapter.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../helpers.h"

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

struct WeightedValue {
  float value{0.0f};
  double weight{0.0};
};

struct RankErrorSummary {
  double max_normalized_error{0.0};
  double max_absolute_error{0.0};
  double target_rank{0.0};
  double rank_lo{0.0};
  double rank_hi{0.0};
  double total_weight{0.0};
  bst_feature_t feature{0};
  std::size_t cut_index{0};
  std::size_t num_cuts{0};
};

inline constexpr double kMaxNormalizedRankError = 2.0;
inline constexpr double kMaxWeightedNormalizedRankError = 14.0;

inline double DistanceToInterval(double target, double lo, double hi) {
  if (target < lo) {
    return lo - target;
  }
  if (target > hi) {
    return target - hi;
  }
  return 0.0;
}

inline auto CollectWeightedColumns(DMatrix* dmat) -> std::vector<std::vector<WeightedValue>> {
  std::vector<std::vector<WeightedValue>> columns(dmat->Info().num_col_);
  std::vector<float> weights = dmat->Info().group_ptr_.empty()
                                   ? dmat->Info().weights_.HostVector()
                                   : detail::UnrollGroupWeights(dmat->Info());

  bst_idx_t row_idx{0};
  for (auto& batch : dmat->GetBatches<SparsePage>()) {
    auto page = batch.GetView();
    CHECK_GT(batch.Size(), 0ul);
    for (auto i = 0ull; i < batch.Size(); ++i) {
      auto row_weight =
          weights.empty() ? 1.0
                          : static_cast<double>(weights.at(static_cast<std::size_t>(row_idx + i)));
      for (auto e : page[i]) {
        columns[e.index].push_back({e.fvalue, row_weight});
      }
    }
    row_idx += batch.Size();
  }
  CHECK_EQ(row_idx, dmat->Info().num_row_);

  for (auto& column : columns) {
    std::sort(column.begin(), column.end(),
              [](auto const& lhs, auto const& rhs) { return lhs.value < rhs.value; });
  }
  return columns;
}

inline auto MeasureRankError(const HistogramCuts& cuts, bst_feature_t column_idx,
                             std::vector<WeightedValue> const& sorted_column) -> RankErrorSummary {
  RankErrorSummary summary;
  summary.feature = column_idx;
  summary.num_cuts = cuts.Ptrs()[column_idx + 1] - cuts.Ptrs()[column_idx];
  if (summary.num_cuts <= 1 || sorted_column.empty()) {
    return summary;
  }

  std::vector<double> prefix_sum(sorted_column.size() + 1, 0.0);
  for (std::size_t i = 0; i < sorted_column.size(); ++i) {
    prefix_sum[i + 1] = prefix_sum[i] + sorted_column[i].weight;
  }
  summary.total_weight = prefix_sum.back();
  if (summary.total_weight == 0.0) {
    return summary;
  }

  auto const cuts_begin = cuts.Values().cbegin() + cuts.Ptrs()[column_idx];
  auto const cuts_end = cuts.Values().cbegin() + cuts.Ptrs()[column_idx + 1];
  auto avg_bin_weight = summary.total_weight / static_cast<double>(summary.num_cuts);

  std::size_t cut_idx = 0;
  for (auto cut_it = cuts_begin; cut_it + 1 != cuts_end; ++cut_it, ++cut_idx) {
    auto cut_value = *cut_it;
    auto lb = std::lower_bound(sorted_column.cbegin(), sorted_column.cend(), cut_value,
                               [](auto const& lhs, float rhs) { return lhs.value < rhs; });
    auto ub = std::upper_bound(sorted_column.cbegin(), sorted_column.cend(), cut_value,
                               [](float lhs, auto const& rhs) { return lhs < rhs.value; });
    auto rank_lo = prefix_sum[std::distance(sorted_column.cbegin(), lb)];
    auto rank_hi = prefix_sum[std::distance(sorted_column.cbegin(), ub)];
    auto target_rank = static_cast<double>(cut_idx + 1) * summary.total_weight /
                       static_cast<double>(summary.num_cuts);
    auto absolute_error = DistanceToInterval(target_rank, rank_lo, rank_hi);
    auto normalized_error = absolute_error / avg_bin_weight;
    if (normalized_error > summary.max_normalized_error) {
      summary.max_normalized_error = normalized_error;
      summary.max_absolute_error = absolute_error;
      summary.target_rank = target_rank;
      summary.rank_lo = rank_lo;
      summary.rank_hi = rank_hi;
      summary.cut_index = cut_idx;
    }
  }

  return summary;
}

inline void ValidateColumn(const HistogramCuts& cuts, int column_idx,
                           std::vector<WeightedValue> const& sorted_column, size_t num_bins,
                           double max_normalized_rank_error) {
  // Check the endpoints are correct
  CHECK_GT(sorted_column.size(), 0);
  auto first_bin = common::HistogramCuts::NumericBinLowerBound(
      cuts.Ptrs(), cuts.Values(), column_idx, cuts.Ptrs().at(column_idx));
  EXPECT_TRUE(std::isinf(first_bin));
  EXPECT_LT(first_bin, 0.0f);
  EXPECT_GT(cuts.Values()[cuts.Ptrs()[column_idx]], sorted_column.front().value);
  EXPECT_GE(cuts.Values()[cuts.Ptrs()[column_idx + 1] - 1], sorted_column.back().value);

  // Check the cuts are sorted
  auto cuts_begin = cuts.Values().begin() + cuts.Ptrs()[column_idx];
  auto cuts_end = cuts.Values().begin() + cuts.Ptrs()[column_idx + 1];
  EXPECT_TRUE(std::is_sorted(cuts_begin, cuts_end));

  // Check all cut points are unique
  EXPECT_EQ(std::set<float>(cuts_begin, cuts_end).size(),
            static_cast<size_t>(cuts_end - cuts_begin));

  std::set<float> unique;
  for (auto const& entry : sorted_column) {
    unique.insert(entry.value);
  }
  auto const all_unit_weights = std::all_of(sorted_column.cbegin(), sorted_column.cend(),
                                            [](auto const& entry) { return entry.weight == 1.0; });
  if (unique.size() <= num_bins && all_unit_weights) {
    // Less unique values than number of bins
    // Each value should get its own bin
    int i = 0;
    for (auto v : unique) {
      ASSERT_EQ(cuts.SearchBin(v, column_idx), cuts.Ptrs()[column_idx] + i);
      i++;
    }
  } else {
    auto stats = MeasureRankError(cuts, column_idx, sorted_column);
    EXPECT_LE(stats.max_normalized_error, max_normalized_rank_error)
        << "feature=" << column_idx << ", cut=" << stats.cut_index
        << ", normalized_error=" << stats.max_normalized_error
        << ", absolute_error=" << stats.max_absolute_error << ", target_rank=" << stats.target_rank
        << ", rank_lo=" << stats.rank_lo << ", rank_hi=" << stats.rank_hi
        << ", total_weight=" << stats.total_weight << ", num_cuts=" << stats.num_cuts;
  }
}

inline void ValidateCuts(const HistogramCuts& cuts, DMatrix* dmat, int num_bins,
                         double max_normalized_rank_error = kMaxNormalizedRankError) {
  auto columns = CollectWeightedColumns(dmat);
  auto ft = dmat->Info().feature_types.ConstHostSpan();

  for (auto i = 0ull; i < columns.size(); i++) {
    if (columns[i].empty()) {
      continue;
    }
    if (!ft.empty() && IsCat(ft, i)) {
      continue;
    }
    ValidateColumn(cuts, i, columns.at(i), num_bins, max_normalized_rank_error);
  }
}

inline void ValidateCutsGpu(const HistogramCuts& cuts, DMatrix* dmat, int num_bins) {
  auto max_rank_error =
      dmat->Info().weights_.Empty() ? kMaxNormalizedRankError : kMaxWeightedNormalizedRankError;
  ValidateCuts(cuts, dmat, num_bins, max_rank_error);
}

/**
 * \brief Test for sketching on categorical data.
 *
 * \param sketch Sketch function, can be on device or on host.
 */
template <typename Fn>
void TestCategoricalSketch(size_t n, size_t num_categories, int32_t num_bins, bool weighted,
                           Fn sketch) {
  auto x = GenerateRandomCategoricalSingleColumn(n, num_categories);
  auto dmat = GetDMatrixFromData(x, n, 1);
  dmat->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);

  if (weighted) {
    std::vector<float> weights(n, 0);
    SimpleLCG lcg;
    SimpleRealUniformDistribution<float> dist(0, 1);
    for (auto& v : weights) {
      v = dist(&lcg);
    }
    dmat->Info().weights_.HostVector() = weights;
  }

  ASSERT_EQ(dmat->Info().feature_types.Size(), 1);
  auto cuts = sketch(dmat.get(), num_bins);
  ASSERT_EQ(cuts.MaxCategory(), num_categories - 1);
  std::sort(x.begin(), x.end());
  auto n_uniques = std::unique(x.begin(), x.end()) - x.begin();
  ASSERT_NE(n_uniques, x.size());
  ASSERT_EQ(cuts.TotalBins(), n_uniques);
  ASSERT_EQ(n_uniques, num_categories);

  auto& values = cuts.cut_values_.HostVector();
  ASSERT_TRUE(std::is_sorted(values.cbegin(), values.cend()));
  auto is_unique = (std::unique(values.begin(), values.end()) - values.begin()) == n_uniques;
  ASSERT_TRUE(is_unique);

  x.resize(n_uniques);
  for (decltype(n_uniques) i = 0; i < n_uniques; ++i) {
    ASSERT_EQ(x[i], values[i]);
  }
}
}  // namespace xgboost::common
