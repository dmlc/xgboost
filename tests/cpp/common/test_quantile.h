/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef TESTS_CPP_COMMON_TEST_QUANTILE_H_
#define TESTS_CPP_COMMON_TEST_QUANTILE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "../../../src/common/hist_util.h"
#include "../../../src/common/quantile.h"
#include "../helpers.h"

namespace xgboost::common::quantile_test {
enum class WeightKind { kNone, kRow };

enum class DataKind { kClustered, kDuplicateHeavy, kExactUnique, kStaircaseMass };

enum class FeatureKind { kNumerical, kMixed };

inline constexpr double kMaxNormalizedRankError = 2.0;
inline constexpr double kMaxWeightedNormalizedRankError = 10.0;

struct SummaryCase {
  std::string name;
  std::size_t rows{0};
  bst_bin_t max_bin{0};
  DataKind data{DataKind::kClustered};
  WeightKind weights{WeightKind::kNone};
  std::uint32_t seed{0};
};

struct GeneratedColumn {
  std::vector<float> values;
  std::vector<float> weights;
};

struct WeightedValue {
  float value;
  double weight;
};

struct ReferenceColumn {
  std::vector<float> values;
  std::vector<double> prefix_weights;
};

struct ContainerCase {
  std::string name;
  std::size_t rows{0};
  std::size_t cols{0};
  float sparsity{0.0f};
  bst_bin_t max_bin{0};
  WeightKind weights{WeightKind::kNone};
  FeatureKind features{FeatureKind::kNumerical};
  std::uint32_t seed{0};
};

inline bool IsExactUniqueCase(SummaryCase const& c) { return c.data == DataKind::kExactUnique; }

inline std::string SummaryCaseName(testing::TestParamInfo<SummaryCase> const& info) {
  return info.param.name;
}

inline std::string ContainerCaseName(testing::TestParamInfo<ContainerCase> const& info) {
  return info.param.name;
}

inline std::vector<SummaryCase> SummaryAnchorCases() {
  return {
      {"empty_unweighted", 0, 16, DataKind::kClustered, WeightKind::kNone, 10},
      {"clustered_unweighted_small", 128, 16, DataKind::kClustered, WeightKind::kNone, 0},
      {"clustered_weighted_small", 128, 16, DataKind::kClustered, WeightKind::kRow, 1},
      {"duplicate_weighted_small", 128, 16, DataKind::kDuplicateHeavy, WeightKind::kRow, 2},
      {"staircase_unweighted_large", 4096, 16, DataKind::kStaircaseMass, WeightKind::kNone, 5},
      {"clustered_weighted_large", 4096, 16, DataKind::kClustered, WeightKind::kRow, 6},
      {"duplicate_weighted_large", 4096, 16, DataKind::kDuplicateHeavy, WeightKind::kRow, 7},
      {"clustered_unweighted_wide_budget_gap", 16384, 32, DataKind::kClustered, WeightKind::kNone,
       8},
      {"staircase_weighted_wide_budget_gap", 16384, 32, DataKind::kStaircaseMass, WeightKind::kRow,
       9},
      {"exact_unique_unweighted", 16, 16, DataKind::kExactUnique, WeightKind::kNone, 3},
      {"exact_unique_weighted", 16, 16, DataKind::kExactUnique, WeightKind::kRow, 4},
  };
}

inline std::vector<SummaryCase> SummaryRandomCases(std::size_t n_cases) {
  std::vector<SummaryCase> cases;
  cases.reserve(n_cases);

  SimpleLCG lcg;
  auto const data_kinds = std::vector<DataKind>{DataKind::kClustered, DataKind::kDuplicateHeavy,
                                                DataKind::kExactUnique, DataKind::kStaircaseMass};
  auto const weight_kinds = std::vector<WeightKind>{WeightKind::kNone, WeightKind::kRow};
  auto const max_bins_pool = std::vector<bst_bin_t>{8, 16, 32, 64};
  auto const rows_pool = std::vector<std::size_t>{256, 1024, 4096, 16384, 65536};

  for (std::size_t i = 0; i < n_cases; ++i) {
    auto data = data_kinds[lcg() % data_kinds.size()];
    auto weights = weight_kinds[lcg() % weight_kinds.size()];
    auto max_bin = max_bins_pool[lcg() % max_bins_pool.size()];
    auto rows = rows_pool[lcg() % rows_pool.size()];

    if (data == DataKind::kExactUnique) {
      rows = std::min<std::size_t>(rows, max_bin);
      rows = std::max<std::size_t>(rows, 1);
    }

    auto seed = static_cast<std::uint32_t>(lcg() % std::numeric_limits<std::uint32_t>::max());
    cases.push_back(
        {std::string("random_") + std::to_string(i), rows, max_bin, data, weights, seed});
  }

  return cases;
}

inline std::vector<ContainerCase> ContainerAnchorCases() {
  return {
      {"empty_numeric_bins16", 0, 32, 0.0f, 16, WeightKind::kNone, FeatureKind::kNumerical, 10},
      {"dense_numeric_unweighted_bins2", 256, 32, 0.0f, 2, WeightKind::kNone,
       FeatureKind::kNumerical, 11},
      {"dense_numeric_unweighted_bins16", 256, 32, 0.0f, 16, WeightKind::kNone,
       FeatureKind::kNumerical, 12},
      {"dense_numeric_weighted_bins256", 512, 32, 0.0f, 256, WeightKind::kRow,
       FeatureKind::kNumerical, 13},
      {"sparse_numeric_weighted_bins32", 512, 48, 0.7f, 32, WeightKind::kRow,
       FeatureKind::kNumerical, 14},
      {"dense_mixed_unweighted_bins16", 256, 24, 0.0f, 16, WeightKind::kNone, FeatureKind::kMixed,
       15},
      {"sparse_mixed_weighted_bins64", 512, 40, 0.8f, 64, WeightKind::kRow, FeatureKind::kMixed,
       16},
  };
}

inline std::vector<FeatureType> FeatureTypes(ContainerCase const& c) {
  std::vector<FeatureType> ft(c.cols, FeatureType::kNumerical);
  if (c.features == FeatureKind::kMixed) {
    for (std::size_t i = 0; i < ft.size(); ++i) {
      ft[i] = (i % 2 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
    }
  }
  return ft;
}

inline std::vector<float> GenerateWeights(std::size_t rows, std::uint32_t seed) {
  std::vector<float> weights(rows, 1.0f);
  SimpleLCG lcg{seed};
  SimpleRealUniformDistribution<float> unit_dist(0.0f, 1.0f);
  std::generate(weights.begin(), weights.end(), [&] { return std::exp(6.0f * unit_dist(&lcg)); });
  return weights;
}

inline auto CollectWeightedColumns(DMatrix* dmat) -> std::vector<std::vector<WeightedValue>> {
  std::vector<std::vector<WeightedValue>> columns(dmat->Info().num_col_);
  if (dmat->Info().num_row_ == 0) {
    return columns;
  }
  std::vector<float> weights = dmat->Info().group_ptr_.empty()
                                   ? dmat->Info().weights_.HostVector()
                                   : detail::UnrollGroupWeights(dmat->Info());

  bst_idx_t row_idx{0};
  Context ctx;
  for (auto const& batch : dmat->GetBatches<SparsePage>(&ctx)) {
    auto page = batch.GetView();
    for (std::size_t i = 0; i < batch.Size(); ++i) {
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

inline auto AggregateWeightedColumn(std::vector<WeightedValue> const& sorted_column)
    -> ReferenceColumn {
  ReferenceColumn ref;
  ref.prefix_weights.push_back(0.0);
  for (auto const& entry : sorted_column) {
    if (!ref.values.empty() && ref.values.back() == entry.value) {
      ref.prefix_weights.back() += entry.weight;
    } else {
      ref.values.push_back(entry.value);
      ref.prefix_weights.push_back(ref.prefix_weights.back() + entry.weight);
    }
  }
  return ref;
}

inline double DistanceToInterval(double target, double lo, double hi) {
  if (target < lo) {
    return lo - target;
  }
  if (target > hi) {
    return target - hi;
  }
  return 0.0;
}

struct CutRankErrorSummary {
  double max_normalized_error{0.0};
  double max_absolute_error{0.0};
  double target_rank{0.0};
  double rank_lo{0.0};
  double rank_hi{0.0};
  double total_weight{0.0};
  bst_feature_t feature{0};
  std::size_t cut_index{0};
  std::size_t num_interior_cuts{0};
};

inline auto MeasureCutRankError(HistogramCuts const& cuts, bst_feature_t column_idx,
                                ReferenceColumn const& ref) -> CutRankErrorSummary {
  CutRankErrorSummary summary;
  summary.feature = column_idx;
  if (ref.values.empty()) {
    return summary;
  }

  auto beg = cuts.Ptrs()[column_idx];
  auto end = cuts.Ptrs()[column_idx + 1];
  auto num_cuts = end - beg;
  if (num_cuts <= 1) {
    return summary;
  }
  summary.num_interior_cuts = num_cuts - 1;  // Final cut is the sentinel upper bound.
  summary.total_weight = ref.prefix_weights.back();
  if (summary.total_weight == 0.0 || summary.num_interior_cuts == 0) {
    return summary;
  }

  auto avg_bin_weight = summary.total_weight / static_cast<double>(summary.num_interior_cuts);
  for (std::size_t cut_idx = 0; cut_idx < summary.num_interior_cuts; ++cut_idx) {
    auto cut_value = cuts.Values()[beg + cut_idx];
    auto lb = std::lower_bound(ref.values.cbegin(), ref.values.cend(), cut_value);
    auto ub = std::upper_bound(ref.values.cbegin(), ref.values.cend(), cut_value);
    auto rank_lo = ref.prefix_weights[std::distance(ref.values.cbegin(), lb)];
    auto rank_hi = ref.prefix_weights[std::distance(ref.values.cbegin(), ub)];
    auto target_rank = static_cast<double>(cut_idx + 1) * summary.total_weight /
                       static_cast<double>(summary.num_interior_cuts);
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

inline void ValidateNumericalCuts(HistogramCuts const& cuts, bst_feature_t column_idx,
                                  std::vector<WeightedValue> const& sorted_column,
                                  std::size_t num_bins, double max_normalized_rank_error) {
  auto ref = AggregateWeightedColumn(sorted_column);
  CHECK(!ref.values.empty());

  auto beg = cuts.Ptrs()[column_idx];
  auto end = cuts.Ptrs()[column_idx + 1];
  auto first_bin = HistogramCuts::NumericBinLowerBound(cuts.Ptrs(), cuts.Values(), column_idx, beg);
  EXPECT_TRUE(std::isinf(first_bin));
  EXPECT_LT(first_bin, 0.0f);
  EXPECT_GT(cuts.Values()[beg], ref.values.front());
  EXPECT_GE(cuts.Values()[end - 1], ref.values.back());

  if (ref.values.size() <= num_bins) {
    for (std::size_t i = 0; i < ref.values.size(); ++i) {
      ASSERT_EQ(cuts.SearchBin(ref.values[i], column_idx), beg + i)
          << "feature=" << column_idx << ", value_index=" << i;
    }
  } else {
    auto stats = MeasureCutRankError(cuts, column_idx, ref);
    EXPECT_LE(stats.max_normalized_error, max_normalized_rank_error)
        << "feature=" << column_idx << ", cut=" << stats.cut_index
        << ", normalized_error=" << stats.max_normalized_error
        << ", absolute_error=" << stats.max_absolute_error << ", target_rank=" << stats.target_rank
        << ", rank_lo=" << stats.rank_lo << ", rank_hi=" << stats.rank_hi
        << ", total_weight=" << stats.total_weight
        << ", num_interior_cuts=" << stats.num_interior_cuts;
  }
}

inline void ValidateCategoricalCuts(HistogramCuts const& cuts, bst_feature_t column_idx,
                                    std::vector<WeightedValue> const& sorted_column) {
  std::vector<float> categories;
  categories.reserve(sorted_column.size());
  for (auto const& entry : sorted_column) {
    categories.push_back(entry.value);
  }
  std::sort(categories.begin(), categories.end());
  categories.erase(std::unique(categories.begin(), categories.end()), categories.end());

  auto beg = cuts.Ptrs()[column_idx];
  auto end = cuts.Ptrs()[column_idx + 1];
  ASSERT_EQ(static_cast<std::size_t>(end - beg), categories.size()) << "feature=" << column_idx;
  for (std::size_t i = 0; i < categories.size(); ++i) {
    EXPECT_EQ(cuts.Values()[beg + i], categories[i]) << "feature=" << column_idx;
  }
}

inline void ValidateContainerCuts(ContainerCase const& c, HistogramCuts const& cuts, DMatrix* dmat,
                                  std::vector<std::vector<WeightedValue>> const& columns,
                                  std::size_t f_begin = 0,
                                  std::size_t f_end = std::numeric_limits<std::size_t>::max()) {
  ASSERT_EQ(cuts.Ptrs().size(), c.cols + 1) << "case=" << c.name;
  auto ft = dmat->Info().feature_types.ConstHostSpan();
  auto max_error =
      c.weights == WeightKind::kRow ? kMaxWeightedNormalizedRankError : kMaxNormalizedRankError;
  f_end = std::min(f_end, columns.size());
  for (std::size_t i = f_begin; i < f_end; ++i) {
    auto beg = cuts.Ptrs()[i];
    auto end = cuts.Ptrs()[i + 1];
    ASSERT_LT(beg, end) << "case=" << c.name << ", feature=" << i;
    for (auto j = beg + 1; j < end; ++j) {
      EXPECT_LT(cuts.Values()[j - 1], cuts.Values()[j]) << "case=" << c.name << ", feature=" << i;
    }
    if (columns[i].empty()) {
      continue;
    }
    if (!ft.empty() && IsCat(ft, i)) {
      ValidateCategoricalCuts(cuts, i, columns[i]);
    } else {
      ValidateNumericalCuts(cuts, i, columns[i], c.max_bin, max_error);
    }
  }
}

inline GeneratedColumn GenerateSummaryColumn(SummaryCase const& c) {
  GeneratedColumn out;
  out.values.resize(c.rows);
  out.weights.resize(c.rows, 1.0f);

  SimpleLCG lcg{c.seed};

  switch (c.data) {
    case DataKind::kClustered: {
      SimpleRealUniformDistribution<float> jitter(-1e-4f, 1e-4f);
      for (std::size_t i = 0; i < c.rows; ++i) {
        auto base = static_cast<float>(lcg() % 4);
        out.values[i] = base + jitter(&lcg);
      }
      break;
    }
    case DataKind::kDuplicateHeavy: {
      std::size_t buckets = std::min<std::size_t>(8, std::max<std::size_t>(1, c.max_bin / 2));
      for (std::size_t i = 0; i < c.rows; ++i) {
        out.values[i] = static_cast<float>(i % buckets);
      }
      break;
    }
    case DataKind::kExactUnique: {
      CHECK_LE(c.rows, static_cast<std::size_t>(c.max_bin));
      std::iota(out.values.begin(), out.values.end(), 0.0f);
      for (std::size_t i = out.values.size(); i > 1; --i) {
        auto j = lcg() % i;
        std::swap(out.values[i - 1], out.values[j]);
      }
      break;
    }
    case DataKind::kStaircaseMass: {
      for (std::size_t i = 0; i < c.rows; ++i) {
        out.values[i] =
            static_cast<float>(i) / static_cast<float>(std::max<std::size_t>(c.rows, 1));
      }
      break;
    }
  }

  if (c.weights == WeightKind::kRow) {
    switch (c.data) {
      case DataKind::kClustered: {
        SimpleRealUniformDistribution<float> unit_dist(0.0f, 1.0f);
        std::generate(out.weights.begin(), out.weights.end(),
                      [&] { return std::exp(6.0f * unit_dist(&lcg)); });
        break;
      }
      case DataKind::kDuplicateHeavy: {
        SimpleRealUniformDistribution<float> unit_dist(0.0f, 1.0f);
        std::generate(out.weights.begin(), out.weights.end(),
                      [&] { return unit_dist(&lcg) < 0.01f ? 1000.0f : 1.0f; });
        break;
      }
      case DataKind::kExactUnique: {
        SimpleRealUniformDistribution<float> unit_dist(0.0f, 1.0f);
        std::generate(out.weights.begin(), out.weights.end(),
                      [&] { return std::exp(6.0f * unit_dist(&lcg)); });
        break;
      }
      case DataKind::kStaircaseMass: {
        auto period = std::max<std::size_t>(2, static_cast<std::size_t>(c.max_bin));
        for (std::size_t i = 0; i < c.rows; ++i) {
          auto phase = i % period;
          auto exponent = static_cast<float>((phase * 8) / period);
          out.weights[i] = std::exp2(exponent);
        }
        break;
      }
    }
  }

  return out;
}

inline ReferenceColumn AggregateReferenceColumn(GeneratedColumn const& col) {
  std::vector<std::pair<float, double>> pairs;
  pairs.reserve(col.values.size());
  for (std::size_t i = 0; i < col.values.size(); ++i) {
    if (col.weights[i] == 0.0f) {
      continue;
    }
    pairs.emplace_back(col.values[i], static_cast<double>(col.weights[i]));
  }
  std::sort(pairs.begin(), pairs.end(),
            [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });

  std::vector<WeightedValue> out;
  for (auto const& [value, weight] : pairs) {
    if (!out.empty() && out.back().value == value) {
      out.back().weight += weight;
    } else {
      out.push_back({value, weight});
    }
  }

  ReferenceColumn ref;
  ref.values.reserve(out.size());
  ref.prefix_weights.reserve(out.size() + 1);
  ref.prefix_weights.push_back(0.0);
  for (auto const& v : out) {
    ref.values.push_back(v.value);
    ref.prefix_weights.push_back(ref.prefix_weights.back() + v.weight);
  }
  return ref;
}

inline double TotalWeight(ReferenceColumn const& col) { return col.prefix_weights.back(); }

inline std::size_t UniqueValueCount(ReferenceColumn const& col) { return col.values.size(); }

inline bool EmptyReference(ReferenceColumn const& col) { return col.values.empty(); }

inline Span<float const> ExactValues(ReferenceColumn const& col) {
  return {col.values.data(), col.values.size()};
}

inline std::size_t NonZeroWeightCount(GeneratedColumn const& col) {
  return std::count_if(col.weights.cbegin(), col.weights.cend(),
                       [](float w) { return w != static_cast<float>(0); });
}

template <typename Summary>
inline float QuerySummaryValue(Summary const& summary, double rank) {
  auto entries = summary.Entries();
  CHECK_GE(entries.size(), 1);
  if (entries.size() == 1) {
    return entries.front().value;
  }

  auto rank2 = static_cast<double>(2.0) * rank;
  std::size_t query_cursor = 0;
  while (query_cursor < entries.size() - 2 &&
         rank2 >=
             static_cast<double>(entries[query_cursor + 1].rmin + entries[query_cursor + 1].rmax)) {
    ++query_cursor;
  }
  auto left = entries[query_cursor];
  auto right = entries[query_cursor + 1];
  auto threshold = static_cast<double>(left.RMinNext() + right.RMaxPrev());
  return rank2 < threshold ? left.value : right.value;
}

inline double RankErrorForValue(ReferenceColumn const& col, double target_rank, float queried) {
  auto lo_it = std::lower_bound(col.values.cbegin(), col.values.cend(), queried);
  auto hi_it = std::upper_bound(col.values.cbegin(), col.values.cend(), queried);
  auto lo_idx = static_cast<std::size_t>(std::distance(col.values.cbegin(), lo_it));
  auto hi_idx = static_cast<std::size_t>(std::distance(col.values.cbegin(), hi_it));
  auto rank_lo = col.prefix_weights[lo_idx];
  auto rank_hi = col.prefix_weights[hi_idx];

  if (target_rank < rank_lo) {
    return rank_lo - target_rank;
  }
  if (target_rank > rank_hi) {
    return target_rank - rank_hi;
  }
  return 0.0;
}

template <typename Summary>
double MaxSummaryQueryRankError(Summary const& summary, ReferenceColumn const& reference,
                                std::size_t num_queries) {
  auto total = TotalWeight(reference);
  CHECK_GT(total, 0.0);
  double max_error = 0.0;
  for (std::size_t i = 1; i < num_queries; ++i) {
    auto target = static_cast<double>(i) * total / static_cast<double>(num_queries);
    auto queried = QuerySummaryValue(summary, target);
    max_error = std::max(max_error, RankErrorForValue(reference, target, queried));
  }
  return max_error;
}
}  // namespace xgboost::common::quantile_test

#endif  // TESTS_CPP_COMMON_TEST_QUANTILE_H_
