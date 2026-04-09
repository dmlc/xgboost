/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#ifndef TESTS_CPP_COMMON_TEST_QUANTILE_HELPERS_H_
#define TESTS_CPP_COMMON_TEST_QUANTILE_HELPERS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "../../../src/common/quantile.h"
#include "../helpers.h"

namespace xgboost::common::quantile_test {
enum class WeightKind { kNone, kRow };

enum class DataKind { kClustered, kDuplicateHeavy, kExactUnique, kStaircaseMass };

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

inline bool IsExactUniqueCase(SummaryCase const& c) { return c.data == DataKind::kExactUnique; }

inline std::string CaseName(testing::TestParamInfo<SummaryCase> const& info) {
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

inline std::vector<SummaryCase> ExactUniqueAnchorCases() {
  auto cases = SummaryAnchorCases();
  cases.erase(std::remove_if(cases.begin(), cases.end(),
                             [](SummaryCase const& c) { return !IsExactUniqueCase(c); }),
              cases.end());
  return cases;
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
inline double MaxSummaryQueryRankError(Summary const& summary, ReferenceColumn const& reference,
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

#endif  // TESTS_CPP_COMMON_TEST_QUANTILE_HELPERS_H_
