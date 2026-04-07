/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#include "test_quantile.h"

#include <gtest/gtest.h>

#include <cstdint>  // for int64_t

#include "../../../src/collective/allreduce.h"
#include "../../../src/common/hist_util.h"
#include "../../../src/data/adapter.h"
#include "../collective/test_worker.h"  // for TestDistributedGlobal
#include "test_hist_util.h"
#include "xgboost/context.h"

namespace xgboost::common {
TEST(Quantile, LoadBalance) {
  size_t constexpr kRows = 1000, kCols = 100;
  auto m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();
  std::vector<bst_feature_t> cols_ptr;
  Context ctx;
  for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
    data::SparsePageAdapterBatch adapter{page.GetView()};
    cols_ptr = LoadBalance(adapter, page.data.Size(), kCols, 13, [](auto) { return true; });
  }
  size_t n_cols = 0;
  for (size_t i = 1; i < cols_ptr.size(); ++i) {
    n_cols += cols_ptr[i] - cols_ptr[i - 1];
  }
  CHECK_EQ(n_cols, kCols);
}

TEST(Quantile, InitWithEmptyColumn) {
  WQuantileSketch sketch{0, 0.1};

  auto out = sketch.GetSummary(1);
  ASSERT_EQ(out.Size(), 0);
}

TEST(Quantile, SetPruneInplace) {
  using Summary = WQSummary<>;
  using Entry = Summary::Entry;

  SimpleLCG lcg;
  for (size_t trial = 0; trial < 256; ++trial) {
    size_t n = (lcg() % 256) + 1;
    size_t max_size = (lcg() % n) + 1;

    std::vector<Entry> src_storage(n);
    float running_rank = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      float w = static_cast<float>((lcg() % 7) + 1);
      float value = static_cast<float>(i);
      src_storage[i] = Entry{running_rank, running_rank + w, w, value};
      running_rank += w;
    }

    std::vector<Entry> ref_storage(n);
    Summary src_ref{Span<Entry>{src_storage.data(), src_storage.size()}, n};
    Summary out_ref{Span<Entry>{ref_storage.data(), ref_storage.size()}, 0};
    out_ref.CopyFrom(src_ref);
    out_ref.SetPrune(max_size);

    Summary in_place{Span<Entry>{src_storage.data(), src_storage.size()}, n};
    in_place.SetPrune(max_size);

    ASSERT_EQ(in_place.Size(), out_ref.Size()) << "trial=" << trial;
    auto const in_entries = in_place.Entries();
    auto const ref_entries = out_ref.Entries();
    for (size_t i = 0; i < in_place.Size(); ++i) {
      EXPECT_FLOAT_EQ(in_entries[i].rmin, ref_entries[i].rmin) << "trial=" << trial;
      EXPECT_FLOAT_EQ(in_entries[i].rmax, ref_entries[i].rmax) << "trial=" << trial;
      EXPECT_FLOAT_EQ(in_entries[i].wmin, ref_entries[i].wmin) << "trial=" << trial;
      EXPECT_FLOAT_EQ(in_entries[i].value, ref_entries[i].value) << "trial=" << trial;
    }
  }
}

namespace {
struct QueryBoundStats {
  double max_absolute_error{0.0};
  double total_weight{0.0};
  double epsilon{0.0};
  double prune_term{0.0};
  double target_rank{0.0};
  double rank_lo{0.0};
  double rank_hi{0.0};
  float queried_value{0.0f};
  std::size_t query_index{0};
  std::size_t n_queries{0};
};

struct WeightedSketchScenario {
  char const* name{nullptr};
  bst_bin_t max_bins{0};
  std::vector<float> values;
  std::vector<float> weights;
};

auto AggregateWeightedData(std::vector<float> const& values, std::vector<float> const& weights)
    -> std::vector<WeightedValue> {
  CHECK_EQ(values.size(), weights.size());
  std::vector<WeightedValue> sorted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    sorted[i] = WeightedValue{values[i], weights[i]};
  }
  std::sort(sorted.begin(), sorted.end(),
            [](auto const& lhs, auto const& rhs) { return lhs.value < rhs.value; });

  std::vector<WeightedValue> unique;
  unique.reserve(sorted.size());
  for (auto const& entry : sorted) {
    if (!unique.empty() && unique.back().value == entry.value) {
      unique.back().weight += entry.weight;
    } else {
      unique.push_back(entry);
    }
  }
  return unique;
}

auto PrefixWeights(std::vector<WeightedValue> const& weighted_uniques) -> std::vector<double> {
  std::vector<double> prefix_sum(weighted_uniques.size() + 1, 0.0);
  for (std::size_t i = 0; i < weighted_uniques.size(); ++i) {
    prefix_sum[i + 1] = prefix_sum[i] + weighted_uniques[i].weight;
  }
  return prefix_sum;
}

auto ExactRankInterval(std::vector<WeightedValue> const& weighted_uniques,
                       std::vector<double> const& prefix_sum, float value)
    -> std::pair<double, double> {
  auto lb = std::lower_bound(weighted_uniques.cbegin(), weighted_uniques.cend(), value,
                             [](auto const& lhs, float rhs) { return lhs.value < rhs; });
  auto ub = std::upper_bound(weighted_uniques.cbegin(), weighted_uniques.cend(), value,
                             [](float lhs, auto const& rhs) { return lhs < rhs.value; });
  auto rank_lo = prefix_sum[std::distance(weighted_uniques.cbegin(), lb)];
  auto rank_hi = prefix_sum[std::distance(weighted_uniques.cbegin(), ub)];
  return {rank_lo, rank_hi};
}

auto MeasurePaperQueryBound(WQSummaryContainer const& summary,
                            std::vector<WeightedValue> const& weighted_uniques, double epsilon,
                            double prune_term) -> QueryBoundStats {
  QueryBoundStats stats;
  stats.epsilon = epsilon;
  stats.prune_term = prune_term;
  auto prefix_sum = PrefixWeights(weighted_uniques);
  stats.total_weight = prefix_sum.back();
  if (stats.total_weight == 0.0) {
    return stats;
  }

  auto const entries = summary.Entries();
  stats.n_queries = std::max<std::size_t>(1, entries.size() * 4);
  for (std::size_t i = 0; i <= stats.n_queries; ++i) {
    auto target_rank =
        static_cast<double>(i) * stats.total_weight / static_cast<double>(stats.n_queries);
    auto const& query = summary.Query(target_rank);
    auto [rank_lo, rank_hi] = ExactRankInterval(weighted_uniques, prefix_sum, query.value);
    auto absolute_error = DistanceToInterval(target_rank, rank_lo, rank_hi);
    if (absolute_error > stats.max_absolute_error) {
      stats.max_absolute_error = absolute_error;
      stats.target_rank = target_rank;
      stats.rank_lo = rank_lo;
      stats.rank_hi = rank_hi;
      stats.queried_value = query.value;
      stats.query_index = i;
    }
  }
  return stats;
}

auto BuildPushSummary(WeightedSketchScenario const& scenario) -> WQSummaryContainer {
  auto eps = SketchEpsilon(scenario.max_bins, scenario.values.size());
  auto budget = SketchSummaryBudget(scenario.max_bins, scenario.values.size());
  WQuantileSketch sketch{scenario.values.size(), eps};
  for (std::size_t i = 0; i < scenario.values.size(); ++i) {
    sketch.Push(scenario.values[i], scenario.weights[i]);
  }
  return sketch.GetSummary(budget);
}

auto BuildSortedSummary(WeightedSketchScenario const& scenario) -> WQSummaryContainer {
  auto eps = SketchEpsilon(scenario.max_bins, scenario.values.size());
  auto budget = SketchSummaryBudget(scenario.max_bins, scenario.values.size());
  WQuantileSketch sketch{scenario.values.size(), eps};

  std::vector<::xgboost::Entry> column(scenario.values.size());
  for (std::size_t i = 0; i < scenario.values.size(); ++i) {
    column[i] = ::xgboost::Entry{static_cast<bst_feature_t>(i), scenario.values[i]};
  }
  std::sort(column.begin(), column.end(),
            [](auto const& lhs, auto const& rhs) { return lhs.fvalue < rhs.fvalue; });
  sketch.PushSorted(Span<::xgboost::Entry const>{column.data(), column.size()}, scenario.weights,
                    budget);
  return sketch.GetSummary(budget);
}

auto BuildMergedSummary(WeightedSketchScenario const& scenario) -> WQSummaryContainer {
  auto eps = SketchEpsilon(scenario.max_bins, scenario.values.size());
  auto budget = SketchSummaryBudget(scenario.max_bins, scenario.values.size());
  auto split = scenario.values.size() / 2;

  WQuantileSketch lhs{scenario.values.size(), eps};
  WQuantileSketch rhs{scenario.values.size(), eps};
  for (std::size_t i = 0; i < split; ++i) {
    lhs.Push(scenario.values[i], scenario.weights[i]);
  }
  for (std::size_t i = split; i < scenario.values.size(); ++i) {
    rhs.Push(scenario.values[i], scenario.weights[i]);
  }

  auto lhs_summary = lhs.GetSummary(budget);
  auto rhs_summary = rhs.GetSummary(budget);
  WQSummaryContainer merged;
  merged.Reserve(lhs_summary.Size() + rhs_summary.Size());
  merged.CopyFrom(lhs_summary);
  merged.SetCombine(rhs_summary);
  merged.SetPrune(budget);
  return merged;
}

auto MakeWeightedSketchScenarios() -> std::vector<WeightedSketchScenario> {
  std::vector<WeightedSketchScenario> scenarios;

  {
    WeightedSketchScenario s;
    s.name = "continuous_random_weights";
    s.max_bins = 64;
    s.values = GenerateRandom(2048, 1);
    s.weights = GenerateRandomWeights(2048);
    scenarios.push_back(std::move(s));
  }

  {
    WeightedSketchScenario s;
    s.name = "bucketed_random_weights";
    s.max_bins = 64;
    s.values = GenerateRandom(2048, 1);
    for (auto& v : s.values) {
      v = std::floor(v * 192.0f) / 8.0f;
    }
    s.weights = GenerateRandomWeights(2048);
    scenarios.push_back(std::move(s));
  }

  {
    WeightedSketchScenario s;
    s.name = "sample_weight_times_hessian";
    s.max_bins = 128;
    s.values = GenerateRandom(4096, 1);
    auto sample_weight = GenerateRandomWeights(4096);
    auto hessian = GenerateRandomWeights(4096);
    s.weights.resize(sample_weight.size());
    for (std::size_t i = 0; i < s.weights.size(); ++i) {
      s.weights[i] = std::max(sample_weight[i] * hessian[s.weights.size() - i - 1], 1e-7f);
    }
    scenarios.push_back(std::move(s));
  }

  return scenarios;
}

template <typename SummaryBuilder>
void CheckWeightedQueryBound(char const* builder_name, SummaryBuilder&& build_summary) {
  for (auto const& scenario : MakeWeightedSketchScenarios()) {
    auto trace = std::string(builder_name) + ":" + scenario.name;
    SCOPED_TRACE(trace);
    auto weighted_uniques = AggregateWeightedData(scenario.values, scenario.weights);
    auto summary = build_summary(scenario);
    auto eps = SketchEpsilon(scenario.max_bins, scenario.values.size());
    auto budget = SketchSummaryBudget(scenario.max_bins, scenario.values.size());
    auto stats =
        MeasurePaperQueryBound(summary, weighted_uniques, eps, 1.0 / static_cast<double>(budget));
    auto bound = (stats.epsilon + stats.prune_term) * stats.total_weight;
    auto tol = std::max(1e-6, 1e-6 * stats.total_weight);
    EXPECT_LE(stats.max_absolute_error, bound + tol)
        << "scenario=" << scenario.name << ", builder=" << builder_name
        << ", query_index=" << stats.query_index << ", queried_value=" << stats.queried_value
        << ", target_rank=" << stats.target_rank << ", rank_lo=" << stats.rank_lo
        << ", rank_hi=" << stats.rank_hi << ", total_weight=" << stats.total_weight
        << ", epsilon=" << stats.epsilon << ", prune_term=" << stats.prune_term
        << ", bound=" << bound;
  }
}

template <bool use_column>
void PushPage(HostSketchContainer* container, SparsePage const& page, MetaInfo const& info,
              Span<float const> hessian) {
  if constexpr (use_column) {
    container->PushColPage(page, info, hessian);
  } else {
    container->PushRowPage(page, info, hessian);
  }
}

template <bool use_column>
void DoTestDistributedQuantile(size_t rows, size_t cols) {
  Context ctx;
  auto const world = collective::GetWorldSize();
  std::vector<MetaInfo> infos(2);
  auto& h_weights = infos.front().weights_.HostVector();
  h_weights.resize(rows);
  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist(3, 1000);
  std::generate(h_weights.begin(), h_weights.end(), [&]() { return dist(&lcg); });
  std::vector<bst_idx_t> column_size(cols, rows);
  bst_bin_t n_bins = 64;

  // Generate cuts for distributed environment.
  auto sparsity = 0.5f;
  auto rank = collective::GetRank();
  std::vector<FeatureType> ft(cols);
  for (size_t i = 0; i < ft.size(); ++i) {
    ft[i] = (i % 2 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
  }

  auto m = RandomDataGenerator{rows, cols, sparsity}
               .Seed(rank)
               .Lower(.0f)
               .Upper(1.0f)
               .Type(ft)
               .MaxCategory(13)
               .GenerateDMatrix();

  std::vector<float> hessian(rows, 1.0);
  auto hess = Span<float const>{hessian};

  HostSketchContainer sketch_distributed(&ctx, n_bins, m->Info().feature_types.ConstHostSpan(),
                                         column_size, false);

  if (use_column) {
    for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
      PushPage<use_column>(&sketch_distributed, page, m->Info(), hess);
    }
  } else {
    for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
      PushPage<use_column>(&sketch_distributed, page, m->Info(), hess);
    }
  }

  auto distributed_cuts = sketch_distributed.MakeCuts(&ctx, m->Info());

  // Generate cuts for single node environment
  collective::Finalize();

  CHECK_EQ(collective::GetWorldSize(), 1);
  std::for_each(column_size.begin(), column_size.end(), [=](auto& size) { size *= world; });
  m->Info().num_row_ = world * rows;
  HostSketchContainer sketch_on_single_node(&ctx, n_bins, m->Info().feature_types.ConstHostSpan(),
                                            column_size, false);
  m->Info().num_row_ = rows;

  for (auto rank = 0; rank < world; ++rank) {
    auto m = RandomDataGenerator{rows, cols, sparsity}
                 .Seed(rank)
                 .Type(ft)
                 .MaxCategory(13)
                 .Lower(.0f)
                 .Upper(1.0f)
                 .GenerateDMatrix();
    if (use_column) {
      for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
        PushPage<use_column>(&sketch_on_single_node, page, m->Info(), hess);
      }
    } else {
      for (auto const& page : m->GetBatches<SparsePage>()) {
        PushPage<use_column>(&sketch_on_single_node, page, m->Info(), hess);
      }
    }
  }

  auto single_node_cuts = sketch_on_single_node.MakeCuts(&ctx, m->Info());

  auto const& sptrs = single_node_cuts.Ptrs();
  auto const& dptrs = distributed_cuts.Ptrs();
  auto const& svals = single_node_cuts.Values();
  auto const& dvals = distributed_cuts.Values();

  ASSERT_EQ(sptrs.size(), dptrs.size());
  for (size_t i = 0; i < sptrs.size(); ++i) {
    ASSERT_EQ(sptrs[i], dptrs[i]) << i;
  }

  ASSERT_EQ(svals.size(), dvals.size());
  for (size_t i = 0; i < svals.size(); ++i) {
    ASSERT_NEAR(svals[i], dvals[i], 2e-2f);
  }
}

template <bool use_column>
void TestDistributedQuantile(size_t const rows, size_t const cols) {
  auto constexpr kWorkers = 4;
  collective::TestDistributedGlobal(
      kWorkers, [=] { DoTestDistributedQuantile<use_column>(rows, cols); }, false);
}
}  // anonymous namespace

TEST(Quantile, DistributedBasic) {
  constexpr size_t kRows = 10, kCols = 10;
  TestDistributedQuantile<false>(kRows, kCols);
}

TEST(Quantile, Distributed) {
  constexpr size_t kRows = 4000, kCols = 200;
  TestDistributedQuantile<false>(kRows, kCols);
}

TEST(Quantile, SortedDistributedBasic) {
  constexpr size_t kRows = 10, kCols = 10;
  TestDistributedQuantile<true>(kRows, kCols);
}

TEST(Quantile, SortedDistributed) {
  constexpr size_t kRows = 4000, kCols = 200;
  TestDistributedQuantile<true>(kRows, kCols);
}

namespace {
template <bool use_column>
void DoTestColSplitQuantile(size_t rows, size_t cols) {
  Context ctx;
  auto const world = collective::GetWorldSize();
  auto const rank = collective::GetRank();

  auto m = std::unique_ptr<DMatrix>{[=]() {
    auto sparsity = 0.5f;
    std::vector<FeatureType> ft(cols);
    for (size_t i = 0; i < ft.size(); ++i) {
      ft[i] = (i % 2 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
    }
    auto dmat = RandomDataGenerator{rows, cols, sparsity}
                    .Seed(0)
                    .Lower(.0f)
                    .Upper(1.0f)
                    .Type(ft)
                    .MaxCategory(13)
                    .GenerateDMatrix();
    return dmat->SliceCol(world, rank);
  }()};

  std::vector<bst_idx_t> column_size(cols, 0);
  auto const slice_size = cols / world;
  auto const slice_start = slice_size * rank;
  auto const slice_end = (rank == world - 1) ? cols : slice_start + slice_size;
  for (auto i = slice_start; i < slice_end; i++) {
    column_size[i] = rows;
  }

  auto const n_bins = 64;

  // Generate cuts for distributed environment.
  HistogramCuts distributed_cuts{0};
  {
    HostSketchContainer sketch_distributed(&ctx, n_bins, m->Info().feature_types.ConstHostSpan(),
                                           column_size, false);

    std::vector<float> hessian(rows, 1.0);
    auto hess = Span<float const>{hessian};
    if (use_column) {
      for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
        PushPage<use_column>(&sketch_distributed, page, m->Info(), hess);
      }
    } else {
      for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
        PushPage<use_column>(&sketch_distributed, page, m->Info(), hess);
      }
    }

    distributed_cuts = sketch_distributed.MakeCuts(&ctx, m->Info());
  }

  // Generate cuts for single node environment
  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);
  HistogramCuts single_node_cuts{0};
  {
    HostSketchContainer sketch_on_single_node(&ctx, n_bins, m->Info().feature_types.ConstHostSpan(),
                                              column_size, false);

    std::vector<float> hessian(rows, 1.0);
    auto hess = Span<float const>{hessian};
    if (use_column) {
      for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
        PushPage<use_column>(&sketch_on_single_node, page, m->Info(), hess);
      }
    } else {
      for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
        PushPage<use_column>(&sketch_on_single_node, page, m->Info(), hess);
      }
    }

    single_node_cuts = sketch_on_single_node.MakeCuts(&ctx, m->Info());
  }

  auto const& sptrs = single_node_cuts.Ptrs();
  auto const& dptrs = distributed_cuts.Ptrs();
  auto const& svals = single_node_cuts.Values();
  auto const& dvals = distributed_cuts.Values();

  EXPECT_EQ(sptrs.size(), dptrs.size());
  for (size_t i = 0; i < sptrs.size(); ++i) {
    EXPECT_EQ(sptrs[i], dptrs[i]) << "rank: " << rank << ", i: " << i;
  }

  EXPECT_EQ(svals.size(), dvals.size());
  for (size_t i = 0; i < svals.size(); ++i) {
    EXPECT_NEAR(svals[i], dvals[i], 2e-2f) << "rank: " << rank << ", i: " << i;
  }
}

template <bool use_column>
void TestColSplitQuantile(size_t rows, size_t cols) {
  auto constexpr kWorkers = 4;
  collective::TestDistributedGlobal(kWorkers,
                                    [=] { DoTestColSplitQuantile<use_column>(rows, cols); });
}
}  // anonymous namespace

TEST(Quantile, ColumnSplitBasic) {
  constexpr size_t kRows = 10, kCols = 10;
  TestColSplitQuantile<false>(kRows, kCols);
}

TEST(Quantile, ColumnSplit) {
  constexpr size_t kRows = 4000, kCols = 200;
  TestColSplitQuantile<false>(kRows, kCols);
}

TEST(Quantile, ColumnSplitSortedBasic) {
  constexpr size_t kRows = 10, kCols = 10;
  TestColSplitQuantile<true>(kRows, kCols);
}

TEST(Quantile, ColumnSplitSorted) {
  constexpr size_t kRows = 4000, kCols = 200;
  TestColSplitQuantile<true>(kRows, kCols);
}

TEST(Quantile, WeightedSummaryQueryBoundPush) {
  CheckWeightedQueryBound("push", [](auto const& scenario) { return BuildPushSummary(scenario); });
}

TEST(Quantile, WeightedSummaryQueryBoundSortedPush) {
  CheckWeightedQueryBound("sorted_push",
                          [](auto const& scenario) { return BuildSortedSummary(scenario); });
}

TEST(Quantile, WeightedSummaryQueryBoundMerged) {
  CheckWeightedQueryBound("merge",
                          [](auto const& scenario) { return BuildMergedSummary(scenario); });
}

namespace {
void TestSameOnAllWorkers() {
  auto const world = collective::GetWorldSize();
  constexpr size_t kRows = 1000, kCols = 100;
  Context ctx;

  RunWithSeedsAndBins(kRows, [=, &ctx](int32_t seed, size_t n_bins, MetaInfo const&) {
    auto rank = collective::GetRank();
    HostDeviceVector<float> storage;
    std::vector<FeatureType> ft(kCols);
    for (size_t i = 0; i < ft.size(); ++i) {
      ft[i] = (i % 2 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
    }

    auto m = RandomDataGenerator{kRows, kCols, 0}
                 .Device(DeviceOrd::CPU())
                 .Type(ft)
                 .MaxCategory(17)
                 .Seed(rank + seed)
                 .GenerateDMatrix();
    auto cuts = SketchOnDMatrix(&ctx, m.get(), n_bins);
    std::vector<float> cut_values(cuts.Values().size() * world, 0);
    std::vector<typename std::remove_reference_t<decltype(cuts.Ptrs())>::value_type> cut_ptrs(
        cuts.Ptrs().size() * world, 0);

    std::int64_t value_size = cuts.Values().size();
    std::int64_t ptr_size = cuts.Ptrs().size();

    auto rc = collective::Success() << [&] {
      return collective::Allreduce(&ctx, &value_size, collective::Op::kMax);
    } << [&] {
      return collective::Allreduce(&ctx, &ptr_size, collective::Op::kMax);
    };
    collective::SafeColl(rc);
    ASSERT_EQ(ptr_size, kCols + 1);

    std::size_t value_offset = value_size * rank;
    std::copy(cuts.Values().begin(), cuts.Values().end(), cut_values.begin() + value_offset);
    std::size_t ptr_offset = ptr_size * rank;
    std::copy(cuts.Ptrs().cbegin(), cuts.Ptrs().cend(), cut_ptrs.begin() + ptr_offset);

    rc = std::move(rc) << [&] {
      return collective::Allreduce(&ctx, linalg::MakeVec(cut_values.data(), cut_values.size()),
                                   collective::Op::kSum);
    } << [&] {
      return collective::Allreduce(&ctx, linalg::MakeVec(cut_ptrs.data(), cut_ptrs.size()),
                                   collective::Op::kSum);
    };
    collective::SafeColl(rc);

    for (std::int32_t i = 0; i < world; i++) {
      for (std::int64_t j = 0; j < value_size; ++j) {
        size_t idx = i * value_size + j;
        ASSERT_NEAR(cuts.Values().at(j), cut_values.at(idx), kRtEps);
      }

      for (std::int64_t j = 0; j < ptr_size; ++j) {
        size_t idx = i * ptr_size + j;
        EXPECT_EQ(cuts.Ptrs().at(j), cut_ptrs.at(idx));
      }
    }
  });
}
}  // anonymous namespace

TEST(Quantile, SameOnAllWorkers) {
  auto constexpr kWorkers = 4;
  collective::TestDistributedGlobal(kWorkers, [] { TestSameOnAllWorkers(); });
}
}  // namespace xgboost::common
