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
#include "test_quantile_helpers.h"
#include "xgboost/context.h"

namespace xgboost::common {
namespace quantile_test {
class QuantileSummaryTest : public ::testing::TestWithParam<SummaryCase> {};
class QuantileSummarySortedTest : public ::testing::TestWithParam<SummaryCase> {};

TEST_P(QuantileSummaryTest, Invariants) {
  auto c = GetParam();
  auto col = GenerateSummaryColumn(c);
  WQuantileSketch sketch{c.rows, SketchEpsilon(c.max_bin, c.rows)};
  for (std::size_t i = 0; i < col.values.size(); ++i) {
    sketch.Push(col.values[i], col.weights[i]);
  }
  auto budget = SketchSummaryBudget(c.max_bin, c.rows);
  auto summary = sketch.GetSummary(budget);
  auto entries = summary.Entries();
  auto ref = AggregateReferenceColumn(col);
  auto nonzero_samples = NonZeroWeightCount(col);

  // An empty sketch should remain empty after finalization.
  if (EmptyReference(ref)) {
    ASSERT_TRUE(entries.empty()) << "case=" << c.name;
    return;
  }

  // A numerical summary should remain a strictly increasing support set.
  ASSERT_FALSE(entries.empty()) << "case=" << c.name;
  for (std::size_t i = 1; i < entries.size(); ++i) {
    EXPECT_LT(entries[i - 1].value, entries[i].value) << "case=" << c.name;
  }

  // Large-n anchors should exercise actual compression rather than exact retention.
  if (c.rows > static_cast<std::size_t>(c.max_bin) * 8) {
    ASSERT_LT(summary.Size(), nonzero_samples)
        << "case=" << c.name << " should exercise sketch compression.";
  }

  // The summary query rule should satisfy the target rank bound plus the final prune term.
  auto total = TotalWeight(ref);
  auto max_error = MaxSummaryQueryRankError(summary, ref, c.max_bin);
  auto eps = SketchEpsilon(c.max_bin, c.rows);
  auto bound = (eps + 1.0 / static_cast<double>(budget)) * total;

  EXPECT_LE(max_error, bound) << "case=" << c.name << ", total=" << total << ", budget=" << budget
                              << ", eps=" << eps;

  // If the target bin count can already represent all distinct values, the summary should
  // preserve the exact support instead of approximating it.
  if (UniqueValueCount(ref) <= static_cast<std::size_t>(c.max_bin)) {
    auto exact_values = ExactValues(ref);
    ASSERT_EQ(entries.size(), exact_values.size()) << "case=" << c.name;
    for (std::size_t i = 0; i < exact_values.size(); ++i) {
      EXPECT_FLOAT_EQ(entries[i].value, exact_values[i]) << "case=" << c.name;
    }
  }
}

TEST_P(QuantileSummarySortedTest, QueryBound) {
  auto c = GetParam();
  auto col = GenerateSummaryColumn(c);
  WQuantileSketch sketch{c.rows, SketchEpsilon(c.max_bin, c.rows)};
  std::vector<::xgboost::Entry> sorted_col;
  sorted_col.reserve(col.values.size());
  for (std::size_t i = 0; i < col.values.size(); ++i) {
    sorted_col.emplace_back(i, col.values[i]);
  }
  std::sort(sorted_col.begin(), sorted_col.end(), ::xgboost::Entry::CmpValue);
  sketch.PushSorted(Span<::xgboost::Entry const>{sorted_col.data(), sorted_col.size()}, col.weights,
                    c.max_bin);

  auto budget = SketchSummaryBudget(c.max_bin, c.rows);
  auto summary = sketch.GetSummary(budget);
  auto entries = summary.Entries();
  auto ref = AggregateReferenceColumn(col);

  if (EmptyReference(ref)) {
    ASSERT_TRUE(entries.empty()) << "case=" << c.name;
    return;
  }

  auto total = TotalWeight(ref);
  auto max_error = MaxSummaryQueryRankError(summary, ref, c.max_bin);
  auto eps = SketchEpsilon(c.max_bin, c.rows);
  auto bound = (eps + 1.0 / static_cast<double>(budget)) * total;

  EXPECT_LE(max_error, bound) << "case=" << c.name << ", total=" << total << ", budget=" << budget
                              << ", eps=" << eps;
}

INSTANTIATE_TEST_SUITE_P(Anchors, QuantileSummaryTest, ::testing::ValuesIn(SummaryAnchorCases()),
                         CaseName);
INSTANTIATE_TEST_SUITE_P(RandomSamples, QuantileSummaryTest,
                         ::testing::ValuesIn(SummaryRandomCases(100)), CaseName);
INSTANTIATE_TEST_SUITE_P(Anchors, QuantileSummarySortedTest,
                         ::testing::ValuesIn(SummaryAnchorCases()), CaseName);
}  // namespace quantile_test

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

TEST(Quantile, TrackSketchElements) {
  WQuantileSketch sketch{16, 0.1};
  ASSERT_EQ(sketch.NumElements(), 0);

  sketch.Push(0.1f);
  sketch.Push(0.2f, 0.0f);
  sketch.Push(0.3f, 2.0f);
  sketch.Push(0.9f);

  ASSERT_EQ(sketch.NumElements(), 3);
  auto out = sketch.GetSummary(4);
  ASSERT_GT(out.Size(), 0);
  ASSERT_EQ(sketch.NumElements(), 3);
}

TEST(Quantile, TrackSketchElementsSorted) {
  WQuantileSketch sketch{16, 0.1};
  std::vector<::xgboost::Entry> column{{0, 0.1f}, {1, 0.2f}, {2, 0.8f}, {3, 0.9f}};
  std::vector<float> weights{1.0f, 0.0f, 1.0f, 1.0f};

  sketch.PushSorted(Span<::xgboost::Entry const>{column.data(), column.size()}, weights, 2);

  ASSERT_EQ(sketch.NumElements(), 3);
  auto out = sketch.GetSummary(4);
  ASSERT_GT(out.Size(), 0);
  ASSERT_EQ(sketch.NumElements(), 3);
}
namespace {
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
