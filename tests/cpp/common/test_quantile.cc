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
#include "xgboost/context.h"

namespace xgboost::common {
namespace quantile_test {
class QuantileSummaryTest : public ::testing::TestWithParam<SummaryCase> {};
class QuantileContainerTest : public ::testing::TestWithParam<ContainerCase> {};
class QuantileDistributedContainerTest : public ::testing::TestWithParam<ContainerCase> {};
class QuantileSketchOnDMatrixTest : public ::testing::TestWithParam<ContainerCase> {};

namespace {
void TestSummaryInvariants(SummaryCase const& c, WQSummaryContainer const& summary,
                           GeneratedColumn const& col) {
  auto entries = summary.Entries();
  auto ref = AggregateReferenceColumn(col);
  auto nonzero_samples = NonZeroWeightCount(col);
  auto budget = SketchSummaryBudget(c.max_bin, c.rows);
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
void AssertSameOnAllWorkers(Context const* ctx, HistogramCuts const& cuts) {
  auto const world = collective::GetWorldSize();
  if (world <= 1) {
    return;
  }

  auto const rank = collective::GetRank();
  auto const local_value_size = static_cast<std::int64_t>(cuts.Values().size());
  auto const local_ptr_size = static_cast<std::int64_t>(cuts.Ptrs().size());

  std::vector<std::int64_t> value_sizes(world, 0);
  std::vector<std::int64_t> ptr_sizes(world, 0);
  value_sizes.at(rank) = local_value_size;
  ptr_sizes.at(rank) = local_ptr_size;

  auto rc = collective::Success() << [&] {
    return collective::Allreduce(ctx, linalg::MakeVec(value_sizes.data(), value_sizes.size()),
                                 collective::Op::kSum);
  } << [&] {
    return collective::Allreduce(ctx, linalg::MakeVec(ptr_sizes.data(), ptr_sizes.size()),
                                 collective::Op::kSum);
  };
  collective::SafeColl(rc);

  auto const max_value_size = *std::max_element(value_sizes.cbegin(), value_sizes.cend());
  auto const max_ptr_size = *std::max_element(ptr_sizes.cbegin(), ptr_sizes.cend());
  std::vector<float> cut_values(static_cast<std::size_t>(max_value_size) * world, 0.0f);
  std::vector<typename std::remove_reference_t<decltype(cuts.Ptrs())>::value_type> cut_ptrs(
      static_cast<std::size_t>(max_ptr_size) * world, 0);

  auto const value_offset = static_cast<std::size_t>(max_value_size) * rank;
  auto const ptr_offset = static_cast<std::size_t>(max_ptr_size) * rank;
  std::copy(cuts.Values().cbegin(), cuts.Values().cend(), cut_values.begin() + value_offset);
  std::copy(cuts.Ptrs().cbegin(), cuts.Ptrs().cend(), cut_ptrs.begin() + ptr_offset);

  rc = collective::Success() << [&] {
    return collective::Allreduce(ctx, linalg::MakeVec(cut_values.data(), cut_values.size()),
                                 collective::Op::kSum);
  } << [&] {
    return collective::Allreduce(ctx, linalg::MakeVec(cut_ptrs.data(), cut_ptrs.size()),
                                 collective::Op::kSum);
  };
  collective::SafeColl(rc);

  for (std::int32_t worker = 0; worker < world; ++worker) {
    ASSERT_EQ(value_sizes.at(worker), local_value_size);
    ASSERT_EQ(ptr_sizes.at(worker), local_ptr_size);
    for (std::int64_t j = 0; j < local_value_size; ++j) {
      auto idx = static_cast<std::size_t>(worker) * max_value_size + static_cast<std::size_t>(j);
      ASSERT_NEAR(cuts.Values().at(j), cut_values.at(idx), kRtEps);
    }
    for (std::int64_t j = 0; j < local_ptr_size; ++j) {
      auto idx = static_cast<std::size_t>(worker) * max_ptr_size + static_cast<std::size_t>(j);
      ASSERT_EQ(cuts.Ptrs().at(j), cut_ptrs.at(idx));
    }
  }
}

}  // namespace

TEST_P(QuantileSummaryTest, Invariants) {
  auto c = GetParam();
  auto col = GenerateSummaryColumn(c);

  WQuantileSketch row_sketch{c.rows, SketchEpsilon(c.max_bin, c.rows)};
  for (std::size_t i = 0; i < col.values.size(); ++i) {
    row_sketch.Push(col.values[i], col.weights[i]);
  }
  auto row_summary = row_sketch.GetSummary(SketchSummaryBudget(c.max_bin, c.rows));
  TestSummaryInvariants(c, row_summary, col);

  WQuantileSketch sorted_sketch{c.rows, SketchEpsilon(c.max_bin, c.rows)};
  std::vector<::xgboost::Entry> sorted_col;
  sorted_col.reserve(col.values.size());
  for (std::size_t i = 0; i < col.values.size(); ++i) {
    sorted_col.emplace_back(i, col.values[i]);
  }
  std::sort(sorted_col.begin(), sorted_col.end(), ::xgboost::Entry::CmpValue);
  sorted_sketch.PushSorted(Span<::xgboost::Entry const>{sorted_col.data(), sorted_col.size()},
                           col.weights, c.max_bin);
  auto sorted_summary = sorted_sketch.GetSummary(SketchSummaryBudget(c.max_bin, c.rows));
  TestSummaryInvariants(c, sorted_summary, col);
}

INSTANTIATE_TEST_SUITE_P(Anchors, QuantileSummaryTest, ::testing::ValuesIn(SummaryAnchorCases()),
                         SummaryCaseName);
INSTANTIATE_TEST_SUITE_P(RandomSamples, QuantileSummaryTest,
                         ::testing::ValuesIn(SummaryRandomCases(100)), SummaryCaseName);

TEST_P(QuantileContainerTest, Invariants) {
  auto c = GetParam();
  Context ctx;
  auto ft = FeatureTypes(c);
  auto m = RandomDataGenerator{c.rows, c.cols, c.sparsity}
               .Seed(c.seed)
               .Lower(.0f)
               .Upper(1.0f)
               .Type(ft)
               .MaxCategory(13)
               .GenerateDMatrix();
  if (c.weights == WeightKind::kRow) {
    m->Info().weights_.HostVector() = GenerateWeights(c.rows, c.seed + 1024);
  }

  std::vector<bst_idx_t> column_size(c.cols, c.rows);
  std::vector<float> hessian(c.rows, 1.0f);
  auto hess = Span<float const>{hessian};

  HostSketchContainer row_sketch(&ctx, c.max_bin, m->Info().feature_types.ConstHostSpan(),
                                 column_size, false);
  for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
    row_sketch.PushRowPage(page, m->Info(), hess);
  }
  auto row_cuts = row_sketch.MakeCuts(&ctx, m->Info());
  auto columns = CollectWeightedColumns(m.get());
  ValidateContainerCuts(c, row_cuts, m.get(), columns);

  HostSketchContainer sorted_sketch(&ctx, c.max_bin, m->Info().feature_types.ConstHostSpan(),
                                    column_size, false);
  for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
    sorted_sketch.PushColPage(page, m->Info(), hess);
  }
  auto sorted_cuts = sorted_sketch.MakeCuts(&ctx, m->Info());
  ValidateContainerCuts(c, sorted_cuts, m.get(), columns);
}

TEST_P(QuantileSketchOnDMatrixTest, Invariants) {
  auto c = GetParam();
  Context ctx;
  auto ft = FeatureTypes(c);
  auto m = RandomDataGenerator{c.rows, c.cols, c.sparsity}
               .Seed(c.seed)
               .Lower(.0f)
               .Upper(1.0f)
               .Type(ft)
               .MaxCategory(13)
               .GenerateDMatrix();
  if (c.weights == WeightKind::kRow) {
    m->Info().weights_.HostVector() = GenerateWeights(c.rows, c.seed + 2048);
  }

  auto columns = CollectWeightedColumns(m.get());
  std::vector<float> hessian(c.rows, 1.0f);
  auto hess = Span<float const>{hessian};
  auto row_cuts = SketchOnDMatrix(&ctx, m.get(), c.max_bin, false, hess);
  ValidateContainerCuts(c, row_cuts, m.get(), columns);

  auto sorted_cuts = SketchOnDMatrix(&ctx, m.get(), c.max_bin, true, hess);
  ValidateContainerCuts(c, sorted_cuts, m.get(), columns);
}

namespace {
void DoPropertyDistributedQuantile(ContainerCase const& c) {
  Context ctx;
  auto const world = collective::GetWorldSize();
  auto ft = FeatureTypes(c);
  auto rank = collective::GetRank();
  auto full_m = RandomDataGenerator{c.rows * static_cast<std::size_t>(world), c.cols, c.sparsity}
                    .Seed(c.seed)
                    .Lower(.0f)
                    .Upper(1.0f)
                    .Type(ft)
                    .MaxCategory(13)
                    .GenerateDMatrix();
  if (c.weights == WeightKind::kRow) {
    full_m->Info().weights_.HostVector() =
        GenerateWeights(c.rows * static_cast<std::size_t>(world), c.seed + 4096);
  }
  std::vector<std::int32_t> ridxs(c.rows);
  auto row_begin = static_cast<std::size_t>(rank) * c.rows;
  std::iota(ridxs.begin(), ridxs.end(), static_cast<std::int32_t>(row_begin));
  std::shared_ptr<DMatrix> m{full_m->Slice(Span<std::int32_t const>{ridxs.data(), ridxs.size()})};

  std::vector<bst_idx_t> column_size(c.cols, c.rows);
  std::vector<float> hessian(c.rows, 1.0f);
  auto hess = Span<float const>{hessian};
  HostSketchContainer row_sketch(&ctx, c.max_bin, m->Info().feature_types.ConstHostSpan(),
                                 column_size, false);
  for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
    row_sketch.PushRowPage(page, m->Info(), hess);
  }
  auto row_cuts = row_sketch.MakeCuts(&ctx, m->Info());

  HostSketchContainer sorted_sketch(&ctx, c.max_bin, m->Info().feature_types.ConstHostSpan(),
                                    column_size, false);
  for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
    sorted_sketch.PushColPage(page, m->Info(), hess);
  }
  auto sorted_cuts = sorted_sketch.MakeCuts(&ctx, m->Info());

  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);
  auto columns = CollectWeightedColumns(full_m.get());
  ValidateContainerCuts(c, row_cuts, full_m.get(), columns);
  ValidateContainerCuts(c, sorted_cuts, full_m.get(), columns);
}

void DoSameOnAllWorkersDistributedQuantile(ContainerCase const& c) {
  Context ctx;
  auto const world = collective::GetWorldSize();
  auto ft = FeatureTypes(c);
  auto rank = collective::GetRank();
  auto full_m = RandomDataGenerator{c.rows * static_cast<std::size_t>(world), c.cols, c.sparsity}
                    .Seed(c.seed)
                    .Lower(.0f)
                    .Upper(1.0f)
                    .Type(ft)
                    .MaxCategory(13)
                    .GenerateDMatrix();
  if (c.weights == WeightKind::kRow) {
    full_m->Info().weights_.HostVector() =
        GenerateWeights(c.rows * static_cast<std::size_t>(world), c.seed + 4096);
  }
  std::vector<std::int32_t> ridxs(c.rows);
  auto row_begin = static_cast<std::size_t>(rank) * c.rows;
  std::iota(ridxs.begin(), ridxs.end(), static_cast<std::int32_t>(row_begin));
  std::shared_ptr<DMatrix> m{full_m->Slice(Span<std::int32_t const>{ridxs.data(), ridxs.size()})};

  std::vector<bst_idx_t> column_size(c.cols, c.rows);
  std::vector<float> hessian(c.rows, 1.0f);
  auto hess = Span<float const>{hessian};
  HostSketchContainer row_sketch(&ctx, c.max_bin, m->Info().feature_types.ConstHostSpan(),
                                 column_size, false);
  for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
    row_sketch.PushRowPage(page, m->Info(), hess);
  }
  auto row_cuts = row_sketch.MakeCuts(&ctx, m->Info());
  AssertSameOnAllWorkers(&ctx, row_cuts);

  HostSketchContainer sorted_sketch(&ctx, c.max_bin, m->Info().feature_types.ConstHostSpan(),
                                    column_size, false);
  for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
    sorted_sketch.PushColPage(page, m->Info(), hess);
  }
  auto sorted_cuts = sorted_sketch.MakeCuts(&ctx, m->Info());
  AssertSameOnAllWorkers(&ctx, sorted_cuts);

  collective::Finalize();
}
}  // namespace

TEST_P(QuantileDistributedContainerTest, Invariants) {
  auto c = GetParam();
  collective::TestDistributedGlobal(4, [&] { DoPropertyDistributedQuantile(c); }, false);
}

TEST(Quantile, SameOnAllWorkers) {
  auto c = ContainerCase{};
  c.name = "same_on_all_workers";
  c.rows = 1024;
  c.cols = 8;
  c.sparsity = 0.2f;
  c.max_bin = 256;
  c.weights = WeightKind::kRow;
  c.features = FeatureKind::kNumerical;
  c.seed = 11;
  collective::TestDistributedGlobal(4, [&] { DoSameOnAllWorkersDistributedQuantile(c); }, false);
}

INSTANTIATE_TEST_SUITE_P(Anchors, QuantileContainerTest,
                         ::testing::ValuesIn(ContainerAnchorCases()), ContainerCaseName);
INSTANTIATE_TEST_SUITE_P(Anchors, QuantileDistributedContainerTest,
                         ::testing::ValuesIn(ContainerAnchorCases()), ContainerCaseName);
INSTANTIATE_TEST_SUITE_P(Anchors, QuantileSketchOnDMatrixTest,
                         ::testing::ValuesIn(ContainerAnchorCases()), ContainerCaseName);
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
void DoPropertyColumnSplitQuantile(size_t rows, size_t cols) {
  Context ctx;
  auto const world = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  auto sparsity = 0.5f;
  std::vector<FeatureType> ft(cols);
  for (size_t i = 0; i < ft.size(); ++i) {
    ft[i] = (i % 2 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
  }
  auto full_m = RandomDataGenerator{rows, cols, sparsity}
                    .Seed(0)
                    .Lower(.0f)
                    .Upper(1.0f)
                    .Type(ft)
                    .MaxCategory(13)
                    .GenerateDMatrix();
  auto m = std::shared_ptr<DMatrix>{full_m->SliceCol(world, rank)};

  std::vector<bst_idx_t> column_size(cols, 0);
  auto const slice_size = cols / world;
  auto const slice_start = slice_size * rank;
  auto const slice_end = (rank == world - 1) ? cols : slice_start + slice_size;
  for (auto i = slice_start; i < slice_end; i++) {
    column_size[i] = rows;
  }

  auto const n_bins = 64;
  quantile_test::ContainerCase c;
  c.name = rows == 10 ? "column_split_basic" : "column_split_large";
  c.rows = rows;
  c.cols = cols;
  c.sparsity = sparsity;
  c.max_bin = n_bins;
  c.weights = quantile_test::WeightKind::kNone;
  c.features = quantile_test::FeatureKind::kMixed;
  c.seed = 0;
  auto columns = quantile_test::CollectWeightedColumns(full_m.get());
  std::vector<float> hessian(rows, 1.0f);
  auto hess = Span<float const>{hessian};

  HistogramCuts row_cuts{0};
  {
    HostSketchContainer sketch_distributed(&ctx, n_bins, m->Info().feature_types.ConstHostSpan(),
                                           column_size, false);
    for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
      sketch_distributed.PushRowPage(page, m->Info(), hess);
    }
    row_cuts = sketch_distributed.MakeCuts(&ctx, m->Info());
  }

  HistogramCuts sorted_cuts{0};
  {
    HostSketchContainer sketch_distributed(&ctx, n_bins, m->Info().feature_types.ConstHostSpan(),
                                           column_size, false);
    for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
      sketch_distributed.PushColPage(page, m->Info(), hess);
    }
    sorted_cuts = sketch_distributed.MakeCuts(&ctx, m->Info());
  }

  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);
  quantile_test::ValidateContainerCuts(c, row_cuts, full_m.get(), columns, slice_start, slice_end);
  quantile_test::ValidateContainerCuts(c, sorted_cuts, full_m.get(), columns, slice_start,
                                       slice_end);
}
}  // anonymous namespace

TEST(Quantile, ColumnSplit) {
  constexpr size_t kRows = 4000, kCols = 200;
  collective::TestDistributedGlobal(4, [&] { DoPropertyColumnSplitQuantile(kRows, kCols); });
}

}  // namespace xgboost::common
