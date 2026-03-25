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

TEST(Quantile, TrackSketchElements) {
  WQuantileSketch sketch{16, 0.1};
  ASSERT_EQ(sketch.NumElements(), 0);

  sketch.Push(0.1f);
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
  std::vector<float> weights(column.size(), 1.0f);

  sketch.PushSorted(Span<::xgboost::Entry const>{column.data(), column.size()}, weights, 2);

  ASSERT_EQ(sketch.NumElements(), column.size());
  auto out = sketch.GetSummary(4);
  ASSERT_GT(out.Size(), 0);
  ASSERT_EQ(sketch.NumElements(), column.size());
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
template <bool use_column>
void PushPage(HostSketchContainer* container, SparsePage const& page, MetaInfo const& info,
              Span<float const> hessian) {
  if constexpr (use_column) {
    container->PushColPage(page, info, hessian);
  } else {
    container->PushRowPage(page, info, hessian);
  }
}

auto RowSplitBounds(std::size_t n_rows, std::size_t world, std::size_t rank) {
  auto rows_per_worker = n_rows / world;
  CHECK_EQ(rows_per_worker * world, n_rows);
  auto row_begin = rank * rows_per_worker;
  auto row_end = row_begin + rows_per_worker;
  return std::pair{row_begin, row_end};
}

auto SliceRows(std::vector<float> const& data, std::size_t n_cols, std::size_t row_begin,
               std::size_t row_end) -> std::vector<float> {
  auto begin = data.cbegin() + row_begin * n_cols;
  auto end = data.cbegin() + row_end * n_cols;
  return {begin, end};
}

template <bool use_column>
auto SketchDistributedCuts(Context const* ctx, DMatrix* m,
                           std::vector<bst_idx_t> const& column_size, bst_bin_t n_bins)
    -> HistogramCuts {
  HostSketchContainer sketch{ctx, n_bins, m->Info().feature_types.ConstHostSpan(), column_size,
                             HostSketchContainer::UseGroup(m->Info())};
  auto hess = Span<float const>{};
  if constexpr (use_column) {
    for (auto const& page : m->GetBatches<SortedCSCPage>(ctx)) {
      PushPage<use_column>(&sketch, page, m->Info(), hess);
    }
  } else {
    for (auto const& page : m->GetBatches<SparsePage>(ctx)) {
      PushPage<use_column>(&sketch, page, m->Info(), hess);
    }
  }
  return sketch.MakeCuts(ctx, m->Info());
}

template <bool use_column>
void DoTestRowSplitRankError(std::size_t rows, std::size_t cols) {
  Context ctx;
  auto const world = static_cast<std::size_t>(collective::GetWorldSize());
  auto const rank = static_cast<std::size_t>(collective::GetRank());
  auto constexpr kBins = 64;

  auto full_data = GenerateRandom(rows, cols);
  auto full_weights = GenerateRandomWeights(rows);
  auto [row_begin, row_end] = RowSplitBounds(rows, world, rank);

  auto local_data = SliceRows(full_data, cols, row_begin, row_end);
  std::vector<float> local_weights(full_weights.cbegin() + row_begin,
                                   full_weights.cbegin() + row_end);
  auto local = GetDMatrixFromData(local_data, row_end - row_begin, cols);
  local->Info().weights_.HostVector() = local_weights;

  std::vector<bst_idx_t> column_size(cols, row_end - row_begin);
  auto distributed_cuts = SketchDistributedCuts<use_column>(&ctx, local.get(), column_size, kBins);

  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);

  auto full = GetDMatrixFromData(full_data, rows, cols);
  full->Info().weights_.HostVector() = full_weights;
  ValidateCuts(distributed_cuts, full.get(), kBins);
}

template <bool use_column>
void TestRowSplitRankError(std::size_t rows, std::size_t cols) {
  auto constexpr kWorkers = 4;
  collective::TestDistributedGlobal(
      kWorkers, [=] { DoTestRowSplitRankError<use_column>(rows, cols); }, false);
}

template <bool use_column>
void DoTestColumnSplitRankError(std::size_t rows, std::size_t cols) {
  Context ctx;
  auto const world = static_cast<std::size_t>(collective::GetWorldSize());
  auto const rank = static_cast<std::size_t>(collective::GetRank());
  auto constexpr kBins = 64;

  auto full_data = GenerateRandom(rows, cols);
  auto full_weights = GenerateRandomWeights(rows);
  auto base = GetDMatrixFromData(full_data, rows, cols);
  base->Info().weights_.HostVector() = full_weights;
  auto m = std::unique_ptr<DMatrix>{base->SliceCol(world, rank)};

  std::vector<bst_idx_t> column_size(cols, 0);
  auto const slice_size = cols / world;
  auto const slice_start = slice_size * rank;
  auto const slice_end = (rank == world - 1) ? cols : slice_start + slice_size;
  for (auto i = slice_start; i < slice_end; ++i) {
    column_size[i] = rows;
  }

  auto distributed_cuts = SketchDistributedCuts<use_column>(&ctx, m.get(), column_size, kBins);

  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);

  ValidateCuts(distributed_cuts, m.get(), kBins);
}

template <bool use_column>
void TestColumnSplitRankError(std::size_t rows, std::size_t cols) {
  auto constexpr kWorkers = 4;
  collective::TestDistributedGlobal(
      kWorkers, [=] { DoTestColumnSplitRankError<use_column>(rows, cols); }, false);
}
}  // anonymous namespace

TEST(Quantile, DistributedRankError) {
  constexpr std::size_t kRows = 1024, kCols = 64;
  TestRowSplitRankError<false>(kRows, kCols);
}

TEST(Quantile, SortedDistributedRankError) {
  constexpr std::size_t kRows = 1024, kCols = 64;
  TestRowSplitRankError<true>(kRows, kCols);
}

TEST(Quantile, ColumnSplitRankError) {
  constexpr std::size_t kRows = 1024, kCols = 64;
  TestColumnSplitRankError<false>(kRows, kCols);
}

TEST(Quantile, ColumnSplitSortedRankError) {
  constexpr std::size_t kRows = 1024, kCols = 64;
  TestColumnSplitRankError<true>(kRows, kCols);
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
