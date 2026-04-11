/**
 * Copyright 2020-2026, XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/collective/allreduce.h"
#include "../../../src/common/hist_util.cuh"
#include "../../../src/common/quantile.cuh"
#include "../../../src/data/device_adapter.cuh"  // CupyAdapter
#include "../collective/test_worker.h"           // for BaseMGPUTest
#include "../helpers.h"
#include "test_hist_util.h"
#include "test_quantile.h"
#include "test_quantile_helpers.h"

namespace xgboost {
namespace {
struct IsSorted {
  XGBOOST_DEVICE bool operator()(common::SketchEntry const& a, common::SketchEntry const& b) const {
    return a.value < b.value;
  }
};

auto MakeHostSummary(std::vector<std::pair<float, float>> const& items)
    -> common::WQSummaryContainer {
  common::WQSummaryContainer summary;
  summary.Reserve(items.size());
  summary.SetFromSorted(items);
  return summary;
}

auto CopySummaryEntries(common::WQSummaryContainer const& summary)
    -> std::vector<common::SketchEntry> {
  auto entries = summary.Entries();
  return {entries.cbegin(), entries.cend()};
}

struct HostSketchView {
  std::vector<common::SketchEntry> data;
  std::vector<bst_idx_t> columns_ptr;
};

struct HostEntryBatch {
  std::vector<Entry> entries;
  std::vector<size_t> columns_ptr;
};

auto MakeEntryBatch(std::vector<std::vector<float>> const& columns) -> HostEntryBatch {
  HostEntryBatch batch;
  batch.columns_ptr.push_back(0);
  for (bst_feature_t c = 0; c < columns.size(); ++c) {
    for (auto value : columns[c]) {
      batch.entries.push_back(Entry{c, value});
    }
    batch.columns_ptr.push_back(batch.entries.size());
  }
  return batch;
}

auto MakePruneBatch(std::size_t rows, bst_feature_t cols, bool with_duplicates) -> HostEntryBatch {
  std::vector<std::vector<float>> columns(cols);
  for (size_t i = 0; i < rows; ++i) {
    if (with_duplicates) {
      columns[0].push_back(static_cast<float>(i / 4));
      columns[1].push_back(static_cast<float>(i / 8) + 10.0f);
      columns[2].push_back(static_cast<float>(i / 2) + 100.0f);
    } else {
      columns[0].push_back(static_cast<float>(i));
      columns[1].push_back(static_cast<float>(i) * 0.5f + 10.0f);
      columns[2].push_back(static_cast<float>(i) * 0.25f + 100.0f);
    }
  }
  return MakeEntryBatch(columns);
}

auto CopySketchToHost(xgboost::common::Span<common::SketchEntry const> data,
                      xgboost::common::Span<bst_idx_t const> columns_ptr) -> HostSketchView {
  HostSketchView out;
  out.data.resize(data.size());
  out.columns_ptr.resize(columns_ptr.size());
  dh::CopyDeviceSpanToVector(&out.data, data);
  dh::CopyDeviceSpanToVector(&out.columns_ptr, columns_ptr);
  return out;
}
}  // namespace

namespace common {
class MGPUQuantileTest : public collective::BaseMGPUTest {};

namespace {
void DoGPUContainerProperty(quantile_test::ContainerCase const& c) {
  auto ctx = MakeCUDACtx(0);
  auto ft = quantile_test::FeatureTypes(c);
  auto m = RandomDataGenerator{c.rows, c.cols, c.sparsity}
               .Seed(c.seed)
               .Lower(.0f)
               .Upper(1.0f)
               .Type(ft)
               .MaxCategory(13)
               .GenerateDMatrix();
  if (c.weights == quantile_test::WeightKind::kRow) {
    m->Info().weights_.HostVector() = quantile_test::GenerateWeights(c.rows, c.seed + 1024);
  }
  auto cuts = DeviceSketch(&ctx, m.get(), c.max_bin);
  auto columns = quantile_test::CollectWeightedColumns(m.get());
  quantile_test::ValidateContainerCuts(c, cuts, m.get(), columns);
}

void DoMGPURowSplitProperty(quantile_test::ContainerCase const& c) {
  auto const world = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  auto ctx = MakeCUDACtx(GPUIDX);
  auto ft = quantile_test::FeatureTypes(c);
  auto full_m = RandomDataGenerator{c.rows * static_cast<std::size_t>(world), c.cols, c.sparsity}
                    .Seed(c.seed)
                    .Lower(.0f)
                    .Upper(1.0f)
                    .Type(ft)
                    .MaxCategory(13)
                    .GenerateDMatrix();
  if (c.weights == quantile_test::WeightKind::kRow) {
    full_m->Info().weights_.HostVector() =
        quantile_test::GenerateWeights(c.rows * static_cast<std::size_t>(world), c.seed + 4096);
  }

  std::vector<std::int32_t> ridxs(c.rows);
  auto row_begin = static_cast<std::size_t>(rank) * c.rows;
  std::iota(ridxs.begin(), ridxs.end(), static_cast<std::int32_t>(row_begin));
  auto m =
      std::shared_ptr<DMatrix>{full_m->Slice(Span<std::int32_t const>{ridxs.data(), ridxs.size()})};
  m->Info().data_split_mode = DataSplitMode::kRow;

  auto cuts = DeviceSketch(&ctx, m.get(), c.max_bin);
  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);

  auto columns = quantile_test::CollectWeightedColumns(full_m.get());
  quantile_test::ValidateContainerCuts(c, cuts, full_m.get(), columns);
}

void DoMGPUColumnSplitProperty(quantile_test::ContainerCase const& c) {
  auto const world = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  auto ctx = MakeCUDACtx(GPUIDX);
  auto ft = quantile_test::FeatureTypes(c);
  auto full_m = RandomDataGenerator{c.rows, c.cols, c.sparsity}
                    .Seed(c.seed)
                    .Lower(.0f)
                    .Upper(1.0f)
                    .Type(ft)
                    .MaxCategory(13)
                    .GenerateDMatrix();
  if (c.weights == quantile_test::WeightKind::kRow) {
    full_m->Info().weights_.HostVector() = quantile_test::GenerateWeights(c.rows, c.seed + 2048);
  }
  auto m = std::shared_ptr<DMatrix>{full_m->SliceCol(world, rank)};

  auto cuts = DeviceSketch(&ctx, m.get(), c.max_bin);
  auto const slice_size = c.cols / world;
  auto const slice_start = slice_size * rank;
  auto const slice_end = (rank == world - 1) ? c.cols : slice_start + slice_size;

  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);

  auto columns = quantile_test::CollectWeightedColumns(full_m.get());
  quantile_test::ValidateContainerCuts(c, cuts, full_m.get(), columns, slice_start, slice_end);
}
}  // namespace

TEST(GPUQuantileProperty, Invariants) {
  for (auto const& c : quantile_test::ContainerAnchorCases()) {
    SCOPED_TRACE(c.name);
    DoGPUContainerProperty(c);
  }
}

TEST_F(MGPUQuantileTest, RowSplitProperty) {
  for (auto const& c : quantile_test::ContainerAnchorCases()) {
    SCOPED_TRACE(c.name);
    this->DoTest([&] { DoMGPURowSplitProperty(c); }, true);
    this->DoTest([&] { DoMGPURowSplitProperty(c); }, false);
  }
}

TEST_F(MGPUQuantileTest, ColumnSplitProperty) {
  for (auto const& c : quantile_test::ContainerAnchorCases()) {
    SCOPED_TRACE(c.name);
    this->DoTest([&] { DoMGPUColumnSplitProperty(c); }, true);
    this->DoTest([&] { DoMGPUColumnSplitProperty(c); }, false);
  }
}

TEST(GPUQuantile, EmptyPush) {
  auto ctx = MakeCUDACtx(0);
  constexpr size_t kCols = 100, kBins = 256;
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch(ft, kBins, kCols, ctx.Device());
  dh::caching_device_vector<Entry> entries;
  dh::device_vector<bst_idx_t> cuts_ptr(kCols + 1);
  thrust::fill(cuts_ptr.begin(), cuts_ptr.end(), 0);
  // Push empty
  sketch.Push(&ctx, dh::ToSpan(entries), dh::ToSpan(cuts_ptr), dh::ToSpan(cuts_ptr), 0, 0);
  ASSERT_EQ(sketch.Data().size(), 0);
}

void ValidateSketchInvariants(HostSketchView const& sketch, bool with_error = false) {
  for (size_t i = 1; i < sketch.columns_ptr.size(); ++i) {
    auto column_id = i - 1;
    auto beg = sketch.columns_ptr[column_id];
    auto end = sketch.columns_ptr[i];

    auto column = Span<SketchEntry const>{sketch.data}.subspan(beg, end - beg);
    ASSERT_TRUE(std::is_sorted(column.begin(), column.end(), IsSorted{}));
    ASSERT_TRUE(std::adjacent_find(column.begin(), column.end(),
                                   [](SketchEntry const& l, SketchEntry const& r) {
                                     return l.value == r.value;
                                   }) == column.end());
    for (size_t idx = 1; idx < column.size(); ++idx) {
      float prev_rmin = column[idx - 1].rmin;
      float prev_rmax = column[idx - 1].rmax;
      float rmin_next = column[idx].RMinNext();
      if (with_error) {
        ASSERT_GE(column[idx].rmin + column[idx].rmin * kRtEps, prev_rmin);
        ASSERT_GE(column[idx].rmax + column[idx].rmin * kRtEps, prev_rmax);
        ASSERT_GE(column[idx].rmax + column[idx].rmin * kRtEps, rmin_next);
      } else {
        ASSERT_GE(column[idx].rmin, prev_rmin);
        ASSERT_GE(column[idx].rmax, prev_rmax);
        ASSERT_GE(column[idx].rmax, rmin_next);
      }
    }
  }
}

TEST(GPUQuantile, Prune) {
  constexpr size_t kRows = 64, kCols = 3;
  for (auto with_duplicates : {false, true}) {
    for (auto n_bins : {8, 16, 80}) {
      auto ctx = MakeCUDACtx(0);
      HostDeviceVector<FeatureType> ft;
      SketchContainer sketch(ft, n_bins, kCols, ctx.Device());
      auto batch = MakePruneBatch(kRows, kCols, with_duplicates);
      dh::device_vector<Entry> d_entries{batch.entries};
      dh::device_vector<size_t> d_columns_ptr{batch.columns_ptr};
      dh::device_vector<size_t> d_cuts_ptr{batch.columns_ptr};
      sketch.Push(&ctx, dh::ToSpan(d_entries), dh::ToSpan(d_columns_ptr), dh::ToSpan(d_cuts_ptr),
                  batch.entries.size(), kRows, {});

      sketch.Prune(&ctx, n_bins);
      ASSERT_LE(sketch.Data().size(), static_cast<std::size_t>(n_bins) * kCols);
      auto h_sketch = CopySketchToHost(sketch.Data(), sketch.ColumnsPtr());
      ValidateSketchInvariants(h_sketch);
    }
  }
}

TEST(GPUQuantile, MergeEmpty) {
  constexpr size_t kRows = 1000, kCols = 100;
  size_t n_bins = 10;
  auto ctx = MakeCUDACtx(0);
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch_0(ft, n_bins, kCols, ctx.Device());
  HostDeviceVector<float> storage_0;
  std::string interface_str_0 =
      RandomDataGenerator{kRows, kCols, 0}.Device(ctx.Device()).GenerateArrayInterface(&storage_0);
  data::CupyAdapter adapter_0(interface_str_0);
  MetaInfo info;
  AdapterDeviceSketch(&ctx, adapter_0.Value(), n_bins, info,
                      std::numeric_limits<float>::quiet_NaN(), &sketch_0);

  std::vector<SketchEntry> entries_before(sketch_0.Data().size());
  dh::CopyDeviceSpanToVector(&entries_before, sketch_0.Data());
  std::vector<bst_idx_t> ptrs_before(sketch_0.ColumnsPtr().size());
  dh::CopyDeviceSpanToVector(&ptrs_before, sketch_0.ColumnsPtr());
  thrust::device_vector<size_t> columns_ptr(kCols + 1);
  // Merge an empty sketch
  sketch_0.Merge(&ctx, dh::ToSpan(columns_ptr), Span<SketchEntry>{});

  std::vector<SketchEntry> entries_after(sketch_0.Data().size());
  dh::CopyDeviceSpanToVector(&entries_after, sketch_0.Data());
  std::vector<bst_idx_t> ptrs_after(sketch_0.ColumnsPtr().size());
  dh::CopyDeviceSpanToVector(&ptrs_after, sketch_0.ColumnsPtr());

  CHECK_EQ(entries_before.size(), entries_after.size());
  CHECK_EQ(ptrs_before.size(), ptrs_after.size());
  for (size_t i = 0; i < entries_before.size(); ++i) {
    CHECK_EQ(entries_before[i].value, entries_after[i].value);
    CHECK_EQ(entries_before[i].rmin, entries_after[i].rmin);
    CHECK_EQ(entries_before[i].rmax, entries_after[i].rmax);
    CHECK_EQ(entries_before[i].wmin, entries_after[i].wmin);
  }
  for (size_t i = 0; i < ptrs_before.size(); ++i) {
    CHECK_EQ(ptrs_before[i], ptrs_after[i]);
  }
}

TEST(GPUQuantile, MergeBasic) {
  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [=](std::int32_t seed, bst_bin_t n_bins, MetaInfo const& info) {
    auto ctx = MakeCUDACtx(0);
    HostDeviceVector<FeatureType> ft;
    SketchContainer sketch_0(ft, n_bins, kCols, ctx.Device());
    HostDeviceVector<float> storage_0;
    std::string interface_str_0 = RandomDataGenerator{kRows, kCols, 0}
                                      .Device(ctx.Device())
                                      .Seed(seed)
                                      .GenerateArrayInterface(&storage_0);
    data::CupyAdapter adapter_0(interface_str_0);
    AdapterDeviceSketch(&ctx, adapter_0.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch_0);

    SketchContainer sketch_1(ft, n_bins, kCols, ctx.Device());
    HostDeviceVector<float> storage_1;
    std::string interface_str_1 = RandomDataGenerator{kRows, kCols, 0}
                                      .Device(ctx.Device())
                                      .Seed(seed)
                                      .GenerateArrayInterface(&storage_1);
    data::CupyAdapter adapter_1(interface_str_1);
    AdapterDeviceSketch(&ctx, adapter_1.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch_1);

    size_t size_before_merge = sketch_0.Data().size();
    sketch_0.Merge(&ctx, sketch_1.ColumnsPtr(), sketch_1.Data());
    auto h_sketch = CopySketchToHost(sketch_0.Data(), sketch_0.ColumnsPtr());
    ValidateSketchInvariants(h_sketch);
    auto const& h_columns_ptr = h_sketch.columns_ptr;
    ASSERT_LE(h_columns_ptr.back(), sketch_1.Data().size() + size_before_merge);
    ASSERT_EQ(static_cast<std::size_t>(h_columns_ptr.back()), h_sketch.data.size());
  });
}

void TestMergeDuplicated(int32_t n_bins, size_t cols, size_t rows, float frac) {
  auto ctx = MakeCUDACtx(0);
  MetaInfo info;
  int32_t seed = 0;
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch_0(ft, n_bins, cols, ctx.Device());
  HostDeviceVector<float> storage_0;
  std::string interface_str_0 = RandomDataGenerator{rows, cols, 0}
                                    .Device(ctx.Device())
                                    .Seed(seed)
                                    .GenerateArrayInterface(&storage_0);
  data::CupyAdapter adapter_0(interface_str_0);
  AdapterDeviceSketch(&ctx, adapter_0.Value(), n_bins, info,
                      std::numeric_limits<float>::quiet_NaN(), &sketch_0);

  size_t f_rows = rows * frac;
  SketchContainer sketch_1(ft, n_bins, cols, ctx.Device());
  HostDeviceVector<float> storage_1;
  std::string interface_str_1 = RandomDataGenerator{f_rows, cols, 0}
                                    .Device(ctx.Device())
                                    .Seed(seed)
                                    .GenerateArrayInterface(&storage_1);
  auto data_1 = storage_1.DeviceSpan();
  auto tuple_it = cuda::std::make_tuple(thrust::make_counting_iterator<size_t>(0ul), data_1.data());
  using Tuple = cuda::std::tuple<size_t, float>;
  auto it = thrust::make_zip_iterator(tuple_it);
  thrust::transform(ctx.CUDACtx()->CTP(), it, it + data_1.size(), data_1.data(),
                    [=] XGBOOST_DEVICE(Tuple const& tuple) {
                      auto i = cuda::std::get<0>(tuple);
                      if (i % 2 == 0) {
                        return 0.0f;
                      } else {
                        return cuda::std::get<1>(tuple);
                      }
                    });
  data::CupyAdapter adapter_1(interface_str_1);
  AdapterDeviceSketch(&ctx, adapter_1.Value(), n_bins, info,
                      std::numeric_limits<float>::quiet_NaN(), &sketch_1);

  size_t size_before_merge = sketch_0.Data().size();
  sketch_0.Merge(&ctx, sketch_1.ColumnsPtr(), sketch_1.Data());
  auto h_sketch = CopySketchToHost(sketch_0.Data(), sketch_0.ColumnsPtr());
  ValidateSketchInvariants(h_sketch);
  auto const& h_columns_ptr = h_sketch.columns_ptr;
  ASSERT_LE(h_columns_ptr.back(), sketch_1.Data().size() + size_before_merge);
  ASSERT_EQ(static_cast<std::size_t>(h_columns_ptr.back()), h_sketch.data.size());
}

TEST(GPUQuantile, MergeDuplicated) {
  size_t n_bins = 256;
  constexpr size_t kRows = 1000, kCols = 100;
  for (float frac = 0.5; frac < 2.5; frac += 0.5) {
    TestMergeDuplicated(n_bins, kRows, kCols, frac);
  }
}

TEST(GPUQuantile, MergeCategorical) {
  auto ctx = MakeCUDACtx(0);
  constexpr bst_feature_t kCols = 2;
  bst_bin_t n_bins = 16;

  HostDeviceVector<FeatureType> ft;
  ft.HostVector() = {FeatureType::kCategorical, FeatureType::kNumerical};
  SketchContainer sketch_0(ft, n_bins, kCols, ctx.Device());
  SketchContainer sketch_1(ft, n_bins, kCols, ctx.Device());

  std::vector<Entry> entries_0{{0, 0.0f}, {0, 0.0f}, {0, 1.0f}, {0, 2.0f},
                               {0, 2.0f}, {1, 0.1f}, {1, 0.2f}, {1, 0.4f}};
  std::vector<Entry> entries_1{{0, 1.0f}, {0, 1.0f},  {0, 2.0f},  {0, 3.0f},
                               {0, 3.0f}, {1, 0.15f}, {1, 0.25f}, {1, 0.5f}};

  dh::device_vector<Entry> d_entries_0{entries_0};
  dh::device_vector<Entry> d_entries_1{entries_1};
  dh::device_vector<size_t> columns_ptr_0{0, 5, 8};
  dh::device_vector<size_t> columns_ptr_1{0, 5, 8};
  dh::device_vector<size_t> cuts_ptr_0{0, 5, 8};
  dh::device_vector<size_t> cuts_ptr_1{0, 5, 8};

  sketch_0.Push(&ctx, dh::ToSpan(d_entries_0), dh::ToSpan(columns_ptr_0), dh::ToSpan(cuts_ptr_0),
                entries_0.size(), 5, {});
  sketch_1.Push(&ctx, dh::ToSpan(d_entries_1), dh::ToSpan(columns_ptr_1), dh::ToSpan(cuts_ptr_1),
                entries_1.size(), 5, {});

  sketch_0.Merge(&ctx, sketch_1.ColumnsPtr(), sketch_1.Data());
  auto h_sketch = CopySketchToHost(sketch_0.Data(), sketch_0.ColumnsPtr());
  ValidateSketchInvariants(h_sketch);

  auto cat_column = Span<SketchEntry const>{h_sketch.data}.subspan(h_sketch.columns_ptr[0],
                                                                   h_sketch.columns_ptr[1]);
  ASSERT_TRUE(std::adjacent_find(cat_column.begin(), cat_column.end(),
                                 [](SketchEntry const& l, SketchEntry const& r) {
                                   return l.value == r.value;
                                 }) == cat_column.end());
}

TEST(GPUQuantile, MergeSameValue) {
  auto ctx = MakeCUDACtx(0);
  constexpr bst_feature_t kCols = 1;
  bst_bin_t n_bins = 16;

  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch_0(ft, n_bins, kCols, ctx.Device());
  SketchContainer sketch_1(ft, n_bins, kCols, ctx.Device());

  std::vector<Entry> entries_0{{0, 0.5f}};
  std::vector<Entry> entries_1{{0, 0.5f}};
  dh::device_vector<Entry> d_entries_0{entries_0};
  dh::device_vector<Entry> d_entries_1{entries_1};
  dh::device_vector<size_t> columns_ptr{0, 1};
  dh::device_vector<size_t> cuts_ptr{0, 1};

  sketch_0.Push(&ctx, dh::ToSpan(d_entries_0), dh::ToSpan(columns_ptr), dh::ToSpan(cuts_ptr), 1, 1,
                {});
  sketch_1.Push(&ctx, dh::ToSpan(d_entries_1), dh::ToSpan(columns_ptr), dh::ToSpan(cuts_ptr), 1, 1,
                {});

  sketch_0.Merge(&ctx, sketch_1.ColumnsPtr(), sketch_1.Data());

  std::vector<bst_idx_t> h_columns_ptr(sketch_0.ColumnsPtr().size());
  dh::CopyDeviceSpanToVector(&h_columns_ptr, sketch_0.ColumnsPtr());
  std::vector<SketchEntry> h_data(sketch_0.Data().size());
  dh::CopyDeviceSpanToVector(&h_data, sketch_0.Data());

  ASSERT_EQ(h_columns_ptr.back(), 1);
  ASSERT_EQ(h_data.size(), 1);
  EXPECT_FLOAT_EQ(h_data.front().value, 0.5f);
  EXPECT_FLOAT_EQ(h_data.front().rmin, 0.0f);
  EXPECT_FLOAT_EQ(h_data.front().wmin, 2.0f);
  EXPECT_FLOAT_EQ(h_data.front().rmax, 2.0f);
}

TEST(GPUQuantile, MergeMatchesCpuCombine) {
  auto ctx = MakeCUDACtx(0);
  constexpr bst_feature_t kCols = 1;
  bst_bin_t n_bins = 16;

  auto lhs = MakeHostSummary({{0.1f, 1.0f}, {0.3f, 2.0f}, {0.5f, 1.0f}});
  auto rhs = MakeHostSummary({{0.3f, 1.5f}, {0.4f, 1.0f}, {0.5f, 0.5f}});

  common::WQSummaryContainer expected;
  expected.Reserve(lhs.Size() + rhs.Size());
  expected.CopyFrom(lhs);
  expected.SetCombine(rhs);

  auto lhs_entries = CopySummaryEntries(lhs);
  auto rhs_entries = CopySummaryEntries(rhs);

  dh::device_vector<SketchEntry> d_lhs{lhs_entries};
  dh::device_vector<SketchEntry> d_rhs{rhs_entries};
  dh::device_vector<size_t> lhs_ptr{0, lhs.Size()};
  dh::device_vector<size_t> rhs_ptr{0, rhs.Size()};

  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch(ft, n_bins, kCols, ctx.Device());
  sketch.Merge(&ctx, dh::ToSpan(lhs_ptr), dh::ToSpan(d_lhs));
  sketch.Merge(&ctx, dh::ToSpan(rhs_ptr), dh::ToSpan(d_rhs));

  std::vector<bst_idx_t> h_columns_ptr(sketch.ColumnsPtr().size());
  dh::CopyDeviceSpanToVector(&h_columns_ptr, sketch.ColumnsPtr());
  auto h_data = std::vector<SketchEntry>(sketch.Data().size());
  dh::CopyDeviceSpanToVector(&h_data, sketch.Data());

  ASSERT_EQ(h_columns_ptr.back(), expected.Size());
  auto expected_entries = expected.Entries();
  ASSERT_EQ(h_data.size(), expected_entries.size());
  for (std::size_t i = 0; i < h_data.size(); ++i) {
    EXPECT_FLOAT_EQ(h_data[i].value, expected_entries[i].value);
    EXPECT_FLOAT_EQ(h_data[i].rmin, expected_entries[i].rmin);
    EXPECT_FLOAT_EQ(h_data[i].rmax, expected_entries[i].rmax);
    EXPECT_FLOAT_EQ(h_data[i].wmin, expected_entries[i].wmin);
  }
}

TEST(GPUQuantile, MultiMerge) {
  constexpr size_t kRows = 20, kCols = 1;
  int32_t world = 2;
  RunWithSeedsAndBins(kRows, [=](std::int32_t seed, bst_bin_t n_bins, MetaInfo const& info) {
    // Set up single node version
    HostDeviceVector<FeatureType> ft;
    auto ctx = MakeCUDACtx(0);
    SketchContainer sketch_on_single_node(ft, n_bins, kCols, ctx.Device());

    auto intermediate_num_cuts = SketchSummaryBudget(n_bins, kRows * world);
    std::vector<SketchContainer> containers;
    for (auto rank = 0; rank < world; ++rank) {
      HostDeviceVector<float> storage;
      std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                      .Device(ctx.Device())
                                      .Seed(rank + seed)
                                      .GenerateArrayInterface(&storage);
      data::CupyAdapter adapter(interface_str);
      HostDeviceVector<FeatureType> ft;
      containers.emplace_back(ft, n_bins, kCols, ctx.Device());
      AdapterDeviceSketch(&ctx, adapter.Value(), n_bins, info,
                          std::numeric_limits<float>::quiet_NaN(), &containers.back());
    }
    for (auto& sketch : containers) {
      sketch.Prune(&ctx, intermediate_num_cuts);
      sketch_on_single_node.Merge(&ctx, sketch.ColumnsPtr(), sketch.Data());
    }
    auto h_sketch =
        CopySketchToHost(sketch_on_single_node.Data(), sketch_on_single_node.ColumnsPtr());
    ValidateSketchInvariants(h_sketch);
  });
}

TEST(GPUQuantile, MissingColumns) {
  auto dmat = std::unique_ptr<DMatrix>{[=]() {
    std::size_t constexpr kRows = 1000, kCols = 100;
    auto sparsity = 0.5f;
    std::vector<FeatureType> ft(kCols);
    for (size_t i = 0; i < ft.size(); ++i) {
      ft[i] = (i % 2 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
    }
    auto dmat = RandomDataGenerator{kRows, kCols, sparsity}
                    .Seed(0)
                    .Lower(.0f)
                    .Upper(1.0f)
                    .Type(ft)
                    .MaxCategory(13)
                    .GenerateDMatrix();
    return dmat->SliceCol(2, 1);
  }()};
  dmat->Info().data_split_mode = DataSplitMode::kRow;

  auto ctx = MakeCUDACtx(0);
  std::size_t constexpr kBins = 64;
  HistogramCuts cuts = common::DeviceSketch(&ctx, dmat.get(), kBins);
  ASSERT_TRUE(cuts.HasCategorical());
}

namespace {
void TestSameOnAllWorkers() {
  auto world = collective::GetWorldSize();
  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [=](std::int32_t seed, bst_bin_t n_bins, MetaInfo const& info) {
    auto const rank = collective::GetRank();
    auto const device = DeviceOrd::CUDA(GPUIDX);
    Context ctx = MakeCUDACtx(device.ordinal);
    HostDeviceVector<FeatureType> ft({}, device);
    SketchContainer sketch_distributed(ft, n_bins, kCols, device);
    HostDeviceVector<float> storage({}, device);
    std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                    .Device(device)
                                    .Seed(rank + seed)
                                    .GenerateArrayInterface(&storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(&ctx, adapter.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch_distributed);
    sketch_distributed.AllReduce(&ctx, false);
    auto h_sketch = CopySketchToHost(sketch_distributed.Data(), sketch_distributed.ColumnsPtr());
    ValidateSketchInvariants(h_sketch, true);

    // Test for all workers having the same sketch.
    size_t n_data = sketch_distributed.Data().size();
    auto rc = collective::Allreduce(&ctx, linalg::MakeVec(&n_data, 1), collective::Op::kMax);
    SafeColl(rc);
    ASSERT_EQ(n_data, sketch_distributed.Data().size());
    size_t size_as_float = sketch_distributed.Data().size_bytes() / sizeof(float);
    auto local_data = Span<float const>{
        reinterpret_cast<float const*>(sketch_distributed.Data().data()), size_as_float};

    dh::caching_device_vector<float> all_workers(size_as_float * world);
    thrust::fill(all_workers.begin(), all_workers.end(), 0);
    thrust::copy(thrust::device, local_data.data(), local_data.data() + local_data.size(),
                 all_workers.begin() + local_data.size() * rank);
    rc = collective::Allreduce(
        &ctx, linalg::MakeVec(all_workers.data().get(), all_workers.size(), ctx.Device()),
        collective::Op::kSum);
    SafeColl(rc);

    auto base_line = dh::ToSpan(all_workers).subspan(0, size_as_float);
    std::vector<float> h_base_line(base_line.size());
    dh::CopyDeviceSpanToVector(&h_base_line, base_line);

    size_t offset = 0;
    for (decltype(world) i = 0; i < world; ++i) {
      auto comp = dh::ToSpan(all_workers).subspan(offset, size_as_float);
      std::vector<float> h_comp(comp.size());
      dh::CopyDeviceSpanToVector(&h_comp, comp);
      ASSERT_EQ(comp.size(), base_line.size());
      for (size_t j = 0; j < h_comp.size(); ++j) {
        ASSERT_NEAR(h_base_line[j], h_comp[j], kRtEps);
      }
      offset += size_as_float;
    }
  });
}
}  // anonymous namespace

TEST_F(MGPUQuantileTest, SameOnAllWorkers) {
  this->DoTest([] { TestSameOnAllWorkers(); }, true);
  this->DoTest([] { TestSameOnAllWorkers(); }, false);
}

TEST(GPUQuantile, Push) {
  size_t constexpr kRows = 100;
  std::vector<float> data(kRows);
  auto ctx = MakeCUDACtx(0);
  std::fill(data.begin(), data.begin() + (data.size() / 2), 0.3f);
  std::fill(data.begin() + (data.size() / 2), data.end(), 0.5f);
  int32_t n_bins = 128;
  bst_feature_t constexpr kCols = 1;

  std::vector<Entry> entries(kRows);
  for (bst_feature_t i = 0; i < entries.size(); ++i) {
    Entry e{i, data[i]};
    entries[i] = e;
  }

  dh::device_vector<Entry> d_entries(entries);
  dh::device_vector<size_t> columns_ptr(2);
  columns_ptr[0] = 0;
  columns_ptr[1] = kRows;

  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch(ft, n_bins, kCols, ctx.Device());
  sketch.Push(&ctx, dh::ToSpan(d_entries), dh::ToSpan(columns_ptr), dh::ToSpan(columns_ptr), kRows,
              kRows, {});

  auto sketch_data = sketch.Data();

  thrust::host_vector<SketchEntry> h_sketch_data(sketch_data.size());

  auto ptr = thrust::device_ptr<SketchEntry const>(sketch_data.data());
  thrust::copy(ptr, ptr + sketch_data.size(), h_sketch_data.begin());
  ASSERT_EQ(h_sketch_data.size(), 2);

  auto v_0 = h_sketch_data[0];
  ASSERT_EQ(v_0.rmin, 0);
  ASSERT_EQ(v_0.wmin, kRows / 2.0f);
  ASSERT_EQ(v_0.rmax, kRows / 2.0f);

  auto v_1 = h_sketch_data[1];
  ASSERT_EQ(v_1.rmin, kRows / 2.0f);
  ASSERT_EQ(v_1.wmin, kRows / 2.0f);
  ASSERT_EQ(v_1.rmax, static_cast<float>(kRows));
}

TEST(GPUQuantile, MultiColPush) {
  size_t constexpr kRows = 100, kCols = 4;
  std::vector<float> data(kRows * kCols);
  std::fill(data.begin(), data.begin() + (data.size() / 2), 0.3f);

  auto ctx = MakeCUDACtx(0);
  std::vector<Entry> entries(kRows * kCols);

  for (bst_feature_t c = 0; c < kCols; ++c) {
    for (size_t r = 0; r < kRows; ++r) {
      float v = (r >= kRows / 2) ? 0.7 : 0.4;
      auto e = Entry{c, v};
      entries[c * kRows + r] = e;
    }
  }

  int32_t n_bins = 16;
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch(ft, n_bins, kCols, ctx.Device());

  dh::device_vector<Entry> d_entries{entries};
  dh::device_vector<size_t> columns_ptr(kCols + 1, 0);
  for (size_t i = 1; i < kCols + 1; ++i) {
    columns_ptr[i] = kRows;
  }
  thrust::inclusive_scan(thrust::device, columns_ptr.begin(), columns_ptr.end(),
                         columns_ptr.begin());
  dh::device_vector<size_t> cuts_ptr(columns_ptr);

  sketch.Push(&ctx, dh::ToSpan(d_entries), dh::ToSpan(columns_ptr), dh::ToSpan(cuts_ptr),
              kRows * kCols, kRows, {});

  auto sketch_data = sketch.Data();
  ASSERT_EQ(sketch_data.size(), kCols * 2);
  auto ptr = thrust::device_ptr<SketchEntry const>(sketch_data.data());
  std::vector<SketchEntry> h_sketch_data(sketch_data.size());
  thrust::copy(ptr, ptr + sketch_data.size(), h_sketch_data.begin());

  for (size_t i = 0; i < kCols; ++i) {
    auto v_0 = h_sketch_data[i * 2];
    ASSERT_EQ(v_0.rmin, 0);
    ASSERT_EQ(v_0.wmin, kRows / 2.0f);
    ASSERT_EQ(v_0.rmax, kRows / 2.0f);

    auto v_1 = h_sketch_data[i * 2 + 1];
    ASSERT_EQ(v_1.rmin, kRows / 2.0f);
    ASSERT_EQ(v_1.wmin, kRows / 2.0f);
    ASSERT_EQ(v_1.rmax, static_cast<float>(kRows));
  }
}
}  // namespace common
}  // namespace xgboost
