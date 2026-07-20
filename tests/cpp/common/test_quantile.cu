/**
 * Copyright 2020-2026, XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/collective/allreduce.h"
#include "../../../src/common/hist_util.cuh"
#include "../../../src/common/quantile.cuh"
#include "../collective/test_worker.h"  // for BaseMGPUTest
#include "../helpers.h"
#include "test_quantile.h"

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

struct DeviceEntryBatch {
  dh::device_vector<Entry> entries;
  dh::device_vector<size_t> columns_ptr;
  dh::device_vector<float> weights_scan;
  std::size_t rows;
};

auto MakeEntryBatch(std::vector<std::vector<float>> const& columns,
                    std::vector<std::vector<float>> const* weights = nullptr) -> DeviceEntryBatch {
  std::vector<Entry> h_entries;
  std::vector<size_t> h_columns_ptr;
  std::vector<float> h_weights_scan;
  h_columns_ptr.emplace_back(0);
  for (bst_feature_t c = 0; c < columns.size(); ++c) {
    float prefix_sum = 0.0f;
    for (auto value : columns[c]) {
      h_entries.emplace_back(c, value);
    }
    if (weights) {
      CHECK_EQ(columns[c].size(), (*weights)[c].size());
      for (auto w : (*weights)[c]) {
        prefix_sum += w;
        h_weights_scan.emplace_back(prefix_sum);
      }
    }
    h_columns_ptr.emplace_back(h_entries.size());
  }
  return {dh::device_vector<Entry>{h_entries}, dh::device_vector<size_t>{h_columns_ptr},
          dh::device_vector<float>{h_weights_scan}, columns.empty() ? 0 : columns.front().size()};
}

auto MakeSyntheticBatch(std::size_t rows, bst_feature_t cols, std::int32_t seed = 0,
                        bool weighted = false, bool with_duplicates = true,
                        std::size_t batch_idx = 0) -> DeviceEntryBatch {
  std::vector<std::vector<float>> columns(cols);
  std::vector<std::vector<float>> weights(cols);
  for (bst_feature_t c = 0; c < cols; ++c) {
    auto base = static_cast<float>(c) * 1000.0f + static_cast<float>(seed % 97) * 10.0f +
                static_cast<float>(batch_idx) * 1000.0f;
    for (std::size_t r = 0; r < rows; ++r) {
      float value;
      if (with_duplicates) {
        value = base + static_cast<float>(r / 2);
      } else {
        auto jitter = static_cast<float>((seed + c * 17 + r * 13) % 7) * 1e-3f;
        value = base + static_cast<float>(r) + jitter;
      }
      columns[c].push_back(value);
      if (weighted) {
        auto weight = 0.5f + static_cast<float>((seed + c * 19 + r * 23) % 11) * 0.25f;
        weights[c].push_back(weight);
      }
    }
  }
  return weighted ? MakeEntryBatch(columns, &weights) : MakeEntryBatch(columns);
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

[[nodiscard]] auto ExpectedSketchEntriesPerFeature(bst_bin_t n_bins, std::size_t rows_seen)
    -> std::size_t {
  auto num_rows = std::max<std::size_t>(1, rows_seen);
  auto eps = common::SketchEpsilon(n_bins, num_rows);
  auto limit = common::WQSketch::LimitSizeLevel(num_rows, eps);
  return std::min(limit, num_rows);
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

void TestSameOnAllWorkers() {
  constexpr size_t kRows = 1000, kCols = 100;
  for (auto n_bins : {bst_bin_t{2}, bst_bin_t{16}, static_cast<bst_bin_t>(kRows + 160)}) {
    for (auto weighted : {false, true}) {
      auto const rank = collective::GetRank();
      auto const device = DeviceOrd::CUDA(GPUIDX);
      Context ctx = MakeCUDACtx(device.ordinal);
      HostDeviceVector<FeatureType> ft({}, device);
      SketchContainer sketch_distributed(ft, n_bins, kCols, device);
      auto batch = MakeSyntheticBatch(kRows, kCols, rank + 29, weighted, false, rank);
      sketch_distributed.Push(&ctx, dh::ToSpan(batch.entries), dh::ToSpan(batch.columns_ptr),
                              batch.rows, dh::ToSpan(batch.weights_scan));
      sketch_distributed.AllReduce(&ctx);
      auto h_sketch = CopySketchToHost(sketch_distributed.Data(), sketch_distributed.ColumnsPtr());
      ValidateSketchInvariants(h_sketch, true);
      AssertSameSketchOnAllWorkers(h_sketch);
    }
  }
}
}  // anonymous namespace

TEST_F(MGPUQuantileTest, SameOnAllWorkers) {
  this->DoTest([] { TestSameOnAllWorkers(); }, true);
  this->DoTest([] { TestSameOnAllWorkers(); }, false);
}

TEST(GPUQuantile, Push) {
  constexpr size_t kRows = 100, kBatches = 3;
  auto ctx = MakeCUDACtx(0);
  {
    HostDeviceVector<FeatureType> ft;
    SketchContainer sketch(ft, 128, 1, ctx.Device());
    auto batch = MakeEntryBatch({{0.3f, 0.3f, 0.3f, 0.3f, 0.5f, 0.5f, 0.5f, 0.5f}});
    sketch.Push(&ctx, dh::ToSpan(batch.entries), dh::ToSpan(batch.columns_ptr), batch.rows,
                dh::ToSpan(batch.weights_scan));

    auto h_sketch = CopySketchToHost(sketch.Data(), sketch.ColumnsPtr());
    ValidateSketchInvariants(h_sketch);
    ASSERT_EQ(h_sketch.columns_ptr, (std::vector<bst_idx_t>{0, 2}));
    ASSERT_EQ(h_sketch.data.size(), 2);

    auto v_0 = h_sketch.data[0];
    EXPECT_FLOAT_EQ(v_0.value, 0.3f);
    EXPECT_FLOAT_EQ(v_0.rmin, 0.0f);
    EXPECT_FLOAT_EQ(v_0.wmin, 4.0f);
    EXPECT_FLOAT_EQ(v_0.rmax, 4.0f);

    auto v_1 = h_sketch.data[1];
    EXPECT_FLOAT_EQ(v_1.value, 0.5f);
    EXPECT_FLOAT_EQ(v_1.rmin, 4.0f);
    EXPECT_FLOAT_EQ(v_1.wmin, 4.0f);
    EXPECT_FLOAT_EQ(v_1.rmax, 8.0f);
  }

  for (auto [n_bins, kCols] : {std::pair{128, 1}, std::pair{16, 4}}) {
    HostDeviceVector<FeatureType> ft;
    SketchContainer sketch(ft, n_bins, kCols, ctx.Device());
    for (size_t batch_idx = 0; batch_idx < kBatches; ++batch_idx) {
      auto batch = MakeSyntheticBatch(kRows, kCols, 0, false, true, batch_idx);
      sketch.Push(&ctx, dh::ToSpan(batch.entries), dh::ToSpan(batch.columns_ptr), batch.rows,
                  dh::ToSpan(batch.weights_scan));
      auto rows_seen = kRows * (batch_idx + 1);
      ASSERT_LE(sketch.Data().size(), ExpectedSketchEntriesPerFeature(n_bins, rows_seen) * kCols);
    }

    auto h_sketch = CopySketchToHost(sketch.Data(), sketch.ColumnsPtr());
    ValidateSketchInvariants(h_sketch);
    ASSERT_EQ(h_sketch.columns_ptr.size(), static_cast<std::size_t>(kCols + 1));
    for (size_t i = 0; i < static_cast<std::size_t>(kCols); ++i) {
      ASSERT_LT(h_sketch.columns_ptr[i], h_sketch.columns_ptr[i + 1]);
    }
  }
}
}  // namespace common
}  // namespace xgboost
