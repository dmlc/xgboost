/**
 * Copyright 2020-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/iterator/zip_iterator.h>  // for make_zip_iterator

#include <cuda/std/tuple>  // for make_tuple, tuple
#include <string>
#include <vector>

#include "../../../src/collective/allreduce.h"
#include "../../../src/common/hist_util.cuh"
#include "../../../src/common/quantile.cuh"
#include "../../../src/data/device_adapter.cuh"  // CupyAdapter
#include "../collective/test_worker.h"           // for BaseMGPUTest
#include "../helpers.h"
#include "test_hist_util.h"
#include "test_quantile.h"

namespace xgboost {
namespace {
struct IsSorted {
  XGBOOST_DEVICE bool operator()(common::SketchEntry const& a, common::SketchEntry const& b) const {
    return a.value < b.value;
  }
};

struct RepeatedValueOp {
  std::size_t cols;

  XGBOOST_DEVICE float operator()(cuda::std::tuple<size_t, float> const& tuple) const {
    auto i = cuda::std::get<0>(tuple);
    auto ridx = i / cols;
    return static_cast<float>((ridx / 8) % 4);
  }
};

auto GenerateDenseData(std::size_t rows, std::size_t cols, std::uint64_t seed)
    -> std::vector<float> {
  HostDeviceVector<float> storage;
  RandomDataGenerator{static_cast<bst_idx_t>(rows), cols, 0}.Seed(seed).GenerateDense(&storage);
  return storage.HostVector();
}

auto MakeFullRowSplitDMatrix(std::size_t rows_per_worker, std::size_t cols, std::int32_t world,
                             std::int32_t seed) -> std::shared_ptr<DMatrix> {
  std::vector<float> full_data;
  full_data.reserve(rows_per_worker * cols * world);
  for (std::int32_t rank = 0; rank < world; ++rank) {
    auto block = GenerateDenseData(rows_per_worker, cols, rank + seed);
    full_data.insert(full_data.end(), block.cbegin(), block.cend());
  }
  return GetDMatrixFromData(full_data, rows_per_worker * world, cols);
}

auto RepeatRowWeights(std::vector<float> const& local_weights, std::int32_t world)
    -> std::vector<float> {
  std::vector<float> full_weights;
  full_weights.reserve(local_weights.size() * static_cast<std::size_t>(world));
  for (std::int32_t rank = 0; rank < world; ++rank) {
    full_weights.insert(full_weights.end(), local_weights.cbegin(), local_weights.cend());
  }
  return full_weights;
}

struct DeviceQueryBoundStats {
  double max_absolute_error{0.0};
  double total_weight{0.0};
  double epsilon{0.0};
  double prune_term{0.0};
  double target_rank{0.0};
  double rank_lo{0.0};
  double rank_hi{0.0};
  float queried_value{0.0f};
  std::size_t feature{0};
  std::size_t query_index{0};
};

struct DeviceWeightedScenario {
  char const* name{nullptr};
  bst_idx_t rows{0};
  bst_feature_t cols{0};
  bst_bin_t bins{0};
  bst_idx_t sketch_batch_num_elements{common::detail::UnknownSketchNumElements()};
  std::vector<float> data;
  std::vector<float> weights;
};

auto CollectDenseWeightedColumns(std::vector<float> const& data, bst_idx_t rows, bst_feature_t cols,
                                 std::vector<float> const& weights)
    -> std::vector<std::vector<common::WeightedValue>> {
  CHECK_EQ(data.size(), static_cast<std::size_t>(rows) * cols);
  CHECK_EQ(weights.size(), static_cast<std::size_t>(rows));
  std::vector<std::vector<common::WeightedValue>> columns(cols);
  for (bst_idx_t r = 0; r < rows; ++r) {
    auto w = static_cast<double>(weights[r]);
    for (bst_feature_t c = 0; c < cols; ++c) {
      columns[c].push_back({data[static_cast<std::size_t>(r) * cols + c], w});
    }
  }
  for (auto& column : columns) {
    std::sort(column.begin(), column.end(),
              [](auto const& lhs, auto const& rhs) { return lhs.value < rhs.value; });
  }
  return columns;
}

auto QueryHostColumn(common::Span<common::SketchEntry const> column, double rank)
    -> common::SketchEntry {
  CHECK(!column.empty());
  if (column.size() == 1) {
    return column.front();
  }
  auto rank2 = static_cast<float>(rank * 2.0);
  auto front = column.front();
  if (rank2 < front.rmin + front.rmax) {
    return front;
  }
  auto back = column.back();
  if (rank2 >= back.rmin + back.rmax) {
    return back;
  }

  auto it = std::upper_bound(
      column.cbegin() + 1, column.cend() - 1, rank2,
      [](float lhs, common::SketchEntry const& rhs) { return lhs < rhs.rmin + rhs.rmax; });
  auto i = static_cast<std::size_t>(std::distance(column.cbegin(), it) - 1);
  if (rank2 < column[i].RMinNext() + column[i + 1].RMaxPrev()) {
    return column[i];
  }
  return column[i + 1];
}

auto MeasureDeviceQueryBound(std::vector<common::SketchEntry> const& entries,
                             std::vector<bst_idx_t> const& columns_ptr,
                             std::vector<std::vector<common::WeightedValue>> const& columns,
                             double epsilon, double prune_term) -> DeviceQueryBoundStats {
  DeviceQueryBoundStats stats;
  stats.epsilon = epsilon;
  stats.prune_term = prune_term;

  for (std::size_t fidx = 0; fidx < columns.size(); ++fidx) {
    auto const& sorted_column = columns[fidx];
    if (sorted_column.empty()) {
      continue;
    }
    auto begin = columns_ptr[fidx];
    auto end = columns_ptr[fidx + 1];
    auto column = common::Span<common::SketchEntry const>{entries.data() + begin, end - begin};
    if (column.empty()) {
      continue;
    }

    std::vector<double> prefix_sum(sorted_column.size() + 1, 0.0);
    for (std::size_t i = 0; i < sorted_column.size(); ++i) {
      prefix_sum[i + 1] = prefix_sum[i] + sorted_column[i].weight;
    }
    auto total_weight = prefix_sum.back();
    auto n_queries = std::max<std::size_t>(1, column.size() * 4);
    for (std::size_t i = 0; i <= n_queries; ++i) {
      auto target_rank = static_cast<double>(i) * total_weight / static_cast<double>(n_queries);
      auto queried = QueryHostColumn(column, target_rank);
      auto lb = std::lower_bound(sorted_column.cbegin(), sorted_column.cend(), queried.value,
                                 [](auto const& lhs, float rhs) { return lhs.value < rhs; });
      auto ub = std::upper_bound(sorted_column.cbegin(), sorted_column.cend(), queried.value,
                                 [](float lhs, auto const& rhs) { return lhs < rhs.value; });
      auto rank_lo = prefix_sum[std::distance(sorted_column.cbegin(), lb)];
      auto rank_hi = prefix_sum[std::distance(sorted_column.cbegin(), ub)];
      auto absolute_error = common::DistanceToInterval(target_rank, rank_lo, rank_hi);
      if (absolute_error > stats.max_absolute_error) {
        stats.max_absolute_error = absolute_error;
        stats.total_weight = total_weight;
        stats.target_rank = target_rank;
        stats.rank_lo = rank_lo;
        stats.rank_hi = rank_hi;
        stats.queried_value = queried.value;
        stats.feature = fidx;
        stats.query_index = i;
      }
    }
  }
  return stats;
}

auto MakeDeviceWeightedScenarios() -> std::vector<DeviceWeightedScenario> {
  std::vector<DeviceWeightedScenario> scenarios;

  {
    DeviceWeightedScenario s;
    s.name = "continuous_random_weights";
    s.rows = 2048;
    s.cols = 5;
    s.bins = 64;
    s.data = common::GenerateRandom(s.rows, s.cols);
    s.weights = common::GenerateRandomWeights(s.rows);
    scenarios.push_back(std::move(s));
  }

  {
    DeviceWeightedScenario s;
    s.name = "bucketed_random_weights";
    s.rows = 2048;
    s.cols = 5;
    s.bins = 64;
    s.data = common::GenerateRandom(s.rows, s.cols);
    for (auto& v : s.data) {
      v = std::floor(v * 96.0f) / 8.0f;
    }
    s.weights = common::GenerateRandomWeights(s.rows);
    scenarios.push_back(std::move(s));
  }

  {
    DeviceWeightedScenario s;
    s.name = "random_weights_batched";
    s.rows = 1500;
    s.cols = 5;
    s.bins = 512;
    s.sketch_batch_num_elements = 512;
    s.data = common::GenerateRandom(s.rows, s.cols);
    s.weights = common::GenerateRandomWeights(s.rows);
    scenarios.push_back(std::move(s));
  }

  {
    DeviceWeightedScenario s;
    s.name = "product_weights_single_batch";
    s.rows = 1500;
    s.cols = 5;
    s.bins = 512;
    s.data = common::GenerateRandom(s.rows, s.cols);
    auto sample_weight = common::GenerateRandomWeights(s.rows);
    auto hessian = common::GenerateRandomWeights(s.rows);
    s.weights.resize(sample_weight.size());
    for (std::size_t i = 0; i < s.weights.size(); ++i) {
      s.weights[i] = sample_weight[i] * hessian[s.weights.size() - i - 1];
    }
    scenarios.push_back(std::move(s));
  }

  {
    DeviceWeightedScenario s;
    s.name = "product_weights_batched";
    s.rows = 1500;
    s.cols = 5;
    s.bins = 512;
    s.sketch_batch_num_elements = 512;
    s.data = common::GenerateRandom(s.rows, s.cols);
    auto sample_weight = common::GenerateRandomWeights(s.rows);
    auto hessian = common::GenerateRandomWeights(s.rows);
    s.weights.resize(sample_weight.size());
    for (std::size_t i = 0; i < s.weights.size(); ++i) {
      s.weights[i] = sample_weight[i] * hessian[s.weights.size() - i - 1];
    }
    scenarios.push_back(std::move(s));
  }

  return scenarios;
}

auto FindDeviceScenario(std::string const& name) -> DeviceWeightedScenario {
  for (auto const& scenario : MakeDeviceWeightedScenarios()) {
    if (name == scenario.name) {
      return scenario;
    }
  }
  LOG(FATAL) << "Unknown device weighted scenario: " << name;
  return {};
}

auto BuildWeightedSketch(Context const* ctx, DeviceWeightedScenario const& scenario,
                         bst_idx_t sketch_batch_num_elements) -> common::SketchContainer {
  thrust::device_vector<float> d_data(scenario.data.begin(), scenario.data.end());
  auto adapter = common::AdapterFromData(d_data, scenario.rows, scenario.cols);

  MetaInfo info;
  info.weights_.HostVector() = scenario.weights;
  HostDeviceVector<FeatureType> ft;
  common::SketchContainer sketch(ft, scenario.bins, scenario.cols, ctx->Device());
  common::AdapterDeviceSketch(ctx, adapter.Value(), scenario.bins, info,
                              std::numeric_limits<float>::quiet_NaN(), &sketch,
                              sketch_batch_num_elements);
  return sketch;
}

struct CutRankStats {
  double max_normalized_error{0.0};
  double max_absolute_error{0.0};
  bst_feature_t feature{0};
  std::size_t cut_index{0};
};

auto MeasureCutRankError(common::HistogramCuts const& cuts, DeviceWeightedScenario const& scenario)
    -> CutRankStats {
  auto columns =
      CollectDenseWeightedColumns(scenario.data, scenario.rows, scenario.cols, scenario.weights);
  CutRankStats out;
  for (bst_feature_t fidx = 0; fidx < scenario.cols; ++fidx) {
    auto stats = common::MeasureRankError(cuts, fidx, columns[fidx]);
    if (stats.max_normalized_error > out.max_normalized_error) {
      out.max_normalized_error = stats.max_normalized_error;
      out.max_absolute_error = stats.max_absolute_error;
      out.feature = fidx;
      out.cut_index = stats.cut_index;
    }
  }
  return out;
}

void CheckDeviceWeightedQueryBound(char const* test_name) {
  auto ctx = MakeCUDACtx(0);
  for (auto const& scenario : MakeDeviceWeightedScenarios()) {
    auto trace = std::string(test_name) + ":" + scenario.name;
    SCOPED_TRACE(trace);
    thrust::device_vector<float> d_data(scenario.data.begin(), scenario.data.end());
    auto adapter = common::AdapterFromData(d_data, scenario.rows, scenario.cols);

    MetaInfo info;
    info.weights_.HostVector() = scenario.weights;
    HostDeviceVector<FeatureType> ft;
    common::SketchContainer sketch(ft, scenario.bins, scenario.cols, ctx.Device());
    common::AdapterDeviceSketch(&ctx, adapter.Value(), scenario.bins, info,
                                std::numeric_limits<float>::quiet_NaN(), &sketch,
                                scenario.sketch_batch_num_elements);

    std::vector<common::SketchEntry> h_data(sketch.Data().size());
    dh::CopyDeviceSpanToVector(&h_data, sketch.Data());
    std::vector<bst_idx_t> h_columns_ptr(sketch.ColumnsPtr().size());
    dh::CopyDeviceSpanToVector(&h_columns_ptr, sketch.ColumnsPtr());

    auto columns =
        CollectDenseWeightedColumns(scenario.data, scenario.rows, scenario.cols, scenario.weights);
    auto eps = common::SketchEpsilon(scenario.bins, scenario.rows);
    auto budget = common::SketchSummaryBudget(scenario.bins, scenario.rows);
    auto stats = MeasureDeviceQueryBound(h_data, h_columns_ptr, columns, eps,
                                         1.0 / static_cast<double>(budget));
    auto bound = (stats.epsilon + stats.prune_term) * stats.total_weight;
    auto tol = std::max(1e-6, 1e-6 * stats.total_weight);
    EXPECT_LE(stats.max_absolute_error, bound + tol)
        << "scenario=" << scenario.name << ", feature=" << stats.feature
        << ", query_index=" << stats.query_index << ", queried_value=" << stats.queried_value
        << ", target_rank=" << stats.target_rank << ", rank_lo=" << stats.rank_lo
        << ", rank_hi=" << stats.rank_hi << ", total_weight=" << stats.total_weight
        << ", epsilon=" << stats.epsilon << ", prune_term=" << stats.prune_term
        << ", bound=" << bound;
  }
}
}  // namespace

namespace common {
class MGPUQuantileTest : public collective::BaseMGPUTest {};

TEST(GPUQuantile, Basic) {
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

// if with_error is true, the test tolerates floating point error
void TestQuantileElemRank(DeviceOrd device, Span<SketchEntry const> in,
                          Span<bst_idx_t const> d_columns_ptr, bool with_error = false) {
  dh::safe_cuda(cudaSetDevice(device.ordinal));
  std::vector<SketchEntry> h_in(in.size());
  dh::CopyDeviceSpanToVector(&h_in, in);
  std::vector<bst_idx_t> h_columns_ptr(d_columns_ptr.size());
  dh::CopyDeviceSpanToVector(&h_columns_ptr, d_columns_ptr);

  for (size_t i = 1; i < d_columns_ptr.size(); ++i) {
    auto column_id = i - 1;
    auto beg = h_columns_ptr[column_id];
    auto end = h_columns_ptr[i];

    auto in_column = Span<SketchEntry>{h_in}.subspan(beg, end - beg);
    for (size_t idx = 1; idx < in_column.size(); ++idx) {
      float prev_rmin = in_column[idx - 1].rmin;
      float prev_rmax = in_column[idx - 1].rmax;
      float rmin_next = in_column[idx].RMinNext();
      if (with_error) {
        ASSERT_GE(in_column[idx].rmin + in_column[idx].rmin * kRtEps, prev_rmin);
        ASSERT_GE(in_column[idx].rmax + in_column[idx].rmin * kRtEps, prev_rmax);
        ASSERT_GE(in_column[idx].rmax + in_column[idx].rmin * kRtEps, rmin_next);
      } else {
        ASSERT_GE(in_column[idx].rmin, prev_rmin);
        ASSERT_GE(in_column[idx].rmax, prev_rmax);
        ASSERT_GE(in_column[idx].rmax, rmin_next);
      }
    }
  }
}

TEST(GPUQuantile, Prune) {
  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [=](std::int32_t seed, bst_bin_t n_bins, MetaInfo const& info) {
    auto ctx = MakeCUDACtx(0);
    HostDeviceVector<FeatureType> ft;
    SketchContainer sketch(ft, n_bins, kCols, ctx.Device());

    HostDeviceVector<float> storage;
    std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                    .Device(ctx.Device())
                                    .Seed(seed)
                                    .GenerateArrayInterface(&storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(&ctx, adapter.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch);
    auto n_cuts = detail::RequiredSampleCutsPerColumn(n_bins, kRows);
    // LE because kRows * kCols is pushed into sketch, after removing
    // duplicated entries we might not have that much inputs for prune.
    ASSERT_LE(sketch.Data().size(), n_cuts * kCols);

    sketch.Prune(&ctx, n_bins);
    ASSERT_LE(sketch.Data().size(), kRows * kCols);
    std::vector<bst_idx_t> h_columns_ptr(sketch.ColumnsPtr().size());
    dh::CopyDeviceSpanToVector(&h_columns_ptr, sketch.ColumnsPtr());
    std::vector<SketchEntry> h_data(sketch.Data().size());
    dh::CopyDeviceSpanToVector(&h_data, sketch.Data());
    for (size_t i = 1; i < h_columns_ptr.size(); ++i) {
      auto begin = h_columns_ptr[i - 1];
      auto column = Span<SketchEntry>{h_data}.subspan(begin, h_columns_ptr[i] - begin);
      ASSERT_TRUE(std::adjacent_find(column.begin(), column.end(),
                                     [](SketchEntry const& l, SketchEntry const& r) {
                                       return l.value == r.value;
                                     }) == column.end());
    }
    TestQuantileElemRank(ctx.Device(), sketch.Data(), sketch.ColumnsPtr());
  });
}

TEST(GPUQuantile, PruneDuplicated) {
  constexpr size_t kRows = 512, kCols = 8;
  RunWithSeedsAndBins(kRows, [=](std::int32_t seed, bst_bin_t n_bins, MetaInfo const& info) {
    auto ctx = MakeCUDACtx(0);
    HostDeviceVector<FeatureType> ft;
    SketchContainer sketch(ft, n_bins, kCols, ctx.Device());

    HostDeviceVector<float> storage;
    std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                    .Device(ctx.Device())
                                    .Seed(seed)
                                    .GenerateArrayInterface(&storage);
    auto d_data = storage.DeviceSpan();
    auto tuple_it =
        cuda::std::make_tuple(thrust::make_counting_iterator<size_t>(0ul), d_data.data());
    auto it = thrust::make_zip_iterator(tuple_it);
    thrust::transform(ctx.CUDACtx()->CTP(), it, it + d_data.size(), d_data.data(),
                      RepeatedValueOp{kCols});

    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(&ctx, adapter.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch);

    sketch.Prune(&ctx, n_bins);

    std::vector<bst_idx_t> h_columns_ptr(sketch.ColumnsPtr().size());
    dh::CopyDeviceSpanToVector(&h_columns_ptr, sketch.ColumnsPtr());
    std::vector<SketchEntry> h_data(sketch.Data().size());
    dh::CopyDeviceSpanToVector(&h_data, sketch.Data());
    for (size_t i = 1; i < h_columns_ptr.size(); ++i) {
      auto begin = h_columns_ptr[i - 1];
      auto column = Span<SketchEntry>{h_data}.subspan(begin, h_columns_ptr[i] - begin);
      ASSERT_TRUE(std::adjacent_find(column.begin(), column.end(),
                                     [](SketchEntry const& l, SketchEntry const& r) {
                                       return l.value == r.value;
                                     }) == column.end());
    }
    TestQuantileElemRank(ctx.Device(), sketch.Data(), sketch.ColumnsPtr());
  });
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
    TestQuantileElemRank(ctx.Device(), sketch_0.Data(), sketch_0.ColumnsPtr());

    auto columns_ptr = sketch_0.ColumnsPtr();
    std::vector<bst_idx_t> h_columns_ptr(columns_ptr.size());
    dh::CopyDeviceSpanToVector(&h_columns_ptr, columns_ptr);
    ASSERT_EQ(h_columns_ptr.back(), sketch_1.Data().size() + size_before_merge);

    std::vector<SketchEntry> h_data(sketch_0.Data().size());
    dh::CopyDeviceSpanToVector(&h_data, sketch_0.Data());
    for (size_t i = 1; i < h_columns_ptr.size(); ++i) {
      auto begin = h_columns_ptr[i - 1];
      auto column = Span<SketchEntry>{h_data}.subspan(begin, h_columns_ptr[i] - begin);
      ASSERT_TRUE(std::is_sorted(column.begin(), column.end(), IsSorted{}));
    }
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
  TestQuantileElemRank(ctx.Device(), sketch_0.Data(), sketch_0.ColumnsPtr());

  auto columns_ptr = sketch_0.ColumnsPtr();
  std::vector<bst_idx_t> h_columns_ptr(columns_ptr.size());
  dh::CopyDeviceSpanToVector(&h_columns_ptr, columns_ptr);
  ASSERT_EQ(h_columns_ptr.back(), sketch_1.Data().size() + size_before_merge);

  std::vector<SketchEntry> h_data(sketch_0.Data().size());
  dh::CopyDeviceSpanToVector(&h_data, sketch_0.Data());
  for (size_t i = 1; i < h_columns_ptr.size(); ++i) {
    auto begin = h_columns_ptr[i - 1];
    auto column = Span<SketchEntry>{h_data}.subspan(begin, h_columns_ptr[i] - begin);
    ASSERT_TRUE(std::is_sorted(column.begin(), column.end(), IsSorted{}));
  }
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
  TestQuantileElemRank(ctx.Device(), sketch_0.Data(), sketch_0.ColumnsPtr());

  std::vector<bst_idx_t> h_columns_ptr(sketch_0.ColumnsPtr().size());
  dh::CopyDeviceSpanToVector(&h_columns_ptr, sketch_0.ColumnsPtr());
  std::vector<SketchEntry> h_data(sketch_0.Data().size());
  dh::CopyDeviceSpanToVector(&h_data, sketch_0.Data());

  auto cat_column = Span<SketchEntry>{h_data}.subspan(h_columns_ptr[0], h_columns_ptr[1]);
  ASSERT_TRUE(std::adjacent_find(cat_column.begin(), cat_column.end(),
                                 [](SketchEntry const& l, SketchEntry const& r) {
                                   return l.value == r.value;
                                 }) == cat_column.end());
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
    TestQuantileElemRank(ctx.Device(), sketch_on_single_node.Data(),
                         sketch_on_single_node.ColumnsPtr());
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
inline constexpr double kMaxDistributedWeightedNormalizedRankError = 20.0;

void TestAllReduceBasic() {
  auto const world = collective::GetWorldSize();
  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [=](std::int32_t seed, bst_bin_t n_bins, MetaInfo const& info) {
    auto const device = DeviceOrd::CUDA(GPUIDX);
    auto ctx = MakeCUDACtx(device.ordinal);

    /**
     * Set up distributed version.  We rely on using rank as seed to generate
     * the exact same copy of data.
     */
    auto rank = collective::GetRank();
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
    auto distributed_cuts = sketch_distributed.MakeCuts(&ctx, false);
    TestQuantileElemRank(device, sketch_distributed.Data(), sketch_distributed.ColumnsPtr(), true);
    auto full = MakeFullRowSplitDMatrix(kRows, kCols, world, seed);
    auto max_rank_error = info.weights_.Empty() ? kMaxNormalizedRankError
                                                : kMaxDistributedWeightedNormalizedRankError;
    ValidateCuts(distributed_cuts, full.get(), n_bins, max_rank_error);
  });
}
}  // anonymous namespace

TEST_F(MGPUQuantileTest, AllReduceBasic) {
  this->DoTest([] { TestAllReduceBasic(); }, true);
  this->DoTest([] { TestAllReduceBasic(); }, false);
}

namespace {
void TestColumnSplit(DMatrix* dmat) {
  auto const world = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  auto m = std::unique_ptr<DMatrix>{dmat->SliceCol(world, rank)};

  // Generate cuts for distributed environment.
  auto ctx = MakeCUDACtx(GPUIDX);
  std::size_t constexpr kBins = 64;
  HistogramCuts distributed_cuts = common::DeviceSketch(&ctx, m.get(), kBins);
  ValidateCuts(distributed_cuts, m.get(), kBins);
}
}  // anonymous namespace

TEST_F(MGPUQuantileTest, ColumnSplitBasic) {
  std::size_t constexpr kRows = 1000, kCols = 100;
  auto dmat = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();
  this->DoTest([&] { TestColumnSplit(dmat.get()); }, true);
  this->DoTest([&] { TestColumnSplit(dmat.get()); }, false);
}

TEST_F(MGPUQuantileTest, ColumnSplitCategorical) {
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
  this->DoTest([&] { TestColumnSplit(dmat.get()); }, true);
  this->DoTest([&] { TestColumnSplit(dmat.get()); }, false);
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
    TestQuantileElemRank(device, sketch_distributed.Data(), sketch_distributed.ColumnsPtr(), true);

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

TEST(GPUQuantile, WeightedSummaryQueryBound) {
  CheckDeviceWeightedQueryBound("gpu_summary_query_bound");
}

TEST(GPUQuantile, WeightedCutRankError) {
  auto scenario = FindDeviceScenario("product_weights_batched");
  auto ctx = MakeCUDACtx(0);

  auto cut_sketch = BuildWeightedSketch(&ctx, scenario, scenario.sketch_batch_num_elements);
  auto cuts = cut_sketch.MakeCuts(&ctx, false);
  auto cut_stats = MeasureCutRankError(cuts, scenario);
  EXPECT_LE(cut_stats.max_normalized_error, 0.5)
      << "feature=" << cut_stats.feature << ", cut=" << cut_stats.cut_index
      << ", absolute_error=" << cut_stats.max_absolute_error;
}
}  // namespace common
}  // namespace xgboost
