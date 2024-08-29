/**
 * Copyright 2020-2024, XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/collective/allreduce.h"
#include "../../../src/common/hist_util.cuh"
#include "../../../src/common/quantile.cuh"
#include "../../../src/data/device_adapter.cuh"  // CupyAdapter
#include "../collective/test_worker.h"           // for BaseMGPUTest
#include "../helpers.h"
#include "test_quantile.h"

namespace xgboost {
namespace {
struct IsSorted {
  XGBOOST_DEVICE bool operator()(common::SketchEntry const& a, common::SketchEntry const& b) const {
    return a.value < b.value;
  }
};
}

namespace common {
class MGPUQuantileTest : public collective::BaseMGPUTest {};

TEST(GPUQuantile, Basic) {
  auto ctx = MakeCUDACtx(0);
  constexpr size_t kRows = 1000, kCols = 100, kBins = 256;
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch(ft, kBins, kCols, kRows, ctx.Device());
  dh::caching_device_vector<Entry> entries;
  dh::device_vector<bst_idx_t> cuts_ptr(kCols+1);
  thrust::fill(cuts_ptr.begin(), cuts_ptr.end(), 0);
  // Push empty
  sketch.Push(&ctx, dh::ToSpan(entries), dh::ToSpan(cuts_ptr), dh::ToSpan(cuts_ptr), 0);
  ASSERT_EQ(sketch.Data().size(), 0);
}

void TestSketchUnique(float sparsity) {
  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [kRows, kCols, sparsity](std::int32_t seed, bst_bin_t n_bins,
                                                      MetaInfo const& info) {
    auto ctx = MakeCUDACtx(0);
    HostDeviceVector<FeatureType> ft;
    SketchContainer sketch(ft, n_bins, kCols, kRows, ctx.Device());

    HostDeviceVector<float> storage;
    std::string interface_str = RandomDataGenerator{kRows, kCols, sparsity}
                                    .Seed(seed)
                                    .Device(ctx.Device())
                                    .GenerateArrayInterface(&storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(&ctx, adapter.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch);
    auto n_cuts = detail::RequiredSampleCutsPerColumn(n_bins, kRows);

    dh::caching_device_vector<size_t> column_sizes_scan;
    HostDeviceVector<size_t> cut_sizes_scan;
    auto batch = adapter.Value();
    data::IsValidFunctor is_valid(std::numeric_limits<float>::quiet_NaN());
    auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
        thrust::make_counting_iterator(0llu),
        [=] __device__(size_t idx) { return batch.GetElement(idx); });
    auto end = kCols * kRows;
    detail::GetColumnSizesScan(ctx.CUDACtx(), ctx.Device(), kCols, n_cuts,
                               IterSpan{batch_iter, end}, is_valid, &cut_sizes_scan,
                               &column_sizes_scan);
    auto const& cut_sizes = cut_sizes_scan.HostVector();
    ASSERT_LE(sketch.Data().size(), cut_sizes.back());

    std::vector<size_t> h_columns_ptr(sketch.ColumnsPtr().size());
    dh::CopyDeviceSpanToVector(&h_columns_ptr, sketch.ColumnsPtr());
    ASSERT_EQ(sketch.Data().size(), h_columns_ptr.back());

    sketch.Unique(&ctx);

    std::vector<SketchEntry> h_data(sketch.Data().size());
    thrust::copy(dh::tcbegin(sketch.Data()), dh::tcend(sketch.Data()), h_data.begin());

    for (size_t i = 1; i < h_columns_ptr.size(); ++i) {
      auto begin = h_columns_ptr[i - 1];
      auto column = common::Span<SketchEntry>(h_data).subspan(begin, h_columns_ptr[i] - begin);
      ASSERT_TRUE(std::is_sorted(column.begin(), column.end(), IsSorted{}));
    }
  });
}

TEST(GPUQuantile, Unique) {
  TestSketchUnique(0);
  TestSketchUnique(0.5);
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
        ASSERT_GE(in_column[idx].rmin + in_column[idx].rmin * kRtEps,
                  prev_rmin);
        ASSERT_GE(in_column[idx].rmax + in_column[idx].rmin * kRtEps, prev_rmax);
        ASSERT_GE(in_column[idx].rmax + in_column[idx].rmin * kRtEps,
                  rmin_next);
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
    SketchContainer sketch(ft, n_bins, kCols, kRows, ctx.Device());

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
    // This is not necessarily true for all inputs without calling unique after
    // prune.
    ASSERT_TRUE(thrust::is_sorted(thrust::device, sketch.Data().data(),
                                  sketch.Data().data() + sketch.Data().size(),
                                  detail::SketchUnique{}));
    TestQuantileElemRank(ctx.Device(), sketch.Data(), sketch.ColumnsPtr());
  });
}

TEST(GPUQuantile, MergeEmpty) {
  constexpr size_t kRows = 1000, kCols = 100;
  size_t n_bins = 10;
  auto ctx = MakeCUDACtx(0);
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch_0(ft, n_bins, kCols, kRows, ctx.Device());
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
    SketchContainer sketch_0(ft, n_bins, kCols, kRows, ctx.Device());
    HostDeviceVector<float> storage_0;
    std::string interface_str_0 = RandomDataGenerator{kRows, kCols, 0}
                                      .Device(ctx.Device())
                                      .Seed(seed)
                                      .GenerateArrayInterface(&storage_0);
    data::CupyAdapter adapter_0(interface_str_0);
    AdapterDeviceSketch(&ctx, adapter_0.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch_0);

    SketchContainer sketch_1(ft, n_bins, kCols, kRows * kRows, ctx.Device());
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
    if (info.weights_.Size() != 0) {
      TestQuantileElemRank(ctx.Device(), sketch_0.Data(), sketch_0.ColumnsPtr(), true);
      sketch_0.FixError();
      TestQuantileElemRank(ctx.Device(), sketch_0.Data(), sketch_0.ColumnsPtr(), false);
    } else {
      TestQuantileElemRank(ctx.Device(), sketch_0.Data(), sketch_0.ColumnsPtr());
    }

    auto columns_ptr = sketch_0.ColumnsPtr();
    std::vector<bst_idx_t> h_columns_ptr(columns_ptr.size());
    dh::CopyDeviceSpanToVector(&h_columns_ptr, columns_ptr);
    ASSERT_EQ(h_columns_ptr.back(), sketch_1.Data().size() + size_before_merge);

    sketch_0.Unique(&ctx);
    ASSERT_TRUE(
        thrust::is_sorted(thrust::device, sketch_0.Data().data(),
                          sketch_0.Data().data() + sketch_0.Data().size(),
                          detail::SketchUnique{}));
  });
}

void TestMergeDuplicated(int32_t n_bins, size_t cols, size_t rows, float frac) {
  auto ctx = MakeCUDACtx(0);
  MetaInfo info;
  int32_t seed = 0;
  HostDeviceVector<FeatureType> ft;
  SketchContainer sketch_0(ft, n_bins, cols, rows, ctx.Device());
  HostDeviceVector<float> storage_0;
  std::string interface_str_0 = RandomDataGenerator{rows, cols, 0}
                                    .Device(ctx.Device())
                                    .Seed(seed)
                                    .GenerateArrayInterface(&storage_0);
  data::CupyAdapter adapter_0(interface_str_0);
  AdapterDeviceSketch(&ctx, adapter_0.Value(), n_bins, info,
                      std::numeric_limits<float>::quiet_NaN(), &sketch_0);

  size_t f_rows = rows * frac;
  SketchContainer sketch_1(ft, n_bins, cols, f_rows, ctx.Device());
  HostDeviceVector<float> storage_1;
  std::string interface_str_1 = RandomDataGenerator{f_rows, cols, 0}
                                    .Device(ctx.Device())
                                    .Seed(seed)
                                    .GenerateArrayInterface(&storage_1);
  auto data_1 = storage_1.DeviceSpan();
  auto tuple_it = thrust::make_tuple(
      thrust::make_counting_iterator<size_t>(0ul), data_1.data());
  using Tuple = thrust::tuple<size_t, float>;
  auto it = thrust::make_zip_iterator(tuple_it);
  thrust::transform(thrust::device, it, it + data_1.size(), data_1.data(),
                    [=] XGBOOST_DEVICE(Tuple const& tuple) {
                      auto i = thrust::get<0>(tuple);
                      if (i % 2 == 0) {
                        return 0.0f;
                      } else {
                        return thrust::get<1>(tuple);
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

  sketch_0.Unique(&ctx);
  columns_ptr = sketch_0.ColumnsPtr();
  dh::CopyDeviceSpanToVector(&h_columns_ptr, columns_ptr);

  std::vector<SketchEntry> h_data(sketch_0.Data().size());
  dh::CopyDeviceSpanToVector(&h_data, sketch_0.Data());
  for (size_t i = 1; i < h_columns_ptr.size(); ++i) {
    auto begin = h_columns_ptr[i - 1];
    auto column = Span<SketchEntry> {h_data}.subspan(begin, h_columns_ptr[i] - begin);
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

TEST(GPUQuantile, MultiMerge) {
  constexpr size_t kRows = 20, kCols = 1;
  int32_t world = 2;
  RunWithSeedsAndBins(kRows, [=](std::int32_t seed, bst_bin_t n_bins, MetaInfo const& info) {
    // Set up single node version
    HostDeviceVector<FeatureType> ft;
    auto ctx = MakeCUDACtx(0);
    SketchContainer sketch_on_single_node(ft, n_bins, kCols, kRows, ctx.Device());

    size_t intermediate_num_cuts = std::min(
        kRows * world, static_cast<size_t>(n_bins * WQSketch::kFactor));
    std::vector<SketchContainer> containers;
    for (auto rank = 0; rank < world; ++rank) {
      HostDeviceVector<float> storage;
      std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                      .Device(ctx.Device())
                                      .Seed(rank + seed)
                                      .GenerateArrayInterface(&storage);
      data::CupyAdapter adapter(interface_str);
      HostDeviceVector<FeatureType> ft;
      containers.emplace_back(ft, n_bins, kCols, kRows, ctx.Device());
      AdapterDeviceSketch(&ctx, adapter.Value(), n_bins, info,
                          std::numeric_limits<float>::quiet_NaN(), &containers.back());
    }
    for (auto &sketch : containers) {
      sketch.Prune(&ctx, intermediate_num_cuts);
      sketch_on_single_node.Merge(&ctx, sketch.ColumnsPtr(), sketch.Data());
      sketch_on_single_node.FixError();
    }
    TestQuantileElemRank(ctx.Device(), sketch_on_single_node.Data(),
                         sketch_on_single_node.ColumnsPtr());

    sketch_on_single_node.Unique(&ctx);
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
void TestAllReduceBasic() {
  auto const world = collective::GetWorldSize();
  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [=](std::int32_t seed, bst_bin_t n_bins, MetaInfo const& info) {
    auto const device = DeviceOrd::CUDA(GPUIDX);
    auto ctx = MakeCUDACtx(device.ordinal);

    /**
     * Set up single node version.
     */
    HostDeviceVector<FeatureType> ft({}, device);
    SketchContainer sketch_on_single_node(ft, n_bins, kCols, kRows, device);

    size_t intermediate_num_cuts =
        std::min(kRows * world, static_cast<size_t>(n_bins * WQSketch::kFactor));
    std::vector<SketchContainer> containers;
    for (auto rank = 0; rank < world; ++rank) {
      HostDeviceVector<float> storage({}, device);
      std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                      .Device(device)
                                      .Seed(rank + seed)
                                      .GenerateArrayInterface(&storage);
      data::CupyAdapter adapter(interface_str);
      HostDeviceVector<FeatureType> ft({}, device);
      containers.emplace_back(ft, n_bins, kCols, kRows, device);
      AdapterDeviceSketch(&ctx, adapter.Value(), n_bins, info,
                          std::numeric_limits<float>::quiet_NaN(), &containers.back());
    }
    for (auto& sketch : containers) {
      sketch.Prune(&ctx, intermediate_num_cuts);
      sketch_on_single_node.Merge(&ctx, sketch.ColumnsPtr(), sketch.Data());
      sketch_on_single_node.FixError();
    }
    sketch_on_single_node.Unique(&ctx);
    TestQuantileElemRank(device, sketch_on_single_node.Data(), sketch_on_single_node.ColumnsPtr(),
                         true);

    /**
     * Set up distributed version.  We rely on using rank as seed to generate
     * the exact same copy of data.
     */
    auto rank = collective::GetRank();
    SketchContainer sketch_distributed(ft, n_bins, kCols, kRows, device);
    HostDeviceVector<float> storage({}, device);
    std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                    .Device(device)
                                    .Seed(rank + seed)
                                    .GenerateArrayInterface(&storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(&ctx, adapter.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch_distributed);
    if (world == 1) {
      auto n_samples_global = kRows * world;
      intermediate_num_cuts =
          std::min(n_samples_global, static_cast<size_t>(n_bins * SketchContainer::kFactor));
      sketch_distributed.Prune(&ctx, intermediate_num_cuts);
    }
    sketch_distributed.AllReduce(&ctx, false);
    sketch_distributed.Unique(&ctx);

    ASSERT_EQ(sketch_distributed.ColumnsPtr().size(), sketch_on_single_node.ColumnsPtr().size());
    ASSERT_EQ(sketch_distributed.Data().size(), sketch_on_single_node.Data().size());

    TestQuantileElemRank(device, sketch_distributed.Data(), sketch_distributed.ColumnsPtr(), true);

    std::vector<SketchEntry> single_node_data(sketch_on_single_node.Data().size());
    dh::CopyDeviceSpanToVector(&single_node_data, sketch_on_single_node.Data());

    std::vector<SketchEntry> distributed_data(sketch_distributed.Data().size());
    dh::CopyDeviceSpanToVector(&distributed_data, sketch_distributed.Data());
    float Eps = 2e-4 * world;

    for (size_t i = 0; i < single_node_data.size(); ++i) {
      ASSERT_NEAR(single_node_data[i].value, distributed_data[i].value, Eps);
      ASSERT_NEAR(single_node_data[i].rmax, distributed_data[i].rmax, Eps);
      ASSERT_NEAR(single_node_data[i].rmin, distributed_data[i].rmin, Eps);
      ASSERT_NEAR(single_node_data[i].wmin, distributed_data[i].wmin, Eps);
    }
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

  // Generate cuts for single node environment
  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);
  HistogramCuts single_node_cuts = common::DeviceSketch(&ctx, m.get(), kBins);

  auto const& sptrs = single_node_cuts.Ptrs();
  auto const& dptrs = distributed_cuts.Ptrs();
  auto const& svals = single_node_cuts.Values();
  auto const& dvals = distributed_cuts.Values();
  auto const& smins = single_node_cuts.MinValues();
  auto const& dmins = distributed_cuts.MinValues();

  EXPECT_EQ(sptrs.size(), dptrs.size());
  for (size_t i = 0; i < sptrs.size(); ++i) {
    EXPECT_EQ(sptrs[i], dptrs[i]) << "rank: " << rank << ", i: " << i;
  }

  EXPECT_EQ(svals.size(), dvals.size());
  for (size_t i = 0; i < svals.size(); ++i) {
    EXPECT_NEAR(svals[i], dvals[i], 2e-2f) << "rank: " << rank << ", i: " << i;
  }

  EXPECT_EQ(smins.size(), dmins.size());
  for (size_t i = 0; i < smins.size(); ++i) {
    EXPECT_FLOAT_EQ(smins[i], dmins[i]) << "rank: " << rank << ", i: " << i;
  }
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
    SketchContainer sketch_distributed(ft, n_bins, kCols, kRows, device);
    HostDeviceVector<float> storage({}, device);
    std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                    .Device(device)
                                    .Seed(rank + seed)
                                    .GenerateArrayInterface(&storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(&ctx, adapter.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch_distributed);
    sketch_distributed.AllReduce(&ctx, false);
    sketch_distributed.Unique(&ctx);
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
  SketchContainer sketch(ft, n_bins, kCols, kRows, ctx.Device());
  sketch.Push(&ctx, dh::ToSpan(d_entries), dh::ToSpan(columns_ptr), dh::ToSpan(columns_ptr), kRows, {});

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
  SketchContainer sketch(ft, n_bins, kCols, kRows, ctx.Device());
  dh::device_vector<Entry> d_entries {entries};

  dh::device_vector<size_t> columns_ptr(kCols + 1, 0);
  for (size_t i = 1; i < kCols + 1; ++i) {
    columns_ptr[i] = kRows;
  }
  thrust::inclusive_scan(thrust::device, columns_ptr.begin(), columns_ptr.end(),
                         columns_ptr.begin());
  dh::device_vector<size_t> cuts_ptr(columns_ptr);

  sketch.Push(&ctx, dh::ToSpan(d_entries), dh::ToSpan(columns_ptr), dh::ToSpan(cuts_ptr),
              kRows * kCols, {});

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
