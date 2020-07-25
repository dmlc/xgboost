#include <gtest/gtest.h>
#include "../helpers.h"
#include "test_quantile.h"
#include "../../../src/common/hist_util.cuh"
#include "../../../src/common/quantile.cuh"

namespace xgboost {
namespace common {
TEST(GPUQuantile, Basic) {
  constexpr size_t kRows = 1000, kCols = 100, kBins = 256;
  SketchContainer sketch(kBins, kCols, kRows, 0);
  dh::caching_device_vector<SketchEntry> entries;
  dh::device_vector<bst_row_t> cuts_ptr(kCols+1);
  thrust::fill(cuts_ptr.begin(), cuts_ptr.end(), 0);
  // Push empty
  sketch.Push(dh::ToSpan(cuts_ptr), &entries);
  ASSERT_EQ(sketch.Data().size(), 0);
}

template <typename Fn> void RunWithSeedsAndBins(size_t rows, Fn fn) {
  std::vector<int32_t> seeds(4);
  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist(3, 1000);
  std::generate(seeds.begin(), seeds.end(), [&](){ return dist(&lcg); });

  std::vector<size_t> bins(8);
  for (size_t i = 0; i < bins.size() - 1; ++i) {
    bins[i] = i * 35 + 2;
  }
  bins.back() = rows + 80;  // provide a bin number greater than rows.

  std::vector<MetaInfo> infos(2);
  auto& h_weights = infos.front().weights_.HostVector();
  h_weights.resize(rows);
  std::generate(h_weights.begin(), h_weights.end(), [&]() { return dist(&lcg); });

  for (auto seed : seeds) {
    for (auto n_bin : bins) {
      for (auto const& info : infos) {
        fn(seed, n_bin, info);
      }
    }
  }
}

void TestSketchUnique(float sparsity) {
  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [kRows, kCols, sparsity](int32_t seed, size_t n_bins, MetaInfo const& info) {
    SketchContainer sketch(n_bins, kCols, kRows, 0);

    HostDeviceVector<float> storage;
    std::string interface_str = RandomDataGenerator{kRows, kCols, sparsity}
                                    .Seed(seed)
                                    .Device(0)
                                    .GenerateArrayInterface(&storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(adapter.Value(), n_bins, info,
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
    detail::GetColumnSizesScan(0, kCols, n_cuts, batch_iter, is_valid, 0, end,
                               &cut_sizes_scan, &column_sizes_scan);
    auto const& cut_sizes = cut_sizes_scan.HostVector();

    if (sparsity == 0) {
      ASSERT_EQ(sketch.Data().size(), n_cuts * kCols);
    } else {
      ASSERT_EQ(sketch.Data().size(), cut_sizes.back());
    }

    sketch.Unique();
    ASSERT_TRUE(thrust::is_sorted(thrust::device, sketch.Data().data(),
                                  sketch.Data().data() + sketch.Data().size(),
                                  detail::SketchUnique{}));
  });
}

TEST(GPUQuantile, Unique) {
  TestSketchUnique(0);
  TestSketchUnique(0.5);
}

// if with_error is true, the test tolerates floating point error
void TestQuantileElemRank(int32_t device, Span<SketchEntry const> in,
                          Span<bst_row_t const> d_columns_ptr, bool with_error = false) {
  dh::LaunchN(device, in.size(), [=]XGBOOST_DEVICE(size_t idx) {
    auto column_id = dh::SegmentId(d_columns_ptr, idx);
    auto in_column = in.subspan(d_columns_ptr[column_id],
                                d_columns_ptr[column_id + 1] -
                                    d_columns_ptr[column_id]);
    auto constexpr kEps = 1e-6f;
    idx -= d_columns_ptr[column_id];
    float prev_rmin = idx == 0 ? 0.0f : in_column[idx-1].rmin;
    float prev_rmax = idx == 0 ? 0.0f : in_column[idx-1].rmax;
    float rmin_next = in_column[idx].RMinNext();

    if (with_error) {
      SPAN_CHECK(in_column[idx].rmin + in_column[idx].rmin * kEps >= prev_rmin);
      SPAN_CHECK(in_column[idx].rmax + in_column[idx].rmin * kEps >= prev_rmax);
      SPAN_CHECK(in_column[idx].rmax + in_column[idx].rmin * kEps >= rmin_next);
    } else {
      SPAN_CHECK(in_column[idx].rmin >= prev_rmin);
      SPAN_CHECK(in_column[idx].rmax >= prev_rmax);
      SPAN_CHECK(in_column[idx].rmax >= rmin_next);
    }
  });
  // Force sync to terminate current test instead of a later one.
  dh::DebugSyncDevice(__FILE__, __LINE__);
}


TEST(GPUQuantile, Prune) {
  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [=](int32_t seed, size_t n_bins, MetaInfo const& info) {
    SketchContainer sketch(n_bins, kCols, kRows, 0);

    HostDeviceVector<float> storage;
    std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                    .Device(0)
                                    .Seed(seed)
                                    .GenerateArrayInterface(&storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(adapter.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch);
    auto n_cuts = detail::RequiredSampleCutsPerColumn(n_bins, kRows);
    ASSERT_EQ(sketch.Data().size(), n_cuts * kCols);

    sketch.Prune(n_bins);
    if (n_bins <= kRows) {
      ASSERT_EQ(sketch.Data().size(), n_bins * kCols);
    } else {
      // LE because kRows * kCols is pushed into sketch, after removing duplicated entries
      // we might not have that much inputs for prune.
      ASSERT_LE(sketch.Data().size(), kRows * kCols);
    }
    // This is not necessarily true for all inputs without calling unique after
    // prune.
    ASSERT_TRUE(thrust::is_sorted(thrust::device, sketch.Data().data(),
                                  sketch.Data().data() + sketch.Data().size(),
                                  detail::SketchUnique{}));
    TestQuantileElemRank(0, sketch.Data(), sketch.ColumnsPtr());
  });
}

TEST(GPUQuantile, MergeEmpty) {
  constexpr size_t kRows = 1000, kCols = 100;
  size_t n_bins = 10;
  SketchContainer sketch_0(n_bins, kCols, kRows, 0);
  HostDeviceVector<float> storage_0;
  std::string interface_str_0 =
      RandomDataGenerator{kRows, kCols, 0}.Device(0).GenerateArrayInterface(
          &storage_0);
  data::CupyAdapter adapter_0(interface_str_0);
  MetaInfo info;
  AdapterDeviceSketch(adapter_0.Value(), n_bins, info,
                      std::numeric_limits<float>::quiet_NaN(), &sketch_0);

  std::vector<SketchEntry> entries_before(sketch_0.Data().size());
  dh::CopyDeviceSpanToVector(&entries_before, sketch_0.Data());
  std::vector<bst_row_t> ptrs_before(sketch_0.ColumnsPtr().size());
  dh::CopyDeviceSpanToVector(&ptrs_before, sketch_0.ColumnsPtr());
  thrust::device_vector<size_t> columns_ptr(kCols + 1);
  // Merge an empty sketch
  sketch_0.Merge(dh::ToSpan(columns_ptr), Span<SketchEntry>{});

  std::vector<SketchEntry> entries_after(sketch_0.Data().size());
  dh::CopyDeviceSpanToVector(&entries_after, sketch_0.Data());
  std::vector<bst_row_t> ptrs_after(sketch_0.ColumnsPtr().size());
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
  RunWithSeedsAndBins(kRows, [=](int32_t seed, size_t n_bins, MetaInfo const& info) {
    SketchContainer sketch_0(n_bins, kCols, kRows, 0);
    HostDeviceVector<float> storage_0;
    std::string interface_str_0 = RandomDataGenerator{kRows, kCols, 0}
                                      .Device(0)
                                      .Seed(seed)
                                      .GenerateArrayInterface(&storage_0);
    data::CupyAdapter adapter_0(interface_str_0);
    AdapterDeviceSketch(adapter_0.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch_0);

    SketchContainer sketch_1(n_bins, kCols, kRows * kRows, 0);
    HostDeviceVector<float> storage_1;
    std::string interface_str_1 = RandomDataGenerator{kRows, kCols, 0}
                                      .Device(0)
                                      .Seed(seed)
                                      .GenerateArrayInterface(&storage_1);
    data::CupyAdapter adapter_1(interface_str_1);
    AdapterDeviceSketch(adapter_1.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(), &sketch_1);

    size_t size_before_merge = sketch_0.Data().size();
    sketch_0.Merge(sketch_1.ColumnsPtr(), sketch_1.Data());
    if (info.weights_.Size() != 0) {
      TestQuantileElemRank(0, sketch_0.Data(), sketch_0.ColumnsPtr(), true);
      sketch_0.FixError();
      TestQuantileElemRank(0, sketch_0.Data(), sketch_0.ColumnsPtr(), false);
    } else {
      TestQuantileElemRank(0, sketch_0.Data(), sketch_0.ColumnsPtr());
    }

    auto columns_ptr = sketch_0.ColumnsPtr();
    std::vector<bst_row_t> h_columns_ptr(columns_ptr.size());
    dh::CopyDeviceSpanToVector(&h_columns_ptr, columns_ptr);
    ASSERT_EQ(h_columns_ptr.back(), sketch_1.Data().size() + size_before_merge);

    sketch_0.Unique();
    ASSERT_TRUE(
        thrust::is_sorted(thrust::device, sketch_0.Data().data(),
                          sketch_0.Data().data() + sketch_0.Data().size(),
                          detail::SketchUnique{}));
  });
}

void TestMergeDuplicated(int32_t n_bins, size_t cols, size_t rows, float frac) {
  MetaInfo info;
  int32_t seed = 0;
  SketchContainer sketch_0(n_bins, cols, rows, 0);
  HostDeviceVector<float> storage_0;
  std::string interface_str_0 = RandomDataGenerator{rows, cols, 0}
                                    .Device(0)
                                    .Seed(seed)
                                    .GenerateArrayInterface(&storage_0);
  data::CupyAdapter adapter_0(interface_str_0);
  AdapterDeviceSketch(adapter_0.Value(), n_bins, info,
                      std::numeric_limits<float>::quiet_NaN(),
                      &sketch_0);

  size_t f_rows = rows * frac;
  SketchContainer sketch_1(n_bins, cols, f_rows, 0);
  HostDeviceVector<float> storage_1;
  std::string interface_str_1 = RandomDataGenerator{f_rows, cols, 0}
                                    .Device(0)
                                    .Seed(seed)
                                    .GenerateArrayInterface(&storage_1);
  auto data_1 = storage_1.DeviceSpan();
  auto tuple_it = thrust::make_tuple(
      thrust::make_counting_iterator<size_t>(0ul), data_1.data());
  using Tuple = thrust::tuple<size_t, float>;
  auto it = thrust::make_zip_iterator(tuple_it);
  thrust::transform(thrust::device, it, it + data_1.size(), data_1.data(),
                    [=] __device__(Tuple const &tuple) {
                      auto i = thrust::get<0>(tuple);
                      if (thrust::get<0>(tuple) % 2 == 0) {
                        return 0.0f;
                      } else {
                        return thrust::get<1>(tuple);
                      }
                    });
  data::CupyAdapter adapter_1(interface_str_1);
  AdapterDeviceSketch(adapter_1.Value(), n_bins, info,
                      std::numeric_limits<float>::quiet_NaN(),
                      &sketch_1);

  size_t size_before_merge = sketch_0.Data().size();
  sketch_0.Merge(sketch_1.ColumnsPtr(), sketch_1.Data());
  TestQuantileElemRank(0, sketch_0.Data(), sketch_0.ColumnsPtr());

  auto columns_ptr = sketch_0.ColumnsPtr();
  std::vector<bst_row_t> h_columns_ptr(columns_ptr.size());
  dh::CopyDeviceSpanToVector(&h_columns_ptr, columns_ptr);
  ASSERT_EQ(h_columns_ptr.back(), sketch_1.Data().size() + size_before_merge);

  sketch_0.Unique();
  ASSERT_TRUE(thrust::is_sorted(thrust::device, sketch_0.Data().data(),
                                sketch_0.Data().data() + sketch_0.Data().size(),
                                detail::SketchUnique{}));
}

TEST(GPUQuantile, MergeDuplicated) {
  size_t n_bins = 256;
  constexpr size_t kRows = 1000, kCols = 100;
  for (float frac = 0.5; frac < 2.5; frac += 0.5) {
    TestMergeDuplicated(n_bins, kRows, kCols, frac);
  }
}

void InitRabitContext(std::string msg) {
  auto n_gpus = AllVisibleGPUs();
  auto port = std::getenv("DMLC_TRACKER_PORT");
  std::string port_str;
  if (port) {
    port_str = port;
  } else {
    LOG(WARNING) << msg << " as `DMLC_TRACKER_PORT` is not set up.";
    return;
  }

  std::vector<std::string> envs{
      "DMLC_TRACKER_PORT=" + port_str,
      "DMLC_TRACKER_URI=127.0.0.1",
      "DMLC_NUM_WORKER=" + std::to_string(n_gpus)};
  char* c_envs[] {&(envs[0][0]), &(envs[1][0]), &(envs[2][0])};
  rabit::Init(3, c_envs);
}

TEST(GPUQuantile, AllReduceBasic) {
  // This test is supposed to run by a python test that setups the environment.
  std::string msg {"Skipping AllReduce test"};
#if defined(__linux__) && defined(XGBOOST_USE_NCCL)
  InitRabitContext(msg);
  auto n_gpus = AllVisibleGPUs();
  auto world = rabit::GetWorldSize();
  if (world != 1) {
    ASSERT_EQ(world, n_gpus);
  } else {
    return;
  }

  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [=](int32_t seed, size_t n_bins, MetaInfo const& info) {
    // Set up single node version;
    SketchContainer sketch_on_single_node(n_bins, kCols, kRows, 0);

    size_t intermediate_num_cuts =
        std::min(kRows * world, static_cast<size_t>(n_bins * WQSketch::kFactor));
    std::vector<SketchContainer> containers;
    for (auto rank = 0; rank < world; ++rank) {
      HostDeviceVector<float> storage;
      std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                      .Device(0)
                                      .Seed(rank + seed)
                                      .GenerateArrayInterface(&storage);
      data::CupyAdapter adapter(interface_str);
      containers.emplace_back(n_bins, kCols, kRows, 0);
      AdapterDeviceSketch(adapter.Value(), n_bins, info,
                          std::numeric_limits<float>::quiet_NaN(),
                          &containers.back());
    }
    for (auto& sketch : containers) {
      sketch.Prune(intermediate_num_cuts);
      sketch_on_single_node.Merge(sketch.ColumnsPtr(), sketch.Data());
      sketch_on_single_node.FixError();
    }
    sketch_on_single_node.Unique();
    TestQuantileElemRank(0, sketch_on_single_node.Data(),
                         sketch_on_single_node.ColumnsPtr());

    // Set up distributed version.  We rely on using rank as seed to generate
    // the exact same copy of data.
    auto rank = rabit::GetRank();
    SketchContainer sketch_distributed(n_bins, kCols, kRows, 0);
    HostDeviceVector<float> storage;
    std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                    .Device(0)
                                    .Seed(rank + seed)
                                    .GenerateArrayInterface(&storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(adapter.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(),
                        &sketch_distributed);
    sketch_distributed.AllReduce();
    sketch_distributed.Unique();

    ASSERT_EQ(sketch_distributed.ColumnsPtr().size(),
              sketch_on_single_node.ColumnsPtr().size());
    ASSERT_EQ(sketch_distributed.Data().size(),
              sketch_on_single_node.Data().size());

    TestQuantileElemRank(0, sketch_distributed.Data(),
                         sketch_distributed.ColumnsPtr());

    std::vector<SketchEntry> single_node_data(
        sketch_on_single_node.Data().size());
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
  rabit::Finalize();
#else
  LOG(WARNING) << msg;
  return;
#endif  // !defined(__linux__) && defined(XGBOOST_USE_NCCL)
}

TEST(GPUQuantile, SameOnAllWorkers) {
  std::string msg {"Skipping SameOnAllWorkers test"};
#if defined(__linux__) && defined(XGBOOST_USE_NCCL)
  InitRabitContext(msg);
  auto world = rabit::GetWorldSize();
  auto n_gpus = AllVisibleGPUs();
  if (world != 1) {
    ASSERT_EQ(world, n_gpus);
  } else {
    return;
  }

  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(kRows, [=](int32_t seed, size_t n_bins,
                                 MetaInfo const &info) {
    auto rank = rabit::GetRank();
    SketchContainer sketch_distributed(n_bins, kCols, kRows, 0);
    HostDeviceVector<float> storage;
    std::string interface_str = RandomDataGenerator{kRows, kCols, 0}
                                    .Device(0)
                                    .Seed(rank + seed)
                                    .GenerateArrayInterface(&storage);
    data::CupyAdapter adapter(interface_str);
    AdapterDeviceSketch(adapter.Value(), n_bins, info,
                        std::numeric_limits<float>::quiet_NaN(),
                        &sketch_distributed);
    sketch_distributed.AllReduce();
    sketch_distributed.Unique();
    TestQuantileElemRank(0, sketch_distributed.Data(), sketch_distributed.ColumnsPtr());

    // Test for all workers having the same sketch.
    size_t n_data = sketch_distributed.Data().size();
    rabit::Allreduce<rabit::op::Max>(&n_data, 1);
    ASSERT_EQ(n_data, sketch_distributed.Data().size());
    size_t size_as_float =
        sketch_distributed.Data().size_bytes() / sizeof(float);
    auto local_data = Span<float const>{
        reinterpret_cast<float const *>(sketch_distributed.Data().data()),
        size_as_float};

    dh::caching_device_vector<float> all_workers(size_as_float * world);
    thrust::fill(all_workers.begin(), all_workers.end(), 0);
    thrust::copy(thrust::device, local_data.data(),
                 local_data.data() + local_data.size(),
                 all_workers.begin() + local_data.size() * rank);
    dh::AllReducer reducer;
    reducer.Init(0);

    reducer.AllReduceSum(all_workers.data().get(), all_workers.data().get(),
                         all_workers.size());
    reducer.Synchronize();

    auto base_line = dh::ToSpan(all_workers).subspan(0, size_as_float);
    std::vector<float> h_base_line(base_line.size());
    dh::CopyDeviceSpanToVector(&h_base_line, base_line);

    size_t offset = 0;
    for (size_t i = 0; i < world; ++i) {
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
#else
  LOG(WARNING) << msg;
  return;
#endif  // !defined(__linux__) && defined(XGBOOST_USE_NCCL)
}

TEST(GPUQuantile, FromOneHot) {
  std::vector<float> x = BasicOneHotEncodedData();
  auto m = GetDMatrixFromData(x, 5, 3);
  int32_t max_bins = 16;
  auto cuts = DeviceSketch(0, m.get(), max_bins);

  std::vector<uint32_t> const& h_cuts_ptr = cuts.Ptrs();
  std::vector<float> h_cuts_values = cuts.Values();

  ValidateBasicOneHot(h_cuts_ptr, h_cuts_values);
}
}  // namespace common
}  // namespace xgboost
