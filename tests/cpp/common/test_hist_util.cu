/**
 * Copyright 2019-2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <xgboost/base.h>  // for bst_bin_t
#include <xgboost/data.h>

#include <algorithm>  // for transform
#include <array>
#include <cmath>    // for floor
#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits
#include <vector>   // for vector

#include "../../../include/xgboost/logging.h"
#include "../../../src/common/cuda_context.cuh"
#include "../../../src/common/cuda_rt_utils.h"  // for SetDevice
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/hist_util.cuh"
#include "../../../src/common/hist_util.h"
#include "../data/test_array_interface.h"
#include "../filesystem.h"  // for TemporaryDirectory
#include "../helpers.h"
#include "test_hist_util.h"

namespace xgboost::common {

template <typename Batch>
struct GetBatchElementOp {
  Batch batch;

  __device__ data::COOTuple operator()(std::size_t idx) const { return batch.GetElement(idx); }
};

TEST(HistUtil, DeviceSketchPeakMemory) {
  auto ctx = MakeCUDACtx(0);
  int num_columns = 2048;
  int num_rows = 32768;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  [[maybe_unused]] auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
  ConsoleLogger::Configure({{"verbosity", "0"}});

  auto constexpr kPeakMemoryCeiling = static_cast<std::size_t>(1400) << 20;
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), kPeakMemoryCeiling);
}

TEST(HistUtil, DeviceSketchWeightsPeakMemory) {
  auto ctx = MakeCUDACtx(0);
  int num_columns = 2048;
  int num_rows = 32768;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
  dmat->Info().weights_.HostVector() = GenerateRandomWeights(num_rows);

  dh::GlobalMemoryLogger().Clear();
  ConsoleLogger::Configure({{"verbosity", "3"}});
  [[maybe_unused]] auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
  ConsoleLogger::Configure({{"verbosity", "0"}});

  auto constexpr kPeakMemoryCeiling = static_cast<std::size_t>(2100) << 20;
  EXPECT_LE(dh::GlobalMemoryLogger().PeakMemory(), kPeakMemoryCeiling);
}

TEST(HistUtil, DeviceSketchDeterminism) {
  auto ctx = MakeCUDACtx(0);
  int num_rows = 500;
  int num_columns = 5;
  int num_bins = 256;
  auto x = GenerateRandom(num_rows, num_columns);
  auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
  auto reference_sketch = DeviceSketch(&ctx, dmat.get(), num_bins);
  size_t constexpr kRounds{100};
  for (size_t r = 0; r < kRounds; ++r) {
    auto new_sketch = DeviceSketch(&ctx, dmat.get(), num_bins);
    ASSERT_EQ(reference_sketch.Values(), new_sketch.Values());
  }
}

TEST(HistUtil, RemoveDuplicatedCategories) {
  bst_idx_t n_samples = 512;
  bst_feature_t n_features = 3;
  bst_cat_t n_categories = 5;

  auto ctx = MakeCUDACtx(0);
  SimpleLCG rng;
  SimpleRealUniformDistribution<float> cat_d{0.0f, static_cast<float>(n_categories)};

  dh::device_vector<Entry> sorted_entries(n_samples * n_features);
  for (std::size_t i = 0; i < n_samples; ++i) {
    for (bst_feature_t j = 0; j < n_features; ++j) {
      float fvalue{0.0f};
      // The second column is categorical
      if (j == 1) {
        fvalue = std::floor(cat_d(&rng));
      } else {
        fvalue = i;
      }
      sorted_entries[i * n_features + j] = Entry{j, fvalue};
    }
  }

  MetaInfo info;
  info.num_col_ = n_features;
  info.num_row_ = n_samples;
  info.feature_types.HostVector() = std::vector<FeatureType>{
      FeatureType::kNumerical, FeatureType::kCategorical, FeatureType::kNumerical};
  ASSERT_EQ(info.feature_types.Size(), n_features);

  dh::device_vector<float> weight(n_samples * n_features, 0);
  dh::Iota(dh::ToSpan(weight), ctx.CUDACtx()->Stream());

  dh::caching_device_vector<bst_idx_t> columns_ptr(4);
  for (std::size_t i = 0; i < columns_ptr.size(); ++i) {
    columns_ptr[i] = i * n_samples;
  }
  // sort into column major
  thrust::sort_by_key(sorted_entries.begin(), sorted_entries.end(), weight.begin(),
                      detail::EntryCompareOp());

  detail::RemoveDuplicatedCategories(&ctx, info, &sorted_entries, &weight, &columns_ptr);

  auto const& h_cptr = columns_ptr;
  ASSERT_EQ(h_cptr.back(), n_samples * 2 + n_categories);
  // check numerical
  for (std::size_t i = 0; i < n_samples; ++i) {
    ASSERT_EQ(weight[i], i * 3);
  }
  auto beg = n_samples + n_categories;
  for (std::size_t i = 0; i < n_samples; ++i) {
    ASSERT_EQ(weight[i + beg], i * 3 + 2);
  }
  // check categorical
  beg = n_samples;
  for (bst_cat_t i = 0; i < n_categories; ++i) {
    // all from the second column
    ASSERT_EQ(static_cast<bst_feature_t>(weight[i + beg]) % n_features, 1);
  }
}

TEST(HistUtil, DeviceSketchCutInvariants) {
  auto ctx = MakeCUDACtx(0);
  auto bin_sizes = {2, 16, 256, 512};
  auto sizes = {100, 1000, 1500};
  std::array<bool, 2> weighted = {false, true};
  int constexpr kNumColumns = 5;
  for (auto num_rows : sizes) {
    auto data = GenerateRandom(num_rows, kNumColumns);
    std::vector<FeatureType> ft(kNumColumns, FeatureType::kNumerical);
    for (std::size_t ridx = 0; ridx < static_cast<std::size_t>(num_rows); ++ridx) {
      data[ridx * kNumColumns + 1] = static_cast<float>(ridx % 7);
      data[ridx * kNumColumns + 2] = static_cast<float>(ridx % 5);
      data[ridx * kNumColumns + 4] = static_cast<float>(ridx % 17);
    }
    ft[1] = FeatureType::kCategorical;
    ft[4] = FeatureType::kCategorical;

    HostDeviceVector<float> x{data};
    common::TemporaryDirectory temp;
    std::vector<std::shared_ptr<DMatrix>> matrices{
        GetDMatrixFromData(data, num_rows, kNumColumns),
        GetExternalMemoryDMatrixFromData(x, num_rows, kNumColumns, temp)};
    for (auto const& dmat : matrices) {
      dmat->Info().feature_types.HostVector() = ft;
      for (bool use_weights : weighted) {
        if (use_weights) {
          dmat->Info().weights_.HostVector() = GenerateRandomWeights(num_rows);
        } else {
          dmat->Info().weights_.HostVector().clear();
        }
        for (auto num_bins : bin_sizes) {
          auto cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
          ValidateCuts(cuts, dmat.get(), num_bins);
        }
      }
    }
  }
}

TEST(HistUtil, GetColumnSize) {
  bst_idx_t n_samples = 4096;
  auto ctx = MakeCUDACtx(0);
  bst_feature_t n_features = 12;
  curt::SetDevice(ctx.Ordinal());
  auto n = n_samples * n_features;
  std::vector<float> x(n);
  std::iota(x.begin(), x.end(), 0.0f);
  float missing = n_samples * n_features;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (i % 5 == 0) {
      x[i] = missing;
    }
  }
  thrust::device_vector<float> d_x = x;
  auto is_valid = data::IsValidFunctor{missing};

  auto adapter = AdapterFromData(d_x, n_samples, n_features);
  auto batch = adapter.Value();
  using Batch = decltype(batch);

  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(thrust::make_counting_iterator(0llu),
                                                              GetBatchElementOp<Batch>{batch});

  dh::caching_device_vector<std::size_t> column_sizes_scan;
  column_sizes_scan.resize(n_features + 1);
  std::vector<std::size_t> h_column_size(column_sizes_scan.size());
  std::vector<std::size_t> h_column_size_1(column_sizes_scan.size());

  auto cuctx = ctx.CUDACtx();
  detail::LaunchGetColumnSizeKernel<decltype(batch_iter), true, true>(
      cuctx, ctx.Device(), IterSpan{batch_iter, batch.Size()}, is_valid,
      dh::ToSpan(column_sizes_scan));
  thrust::copy(column_sizes_scan.begin(), column_sizes_scan.end(), h_column_size.begin());

  detail::LaunchGetColumnSizeKernel<decltype(batch_iter), true, false>(
      cuctx, ctx.Device(), IterSpan{batch_iter, batch.Size()}, is_valid,
      dh::ToSpan(column_sizes_scan));
  thrust::copy(column_sizes_scan.begin(), column_sizes_scan.end(), h_column_size_1.begin());
  ASSERT_EQ(h_column_size, h_column_size_1);

  detail::LaunchGetColumnSizeKernel<decltype(batch_iter), false, true>(
      cuctx, ctx.Device(), IterSpan{batch_iter, batch.Size()}, is_valid,
      dh::ToSpan(column_sizes_scan));
  thrust::copy(column_sizes_scan.begin(), column_sizes_scan.end(), h_column_size_1.begin());
  ASSERT_EQ(h_column_size, h_column_size_1);

  detail::LaunchGetColumnSizeKernel<decltype(batch_iter), false, false>(
      cuctx, ctx.Device(), IterSpan{batch_iter, batch.Size()}, is_valid,
      dh::ToSpan(column_sizes_scan));
  thrust::copy(column_sizes_scan.begin(), column_sizes_scan.end(), h_column_size_1.begin());
  ASSERT_EQ(h_column_size, h_column_size_1);
}

TEST(HistUtil, AdapterSketch) {
  auto ctx = MakeCUDACtx(0);
  auto bin_sizes = {2, 16, 256, 512};
  auto sizes = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      auto dmat_cuts = DeviceSketch(&ctx, dmat.get(), num_bins);
      auto x_device = thrust::device_vector<float>(x);
      auto adapter = AdapterFromData(x_device, num_rows, num_columns);
      HostDeviceVector<FeatureType> ft;
      SketchContainer sketch_container(ft, num_bins, adapter.NumColumns(), DeviceOrd::CUDA(0));
      MetaInfo info;
      AdapterDeviceSketch(&ctx, adapter.Value(), num_bins, info,
                          std::numeric_limits<float>::quiet_NaN(), &sketch_container);
      common::HistogramCuts adapter_cuts = sketch_container.MakeCuts(&ctx, info.IsColumnSplit());
      EXPECT_EQ(dmat_cuts.Values(), adapter_cuts.Values());
      EXPECT_EQ(dmat_cuts.Ptrs(), adapter_cuts.Ptrs());
    }
  }
}

TEST(HistUtil, GroupWeightsEquivalentToRowWeightsOnDevice) {
  size_t constexpr kRows = 300, kCols = 20, kBins = 256;
  size_t constexpr kGroups = 10;
  HostDeviceVector<float> storage;
  auto interface_str = RandomDataGenerator{kRows, kCols, 0}
                           .Device(DeviceOrd::CUDA(0))
                           .GenerateArrayInterface(&storage);
  auto ctx = MakeCUDACtx(0);
  std::vector<bst_group_t> group_sizes(kGroups, kRows / kGroups);
  std::vector<bst_group_t> group_ptr(kGroups + 1, 0);
  std::partial_sum(group_sizes.begin(), group_sizes.end(), group_ptr.begin() + 1);
  std::vector<float> group_weights(kGroups);
  for (size_t i = 0; i < group_weights.size(); ++i) {
    group_weights[i] = static_cast<float>(i + 1) / static_cast<float>(kGroups);
  }

  auto make_adapter_cuts = [&] {
    MetaInfo info;
    info.SetInfo(ctx, "group", Make1dInterfaceTest(group_sizes.data(), group_sizes.size()));
    info.weights_.HostVector() = group_weights;
    info.weights_.SetDevice(DeviceOrd::CUDA(0));
    info.num_row_ = kRows;
    info.num_col_ = kCols;

    data::CupyAdapter adapter(interface_str);
    HostDeviceVector<FeatureType> ft;
    SketchContainer sketch{ft, static_cast<bst_bin_t>(kBins), static_cast<bst_feature_t>(kCols),
                           DeviceOrd::CUDA(0)};
    AdapterDeviceSketch(&ctx, adapter.Value(), kBins, info, std::numeric_limits<float>::quiet_NaN(),
                        &sketch);
    return sketch.MakeCuts(&ctx, info.IsColumnSplit());
  };

  auto grouped_cuts = make_adapter_cuts();

  auto grouped = GetDMatrixFromData(storage.HostVector(), kRows, kCols);
  grouped->Info().SetInfo(ctx, "group",
                          Make1dInterfaceTest(group_sizes.data(), group_sizes.size()));
  grouped->Info().SetInfo(ctx, "weight",
                          Make1dInterfaceTest(group_weights.data(), group_weights.size()));
  auto device_grouped_cuts = DeviceSketch(&ctx, grouped.get(), kBins);

  auto per_row = GetDMatrixFromData(storage.HostVector(), kRows, kCols);
  std::vector<float> row_weights(kRows);
  for (size_t i = 0; i < row_weights.size(); ++i) {
    auto gidx = dh::SegmentId(Span{group_ptr.data(), group_ptr.size()}, i);
    row_weights[i] = group_weights[gidx];
  }
  per_row->Info().SetInfo(ctx, "weight",
                          Make1dInterfaceTest(row_weights.data(), row_weights.size()));
  auto row_cuts = DeviceSketch(&ctx, per_row.get(), kBins);

  ASSERT_EQ(device_grouped_cuts.Values(), row_cuts.Values());
  ASSERT_EQ(device_grouped_cuts.Ptrs(), row_cuts.Ptrs());
  ASSERT_EQ(grouped_cuts.Values(), row_cuts.Values());
  ASSERT_EQ(grouped_cuts.Ptrs(), row_cuts.Ptrs());
}

// For ranking data on GPU, sketching with group weights and Hessian should match sketching
// with the equivalent per-row weights after expanding each group weight onto its rows.
TEST(HistUtil, DeviceSketchWithHessianLTR) {
  auto ctx = MakeCUDACtx(0);
  bst_feature_t constexpr kFeatures = 5;
  bst_group_t constexpr kGroups = 3;
  std::vector<bst_idx_t> sizes = {1, 2, 256, 512, 1000, 1500};
  std::vector<bst_bin_t> bin_sizes = {2, 16, 256, 512};

  for (auto n_samples : sizes) {
    auto x = GenerateRandom(n_samples, kFeatures);
    std::vector<bst_group_t> gptr(kGroups + 1, 0);
    gptr[1] = n_samples / kGroups;
    gptr[2] = n_samples / kGroups + gptr[1];
    gptr.back() = n_samples;

    HostDeviceVector<float> hessian;
    auto& h_hess = hessian.HostVector();
    h_hess = GenerateRandomWeights(n_samples);
    std::mt19937 rng(0);
    std::shuffle(h_hess.begin(), h_hess.end(), rng);
    hessian.SetDevice(ctx.Device());
    auto p_fmat = GetDMatrixFromData(x, n_samples, kFeatures);
    std::vector<std::vector<float>> group_weight_cases = {std::vector<float>(kGroups, 1.0f),
                                                          GenerateRandomWeights(kGroups)};
    for (auto n_bins : bin_sizes) {
      for (auto const& group_weights : group_weight_cases) {
        p_fmat->Info().group_ptr_ = gptr;
        p_fmat->Info().weights_.HostVector() = group_weights;

        auto cuts_hess =
            DeviceSketchWithHessian(&ctx, p_fmat.get(), n_bins, hessian.ConstDeviceSpan());
        p_fmat->Info().weights_.Resize(p_fmat->Info().num_row_);
        for (std::size_t i = 0; i < h_hess.size(); ++i) {
          auto gidx = dh::SegmentId(Span{gptr.data(), gptr.size()}, i);
          p_fmat->Info().weights_.HostVector()[i] = group_weights[gidx] * h_hess[i];
        }
        p_fmat->Info().group_ptr_.clear();
        ValidateCuts(cuts_hess, p_fmat.get(), n_bins);

        auto cuts = DeviceSketch(&ctx, p_fmat.get(), n_bins);
        ValidateCuts(cuts, p_fmat.get(), n_bins);
        ASSERT_EQ(cuts.Values().size(), cuts_hess.Values().size());
        for (std::size_t i = 0; i < cuts.Values().size(); ++i) {
          EXPECT_NEAR(cuts.Values()[i], cuts_hess.Values()[i], 1e-4f);
        }
      }
    }
  }
}
}  // namespace xgboost::common
