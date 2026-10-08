/**
 * Copyright 2019-2025, XGBoost Contributors
 */
#include "test_hist_util.h"

#include <gtest/gtest.h>
#include <xgboost/data.h>                // for ExtMemConfig
#include <xgboost/host_device_vector.h>  // for HostDeviceVector

#include <memory>  // for shared_ptr
#include <string>
#include <vector>

#include "../../../src/common/hist_util.h"
#include "../../../src/data/gradient_index.h"
#include "../helpers.h"

namespace xgboost::common {
TEST(ParallelGHistBuilder, Reset) {
  constexpr size_t kBins = 10;
  constexpr size_t kNodes = 5;
  constexpr size_t kNodesExtended = 10;
  constexpr size_t kTasksPerNode = 10;
  constexpr double kValue = 1.0;
  const size_t nthreads = AllThreadsForTest();

  HistCollection collection;
  collection.Init(kBins);

  for (size_t inode = 0; inode < kNodesExtended; inode++) {
    collection.AddHistRow(inode);
    collection.AllocateData(inode);
  }
  ParallelGHistBuilder hist_builder;
  hist_builder.Init(kBins);
  std::vector<GHistRow> target_hist(kNodes);
  for (size_t i = 0; i < target_hist.size(); ++i) {
    target_hist[i] = collection[i];
  }

  common::BlockedSpace2d space(kNodes, [&](size_t /* node*/) { return kTasksPerNode; }, 1);
  hist_builder.Reset(nthreads, kNodes, space, target_hist);

  common::ParallelFor2d(space, nthreads, [&](size_t inode, common::Range1d) {
    const size_t tid = omp_get_thread_num();

    GHistRow hist = hist_builder.GetInitializedHist(tid, inode);
    // fill hist by some non-null values
    for (size_t j = 0; j < kBins; ++j) {
      hist[j].Add(kValue, kValue);
    }
  });

  // reset and extend buffer
  target_hist.resize(kNodesExtended);
  for (size_t i = 0; i < target_hist.size(); ++i) {
    target_hist[i] = collection[i];
  }
  common::BlockedSpace2d space2(kNodesExtended, [&](size_t /*node*/) { return kTasksPerNode; }, 1);
  hist_builder.Reset(nthreads, kNodesExtended, space2, target_hist);

  common::ParallelFor2d(space2, nthreads, [&](size_t inode, common::Range1d) {
    const size_t tid = omp_get_thread_num();

    GHistRow hist = hist_builder.GetInitializedHist(tid, inode);
    // fill hist by some non-null values
    for (size_t j = 0; j < kBins; ++j) {
      ASSERT_EQ(0.0, hist[j].GetGrad());
      ASSERT_EQ(0.0, hist[j].GetHess());
    }
  });
}

TEST(ParallelGHistBuilder, ReduceHist) {
  constexpr size_t kBins = 10;
  constexpr size_t kNodes = 5;
  constexpr size_t kTasksPerNode = 10;
  constexpr double kValue = 1.0;
  const size_t nthreads = AllThreadsForTest();

  HistCollection collection;
  collection.Init(kBins);

  for (size_t inode = 0; inode < kNodes; inode++) {
    collection.AddHistRow(inode);
    collection.AllocateData(inode);
  }
  ParallelGHistBuilder hist_builder;
  hist_builder.Init(kBins);
  std::vector<GHistRow> target_hist(kNodes);
  for (size_t i = 0; i < target_hist.size(); ++i) {
    target_hist[i] = collection[i];
  }

  common::BlockedSpace2d space(kNodes, [&](size_t /*node*/) { return kTasksPerNode; }, 1);
  hist_builder.Reset(nthreads, kNodes, space, target_hist);

  // Simple analog of BuildHist function, works in parallel for both tree-nodes and data in node
  common::ParallelFor2d(space, nthreads, [&](size_t inode, common::Range1d) {
    const size_t tid = omp_get_thread_num();

    GHistRow hist = hist_builder.GetInitializedHist(tid, inode);
    for (size_t i = 0; i < kBins; ++i) {
      hist[i].Add(kValue, kValue);
    }
  });

  for (size_t inode = 0; inode < kNodes; inode++) {
    hist_builder.ReduceHist(inode, 0, kBins);

    // We had kTasksPerNode tasks to add kValue to each bin for each node
    // So, after reducing we expect to have (kValue * kTasksPerNode) in each node
    for (size_t i = 0; i < kBins; ++i) {
      ASSERT_EQ(kValue * kTasksPerNode, collection[inode][i].GetGrad());
      ASSERT_EQ(kValue * kTasksPerNode, collection[inode][i].GetHess());
    }
  }
}

TEST(HistUtil, IndexBinBound) {
  uint64_t bin_sizes[] = {static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) + 1,
                          static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1,
                          static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 2};
  BinTypeSize expected_bin_type_sizes[] = {kUint8BinsTypeSize, kUint16BinsTypeSize,
                                           kUint32BinsTypeSize};
  size_t constexpr kRows = 100;
  size_t constexpr kCols = 10;
  Context ctx;
  size_t bin_id = 0;
  for (auto max_bin : bin_sizes) {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

    GHistIndexMatrix hmat(&ctx, p_fmat.get(), max_bin, 0.5, false);
    EXPECT_EQ(hmat.index.Size(), kRows * kCols);
    EXPECT_EQ(expected_bin_type_sizes[bin_id++], hmat.index.GetBinTypeSize());
  }
}

template <typename T>
void CheckIndexData(T const* data_ptr, uint32_t const* offsets, const GHistIndexMatrix& hmat,
                    size_t n_cols) {
  for (size_t i = 0; i < hmat.index.Size(); ++i) {
    EXPECT_EQ(data_ptr[i] + offsets[i % n_cols], hmat.index[i]);
  }
}

TEST(HistUtil, IndexBinData) {
  uint64_t constexpr kBinSizes[] = {
      static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) + 1,
      static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1,
      static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 2};
  size_t constexpr kRows = 100;
  size_t constexpr kCols = 10;
  Context ctx;

  for (auto max_bin : kBinSizes) {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
    GHistIndexMatrix hmat(&ctx, p_fmat.get(), max_bin, 0.5, false);
    uint32_t const* offsets = hmat.index.Offset();
    EXPECT_EQ(hmat.index.Size(), kRows * kCols);
    switch (max_bin) {
      case kBinSizes[0]:
        CheckIndexData(hmat.index.data<uint8_t>(), offsets, hmat, kCols);
        break;
      case kBinSizes[1]:
        CheckIndexData(hmat.index.data<uint16_t>(), offsets, hmat, kCols);
        break;
      case kBinSizes[2]:
        CheckIndexData(hmat.index.data<uint32_t>(), offsets, hmat, kCols);
        break;
    }
  }
}

// Sketching with a separate Hessian input should match sketching with sample weights after
// folding the Hessian into the per-row weights, for both sorted and unsorted CPU paths.
TEST(HistUtil, QuantileWithHessian) {
  int constexpr kRows = 1500;
  int constexpr kCols = 5;
  int constexpr kBins = 256;
  Context ctx;
  auto x = GenerateRandom(kRows, kCols);
  auto dmat = GetDMatrixFromData(x, kRows, kCols);
  auto w = GenerateRandomWeights(kRows);
  auto hessian = GenerateRandomWeights(kRows);
  std::mt19937 rng(0);
  std::shuffle(hessian.begin(), hessian.end(), rng);
  dmat->Info().weights_.HostVector() = w;

  HistogramCuts cuts_hess = SketchOnDMatrix(&ctx, dmat.get(), kBins, false, hessian);
  HistogramCuts sorted_cuts_hess = SketchOnDMatrix(&ctx, dmat.get(), kBins, true, hessian);
  for (size_t i = 0; i < w.size(); ++i) {
    dmat->Info().weights_.HostVector()[i] = w[i] * hessian[i];
  }
  ValidateCuts(cuts_hess, dmat.get(), kBins);
  ValidateCuts(sorted_cuts_hess, dmat.get(), kBins);

  HistogramCuts cuts_wh = SketchOnDMatrix(&ctx, dmat.get(), kBins, false);
  ValidateCuts(cuts_wh, dmat.get(), kBins);
  HistogramCuts sorted_cuts_wh = SketchOnDMatrix(&ctx, dmat.get(), kBins, true);
  ValidateCuts(sorted_cuts_wh, dmat.get(), kBins);

  ASSERT_EQ(cuts_hess.Values().size(), cuts_wh.Values().size());
  for (size_t i = 0; i < cuts_hess.Values().size(); ++i) {
    ASSERT_NEAR(cuts_wh.Values()[i], cuts_hess.Values()[i], kRtEps);
    ASSERT_NEAR(sorted_cuts_wh.Values()[i], sorted_cuts_hess.Values()[i], kRtEps);
  }
}

TEST(HistUtil, DenseCutsDMatrixTypes) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  Context ctx;
  for (auto num_rows : sizes) {
    auto data = GenerateRandom(num_rows, num_columns);
    HostDeviceVector<float> x{data};
    common::TemporaryDirectory tmpdir;
    std::vector<std::shared_ptr<DMatrix>> matrices{
        GetDMatrixFromData(data, num_rows, num_columns),
        GetExternalMemoryDMatrixFromData(x, num_rows, num_columns, tmpdir)};
    for (auto num_bins : bin_sizes) {
      for (auto const& dmat : matrices) {
        HistogramCuts cuts = SketchOnDMatrix(&ctx, dmat.get(), num_bins);
        ValidateCuts(cuts, dmat.get(), num_bins);
      }
    }
  }
}

TEST(HistUtil, UnrollGroupWeights) {
  MetaInfo info;
  info.num_row_ = 6;
  info.group_ptr_ = {0, 2, 3, 6};
  info.weights_.HostVector() = {1.0f, 5.0f, 9.0f};

  std::vector<float> expected{1.0f, 1.0f, 5.0f, 9.0f, 9.0f, 9.0f};
  ASSERT_EQ(detail::UnrollGroupWeights(info), expected);
}

TEST(HistUtil, GroupWeightsEquivalentToRowWeights) {
  Context ctx;
  std::vector<float> x{
      0.0f, 5.0f, 1.0f, 4.0f, 2.0f, 3.0f, 3.0f, 2.0f, 4.0f, 1.0f, 5.0f, 0.0f,
  };
  auto grouped = GetDMatrixFromData(x, 6, 2);
  auto per_row = GetDMatrixFromData(x, 6, 2);

  std::vector<bst_group_t> group_sizes{1, 1, 4};
  std::vector<float> group_weights{1.0f, 1000.0f, 1.0f};
  std::vector<float> row_weights{1.0f, 1000.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  grouped->SetInfo("group", Make1dInterfaceTest(group_sizes.data(), group_sizes.size()));
  grouped->SetInfo("weight", Make1dInterfaceTest(group_weights.data(), group_weights.size()));
  per_row->SetInfo("weight", Make1dInterfaceTest(row_weights.data(), row_weights.size()));

  auto grouped_cuts = SketchOnDMatrix(&ctx, grouped.get(), 2, false);
  auto per_row_cuts = SketchOnDMatrix(&ctx, per_row.get(), 2, false);
  auto sorted_grouped_cuts = SketchOnDMatrix(&ctx, grouped.get(), 2, true);
  auto sorted_per_row_cuts = SketchOnDMatrix(&ctx, per_row.get(), 2, true);

  ASSERT_EQ(grouped_cuts.Ptrs(), per_row_cuts.Ptrs());
  ASSERT_EQ(grouped_cuts.Values(), per_row_cuts.Values());
  ASSERT_EQ(sorted_grouped_cuts.Ptrs(), sorted_per_row_cuts.Ptrs());
  ASSERT_EQ(sorted_grouped_cuts.Values(), sorted_per_row_cuts.Values());
}

}  // namespace xgboost::common
