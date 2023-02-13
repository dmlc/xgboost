/**
 * Copyright 2019-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <utility>

#include "../../../src/common/hist_util.h"
#include "../../../src/data/gradient_index.h"
#include "../helpers.h"
#include "test_hist_util.h"

namespace xgboost {
namespace common {

void ParallelGHistBuilderReset() {
  constexpr size_t kBins = 10;
  constexpr size_t kNodes = 5;
  constexpr size_t kNodesExtended = 10;
  constexpr size_t kTasksPerNode = 10;
  constexpr double kValue = 1.0;
  const size_t nthreads = AllThreadsForTest();

  HistCollection collection;
  collection.Init(kBins);

  for(size_t inode = 0; inode < kNodesExtended; inode++) {
    collection.AddHistRow(inode);
  }
  collection.AllocateAllData();
  ParallelGHistBuilder hist_builder;
  hist_builder.Init(kBins);
  std::vector<GHistRow> target_hist(kNodes);
  for(size_t i = 0; i < target_hist.size(); ++i) {
    target_hist[i] = collection[i];
  }

  common::BlockedSpace2d space(
      kNodes, [&](size_t /* node*/) { return kTasksPerNode; }, 1);
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
  for(size_t i = 0; i < target_hist.size(); ++i) {
    target_hist[i] = collection[i];
  }
  common::BlockedSpace2d space2(
      kNodesExtended, [&](size_t /*node*/) { return kTasksPerNode; }, 1);
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

void ParallelGHistBuilderReduceHist(){
  constexpr size_t kBins = 10;
  constexpr size_t kNodes = 5;
  constexpr size_t kTasksPerNode = 10;
  constexpr double kValue = 1.0;
  const size_t nthreads = AllThreadsForTest();

  HistCollection collection;
  collection.Init(kBins);

  for(size_t inode = 0; inode < kNodes; inode++) {
    collection.AddHistRow(inode);
  }
  collection.AllocateAllData();
  ParallelGHistBuilder hist_builder;
  hist_builder.Init(kBins);
  std::vector<GHistRow> target_hist(kNodes);
  for(size_t i = 0; i < target_hist.size(); ++i) {
    target_hist[i] = collection[i];
  }

  common::BlockedSpace2d space(
      kNodes, [&](size_t /*node*/) { return kTasksPerNode; }, 1);
  hist_builder.Reset(nthreads, kNodes, space, target_hist);

  // Simple analog of BuildHist function, works in parallel for both tree-nodes and data in node
  common::ParallelFor2d(space, nthreads, [&](size_t inode, common::Range1d) {
    const size_t tid = omp_get_thread_num();

    GHistRow hist = hist_builder.GetInitializedHist(tid, inode);
    for(size_t i = 0; i < kBins; ++i) {
      hist[i].Add(kValue, kValue);
    }
  });

  for(size_t inode = 0; inode < kNodes; inode++) {
    hist_builder.ReduceHist(inode, 0, kBins);

    // We had kTasksPerNode tasks to add kValue to each bin for each node
    // So, after reducing we expect to have (kValue * kTasksPerNode) in each node
    for(size_t i = 0; i < kBins; ++i) {
      ASSERT_EQ(kValue * kTasksPerNode, collection[inode][i].GetGrad());
      ASSERT_EQ(kValue * kTasksPerNode, collection[inode][i].GetHess());
    }
  }
}

TEST(ParallelGHistBuilder, Reset) { ParallelGHistBuilderReset(); }

TEST(ParallelGHistBuilder, ReduceHist) { ParallelGHistBuilderReduceHist(); }

TEST(CutsBuilder, SearchGroupInd) {
  size_t constexpr kNumGroups = 4;
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;

  auto p_mat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  std::vector<bst_int> group(kNumGroups);
  group[0] = 2;
  group[1] = 3;
  group[2] = 7;
  group[3] = 5;

  p_mat->SetInfo("group", group.data(), DataType::kUInt32, kNumGroups);

  HistogramCuts hmat;

  size_t group_ind = HostSketchContainer::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 0);
  ASSERT_EQ(group_ind, 0ul);

  group_ind = HostSketchContainer::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 5);
  ASSERT_EQ(group_ind, 2ul);

  EXPECT_ANY_THROW(HostSketchContainer::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 17));

  p_mat->Info().Validate(-1);
  EXPECT_THROW(HostSketchContainer::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 17),
               dmlc::Error);

  std::vector<bst_uint> group_ptr {0, 1, 2};
  CHECK_EQ(HostSketchContainer::SearchGroupIndFromRow(group_ptr, 1), 1);
}

TEST(HistUtil, DenseCutsCategorical) {
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
   int categorical_sizes[] = {2, 6, 8, 12};
   int num_bins = 256;
   int sizes[] = {25, 100, 1000};
   for (auto n : sizes) {
     for (auto num_categories : categorical_sizes) {
       auto x = GenerateRandomCategoricalSingleColumn(n, num_categories);
       std::vector<float> x_sorted(x);
       std::sort(x_sorted.begin(), x_sorted.end());
       auto dmat = GetDMatrixFromData(x, n, 1);
       HistogramCuts cuts = SketchOnDMatrix(&ctx, dmat.get(), num_bins);
       auto cuts_from_sketch = cuts.Values();
       EXPECT_LT(cuts.MinValues()[0], x_sorted.front());
       EXPECT_GT(cuts_from_sketch.front(), x_sorted.front());
       EXPECT_GE(cuts_from_sketch.back(), x_sorted.back());
       EXPECT_EQ(cuts_from_sketch.size(), static_cast<size_t>(num_categories));
     }
   }
}

TEST(HistUtil, DenseCutsAccuracyTest) {
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      HistogramCuts cuts = SketchOnDMatrix(&ctx, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, DenseCutsAccuracyTestWeights) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto w = GenerateRandomWeights(num_rows);
    dmat->Info().weights_.HostVector() = w;
    for (auto num_bins : bin_sizes) {
      {
        HistogramCuts cuts = SketchOnDMatrix(&ctx, dmat.get(), num_bins, true);
        ValidateCuts(cuts, dmat.get(), num_bins);
      }
      {
        HistogramCuts cuts = SketchOnDMatrix(&ctx, dmat.get(), num_bins, false);
        ValidateCuts(cuts, dmat.get(), num_bins);
      }
    }
  }
}

void TestQuantileWithHessian(bool use_sorted) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {1000, 1500};
  int num_columns = 5;
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto w = GenerateRandomWeights(num_rows);
    auto hessian = GenerateRandomWeights(num_rows);
    std::mt19937 rng(0);
    std::shuffle(hessian.begin(), hessian.end(), rng);
    dmat->Info().weights_.HostVector() = w;

    for (auto num_bins : bin_sizes) {
      HistogramCuts cuts_hess = SketchOnDMatrix(&ctx, dmat.get(), num_bins, use_sorted, hessian);
      for (size_t i = 0; i < w.size(); ++i) {
        dmat->Info().weights_.HostVector()[i] = w[i] * hessian[i];
      }
      ValidateCuts(cuts_hess, dmat.get(), num_bins);

      HistogramCuts cuts_wh = SketchOnDMatrix(&ctx, dmat.get(), num_bins, use_sorted);
      ValidateCuts(cuts_wh, dmat.get(), num_bins);

      ASSERT_EQ(cuts_hess.Values().size(), cuts_wh.Values().size());
      for (size_t i = 0; i < cuts_hess.Values().size(); ++i) {
        ASSERT_NEAR(cuts_wh.Values()[i], cuts_hess.Values()[i], kRtEps);
      }

      dmat->Info().weights_.HostVector() = w;
    }
  }
}

TEST(HistUtil, QuantileWithHessian) {
  TestQuantileWithHessian(true);
  TestQuantileWithHessian(false);
}

TEST(HistUtil, DenseCutsExternalMemory) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    dmlc::TemporaryDirectory tmpdir;
    auto dmat = GetExternalMemoryDMatrixFromData(x, num_rows, num_columns, tmpdir);
    for (auto num_bins : bin_sizes) {
      HistogramCuts cuts = SketchOnDMatrix(&ctx, dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, IndexBinBound) {
  uint64_t bin_sizes[] = { static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) + 1,
                           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1,
                           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 2 };
  BinTypeSize expected_bin_type_sizes[] = {kUint8BinsTypeSize,
                                           kUint16BinsTypeSize,
                                           kUint32BinsTypeSize};
  size_t constexpr kRows = 100;
  size_t constexpr kCols = 10;
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  size_t bin_id = 0;
  for (auto max_bin : bin_sizes) {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

    GHistIndexMatrix hmat(&ctx, p_fmat.get(), max_bin, 0.5, false);
    EXPECT_EQ(hmat.index.Size(), kRows*kCols);
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
  uint64_t constexpr kBinSizes[] = { static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) + 1,
                                     static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1,
                                     static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 2 };
  size_t constexpr kRows = 100;
  size_t constexpr kCols = 10;
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);

  for (auto max_bin : kBinSizes) {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
    GHistIndexMatrix hmat(&ctx, p_fmat.get(), max_bin, 0.5, false);
    uint32_t const* offsets = hmat.index.Offset();
    EXPECT_EQ(hmat.index.Size(), kRows*kCols);
    switch (max_bin) {
      case kBinSizes[0]:
        CheckIndexData(hmat.index.data<uint8_t>(),
                       offsets, hmat, kCols);
        break;
      case kBinSizes[1]:
        CheckIndexData(hmat.index.data<uint16_t>(),
                       offsets, hmat, kCols);
        break;
      case kBinSizes[2]:
        CheckIndexData(hmat.index.data<uint32_t>(),
                       offsets, hmat, kCols);
        break;
    }
  }
}

void TestSketchFromWeights(bool with_group) {
  size_t constexpr kRows = 300, kCols = 20, kBins = 256;
  size_t constexpr kGroups = 10;
  auto m = RandomDataGenerator{kRows, kCols, 0}.Device(0).GenerateDMatrix();
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  common::HistogramCuts cuts = SketchOnDMatrix(&ctx, m.get(), kBins);

  MetaInfo info;
  auto& h_weights = info.weights_.HostVector();
  if (with_group) {
    h_weights.resize(kGroups);
  } else {
    h_weights.resize(kRows);
  }
  std::fill(h_weights.begin(), h_weights.end(), 1.0f);

  std::vector<bst_group_t> groups(kGroups);
  if (with_group) {
    for (size_t i = 0; i < kGroups; ++i) {
      groups[i] = kRows / kGroups;
    }
    info.SetInfo(ctx, "group", groups.data(), DataType::kUInt32, kGroups);
  }

  info.num_row_ = kRows;
  info.num_col_ = kCols;

  // Assign weights.
  if (with_group) {
    m->SetInfo("group", groups.data(), DataType::kUInt32, kGroups);
  }

  m->SetInfo("weight", h_weights.data(), DataType::kFloat32, h_weights.size());
  m->Info().num_col_ = kCols;
  m->Info().num_row_ = kRows;
  ASSERT_EQ(cuts.Ptrs().size(), kCols + 1);
  ValidateCuts(cuts, m.get(), kBins);

  if (with_group) {
    m->Info().weights_ = decltype(m->Info().weights_)();  // remove weight
    HistogramCuts non_weighted = SketchOnDMatrix(&ctx, m.get(), kBins);
    for (size_t i = 0; i < cuts.Values().size(); ++i) {
      EXPECT_EQ(cuts.Values()[i], non_weighted.Values()[i]);
    }
    for (size_t i = 0; i < cuts.MinValues().size(); ++i) {
      ASSERT_EQ(cuts.MinValues()[i], non_weighted.MinValues()[i]);
    }
    for (size_t i = 0; i < cuts.Ptrs().size(); ++i) {
      ASSERT_EQ(cuts.Ptrs().at(i), non_weighted.Ptrs().at(i));
    }
  }

  if (with_group) {
    auto& h_weights = info.weights_.HostVector();
    h_weights.resize(kGroups);
    // Generate different weight.
    for (size_t i = 0; i < h_weights.size(); ++i) {
      h_weights[i] = static_cast<float>(i + 1) / static_cast<float>(kGroups);
    }
    HistogramCuts weighted = SketchOnDMatrix(&ctx, m.get(), kBins);
    ValidateCuts(weighted, m.get(), kBins);
  }
}

TEST(HistUtil, SketchFromWeights) {
  TestSketchFromWeights(true);
  TestSketchFromWeights(false);
}

TEST(HistUtil, SketchCategoricalFeatures) {
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);
  TestCategoricalSketch(1000, 256, 32, false, [&ctx](DMatrix* p_fmat, int32_t num_bins) {
    return SketchOnDMatrix(&ctx, p_fmat, num_bins);
  });
  TestCategoricalSketch(1000, 256, 32, true, [&ctx](DMatrix* p_fmat, int32_t num_bins) {
    return SketchOnDMatrix(&ctx, p_fmat, num_bins);
  });
}
}  // namespace common
}  // namespace xgboost
