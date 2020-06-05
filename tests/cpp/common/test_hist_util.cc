#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <utility>

#include "../../../src/common/hist_util.h"
#include "../helpers.h"
#include "test_hist_util.h"

namespace xgboost {
namespace common {

size_t GetNThreads() {
  size_t nthreads;
  #pragma omp parallel
  {
    #pragma omp master
    nthreads = omp_get_num_threads();
  }
  return nthreads;
}

template <typename GradientSumT>
void ParallelGHistBuilderReset() {
  constexpr size_t kBins = 10;
  constexpr size_t kNodes = 5;
  constexpr size_t kNodesExtended = 10;
  constexpr size_t kTasksPerNode = 10;
  constexpr double kValue = 1.0;
  const size_t nthreads = GetNThreads();

  HistCollection<GradientSumT> collection;
  collection.Init(kBins);

  for(size_t inode = 0; inode < kNodesExtended; inode++) {
    collection.AddHistRow(inode);
  }

  ParallelGHistBuilder<GradientSumT> hist_builder;
  hist_builder.Init(kBins);
  std::vector<GHistRow<GradientSumT>> target_hist(kNodes);
  for(size_t i = 0; i < target_hist.size(); ++i) {
    target_hist[i] = collection[i];
  }

  common::BlockedSpace2d space(kNodes, [&](size_t node) { return kTasksPerNode; }, 1);
  hist_builder.Reset(nthreads, kNodes, space, target_hist);

  common::ParallelFor2d(space, nthreads, [&](size_t inode, common::Range1d r) {
    const size_t tid = omp_get_thread_num();

    GHistRow<GradientSumT> hist = hist_builder.GetInitializedHist(tid, inode);
    // fill hist by some non-null values
    for(size_t j = 0; j < kBins; ++j) {
      hist[j].Add(kValue, kValue);
    }
  });

  // reset and extend buffer
  target_hist.resize(kNodesExtended);
  for(size_t i = 0; i < target_hist.size(); ++i) {
    target_hist[i] = collection[i];
  }
  common::BlockedSpace2d space2(kNodesExtended, [&](size_t node) { return kTasksPerNode; }, 1);
  hist_builder.Reset(nthreads, kNodesExtended, space2, target_hist);

  common::ParallelFor2d(space2, nthreads, [&](size_t inode, common::Range1d r) {
    const size_t tid = omp_get_thread_num();

    GHistRow<GradientSumT> hist = hist_builder.GetInitializedHist(tid, inode);
    // fill hist by some non-null values
    for(size_t j = 0; j < kBins; ++j) {
      ASSERT_EQ(0.0, hist[j].GetGrad());
      ASSERT_EQ(0.0, hist[j].GetHess());
    }
  });
}


template <typename GradientSumT>
void ParallelGHistBuilderReduceHist(){
  constexpr size_t kBins = 10;
  constexpr size_t kNodes = 5;
  constexpr size_t kTasksPerNode = 10;
  constexpr double kValue = 1.0;
  const size_t nthreads = GetNThreads();

  HistCollection<GradientSumT> collection;
  collection.Init(kBins);

  for(size_t inode = 0; inode < kNodes; inode++) {
    collection.AddHistRow(inode);
  }

  ParallelGHistBuilder<GradientSumT> hist_builder;
  hist_builder.Init(kBins);
  std::vector<GHistRow<GradientSumT>> target_hist(kNodes);
  for(size_t i = 0; i < target_hist.size(); ++i) {
    target_hist[i] = collection[i];
  }

  common::BlockedSpace2d space(kNodes, [&](size_t node) { return kTasksPerNode; }, 1);
  hist_builder.Reset(nthreads, kNodes, space, target_hist);

  // Simple analog of BuildHist function, works in parallel for both tree-nodes and data in node
  common::ParallelFor2d(space, nthreads, [&](size_t inode, common::Range1d r) {
    const size_t tid = omp_get_thread_num();

    GHistRow<GradientSumT> hist = hist_builder.GetInitializedHist(tid, inode);
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

TEST(ParallelGHistBuilder, ResetDouble) {
  ParallelGHistBuilderReset<double>();
}

TEST(ParallelGHistBuilder, ResetFloat) {
  ParallelGHistBuilderReset<float>();
}

TEST(ParallelGHistBuilder, ReduceHistDouble) {
  ParallelGHistBuilderReduceHist<double>();
}

TEST(ParallelGHistBuilder, ReduceHistFloat) {
  ParallelGHistBuilderReduceHist<float>();
}

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

  p_mat->Info().SetInfo(
      "group", group.data(), DataType::kUInt32, kNumGroups);

  HistogramCuts hmat;

  size_t group_ind = CutsBuilder::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 0);
  ASSERT_EQ(group_ind, 0);

  group_ind = CutsBuilder::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 5);
  ASSERT_EQ(group_ind, 2);

  EXPECT_ANY_THROW(CutsBuilder::SearchGroupIndFromRow(p_mat->Info().group_ptr_, 17));
}

TEST(SparseCuts, SingleThreadedBuild) {
  size_t constexpr kRows = 267;
  size_t constexpr kCols = 31;
  size_t constexpr kBins = 256;

  auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  common::GHistIndexMatrix hmat;
  hmat.Init(p_fmat.get(), kBins);

  HistogramCuts cuts;
  SparseCuts indices(&cuts);
  auto const& page = *(p_fmat->GetBatches<xgboost::CSCPage>().begin());
  indices.SingleThreadBuild(page, p_fmat->Info(), kBins, false, 0, page.Size(), 0);

  ASSERT_EQ(hmat.cut.Ptrs().size(), cuts.Ptrs().size());
  ASSERT_EQ(hmat.cut.Ptrs(), cuts.Ptrs());
  ASSERT_EQ(hmat.cut.Values(), cuts.Values());
  ASSERT_EQ(hmat.cut.MinValues(), cuts.MinValues());
}

TEST(SparseCuts, MultiThreadedBuild) {
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;
  size_t constexpr kBins = 255;

  omp_ulong ori_nthreads = omp_get_max_threads();
  omp_set_num_threads(16);

  auto Compare =
#if defined(_MSC_VER)  // msvc fails to capture
      [kBins](DMatrix* p_fmat) {
#else
      [](DMatrix* p_fmat) {
#endif
        HistogramCuts threaded_container;
        SparseCuts threaded_indices(&threaded_container);
        threaded_indices.Build(p_fmat, kBins);

        HistogramCuts container;
        SparseCuts indices(&container);
        auto const& page = *(p_fmat->GetBatches<xgboost::CSCPage>().begin());
        indices.SingleThreadBuild(page, p_fmat->Info(), kBins, false, 0, page.Size(), 0);

        ASSERT_EQ(container.Ptrs().size(), threaded_container.Ptrs().size());
        ASSERT_EQ(container.Values().size(), threaded_container.Values().size());

        for (uint32_t i = 0; i < container.Ptrs().size(); ++i) {
          ASSERT_EQ(container.Ptrs()[i], threaded_container.Ptrs()[i]);
        }
        for (uint32_t i = 0; i < container.Values().size(); ++i) {
          ASSERT_EQ(container.Values()[i], threaded_container.Values()[i]);
        }
      };

  {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
    Compare(p_fmat.get());
  }

  {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0.0001).GenerateDMatrix();
    Compare(p_fmat.get());
  }

  omp_set_num_threads(ori_nthreads);
}

TEST(HistUtil, DenseCutsCategorical) {
   int categorical_sizes[] = {2, 6, 8, 12};
   int num_bins = 256;
   int sizes[] = {25, 100, 1000};
   for (auto n : sizes) {
     for (auto num_categories : categorical_sizes) {
       auto x = GenerateRandomCategoricalSingleColumn(n, num_categories);
       std::vector<float> x_sorted(x);
       std::sort(x_sorted.begin(), x_sorted.end());
       auto dmat = GetDMatrixFromData(x, n, 1);
       HistogramCuts cuts;
       DenseCuts dense(&cuts);
       dense.Build(dmat.get(), num_bins);
       auto cuts_from_sketch = cuts.Values();
       EXPECT_LT(cuts.MinValues()[0], x_sorted.front());
       EXPECT_GT(cuts_from_sketch.front(), x_sorted.front());
       EXPECT_GE(cuts_from_sketch.back(), x_sorted.back());
       EXPECT_EQ(cuts_from_sketch.size(), num_categories);
     }
   }
}

TEST(HistUtil, DenseCutsAccuracyTest) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      HistogramCuts cuts;
      DenseCuts dense(&cuts);
      dense.Build(dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, DenseCutsAccuracyTestWeights) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    auto w = GenerateRandomWeights(num_rows);
    dmat->Info().weights_.HostVector() = w;
    for (auto num_bins : bin_sizes) {
      HistogramCuts cuts;
      DenseCuts dense(&cuts);
      dense.Build(dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, DenseCutsExternalMemory) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    dmlc::TemporaryDirectory tmpdir;
    auto dmat =
        GetExternalMemoryDMatrixFromData(x, num_rows, num_columns, 50, tmpdir);
    for (auto num_bins : bin_sizes) {
      HistogramCuts cuts;
      DenseCuts dense(&cuts);
      dense.Build(dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, SparseCutsAccuracyTest) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    auto dmat = GetDMatrixFromData(x, num_rows, num_columns);
    for (auto num_bins : bin_sizes) {
      HistogramCuts cuts;
      SparseCuts sparse(&cuts);
      sparse.Build(dmat.get(), num_bins);
      ValidateCuts(cuts, dmat.get(), num_bins);
    }
  }
}

TEST(HistUtil, SparseCutsCategorical) {
  int categorical_sizes[] = {2, 6, 8, 12};
  int num_bins = 256;
  int sizes[] = {25, 100, 1000};
  for (auto n : sizes) {
    for (auto num_categories : categorical_sizes) {
      auto x = GenerateRandomCategoricalSingleColumn(n, num_categories);
      std::vector<float> x_sorted(x);
      std::sort(x_sorted.begin(), x_sorted.end());
      auto dmat = GetDMatrixFromData(x, n, 1);
      HistogramCuts cuts;
      SparseCuts sparse(&cuts);
      sparse.Build(dmat.get(), num_bins);
      auto cuts_from_sketch = cuts.Values();
      EXPECT_LT(cuts.MinValues()[0], x_sorted.front());
      EXPECT_GT(cuts_from_sketch.front(), x_sorted.front());
      EXPECT_GE(cuts_from_sketch.back(), x_sorted.back());
      EXPECT_EQ(cuts_from_sketch.size(), num_categories);
    }
  }
}

TEST(HistUtil, SparseCutsExternalMemory) {
  int bin_sizes[] = {2, 16, 256, 512};
  int sizes[] = {100, 1000, 1500};
  int num_columns = 5;
  for (auto num_rows : sizes) {
    auto x = GenerateRandom(num_rows, num_columns);
    dmlc::TemporaryDirectory tmpdir;
    auto dmat =
        GetExternalMemoryDMatrixFromData(x, num_rows, num_columns, 50, tmpdir);
    for (auto num_bins : bin_sizes) {
      HistogramCuts cuts;
      SparseCuts dense(&cuts);
      dense.Build(dmat.get(), num_bins);
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

  size_t bin_id = 0;
  for (auto max_bin : bin_sizes) {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

    common::GHistIndexMatrix hmat;
    hmat.Init(p_fmat.get(), max_bin);
    EXPECT_EQ(hmat.index.Size(), kRows*kCols);
    EXPECT_EQ(expected_bin_type_sizes[bin_id++], hmat.index.GetBinTypeSize());
  }
}

TEST(HistUtil, SparseIndexBinBound) {
  uint64_t bin_sizes[] = { static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) + 1,
                           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1,
                           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 2 };
  BinTypeSize expected_bin_type_sizes[] = { kUint32BinsTypeSize,
                                            kUint32BinsTypeSize,
                                            kUint32BinsTypeSize };
  size_t constexpr kRows = 100;
  size_t constexpr kCols = 10;

  size_t bin_id = 0;
  for (auto max_bin : bin_sizes) {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0.2).GenerateDMatrix();
    common::GHistIndexMatrix hmat;
    hmat.Init(p_fmat.get(), max_bin);
    EXPECT_EQ(expected_bin_type_sizes[bin_id++], hmat.index.GetBinTypeSize());
  }
}

template <typename T>
void CheckIndexData(T* data_ptr, uint32_t* offsets,
                    const common::GHistIndexMatrix& hmat, size_t n_cols) {
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

  for (auto max_bin : kBinSizes) {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
    common::GHistIndexMatrix hmat;
    hmat.Init(p_fmat.get(), max_bin);
    uint32_t* offsets = hmat.index.Offset();
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

TEST(HistUtil, SparseIndexBinData) {
  uint64_t bin_sizes[] = { static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) + 1,
                           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1,
                           static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 2 };
  size_t constexpr kRows = 100;
  size_t constexpr kCols = 10;

  for (auto max_bin : bin_sizes) {
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0.2).GenerateDMatrix();
    common::GHistIndexMatrix hmat;
    hmat.Init(p_fmat.get(), max_bin);
    EXPECT_EQ(hmat.index.Offset(), nullptr);

    uint32_t* data_ptr = hmat.index.data<uint32_t>();
    for (size_t i = 0; i < hmat.index.Size(); ++i) {
      EXPECT_EQ(data_ptr[i], hmat.index[i]);
    }
  }
}

}  // namespace common
}  // namespace xgboost
