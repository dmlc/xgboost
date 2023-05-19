/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include "test_quantile.h"

#include <gtest/gtest.h>

#include "../../../src/common/hist_util.h"
#include "../../../src/data/adapter.h"
#include "xgboost/context.h"

namespace xgboost {
namespace common {

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

namespace {
template <bool use_column>
using ContainerType = std::conditional_t<use_column, SortedSketchContainer, HostSketchContainer>;

// Dispatch for push page.
void PushPage(SortedSketchContainer* container, SparsePage const& page, MetaInfo const& info,
              Span<float const> hessian) {
  container->PushColPage(page, info, hessian);
}
void PushPage(HostSketchContainer* container, SparsePage const& page, MetaInfo const& info,
              Span<float const> hessian) {
  container->PushRowPage(page, info, hessian);
}

template <bool use_column>
void DoTestDistributedQuantile(size_t rows, size_t cols) {
  Context ctx;
  auto const world = collective::GetWorldSize();
  std::vector<MetaInfo> infos(2);
  auto& h_weights = infos.front().weights_.HostVector();
  h_weights.resize(rows);
  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist(3, 1000);
  std::generate(h_weights.begin(), h_weights.end(), [&]() { return dist(&lcg); });
  std::vector<bst_row_t> column_size(cols, rows);
  bst_bin_t n_bins = 64;

  // Generate cuts for distributed environment.
  auto sparsity = 0.5f;
  auto rank = collective::GetRank();
  std::vector<FeatureType> ft(cols);
  for (size_t i = 0; i < ft.size(); ++i) {
    ft[i] = (i % 2 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
  }

  auto m = RandomDataGenerator{rows, cols, sparsity}
               .Seed(rank)
               .Lower(.0f)
               .Upper(1.0f)
               .Type(ft)
               .MaxCategory(13)
               .GenerateDMatrix();

  std::vector<float> hessian(rows, 1.0);
  auto hess = Span<float const>{hessian};

  ContainerType<use_column> sketch_distributed(
      &ctx, n_bins, m->Info().feature_types.ConstHostSpan(), column_size, false);

  if (use_column) {
    for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
      PushPage(&sketch_distributed, page, m->Info(), hess);
    }
  } else {
    for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
      PushPage(&sketch_distributed, page, m->Info(), hess);
    }
  }

  HistogramCuts distributed_cuts;
  sketch_distributed.MakeCuts(m->Info(), &distributed_cuts);

  // Generate cuts for single node environment
  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);
  std::for_each(column_size.begin(), column_size.end(), [=](auto& size) { size *= world; });
  m->Info().num_row_ = world * rows;
  ContainerType<use_column> sketch_on_single_node(
      &ctx, n_bins, m->Info().feature_types.ConstHostSpan(), column_size, false);
  m->Info().num_row_ = rows;

  for (auto rank = 0; rank < world; ++rank) {
    auto m = RandomDataGenerator{rows, cols, sparsity}
                 .Seed(rank)
                 .Type(ft)
                 .MaxCategory(13)
                 .Lower(.0f)
                 .Upper(1.0f)
                 .GenerateDMatrix();
    if (use_column) {
      for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
        PushPage(&sketch_on_single_node, page, m->Info(), hess);
      }
    } else {
      for (auto const& page : m->GetBatches<SparsePage>()) {
        PushPage(&sketch_on_single_node, page, m->Info(), hess);
      }
    }
  }

  HistogramCuts single_node_cuts;
  sketch_on_single_node.MakeCuts(m->Info(), &single_node_cuts);

  auto const& sptrs = single_node_cuts.Ptrs();
  auto const& dptrs = distributed_cuts.Ptrs();
  auto const& svals = single_node_cuts.Values();
  auto const& dvals = distributed_cuts.Values();
  auto const& smins = single_node_cuts.MinValues();
  auto const& dmins = distributed_cuts.MinValues();

  ASSERT_EQ(sptrs.size(), dptrs.size());
  for (size_t i = 0; i < sptrs.size(); ++i) {
    ASSERT_EQ(sptrs[i], dptrs[i]) << i;
  }

  ASSERT_EQ(svals.size(), dvals.size());
  for (size_t i = 0; i < svals.size(); ++i) {
    ASSERT_NEAR(svals[i], dvals[i], 2e-2f);
  }

  ASSERT_EQ(smins.size(), dmins.size());
  for (size_t i = 0; i < smins.size(); ++i) {
    ASSERT_FLOAT_EQ(smins[i], dmins[i]);
  }
}

template <bool use_column>
void TestDistributedQuantile(size_t const rows, size_t const cols) {
  auto constexpr kWorkers = 4;
  RunWithInMemoryCommunicator(kWorkers, DoTestDistributedQuantile<use_column>, rows, cols);
}
}  // anonymous namespace

TEST(Quantile, DistributedBasic) {
  constexpr size_t kRows = 10, kCols = 10;
  TestDistributedQuantile<false>(kRows, kCols);
}

TEST(Quantile, Distributed) {
  constexpr size_t kRows = 4000, kCols = 200;
  TestDistributedQuantile<false>(kRows, kCols);
}

TEST(Quantile, SortedDistributedBasic) {
  constexpr size_t kRows = 10, kCols = 10;
  TestDistributedQuantile<true>(kRows, kCols);
}

TEST(Quantile, SortedDistributed) {
  constexpr size_t kRows = 4000, kCols = 200;
  TestDistributedQuantile<true>(kRows, kCols);
}

namespace {
template <bool use_column>
void DoTestColSplitQuantile(size_t rows, size_t cols) {
  Context ctx;
  auto const world = collective::GetWorldSize();
  auto const rank = collective::GetRank();

  auto m = std::unique_ptr<DMatrix>{[=]() {
    auto sparsity = 0.5f;
    std::vector<FeatureType> ft(cols);
    for (size_t i = 0; i < ft.size(); ++i) {
      ft[i] = (i % 2 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
    }
    auto dmat = RandomDataGenerator{rows, cols, sparsity}
                    .Seed(0)
                    .Lower(.0f)
                    .Upper(1.0f)
                    .Type(ft)
                    .MaxCategory(13)
                    .GenerateDMatrix();
    return dmat->SliceCol(world, rank);
  }()};

  std::vector<bst_row_t> column_size(cols, 0);
  auto const slice_size = cols / world;
  auto const slice_start = slice_size * rank;
  auto const slice_end = (rank == world - 1) ? cols : slice_start + slice_size;
  for (auto i = slice_start; i < slice_end; i++) {
    column_size[i] = rows;
  }

  auto const n_bins = 64;

  // Generate cuts for distributed environment.
  HistogramCuts distributed_cuts;
  {
    ContainerType<use_column> sketch_distributed(
        &ctx, n_bins, m->Info().feature_types.ConstHostSpan(), column_size, false);

    std::vector<float> hessian(rows, 1.0);
    auto hess = Span<float const>{hessian};
    if (use_column) {
      for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
        PushPage(&sketch_distributed, page, m->Info(), hess);
      }
    } else {
      for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
        PushPage(&sketch_distributed, page, m->Info(), hess);
      }
    }

    sketch_distributed.MakeCuts(m->Info(), &distributed_cuts);
  }

  // Generate cuts for single node environment
  collective::Finalize();
  CHECK_EQ(collective::GetWorldSize(), 1);
  HistogramCuts single_node_cuts;
  {
    ContainerType<use_column> sketch_on_single_node(
        &ctx, n_bins, m->Info().feature_types.ConstHostSpan(), column_size, false);

    std::vector<float> hessian(rows, 1.0);
    auto hess = Span<float const>{hessian};
    if (use_column) {
      for (auto const& page : m->GetBatches<SortedCSCPage>(&ctx)) {
        PushPage(&sketch_on_single_node, page, m->Info(), hess);
      }
    } else {
      for (auto const& page : m->GetBatches<SparsePage>(&ctx)) {
        PushPage(&sketch_on_single_node, page, m->Info(), hess);
      }
    }

    sketch_on_single_node.MakeCuts(m->Info(), &single_node_cuts);
  }

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

template <bool use_column>
void TestColSplitQuantile(size_t rows, size_t cols) {
  auto constexpr kWorkers = 4;
  RunWithInMemoryCommunicator(kWorkers, DoTestColSplitQuantile<use_column>, rows, cols);
}
}  // anonymous namespace

TEST(Quantile, ColSplitBasic) {
  constexpr size_t kRows = 10, kCols = 10;
  TestColSplitQuantile<false>(kRows, kCols);
}

TEST(Quantile, ColSplit) {
  constexpr size_t kRows = 4000, kCols = 200;
  TestColSplitQuantile<false>(kRows, kCols);
}

TEST(Quantile, ColSplitSortedBasic) {
  constexpr size_t kRows = 10, kCols = 10;
  TestColSplitQuantile<true>(kRows, kCols);
}

TEST(Quantile, ColSplitSorted) {
  constexpr size_t kRows = 4000, kCols = 200;
  TestColSplitQuantile<true>(kRows, kCols);
}

namespace {
void TestSameOnAllWorkers() {
  auto const world = collective::GetWorldSize();
  constexpr size_t kRows = 1000, kCols = 100;
  auto ctx = CreateEmptyGenericParam(Context::kCpuId);

  RunWithSeedsAndBins(
      kRows, [=, &ctx](int32_t seed, size_t n_bins, MetaInfo const&) {
        auto rank = collective::GetRank();
        HostDeviceVector<float> storage;
        std::vector<FeatureType> ft(kCols);
        for (size_t i = 0; i < ft.size(); ++i) {
          ft[i] = (i % 2 == 0) ? FeatureType::kNumerical : FeatureType::kCategorical;
        }

        auto m = RandomDataGenerator{kRows, kCols, 0}
                     .Device(Context::kCpuId)
                     .Type(ft)
                     .MaxCategory(17)
                     .Seed(rank + seed)
                     .GenerateDMatrix();
        auto cuts = SketchOnDMatrix(&ctx, m.get(), n_bins);
        std::vector<float> cut_values(cuts.Values().size() * world, 0);
        std::vector<
            typename std::remove_reference_t<decltype(cuts.Ptrs())>::value_type>
            cut_ptrs(cuts.Ptrs().size() * world, 0);
        std::vector<float> cut_min_values(cuts.MinValues().size() * world, 0);

        size_t value_size = cuts.Values().size();
        collective::Allreduce<collective::Operation::kMax>(&value_size, 1);
        size_t ptr_size = cuts.Ptrs().size();
        collective::Allreduce<collective::Operation::kMax>(&ptr_size, 1);
        CHECK_EQ(ptr_size, kCols + 1);
        size_t min_value_size = cuts.MinValues().size();
        collective::Allreduce<collective::Operation::kMax>(&min_value_size, 1);
        CHECK_EQ(min_value_size, kCols);

        size_t value_offset = value_size * rank;
        std::copy(cuts.Values().begin(), cuts.Values().end(),
                  cut_values.begin() + value_offset);
        size_t ptr_offset = ptr_size * rank;
        std::copy(cuts.Ptrs().cbegin(), cuts.Ptrs().cend(),
                  cut_ptrs.begin() + ptr_offset);
        size_t min_values_offset = min_value_size * rank;
        std::copy(cuts.MinValues().cbegin(), cuts.MinValues().cend(),
                  cut_min_values.begin() + min_values_offset);

        collective::Allreduce<collective::Operation::kSum>(cut_values.data(), cut_values.size());
        collective::Allreduce<collective::Operation::kSum>(cut_ptrs.data(), cut_ptrs.size());
        collective::Allreduce<collective::Operation::kSum>(cut_min_values.data(), cut_min_values.size());

        for (int32_t i = 0; i < world; i++) {
          for (size_t j = 0; j < value_size; ++j) {
            size_t idx = i * value_size + j;
            EXPECT_NEAR(cuts.Values().at(j), cut_values.at(idx), kRtEps);
          }

          for (size_t j = 0; j < ptr_size; ++j) {
            size_t idx = i * ptr_size + j;
            EXPECT_EQ(cuts.Ptrs().at(j), cut_ptrs.at(idx));
          }

          for (size_t j = 0; j < min_value_size; ++j) {
            size_t idx = i * min_value_size + j;
            EXPECT_EQ(cuts.MinValues().at(j), cut_min_values.at(idx));
          }
        }
      });
}
}  // anonymous namespace

TEST(Quantile, SameOnAllWorkers) {
  auto constexpr kWorkers = 4;
  RunWithInMemoryCommunicator(kWorkers, TestSameOnAllWorkers);
}

}  // namespace common
}  // namespace xgboost
