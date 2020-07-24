#include <gtest/gtest.h>
#include "test_hist_util.h"
#include "test_quantile.h"

#include "../../../src/common/quantile.h"
#include "../../../src/common/hist_util.h"

namespace xgboost {
namespace common {
TEST(Quantile, LoadBalance) {
  size_t constexpr kRows = 1000, kCols = 100;
  auto m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();
  std::vector<bst_feature_t> cols_ptr;
  for (auto const &page : m->GetBatches<SparsePage>()) {
    cols_ptr = HostSketchContainer::LoadBalance(page, kCols, 13);
  }
  size_t n_cols = 0;
  for (size_t i = 1; i < cols_ptr.size(); ++i) {
    n_cols += cols_ptr[i] - cols_ptr[i - 1];
  }
  CHECK_EQ(n_cols, kCols);
}

void TestDistributedQuantile(size_t rows, size_t cols) {
  std::string msg {"Skipping AllReduce test"};
  int32_t constexpr kWorkers = 4;
  InitRabitContext(msg, kWorkers);
  auto world = rabit::GetWorldSize();
  if (world != 1) {
    ASSERT_EQ(world, kWorkers);
  } else {
    return;
  }

  std::vector<MetaInfo> infos(2);
  auto& h_weights = infos.front().weights_.HostVector();
  h_weights.resize(rows);
  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist(3, 1000);
  std::generate(h_weights.begin(), h_weights.end(), [&]() { return dist(&lcg); });
  std::vector<bst_row_t> column_size(cols, rows);
  size_t n_bins = 64;

  // Generate cuts for distributed environment.
  auto sparsity = 0.5f;
  auto rank = rabit::GetRank();
  HostSketchContainer sketch_distributed(column_size, n_bins, false);
  auto m = RandomDataGenerator{rows, cols, sparsity}
               .Seed(rank)
               .Lower(.0f)
               .Upper(1.0f)
               .GenerateDMatrix();
  for (auto const &page : m->GetBatches<SparsePage>()) {
    sketch_distributed.PushRowPage(page, m->Info());
  }
  HistogramCuts distributed_cuts;
  sketch_distributed.MakeCuts(&distributed_cuts);

  // Generate cuts for single node environment
  rabit::Finalize();
  CHECK_EQ(rabit::GetWorldSize(), 1);
  std::for_each(column_size.begin(), column_size.end(), [=](auto& size) { size *= world; });
  HostSketchContainer sketch_on_single_node(column_size, n_bins, false);
  for (auto rank = 0; rank < world; ++rank) {
    auto m = RandomDataGenerator{rows, cols, sparsity}
                 .Seed(rank)
                 .Lower(.0f)
                 .Upper(1.0f)
                 .GenerateDMatrix();
    for (auto const &page : m->GetBatches<SparsePage>()) {
      sketch_on_single_node.PushRowPage(page, m->Info());
    }
  }

  HistogramCuts single_node_cuts;
  sketch_on_single_node.MakeCuts(&single_node_cuts);

  auto const& sptrs = single_node_cuts.Ptrs();
  auto const& dptrs = distributed_cuts.Ptrs();
  auto const& svals = single_node_cuts.Values();
  auto const& dvals = distributed_cuts.Values();
  auto const& smins = single_node_cuts.MinValues();
  auto const& dmins = distributed_cuts.MinValues();

  ASSERT_EQ(sptrs.size(), dptrs.size());
  for (size_t i = 0; i < sptrs.size(); ++i) {
    ASSERT_EQ(sptrs[i], dptrs[i]);
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

TEST(Quantile, DistributedBasic) {
#if defined(__unix__)
  constexpr size_t kRows = 10, kCols = 10;
  TestDistributedQuantile(kRows, kCols);
#endif
}

TEST(Quantile, Distributed) {
#if defined(__unix__)
  constexpr size_t kRows = 1000, kCols = 200;
  TestDistributedQuantile(kRows, kCols);
#endif
}

TEST(Quantile, SameOnAllWorkers) {
#if defined(__unix__)
  std::string msg{"Skipping Quantile AllreduceBasic test"};
  int32_t constexpr kWorkers = 4;
  InitRabitContext(msg, kWorkers);
  auto world = rabit::GetWorldSize();
  if (world != 1) {
    CHECK_EQ(world, kWorkers);
  } else {
    LOG(WARNING) << msg;
    return;
  }

  constexpr size_t kRows = 1000, kCols = 100;
  RunWithSeedsAndBins(
      kRows, [=](int32_t seed, size_t n_bins, MetaInfo const &info) {
        auto rank = rabit::GetRank();
        HostDeviceVector<float> storage;
        auto m = RandomDataGenerator{kRows, kCols, 0}
                     .Device(0)
                     .Seed(rank + seed)
                     .GenerateDMatrix();
        auto cuts = SketchOnDMatrix(m.get(), n_bins);
        std::vector<float> cut_values(cuts.Values().size() * world, 0);
        std::vector<
            typename std::remove_reference_t<decltype(cuts.Ptrs())>::value_type>
            cut_ptrs(cuts.Ptrs().size() * world, 0);
        std::vector<float> cut_min_values(cuts.MinValues().size() * world, 0);

        size_t value_size = cuts.Values().size();
        rabit::Allreduce<rabit::op::Max>(&value_size, 1);
        size_t ptr_size = cuts.Ptrs().size();
        rabit::Allreduce<rabit::op::Max>(&ptr_size, 1);
        CHECK_EQ(ptr_size, kCols + 1);
        size_t min_value_size = cuts.MinValues().size();
        rabit::Allreduce<rabit::op::Max>(&min_value_size, 1);
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

        rabit::Allreduce<rabit::op::Sum>(cut_values.data(), cut_values.size());
        rabit::Allreduce<rabit::op::Sum>(cut_ptrs.data(), cut_ptrs.size());
        rabit::Allreduce<rabit::op::Sum>(cut_min_values.data(), cut_min_values.size());

        for (int32_t i = 0; i < world; i++) {
          for (size_t j = 0; j < value_size; ++j) {
            size_t idx = i * value_size + j;
            ASSERT_NEAR(cuts.Values().at(j), cut_values.at(idx), kRtEps);
          }

          for (size_t j = 0; j < ptr_size; ++j) {
            size_t idx = i * ptr_size + j;
            ASSERT_EQ(cuts.Ptrs().at(j), cut_ptrs.at(idx));
          }

          for (size_t j = 0; j < min_value_size; ++j) {
            size_t idx = i * min_value_size + j;
            ASSERT_EQ(cuts.MinValues().at(j), cut_min_values.at(idx));
          }
        }
      });
  rabit::Finalize();
#endif  // defined(__unix__)
}

TEST(CPUQuantile, FromOneHot) {
  std::vector<float> x = BasicOneHotEncodedData();
  auto m = GetDMatrixFromData(x, 5, 3);

  int32_t max_bins = 16;
  HistogramCuts cuts = SketchOnDMatrix(m.get(), max_bins);

  std::vector<uint32_t> const& h_cuts_ptr = cuts.Ptrs();
  std::vector<float> h_cuts_values = cuts.Values();
  ValidateBasicOneHot(h_cuts_ptr, h_cuts_values);
}
}  // namespace common
}  // namespace xgboost
