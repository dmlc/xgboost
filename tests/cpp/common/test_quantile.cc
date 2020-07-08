#include <gtest/gtest.h>
#include "test_quantile.h"
#include "../../../src/common/quantile.h"
#include "../../../src/common/hist_util.h"

namespace xgboost {
namespace common {
TEST(Quantile, SameOnAllWorkers) {
  std::string msg{"Skipping Quantile AllreduceBasic test"};
  size_t constexpr kWorkers = 4;
  InitRabitContext(msg, kWorkers);
  auto world = rabit::GetWorldSize();
  if (world != 1) {
    CHECK_EQ(world, kWorkers);
  } else {
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
}
}  // namespace common
}  // namespace xgboost
