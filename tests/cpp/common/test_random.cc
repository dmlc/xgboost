#include "../../../src/common/random.h"
#include "../helpers.h"
#include "gtest/gtest.h"

namespace xgboost {
namespace common {
TEST(ColumnSampler, Test) {
  int n = 128;
  ColumnSampler cs;

  // No node sampling
  cs.Init(n, 1.0f, 0.5f, 0.5f);
  auto set0 = *cs.GetFeatureSet(0);
  ASSERT_EQ(set0.Size(), 32);

  auto set1 = *cs.GetFeatureSet(0);

  ASSERT_EQ(set0.HostVector(), set1.HostVector());

  auto set2 = *cs.GetFeatureSet(1);
  ASSERT_NE(set1.HostVector(), set2.HostVector());
  ASSERT_EQ(set2.Size(), 32);

  // Node sampling
  cs.Init(n, 0.5f, 1.0f, 0.5f);
  auto set3 = *cs.GetFeatureSet(0);
  ASSERT_EQ(set3.Size(), 32);

  auto set4 = *cs.GetFeatureSet(0);

  ASSERT_NE(set3.HostVector(), set4.HostVector());
  ASSERT_EQ(set4.Size(), 32);

  // No level or node sampling, should be the same at different depth
  cs.Init(n, 1.0f, 1.0f, 0.5f);
  ASSERT_EQ(cs.GetFeatureSet(0)->HostVector(),
            cs.GetFeatureSet(1)->HostVector());

  cs.Init(n, 1.0f, 1.0f, 1.0f);
  auto set5 = *cs.GetFeatureSet(0);
  ASSERT_EQ(set5.Size(), n);
  cs.Init(n, 1.0f, 1.0f, 1.0f);
  auto set6 = *cs.GetFeatureSet(0);
  ASSERT_EQ(set5.HostVector(), set6.HostVector());

  // Should always be a minimum of one feature
  cs.Init(n, 1e-16f, 1e-16f, 1e-16f);
  ASSERT_EQ(cs.GetFeatureSet(0)->Size(), 1);
}

// Test if different threads using the same seed produce the same result
TEST(ColumnSampler, ThreadSynchronisation) {
  const int64_t num_threads = 10;
  int seed = 7;
  int n = 128;
  std::vector<std::vector<int>> results(num_threads);
#pragma omp parallel for schedule(static, 1)
  for (int64_t i = 0; i < num_threads; ++i) {
    ColumnSampler cs(seed);
    cs.Init(n, 0.5f, 0.5f, 0.5f);
    results.at(i) = cs.GetFeatureSet(0)->ConstHostVector();
  }
  for (int64_t i = 1; i < num_threads; ++i) {
    ASSERT_EQ(results.at(0), results.at(i));
  }
}
}  // namespace common
}  // namespace xgboost
