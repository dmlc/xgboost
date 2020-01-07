#include <valarray>
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
  auto set0 = cs.GetFeatureSet(0);
  ASSERT_EQ(set0->Size(), 32);

  auto set1 = cs.GetFeatureSet(0);

  ASSERT_EQ(set0->HostVector(), set1->HostVector());

  auto set2 = cs.GetFeatureSet(1);
  ASSERT_NE(set1->HostVector(), set2->HostVector());
  ASSERT_EQ(set2->Size(), 32);

  // Node sampling
  cs.Init(n, 0.5f, 1.0f, 0.5f);
  auto set3 = cs.GetFeatureSet(0);
  ASSERT_EQ(set3->Size(), 32);

  auto set4 = cs.GetFeatureSet(0);

  ASSERT_NE(set3->HostVector(), set4->HostVector());
  ASSERT_EQ(set4->Size(), 32);

  // No level or node sampling, should be the same at different depth
  cs.Init(n, 1.0f, 1.0f, 0.5f);
  ASSERT_EQ(cs.GetFeatureSet(0)->HostVector(),
            cs.GetFeatureSet(1)->HostVector());

  cs.Init(n, 1.0f, 1.0f, 1.0f);
  auto set5 = cs.GetFeatureSet(0);
  ASSERT_EQ(set5->Size(), n);
  cs.Init(n, 1.0f, 1.0f, 1.0f);
  auto set6 = cs.GetFeatureSet(0);
  ASSERT_EQ(set5->HostVector(), set6->HostVector());

  // Should always be a minimum of one feature
  cs.Init(n, 1e-16f, 1e-16f, 1e-16f);
  ASSERT_EQ(cs.GetFeatureSet(0)->Size(), 1);
}

// Test if different threads using the same seed produce the same result
TEST(ColumnSampler, ThreadSynchronisation) {
  const int64_t num_threads = 100;
  int n = 128;
  size_t iterations = 10;
  size_t levels = 5;
  std::vector<bst_feature_t> reference_result;
  bool success =
      true;  // Cannot use google test asserts in multithreaded region
#pragma omp parallel num_threads(num_threads)
  {
    for (auto j = 0ull; j < iterations; j++) {
      ColumnSampler cs(j);
      cs.Init(n, 0.5f, 0.5f, 0.5f);
      for (auto level = 0ull; level < levels; level++) {
        auto result = cs.GetFeatureSet(level)->ConstHostVector();
#pragma omp single
        { reference_result = result; }
        if (result != reference_result) {
          success = false;
        }
#pragma omp barrier
      }
    }
  }
  ASSERT_TRUE(success);
}
}  // namespace common
}  // namespace xgboost
