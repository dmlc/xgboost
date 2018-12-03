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
  auto set0 = cs.GetFeatureSet(0)->HostVector();
  ASSERT_EQ(set0.size(), 32);

  auto set1 = cs.GetFeatureSet(0)->HostVector();
  ASSERT_EQ(set0, set1);

  auto set2 = cs.GetFeatureSet(1)->HostVector();
  ASSERT_NE(set1, set2);
  ASSERT_EQ(set2.size(), 32);

  // Node sampling
  cs.Init(n, 0.5f, 1.0f, 0.5f);
  auto set3 = cs.GetFeatureSet(0)->HostVector();
  ASSERT_EQ(set3.size(), 32);

  auto set4 = cs.GetFeatureSet(0)->HostVector();
  ASSERT_NE(set3, set4);
  ASSERT_EQ(set4.size(), 32);

  // No level or node sampling, should be the same at different depth
  cs.Init(n, 1.0f, 1.0f, 0.5f);
  ASSERT_EQ(cs.GetFeatureSet(0)->HostVector(), cs.GetFeatureSet(1)->HostVector());

  cs.Init(n, 1.0f, 1.0f, 1.0f);
  auto set5 = cs.GetFeatureSet(0)->HostVector();
  ASSERT_EQ(set5.size(), n);
  cs.Init(n, 1.0f, 1.0f, 1.0f);
  auto set6 = cs.GetFeatureSet(0)->HostVector();
  ASSERT_EQ(set5, set6);

  // Should always be a minimum of one feature
  cs.Init(n, 1e-16f, 1e-16f, 1e-16f);
  ASSERT_EQ(cs.GetFeatureSet(0)->HostVector().size(), 1);

}
}  // namespace common
}  // namespace xgboost
