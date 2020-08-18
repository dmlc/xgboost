#include <gtest/gtest.h>
#include "../../../src/common/common.h"

namespace xgboost {
namespace common {
TEST(ArgSort, Basic) {
  std::vector<float> inputs {3.0, 2.0, 1.0};
  auto ret = ArgSort<bst_feature_t>(inputs);
  std::vector<bst_feature_t> sol{2, 1, 0};
  ASSERT_EQ(ret, sol);
}
}  // namespace common
}  // namespace xgboost
