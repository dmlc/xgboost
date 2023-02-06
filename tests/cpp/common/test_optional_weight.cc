/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>             // Context
#include <xgboost/host_device_vector.h>  // HostDeviceVector

#include "../../../src/common/optional_weight.h"
namespace xgboost {
namespace common {
TEST(OptionalWeight, Basic) {
  HostDeviceVector<float> weight{{2.0f, 3.0f, 4.0f}};
  Context ctx;
  auto opt_w = MakeOptionalWeights(&ctx, weight);
  ASSERT_EQ(opt_w[0], 2.0f);
  ASSERT_FALSE(opt_w.Empty());

  weight.HostVector().clear();
  opt_w = MakeOptionalWeights(&ctx, weight);
  ASSERT_EQ(opt_w[0], 1.0f);
  ASSERT_TRUE(opt_w.Empty());
}
}  // namespace common
}  // namespace xgboost
