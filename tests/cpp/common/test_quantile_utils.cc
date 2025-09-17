/**
 * Copyright 2023 by XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/common/quantile_loss_utils.h"
#include "xgboost/base.h"  // Args

namespace xgboost {
namespace common {
TEST(QuantileLossParam, Basic) {
  QuantileLossParam param;
  auto& ref = param.quantile_alpha.Get();

  param.UpdateAllowUnknown(Args{{"quantile_alpha", "0.3"}});
  ASSERT_EQ(ref.size(), 1);
  ASSERT_NEAR(ref[0], 0.3, kRtEps);

  param.UpdateAllowUnknown(Args{{"quantile_alpha", "[0.3, 0.6]"}});
  ASSERT_EQ(param.quantile_alpha.Get().size(), 2);
  ASSERT_NEAR(ref[0], 0.3, kRtEps);
  ASSERT_NEAR(ref[1], 0.6, kRtEps);

  param.UpdateAllowUnknown(Args{{"quantile_alpha", "(0.6, 0.3)"}});
  ASSERT_EQ(param.quantile_alpha.Get().size(), 2);
  ASSERT_NEAR(ref[0], 0.6, kRtEps);
  ASSERT_NEAR(ref[1], 0.3, kRtEps);
}
}  // namespace common
}  // namespace xgboost
