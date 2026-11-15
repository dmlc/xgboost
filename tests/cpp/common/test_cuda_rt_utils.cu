/**
 * Copyright 2025-2026, XGBoost contributors
 */

#include <gtest/gtest.h>

#include <array>  // for array

#include "../../../src/common/cuda_rt_utils.h"

namespace xgboost::curt {
TEST(RtUtils, Uuid) {
  std::array<unsigned char, kUuidLength> uuid;
  GetUuid(uuid, 0);
  auto str = PrintUuid(uuid);
  ASSERT_EQ(str.substr(0, 4), "GPU-");
  ASSERT_EQ(str.length(), 40);
  ASSERT_EQ(str.size(), str.length());
}
}  // namespace xgboost::curt
