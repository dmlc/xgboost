/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>

#if defined(XGBOOST_USE_CUDA) && defined(__linux__)
#include "../../../src/common/cuda_dr_utils.h"

namespace xgboost::cudr {
TEST(DrUtils, GetVersionFromSmi) {
  std::int32_t major = 0, minor = 0;
  bool result = GetVersionFromSmi(&major, &minor);

  if (result) {
    EXPECT_GE(major, 0);
    EXPECT_GE(minor, 0);
  } else {
    EXPECT_EQ(major, -1);
    EXPECT_EQ(minor, -1);
  }
}
}  // namespace xgboost::cudr
#endif  // defined(XGBOOST_USE_CUDA)
