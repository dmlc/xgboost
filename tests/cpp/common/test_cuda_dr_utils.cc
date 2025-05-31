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

TEST(DrUtils, GetC2cLinkCountFromSmi) {
  {
    auto out = R"(GPU 0: NVIDIA GH200 480GB (UUID: GPU-********-****-****-****-************)
    C2C Link 0: 44.712 GB/s
    C2C Link 1: 44.712 GB/s
    C2C Link 2: 44.712 GB/s
    C2C Link 3: 44.712 GB/s
    C2C Link 4: 44.712 GB/s
    C2C Link 5: 44.712 GB/s
    C2C Link 6: 44.712 GB/s
    C2C Link 7: 44.712 GB/s
    C2C Link 8: 44.712 GB/s
    C2C Link 9: 44.712 GB/s
  )";
    auto lc = detail::GetC2cLinkCountFromSmiImpl(out);
    ASSERT_EQ(lc, 10);
  }
  {
    auto out = R"(No Devices support C2C.
)";
    auto lc = detail::GetC2cLinkCountFromSmiImpl(out);
    ASSERT_EQ(lc, -1);
  }

  {
    [[maybe_unused]] auto _ = GetC2cLinkCountFromSmi();
  }
  {
    [[maybe_unused]] auto _ = GetC2cLinkCountFromSmiGlobal();
  }
}
}  // namespace xgboost::cudr
#endif  // defined(XGBOOST_USE_CUDA)
