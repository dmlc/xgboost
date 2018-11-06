#include "../../../src/common/common.h"
#include <gtest/gtest.h>

namespace xgboost {

TEST(GPUSet, GPUBasic) {
  GPUSet devices = GPUSet::Empty();
  ASSERT_TRUE(devices.IsEmpty());

  devices = GPUSet{1, 1};
  ASSERT_TRUE(devices != GPUSet::Empty());
  EXPECT_EQ(devices.Size(), 1);
  EXPECT_EQ(*(devices.begin()), 1);

  devices = GPUSet::Range(1, 0);
  EXPECT_EQ(devices, GPUSet::Empty());
  EXPECT_EQ(devices.Size(), 0);
  EXPECT_TRUE(devices.IsEmpty());

  EXPECT_FALSE(devices.Contains(1));

  devices = GPUSet::Range(2, -1);
  EXPECT_EQ(devices, GPUSet::Empty());

  devices = GPUSet::Range(2, 8);
  EXPECT_EQ(devices.Size(), 8);

  EXPECT_EQ(*devices.begin(), 2);
  EXPECT_EQ(*devices.end(), 2 + devices.Size());
  EXPECT_EQ(8, devices.Size());

  ASSERT_NO_THROW(GPUSet::AllVisible());
  devices = GPUSet::AllVisible();
  if (devices.IsEmpty()) {
    LOG(WARNING) << "Empty devices.";
  }
}

#if defined(XGBOOST_USE_NCCL)
TEST(GPUSet, MGPU_GPUBasic) {
  {
    GPUSet devices = GPUSet::All(1, 1);
    ASSERT_EQ(*(devices.begin()), 1);
    ASSERT_EQ(*(devices.end()), 2);
    ASSERT_EQ(devices.Size(), 1);
    ASSERT_TRUE(devices.Contains(1));
  }

  {
    GPUSet devices = GPUSet::All(0, -1);
    ASSERT_GE(devices.Size(), 2);
  }

  // Specify number of rows.
  {
    GPUSet devices = GPUSet::All(0, -1, 1);
    ASSERT_EQ(devices.Size(), 1);
  }
}
#endif

}  // namespace xgboost