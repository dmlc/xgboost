#include "../../../src/common/common.h"
#include <gtest/gtest.h>

namespace xgboost {

TEST(GPUSet, GPUBasic) {
  GPUSet devices = GPUSet::Empty();
  ASSERT_TRUE(devices.IsEmpty());

  devices = GPUSet{0, 1};
  ASSERT_TRUE(devices != GPUSet::Empty());
  EXPECT_EQ(devices.Size(), 1);

  EXPECT_ANY_THROW(devices.Index(1));
  EXPECT_ANY_THROW(devices.Index(-1));

  devices = GPUSet::Range(1, 0);
  EXPECT_EQ(devices, GPUSet::Empty());
  EXPECT_EQ(devices.Size(), 0);
  EXPECT_TRUE(devices.IsEmpty());

  EXPECT_FALSE(devices.Contains(1));

  devices = GPUSet::Range(2, -1);
  EXPECT_EQ(devices, GPUSet::Empty());
  EXPECT_EQ(devices.Size(), 0);
  EXPECT_TRUE(devices.IsEmpty());

  devices = GPUSet::Range(2, 8);
  EXPECT_EQ(devices.Size(), 8);
  devices = devices.Unnormalised();

  EXPECT_EQ(*devices.begin(), 0);
  EXPECT_EQ(*devices.end(), devices.Size());
  EXPECT_EQ(8, devices.Size());

  ASSERT_NO_THROW(GPUSet::AllVisible());
  devices = GPUSet::AllVisible();
  if (devices.IsEmpty()) {
    LOG(WARNING) << "Empty devices.";
  }
}

}  // namespace xgboost
