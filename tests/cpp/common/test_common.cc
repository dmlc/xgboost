#include "../../../src/common/common.h"
#include <gtest/gtest.h>

namespace xgboost {
TEST(GPUSet, Basic) {
  GPUSet devices = GPUSet::Empty();
  ASSERT_TRUE(devices.IsEmpty());

  devices = GPUSet{0, 1};
  ASSERT_TRUE(devices != GPUSet::Empty());
  EXPECT_EQ(devices.Size(), 1);

  devices = GPUSet::Range(1, 0);
  EXPECT_EQ(devices.Size(), 0);
  EXPECT_TRUE(devices.IsEmpty());

  EXPECT_FALSE(devices.Contains(1));

  devices = GPUSet::Range(2, -1);
  EXPECT_EQ(devices, GPUSet::Empty());
  EXPECT_EQ(devices.Size(), 0);
  EXPECT_TRUE(devices.IsEmpty());

  devices = GPUSet::Range(2, 8);  // 2 ~ 10
  EXPECT_EQ(devices.Size(), 8);
  EXPECT_ANY_THROW(devices.DeviceId(8));

  auto device_id = devices.DeviceId(0);
  EXPECT_EQ(device_id, 2);
  auto device_index = devices.Index(2);
  EXPECT_EQ(device_index, 0);

#ifndef XGBOOST_USE_CUDA
  EXPECT_EQ(GPUSet::AllVisible(), GPUSet::Empty());
#endif
}

TEST(Stoui, Parse) {
  std::string const unsigned_int = " 213124  ";
  size_t pos{0};
  uint32_t ures = common::stoui(unsigned_int, &pos);
  ASSERT_EQ(ures, 213124);
  ASSERT_EQ(pos, 7);

  // Same as stoui
  int32_t res = stoi(unsigned_int, &pos);
  ASSERT_EQ(res, 213124);
  ASSERT_EQ(pos, 7);
}

}  // namespace xgboost
