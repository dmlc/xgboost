#include "gtest/gtest.h"
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/monitor.h"

TEST(Monitor, MemStatGlobalUsage) {
  using DeviceMemoryStat = xgboost::common::DeviceMemoryStat;
  DeviceMemoryStat::Ins().Reset();
  DeviceMemoryStat::Ins().SetProfiling(true);
  dh::DeviceVector<float> vec (8);
  float *ptr;
  dh::ProfilingCudaMalloc((void**)&ptr, sizeof(float)*12);

  size_t count = DeviceMemoryStat::Ins().GetAllocationCount();
  size_t peak_usage = DeviceMemoryStat::Ins().GetPeakUsage();

  ASSERT_EQ(count, 2);
  ASSERT_GE(peak_usage, sizeof(float) * 20);

  auto usage = DeviceMemoryStat::Ins().GetPtrUsage(ptr);
  ASSERT_EQ(usage.GetPeak(), sizeof(float)*12);
  ASSERT_EQ(usage.GetAllocCount(), 1);
  ASSERT_EQ(usage.GetRunningSum(), sizeof(float)*12);

  DeviceMemoryStat::Ins().SetProfiling(false);
}

TEST(Monitor, MemStatPtrUsage) {
  using Stat = xgboost::common::DeviceMemoryStat;
  Stat::Ins().Reset();
  Stat::Ins().SetProfiling(true);

  void* ptr0 = (void*)0x10;
  Stat::Ins().Allocate(ptr0, 16);

  auto usage = Stat::Ins().GetPtrUsage(ptr0);
  ASSERT_EQ(usage.GetPeak(), 16);
  ASSERT_EQ(usage.GetAllocCount(), 1);
  ASSERT_EQ(usage.GetRunningSum(), 16);

  Stat::Ins().Deallocate(ptr0, 12);
  usage = Stat::Ins().GetPtrUsage(ptr0);
  ASSERT_EQ(usage.GetPeak(), 16);
  ASSERT_EQ(usage.GetRunningSum(), 4);

  void* ptr1 = (void*)0x20;
  Stat::Ins().Allocate(ptr1, 32);
  Stat::Ins().Replace(ptr0, ptr1);

  usage = Stat::Ins().GetPtrUsage(ptr0);
  ASSERT_EQ(usage.GetPeak(), 32);
  ASSERT_EQ(usage.GetRunningSum(), 32);

  Stat::Ins().SetProfiling(false);
}