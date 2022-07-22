/*!
 * Copyright 2022 XGBoost contributors
 */
#ifdef XGBOOST_USE_NCCL

#include <gtest/gtest.h>

#include "../../../src/collective/nccl_device_communicator.cuh"

namespace xgboost {
namespace collective {

TEST(NcclDeviceCommunicatorSimpleTest, ThrowOnInvalidDeviceOrdinal) {
  auto construct = []() { NcclDeviceCommunicator comm{-1, nullptr}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(NcclDeviceCommunicatorSimpleTest, ThrowOnInvalidCommunicator) {
  auto construct = []() { NcclDeviceCommunicator comm{0, nullptr}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

}  // namespace collective
}  // namespace xgboost

#endif
