/**
 * Copyright 2022-2023, XGBoost contributors
 */
#ifdef XGBOOST_USE_NCCL

#include <gtest/gtest.h>

#include <bitset>
#include <string>  // for string

#include "../../../src/collective/comm.cuh"
#include "../../../src/collective/communicator-inl.cuh"
#include "../../../src/collective/nccl_device_communicator.cuh"
#include "../helpers.h"

namespace xgboost {
namespace collective {

TEST(NcclDeviceCommunicatorSimpleTest, ThrowOnInvalidDeviceOrdinal) {
  auto construct = []() { NcclDeviceCommunicator comm{-1, false, DefaultNcclName()}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(NcclDeviceCommunicatorSimpleTest, SystemError) {
  auto stub = std::make_shared<NcclStub>(DefaultNcclName());
  auto rc = stub->GetNcclResult(ncclSystemError);
  auto msg = rc.Report();
  ASSERT_TRUE(msg.find("environment variables") != std::string::npos);
}

namespace {
void VerifyAllReduceBitwiseAND() {
  auto const rank = collective::GetRank();
  std::bitset<64> original{};
  original[rank] = true;
  HostDeviceVector<uint64_t> buffer({original.to_ullong()}, DeviceOrd::CUDA(rank));
  collective::AllReduce<collective::Operation::kBitwiseAND>(rank, buffer.DevicePointer(), 1);
  collective::Synchronize(rank);
  EXPECT_EQ(buffer.HostVector()[0], 0ULL);
}
}  // anonymous namespace

TEST(NcclDeviceCommunicator, MGPUAllReduceBitwiseAND) {
  auto const n_gpus = common::AllVisibleGPUs();
  if (n_gpus <= 1) {
    GTEST_SKIP() << "Skipping MGPUAllReduceBitwiseAND test with # GPUs = " << n_gpus;
  }
  auto constexpr kUseNccl = true;
  RunWithInMemoryCommunicator<kUseNccl>(n_gpus, VerifyAllReduceBitwiseAND);
}

namespace {
void VerifyAllReduceBitwiseOR() {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  std::bitset<64> original{};
  original[rank] = true;
  HostDeviceVector<uint64_t> buffer({original.to_ullong()}, DeviceOrd::CUDA(rank));
  collective::AllReduce<collective::Operation::kBitwiseOR>(rank, buffer.DevicePointer(), 1);
  collective::Synchronize(rank);
  EXPECT_EQ(buffer.HostVector()[0], (1ULL << world_size) - 1);
}
}  // anonymous namespace

TEST(NcclDeviceCommunicator, MGPUAllReduceBitwiseOR) {
  auto const n_gpus = common::AllVisibleGPUs();
  if (n_gpus <= 1) {
    GTEST_SKIP() << "Skipping MGPUAllReduceBitwiseOR test with # GPUs = " << n_gpus;
  }
  auto constexpr kUseNccl = true;
  RunWithInMemoryCommunicator<kUseNccl>(n_gpus, VerifyAllReduceBitwiseOR);
}

namespace {
void VerifyAllReduceBitwiseXOR() {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  std::bitset<64> original{~0ULL};
  original[rank] = false;
  HostDeviceVector<uint64_t> buffer({original.to_ullong()}, DeviceOrd::CUDA(rank));
  collective::AllReduce<collective::Operation::kBitwiseXOR>(rank, buffer.DevicePointer(), 1);
  collective::Synchronize(rank);
  EXPECT_EQ(buffer.HostVector()[0], (1ULL << world_size) - 1);
}
}  // anonymous namespace

TEST(NcclDeviceCommunicator, MGPUAllReduceBitwiseXOR) {
  auto const n_gpus = common::AllVisibleGPUs();
  if (n_gpus <= 1) {
    GTEST_SKIP() << "Skipping MGPUAllReduceBitwiseXOR test with # GPUs = " << n_gpus;
  }
  auto constexpr kUseNccl = true;
  RunWithInMemoryCommunicator<kUseNccl>(n_gpus, VerifyAllReduceBitwiseXOR);
}

}  // namespace collective
}  // namespace xgboost

#endif  // XGBOOST_USE_NCCL
