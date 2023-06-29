/*!
 * Copyright 2022 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/host_vector.h>

#include <ctime>
#include <iostream>
#include <thread>

#include "../../../plugin/federated/federated_communicator.h"
#include "../../../src/collective/communicator-inl.cuh"
#include "../../../src/collective/device_communicator_adapter.cuh"
#include "./helpers.h"

namespace xgboost::collective {

class FederatedAdapterTest : public BaseFederatedTest {};

TEST(FederatedAdapterSimpleTest, ThrowOnInvalidDeviceOrdinal) {
  auto construct = []() { DeviceCommunicatorAdapter adapter{-1}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

namespace {
void VerifyAllReduceSum() {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  int count = 3;
  thrust::device_vector<double> buffer(count, 0);
  thrust::sequence(buffer.begin(), buffer.end());
  collective::AllReduce<collective::Operation::kSum>(rank, buffer.data().get(), count);
  thrust::host_vector<double> host_buffer = buffer;
  EXPECT_EQ(host_buffer.size(), count);
  for (auto i = 0; i < count; i++) {
    EXPECT_EQ(host_buffer[i], i * world_size);
  }
}
}  // anonymous namespace

TEST_F(FederatedAdapterTest, MGPUAllReduceSum) {
  auto const n_gpus = common::AllVisibleGPUs();
  if (n_gpus <= 1) {
    GTEST_SKIP() << "Skipping MGPUAllReduceSum test with # GPUs = " << n_gpus;
  }
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyAllReduceSum);
}

namespace {
void VerifyAllGatherV() {
  auto const world_size = collective::GetWorldSize();
  auto const rank = collective::GetRank();
  int const count = rank + 2;
  thrust::device_vector<char> buffer(count, 0);
  thrust::sequence(buffer.begin(), buffer.end());
  std::vector<std::size_t> segments(world_size);
  dh::caching_device_vector<char> receive_buffer{};

  collective::AllGatherV(rank, buffer.data().get(), count, &segments, &receive_buffer);

  EXPECT_EQ(segments[0], 2);
  EXPECT_EQ(segments[1], 3);
  thrust::host_vector<char> host_buffer = receive_buffer;
  EXPECT_EQ(host_buffer.size(), 5);
  int expected[] = {0, 1, 0, 1, 2};
  for (auto i = 0; i < 5; i++) {
    EXPECT_EQ(host_buffer[i], expected[i]);
  }
}
}  // anonymous namespace

TEST_F(FederatedAdapterTest, MGPUAllGatherV) {
  auto const n_gpus = common::AllVisibleGPUs();
  if (n_gpus <= 1) {
    GTEST_SKIP() << "Skipping MGPUAllGatherV test with # GPUs = " << n_gpus;
  }
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyAllGatherV);
}

}  // namespace xgboost::collective
