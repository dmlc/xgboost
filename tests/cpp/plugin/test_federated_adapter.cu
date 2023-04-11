/*!
 * Copyright 2022 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/host_vector.h>

#include <ctime>
#include <iostream>
#include <thread>

#include "../../../plugin/federated/federated_communicator.h"
#include "../../../src/collective/device_communicator_adapter.cuh"
#include "./helpers.h"

namespace xgboost::collective {

class FederatedAdapterTest : public BaseFederatedTest {};

TEST(FederatedAdapterSimpleTest, ThrowOnInvalidDeviceOrdinal) {
  auto construct = []() { DeviceCommunicatorAdapter adapter{-1, nullptr}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedAdapterSimpleTest, ThrowOnInvalidCommunicator) {
  auto construct = []() { DeviceCommunicatorAdapter adapter{0, nullptr}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST_F(FederatedAdapterTest, DeviceAllReduceSum) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back([rank, server_address = server_->Address()] {
      FederatedCommunicator comm{kWorldSize, rank, server_address};
      // Assign device 0 to all workers, since we run gtest in a single-GPU machine
      DeviceCommunicatorAdapter adapter{0, &comm};
      int count = 3;
      thrust::device_vector<double> buffer(count, 0);
      thrust::sequence(buffer.begin(), buffer.end());
      adapter.AllReduceSum(buffer.data().get(), count);
      thrust::host_vector<double> host_buffer = buffer;
      EXPECT_EQ(host_buffer.size(), count);
      for (auto i = 0; i < count; i++) {
        EXPECT_EQ(host_buffer[i], i * kWorldSize);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedAdapterTest, DeviceAllGatherV) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back([rank, server_address = server_->Address()] {
      FederatedCommunicator comm{kWorldSize, rank, server_address};
      // Assign device 0 to all workers, since we run gtest in a single-GPU machine
      DeviceCommunicatorAdapter adapter{0, &comm};

      int const count = rank + 2;
      thrust::device_vector<char> buffer(count, 0);
      thrust::sequence(buffer.begin(), buffer.end());
      std::vector<std::size_t> segments(kWorldSize);
      dh::caching_device_vector<char> receive_buffer{};

      adapter.AllGatherV(buffer.data().get(), count, &segments, &receive_buffer);

      EXPECT_EQ(segments[0], 2);
      EXPECT_EQ(segments[1], 3);
      thrust::host_vector<char> host_buffer = receive_buffer;
      EXPECT_EQ(host_buffer.size(), 9);
      int expected[] = {0, 1, 0, 1, 2, 0, 1, 2, 3};
      for (auto i = 0; i < 9; i++) {
        EXPECT_EQ(host_buffer[i], expected[i]);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace xgboost::collective
