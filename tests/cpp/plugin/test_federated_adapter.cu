/*!
 * Copyright 2022 XGBoost contributors
 */
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <thread>
#include <ctime>

#include "./helpers.h"
#include "../../../plugin/federated/federated_communicator.h"
#include "../../../plugin/federated/federated_server.h"
#include "../../../src/collective/device_communicator_adapter.cuh"

namespace {

std::string GetServerAddress() {
  int port = GenerateRandomPort(50000, 60000);
  std::string address = std::string("localhost:") + std::to_string(port);
  return address;
}

}  // anonymous namespace

namespace xgboost {
namespace collective {

class FederatedAdapterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    server_address_ = GetServerAddress();
    server_thread_.reset(new std::thread([this] {
      grpc::ServerBuilder builder;
      federated::FederatedService service{kWorldSize};
      builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
      builder.RegisterService(&service);
      server_ = builder.BuildAndStart();
      server_->Wait();
    }));
  }

  void TearDown() override {
    server_->Shutdown();
    server_thread_->join();
  }

  static int const kWorldSize{2};
  std::string server_address_;
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<grpc::Server> server_;
};

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
    threads.emplace_back(std::thread([rank, server_address=server_address_] {
      FederatedCommunicator comm{kWorldSize, rank, server_address};
      // Assign device 0 to all workers, since we run gtest in a single-GPU machine
      DeviceCommunicatorAdapter adapter{0, &comm};
      int const count = 3;
      thrust::device_vector<double> buffer(count, 0);
      thrust::sequence(buffer.begin(), buffer.end());
      adapter.AllReduceSum(buffer.data().get(), count);
      thrust::host_vector<double> host_buffer = buffer;
      EXPECT_EQ(host_buffer.size(), count);
      for (auto i = 0; i < count; i++) {
        EXPECT_EQ(host_buffer[i], i * 2);
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedAdapterTest, DeviceAllGatherV) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread([rank, server_address=server_address_] {
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
      EXPECT_EQ(host_buffer.size(), 5);
      int expected[] = {0, 1, 0, 1, 2};
      for (auto i = 0; i < 5; i++) {
        EXPECT_EQ(host_buffer[i], expected[i]);
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace collective
}  // namespace xgboost
