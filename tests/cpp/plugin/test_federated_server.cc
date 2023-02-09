/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <ctime>
#include <iostream>
#include <thread>

#include "federated_client.h"
#include "federated_server.h"
#include "helpers.h"

namespace {

std::string GetServerAddress() {
  int port = GenerateRandomPort(50000, 60000);
  std::string address = std::string("localhost:") + std::to_string(port);
  return address;
}

}  // anonymous namespace

namespace xgboost {

class FederatedServerTest : public ::testing::Test {
 public:
  static void VerifyAllgather(int rank, const std::string& server_address) {
    federated::FederatedClient client{server_address, rank};
    CheckAllgather(client, rank);
  }

  static void VerifyAllreduce(int rank, const std::string& server_address) {
    federated::FederatedClient client{server_address, rank};
    CheckAllreduce(client);
  }

  static void VerifyBroadcast(int rank, const std::string& server_address) {
    federated::FederatedClient client{server_address, rank};
    CheckBroadcast(client, rank);
  }

  static void VerifyMixture(int rank, const std::string& server_address) {
    federated::FederatedClient client{server_address, rank};
    for (auto i = 0; i < 10; i++) {
      CheckAllgather(client, rank);
      CheckAllreduce(client);
      CheckBroadcast(client, rank);
    }
  }

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

  static void CheckAllgather(federated::FederatedClient& client, int rank) {
    int data[kWorldSize] = {0, 0, 0};
    data[rank] = rank;
    std::string send_buffer(reinterpret_cast<char const*>(data), sizeof(data));
    auto reply = client.Allgather(send_buffer);
    auto const* result = reinterpret_cast<int const*>(reply.data());
    for (auto i = 0; i < kWorldSize; i++) {
      EXPECT_EQ(result[i], i);
    }
  }

  static void CheckAllreduce(federated::FederatedClient& client) {
    int data[] = {1, 2, 3, 4, 5};
    std::string send_buffer(reinterpret_cast<char const*>(data), sizeof(data));
    auto reply = client.Allreduce(send_buffer, federated::INT32, federated::SUM);
    auto const* result = reinterpret_cast<int const*>(reply.data());
    int expected[] = {3, 6, 9, 12, 15};
    for (auto i = 0; i < 5; i++) {
      EXPECT_EQ(result[i], expected[i]);
    }
  }

  static void CheckBroadcast(federated::FederatedClient& client, int rank) {
    std::string send_buffer{};
    if (rank == 0) {
      send_buffer = "hello broadcast";
    }
    auto reply = client.Broadcast(send_buffer, 0);
    EXPECT_EQ(reply, "hello broadcast") << "rank " << rank;
  }

  static int const kWorldSize{3};
  std::string server_address_;
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<grpc::Server> server_;
};

TEST_F(FederatedServerTest, Allgather) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedServerTest::VerifyAllgather, rank, server_address_));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedServerTest, Allreduce) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedServerTest::VerifyAllreduce, rank, server_address_));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedServerTest, Broadcast) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedServerTest::VerifyBroadcast, rank, server_address_));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedServerTest, Mixture) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedServerTest::VerifyMixture, rank, server_address_));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace xgboost
