/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <thread>

#include "federated_client.h"
#include "federated_server.h"

namespace xgboost {

class FederatedServerTest : public ::testing::Test {
 public:
  static void VerifyAllgather(int rank) {
    federated::FederatedClient client{kServerAddress, rank};
    CheckAllgather(client, rank);
  }

  static void VerifyAllreduce(int rank) {
    federated::FederatedClient client{kServerAddress, rank};
    CheckAllreduce(client);
  }

  static void VerifyBroadcast(int rank) {
    federated::FederatedClient client{kServerAddress, rank};
    CheckBroadcast(client, rank);
  }

  static void VerifyMixture(int rank) {
    federated::FederatedClient client{kServerAddress, rank};
    for (auto i = 0; i < 10; i++) {
      CheckAllgather(client, rank);
      CheckAllreduce(client);
      CheckBroadcast(client, rank);
    }
  }

 protected:
  void SetUp() override {
    server_thread_.reset(new std::thread([this] {
      grpc::ServerBuilder builder;
      federated::FederatedService service{kWorldSize};
      builder.AddListeningPort(kServerAddress, grpc::InsecureServerCredentials());
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
    auto reply = client.Allgather("hello " + std::to_string(rank) + " ");
    EXPECT_EQ(reply, "hello 0 hello 1 hello 2 ");
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
    EXPECT_EQ(reply, "hello broadcast");
  }

  static int const kWorldSize{3};
  static std::string const kServerAddress;
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<grpc::Server> server_;
};

std::string const FederatedServerTest::kServerAddress{"localhost:56789"};  // NOLINT(cert-err58-cpp)

TEST_F(FederatedServerTest, Allgather) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedServerTest::VerifyAllgather, rank));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedServerTest, Allreduce) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedServerTest::VerifyAllreduce, rank));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedServerTest, Broadcast) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedServerTest::VerifyBroadcast, rank));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedServerTest, Mixture) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedServerTest::VerifyMixture, rank));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace xgboost
