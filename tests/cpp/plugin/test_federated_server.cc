/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <gtest/gtest.h>

#include <iostream>
#include <thread>

#include "federated_client.h"
#include "helpers.h"

namespace xgboost {

class FederatedServerTest : public BaseFederatedTest {
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
};

TEST_F(FederatedServerTest, Allgather) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(&FederatedServerTest::VerifyAllgather, rank, server_->Address());
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedServerTest, Allreduce) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(&FederatedServerTest::VerifyAllreduce, rank, server_->Address());
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedServerTest, Broadcast) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(&FederatedServerTest::VerifyBroadcast, rank, server_->Address());
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedServerTest, Mixture) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(&FederatedServerTest::VerifyMixture, rank, server_->Address());
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace xgboost
