/*!
 * Copyright 2022 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <gtest/gtest.h>

#include <iostream>
#include <thread>

#include "../../../plugin/federated/federated_communicator.h"
#include "helpers.h"

namespace xgboost::collective {

class FederatedCommunicatorTest : public BaseFederatedTest {
 public:
  static void VerifyAllgather(int rank, const std::string &server_address) {
    FederatedCommunicator comm{kWorldSize, rank, server_address};
    CheckAllgather(comm, rank);
  }

  static void VerifyAllreduce(int rank, const std::string &server_address) {
    FederatedCommunicator comm{kWorldSize, rank, server_address};
    CheckAllreduce(comm);
  }

  static void VerifyBroadcast(int rank, const std::string &server_address) {
    FederatedCommunicator comm{kWorldSize, rank, server_address};
    CheckBroadcast(comm, rank);
  }

 protected:
  static void CheckAllgather(FederatedCommunicator &comm, int rank) {
    int buffer[kWorldSize] = {0, 0, 0};
    buffer[rank] = rank;
    comm.AllGather(buffer, sizeof(buffer));
    for (auto i = 0; i < kWorldSize; i++) {
      EXPECT_EQ(buffer[i], i);
    }
  }

  static void CheckAllreduce(FederatedCommunicator &comm) {
    int buffer[] = {1, 2, 3, 4, 5};
    comm.AllReduce(buffer, sizeof(buffer) / sizeof(buffer[0]), DataType::kInt32, Operation::kSum);
    int expected[] = {3, 6, 9, 12, 15};
    for (auto i = 0; i < 5; i++) {
      EXPECT_EQ(buffer[i], expected[i]);
    }
  }

  static void CheckBroadcast(FederatedCommunicator &comm, int rank) {
    if (rank == 0) {
      std::string buffer{"hello"};
      comm.Broadcast(&buffer[0], buffer.size(), 0);
      EXPECT_EQ(buffer, "hello");
    } else {
      std::string buffer{"     "};
      comm.Broadcast(&buffer[0], buffer.size(), 0);
      EXPECT_EQ(buffer, "hello");
    }
  }
};

TEST(FederatedCommunicatorSimpleTest, ThrowOnWorldSizeTooSmall) {
  auto construct = [] { FederatedCommunicator comm{0, 0, "localhost:0", "", "", ""}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnRankTooSmall) {
  auto construct = [] { FederatedCommunicator comm{1, -1, "localhost:0", "", "", ""}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnRankTooBig) {
  auto construct = [] { FederatedCommunicator comm{1, 1, "localhost:0", "", "", ""}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnWorldSizeNotInteger) {
  auto construct = [] {
    Json config{JsonObject()};
    config["federated_server_address"] = std::string("localhost:0");
    config["federated_world_size"] = std::string("1");
    config["federated_rank"] = Integer(0);
    FederatedCommunicator::Create(config);
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnRankNotInteger) {
  auto construct = [] {
    Json config{JsonObject()};
    config["federated_server_address"] = std::string("localhost:0");
    config["federated_world_size"] = 1;
    config["federated_rank"] = std::string("0");
    FederatedCommunicator::Create(config);
  };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, GetWorldSizeAndRank) {
  FederatedCommunicator comm{6, 3, "localhost:0"};
  EXPECT_EQ(comm.GetWorldSize(), 6);
  EXPECT_EQ(comm.GetRank(), 3);
}

TEST(FederatedCommunicatorSimpleTest, IsDistributed) {
  FederatedCommunicator comm{2, 1, "localhost:0"};
  EXPECT_TRUE(comm.IsDistributed());
}

TEST_F(FederatedCommunicatorTest, Allgather) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(&FederatedCommunicatorTest::VerifyAllgather, rank, server_->Address());
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedCommunicatorTest, Allreduce) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(&FederatedCommunicatorTest::VerifyAllreduce, rank, server_->Address());
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedCommunicatorTest, Broadcast) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(&FederatedCommunicatorTest::VerifyBroadcast, rank, server_->Address());
  }
  for (auto &thread : threads) {
    thread.join();
  }
}
}  // namespace xgboost::collective
